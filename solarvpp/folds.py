"""Robust fold test: fold별 층화 배정과 champion 대비 네 조건 판정, 배정 후보의 fold 탐색·비교.

Fold 이름·기간과 네 조건 판정(judge)·계절 규칙은 공통 계약층(vppcontract)의 정의를 쓴다.

Fold는 protocol의 V·H와 config.FOLDS이고, 배정은 vppcontract.partition의 기준 층화 배정을 fold cohort에 적용한다.
판정은 fold별 W, seed별 다섯 fold W 합, 계절별 W로 계산한다. screening level(seed 하나)의 보고는 같은 표에 합산 W 순위를 붙이며,
선별은 개선된 후보 중 상위 몇 개를 고르고 네 조건은 confirmation에서 판정한다. 참조: docs/solarvpp-reference.md#robust-folds.

배정 후보는 champion 모델을 그대로 두고 배정만 바꾼다. fold마다 cutoff 이전 정보로 탐색한 배정을 partitions/<label>-<fold>.csv에
쓰고(search_partition), champion을 그 배정으로 다시 학습한 run을 층화 배정 run과 같은 네 조건으로 비교한다(partition_report).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vppcontract.evaluate import ROBUST_SEEDS, judge
from vppcontract.partition import minimum_share, repair, stratified_split
from vppcontract.timeaxis import FOLD_PERIODS, SEASONS, season_index

from .bundling.partition import apply_partition
from .config import Spec
from .data import load_sites
from .experiment import EXPERTS, load_data, log, model_signature, settings_hash, write_report
from .forecast.queries import actual_view
from .store import digest, file_sha, read_json, versions, write_json

# fold 배정 탐색의 추정 기간: fold cutoff 직전 12개월의 발전소 OOT 오차를 주는 기간. V·H는 protocol의 D·V이다.
FOLD_ESTIMATION = {"V": "D", "fold-2309": "fold-2209", "fold-2312": "fold-2212", "fold-2403": "fold-2303", "H": "V"}
SEARCH_SECTIONS = ("primitive", "forecast_input", "risk", "unconditional", "analog", "scenarios", "optimization")


def fold_stratified_partition(spec: Spec, period_name: str, data=None) -> tuple[pd.DataFrame, np.ndarray]:
    """fold cohort(cutoff까지 유효 generation이 있는 발전소)를 기준 층화 배정으로 나눈 배정 표 [site_id, group]와 cohort mask."""
    data = data if data is not None else load_data(spec)
    sites, base = data.sites, data.base
    view = actual_view(np.asarray(base.energy), np.asarray(base.status), spec.period(period_name).cutoff_h)
    cohort = view.valid[:view.cutoff_h + 1].any(axis=0)
    keys = np.stack([sites.station_index["wa"], sites.station_index["wb"]], axis=1)
    capacity, first = sites.capacity, np.zeros(len(sites), dtype=bool)
    inside, outside = np.flatnonzero(cohort), np.flatnonzero(~cohort)
    first[inside] = repair(stratified_split(capacity[inside], keys[inside]), capacity[inside], minimum_share(float(capacity.sum())))
    if len(outside):                                        # 학습·평가에 쓰이지 않는 발전소도 같은 규칙으로 따로 나눈다
        first[outside] = stratified_split(capacity[outside], keys[outside])
    return pd.DataFrame({"site_id": sites.ids, "group": np.where(first, "G1", "G2")}), cohort


def write_partitions(spec: Spec, periods) -> dict:
    """artifact의 partitions/stratified-<fold>.csv를 쓴다. 이미 있으면 덮어쓰지 않고 cohort 발전소의 배정이 같은지 대조한다."""
    data, out = load_data(spec), {}
    for name in periods:
        frame, cohort = fold_stratified_partition(spec, name, data)
        groups = frame.group.to_numpy()
        path = spec.artifact_dir / "partitions" / f"stratified-{name}.csv"
        record = {"path": str(path), "cohort_sites": int(cohort.sum()),
                  "cohort_capacity_kw": {g: float(data.sites.capacity[cohort & (groups == g)].sum()) for g in ("G1", "G2")}}
        if path.exists():
            old = pd.read_csv(path).set_index("site_id").loc[frame.site_id, "group"].to_numpy()
            record.update(written=False, cohort_same=bool((old[cohort] == groups[cohort]).all()))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False, lineterminator="\n")
            record["written"] = True
        out[name] = record
    return out




def fold_runs(spec: Spec, period_name: str, seeds, level: str = "confirmation", partition: str = "stratified") -> dict:
    """fold의 배정 <partition>-<fold>.csv로 돌린 정식 비교 run을 DL 입력별로 모은다. 모델은 champion과 DL 입력만 다를 수 있다.

    정식 비교 run은 계절 학습 run이 아니고, level·seed 목록이 요청과 같고, XGB·DL 두 expert를 모두 돌려 blend를 만들었고, 실행·평가가 끝났고, 학습 예산·분위수·
    블렌드 설정(settings_hash)이 현재와 같고, 배정 파일의 현재 내용과 같은 배정으로 학습한 run이다. 같은 조건의 run이 여럿이면 가장
    늦게 끝난 run을 쓴다.
    """
    path = spec.artifact_dir / "partitions" / f"{partition}-{period_name}.csv"
    if not path.exists():
        return {}
    expected = {"signature": model_signature(spec, spec.champion), "seeds": [int(s) for s in seeds],
                "settings": settings_hash(spec), "partition": apply_partition(load_sites(spec), path)[1]["hash"]}
    found = []
    for run in (spec.artifact_dir / "runs" / period_name).glob("*"):
        if not (run / "manifest.json").exists() or not (run / "report.json").exists():
            continue
        manifest = read_json(run / "manifest.json")
        model = manifest.get("model") or {}
        record = manifest.get("partition") or {}
        if (manifest.get("level") == level and manifest.get("seeds") == expected["seeds"] and not manifest.get("season")
                and manifest.get("execution_status") == "completed" and manifest.get("evaluation_status") == "evaluated"
                and set(manifest.get("experts", [])) == set(EXPERTS) and manifest.get("settings_hash") == expected["settings"]
                and record.get("path") and Path(record["path"]).name == path.name and record.get("hash") == expected["partition"]
                and model == {**expected["signature"], "dl_input": model.get("dl_input")}):
            found.append((manifest.get("finished", ""), model["dl_input"], run))
    return {dl_input: run for _, dl_input, run in sorted(found, key=lambda item: item[0])}


def _measure(spec: Spec, period_name: str, run: Path) -> dict:
    """정식 비교 run의 blend W, seed별 blend W(seed 하나인 run은 결합 예측 W), 계절별 W, 실제 발전량 수익.
    blend 성분은 manifest의 칸 이름으로 찾고, XGB·DL 성분이 모두 있는 blend run이 아니면 멈춘다."""
    report, manifest = read_json(run / "report.json"), read_json(run / "manifest.json")
    label = manifest.get("cell")
    if label not in report["components"] or not {"xgb_component", "dl_component"} <= set(report["components"]):
        raise ValueError(f"{run}은 XGB·DL 두 expert의 blend run이 아닙니다.")
    daily = np.load(run / "daily_penalty.npz")[label].sum(axis=1)
    days = spec.period(period_name).days
    if len(daily) != len(days):
        raise ValueError(f"{run}의 일별 W가 평가 기간과 맞지 않습니다.")
    season = season_index(days)
    total = float(report["components"][label]["W_cohort_krw"])
    seeds = ({s: float(v["blend"]) for s, v in report["per_seed_W_krw"].items()} if "per_seed_W_krw" in report
             else {str(s): total for s in manifest["seeds"]})
    return {"W": total, "seeds": seeds, "revenue": float(report["components"][label].get("revenue_krw", np.nan)),
            "seasons": {k: float(daily[season == i].sum()) for i, k in enumerate(SEASONS)}}


def _collect(measures: dict, folds: list[str], seeds) -> dict:
    """fold별 측정값을 judge 입력(fold W, seed별 fold W, 계절 W 합)으로 모은다."""
    return {"folds": {f: measures[f]["W"] for f in folds}, "seeds": {str(s): {f: measures[f]["seeds"][str(s)] for f in folds} for s in seeds},
            "seasons": {k: sum(measures[f]["seasons"][k] for f in folds) for k in SEASONS}}


def fold_report(spec: Spec, candidates, seeds=ROBUST_SEEDS, level: str = "confirmation") -> dict:
    """fold마다 champion과 후보의 완료 run을 찾아 네 조건을 판정하고 보고서로 저장한다. 빠진 run은 missing에 적고, 다섯 fold가
    모두 있는 후보를 합산 W 변화로 줄 세운다(ranking)."""
    champion = model_signature(spec, spec.champion)["dl_input"]
    runs = {f: fold_runs(spec, f, seeds, level) for f in FOLD_PERIODS}
    names = [champion, *candidates]
    measures = {n: {f: _measure(spec, f, runs[f][n]) for f in FOLD_PERIODS if n in runs[f]} for n in names}
    result = {"champion": champion, "level": level, "folds": list(FOLD_PERIODS), "seeds": list(seeds),
              "missing": [f"{n}@{f}" for f in FOLD_PERIODS for n in names if n not in runs[f]],
              "champion_runs": {f: str(runs[f][champion]) for f in FOLD_PERIODS if champion in runs[f]}, "candidates": {}}
    for cand in candidates:
        folds = [f for f in FOLD_PERIODS if champion in runs[f] and cand in runs[f]]
        for f in folds:                                     # 같은 배정·cohort·평가 시각이면 evaluation panel이 같다
            if read_json(runs[f][cand] / "manifest.json")["panel_hash"] != read_json(runs[f][champion] / "manifest.json")["panel_hash"]:
                raise ValueError(f"{f}: {cand} run과 champion run의 evaluation panel이 다릅니다.")
        if folds:
            verdict = judge(_collect(measures[champion], folds, seeds), _collect(measures[cand], folds, seeds))
            result["candidates"][cand] = {**verdict, "complete": len(folds) == len(FOLD_PERIODS),
                                          "runs": {f: str(runs[f][cand]) for f in folds}}
    complete = sorted((v["total"], c) for c, v in result["candidates"].items() if v["complete"])
    result["ranking"] = [{"candidate": c, "total": t, "improved": t < 0} for t, c in complete]
    key = [list(candidates), list(seeds)] + ([] if level == "confirmation" else [level])
    result["report"] = str(write_report(spec, f"folds-{digest(key)[:10]}", result))
    return result


def partition_report(spec: Spec, labels, seeds=ROBUST_SEEDS, level: str = "confirmation") -> dict:
    """fold마다 champion을 배정 <label>-<fold>.csv로 학습한 정식 비교 run과 층화 배정 run을 비교해 네 조건을 판정하고 보고서로 저장한다.

    두 run은 같은 cohort 발전소와 평가 시각을 써야 한다. 그룹 채점 mask는 구성원이 모두 유효한 시각이라 배정에 따라 달라질 수 있으므로,
    cohort 발전소 집합이 같고 실제 발전량 수익(채점한 시각·발전소의 실적 합)이 같을 때만 비교하고 다르면 멈춘다.
    """
    champion = model_signature(spec, spec.champion)["dl_input"]
    base = {f: fold_runs(spec, f, seeds, level).get(champion) for f in FOLD_PERIODS}
    base_measures = {f: _measure(spec, f, run) for f, run in base.items() if run is not None}
    result = {"kind": "partition", "champion": champion, "level": level, "folds": list(FOLD_PERIODS), "seeds": list(seeds),
              "missing": [f"stratified@{f}" for f, run in base.items() if run is None],
              "champion_runs": {f: str(run) for f, run in base.items() if run is not None}, "candidates": {}}
    for label in labels:
        runs = {f: fold_runs(spec, f, seeds, level, label).get(champion) for f in FOLD_PERIODS}
        result["missing"] += [f"{label}@{f}" for f, run in runs.items() if run is None]
        folds = [f for f in FOLD_PERIODS if base[f] is not None and runs[f] is not None]
        if not folds:
            continue
        measures = {f: _measure(spec, f, runs[f]) for f in folds}
        for f in folds:
            cohorts = [sorted(s for ids in read_json(r / "manifest.json")["cohort"]["site_ids"].values() for s in ids) for r in (base[f], runs[f])]
            if cohorts[0] != cohorts[1]:
                raise ValueError(f"{f}: {label} 배정과 층화 배정 run의 cohort 발전소가 다릅니다.")
            if not np.isclose(measures[f]["revenue"], base_measures[f]["revenue"], rtol=0, atol=1e-3):
                raise ValueError(f"{f}: {label} 배정과 층화 배정 run의 실제 발전량 수익이 달라 채점 시각이 같지 않습니다.")
        verdict = judge(_collect(base_measures, folds, seeds), _collect(measures, folds, seeds))
        result["candidates"][label] = {**verdict, "complete": len(folds) == len(FOLD_PERIODS), "runs": {f: str(runs[f]) for f in folds}}
    complete = sorted((v["total"], c) for c, v in result["candidates"].items() if v["complete"])
    result["ranking"] = [{"candidate": c, "total": t, "improved": t < 0} for t, c in complete]
    key = ["partition", list(labels), list(seeds)] + ([] if level == "confirmation" else [level])
    result["report"] = str(write_report(spec, f"folds-{digest(key)[:10]}", result))
    return result


def search_settings(spec: Spec, path: str | Path) -> dict:
    """fold 배정 탐색 설정. base 설정(조건부 bundling의 공동 오차·시나리오·최적화)에 이 파일의 항목을 덮어쓴다."""
    own = read_json(spec.root / path)
    return {**read_json(spec.root / own["base"]), **{k: v for k, v in own.items() if k != "base"}}


def search_partition(spec: Spec, period_name: str, method: str, cfg: dict, data=None) -> dict:
    """fold cutoff 이전 정보로만 배정을 탐색해 partitions/search-<method>-<fold>.csv와 탐색 기록(.json)을 쓴다.

    발전소 오차는 primitive MK의 발전소 단위 예측을 추정 기간(FOLD_ESTIMATION)에서 fold cutoff까지 모은 OOT 오차이다. 같은 오차로
    공동 오차 모델과 시나리오(method)를 만들고, 층화 배정·QUBO 해·무작위 배정에서 시작한 flip·swap 탐색 가운데 목적이 가장 낮은 해를
    고른다. 두 그룹이 cohort 용량의 minimum_share(층화 배정과 같은 하한) 이상이 되도록 목적 증가가 가장 작은 발전소를 옮겨 복구한다.
    cohort 밖 발전소는 용량이 작은 그룹에 큰 순서로 배정한다(학습·평가에 쓰이지 않는다). 같은 설정의 기록이 있으면 다시 계산하지 않고,
    설정이 다른 기록은 덮어쓰지 않는다.
    """
    from .bundling.error_model import ScenarioSource, fit_risk
    from .bundling.optimize import ScenarioObjective, optimize_scenarios, restore_feasibility
    from .bundling.partition import build_instance, full_assignment, write_partition
    from .bundling.residuals import primitive_descriptor, site_primitive
    from .experiment import cuda_lock

    if period_name not in FOLD_PERIODS:
        raise ValueError(f"배정 탐색은 fold 기간에서 합니다: {period_name}")
    data = data if data is not None else load_data(spec)
    estimation = FOLD_ESTIMATION[period_name]
    path = spec.artifact_dir / "partitions" / f"search-{method}-{period_name}.csv"
    record_path = path.with_suffix(".json")
    identity = {"fold": period_name, "method": method, "estimation": estimation, "settings": {k: cfg[k] for k in SEARCH_SECTIONS},
                "residuals": primitive_descriptor(spec, data, estimation, cfg["primitive"]), "base": data.base.key,
                "weather": data.weather.key, "versions": versions("residuals", "error_model", "optimize", "partition")}
    key = digest(identity)
    if record_path.exists():
        record = read_json(record_path)
        if record.get("key") != key or not path.exists() or file_sha(path) != record.get("partition_sha"):
            raise ValueError(f"{record_path}는 다른 설정이나 바뀐 배정의 탐색 기록입니다. 덮어쓰지 않습니다.")
        return record
    log(f"fold 배정 탐색 {period_name} {method}: 추정 기간 {estimation}의 발전소 예측")
    forecasts = {estimation: site_primitive(spec, data, estimation, cfg["primitive"])}
    instance = build_instance(spec, data, period_name, forecasts, {"cutoff": period_name, "estimation": [estimation], "evaluation": None})
    frame, cohort = fold_stratified_partition(spec, period_name, data)
    if not np.array_equal(np.flatnonzero(cohort), instance.sites):
        raise ValueError("탐색 instance와 fold 층화 배정의 cohort가 다릅니다.")
    start = frame.group.to_numpy()[instance.sites] == "G1"
    instance.original = start                                     # 탐색 시작점이자 비교 기준은 층화 배정이다
    log(f"fold 배정 탐색 {period_name} {method}: 공동 오차 모델(cohort {len(instance.sites)}개소, 추정 {len(instance.est_hours)}시각)")
    risk = fit_risk(spec, data, instance, cfg)
    with cuda_lock(spec.artifact_dir / ".cuda.lock"):
        source = ScenarioSource(spec, data, instance, risk, cfg)
    scenarios, opt = cfg["scenarios"], cfg["optimization"]
    e, weight = source.materialize(method, scenarios["search_samples_per_hour"], scenarios["search_seed"])
    minimum_kw = minimum_share(float(data.sites.capacity.sum())) * instance.total
    log(f"fold 배정 탐색 {period_name} {method}: QUBO·flip·swap 탐색")
    with cuda_lock(spec.artifact_dir / ".cuda.lock"):
        obj = ScenarioObjective(e, weight, instance.capacity, float(opt["fixed_lambda"]), opt["swap_candidates"], opt["device"],
                                source.count(method, scenarios["search_samples_per_hour"]), opt["center"])
        del e, weight
        found, search = optimize_scenarios(obj, instance.capacity, {"stratified": start}, opt["random_starts"], 42, opt["max_iterations"],
                                           opt["tolerance"], opt["qubo"])
        x, feasibility = restore_feasibility(obj, found, instance.capacity, minimum_kw)
        objective = {"stratified": obj.value(start), "search": obj.value(x)}
        obj.close()
    expected = source.score_partitions(method, np.stack([start, x]), scenarios["evaluation_samples_per_hour"], scenarios["evaluation_seed"],
                                       opt["center"])
    write_partition(path, data.sites, full_assignment(len(data.sites), data.sites.capacity, instance.sites, x))
    log(f"fold 배정 탐색 {period_name} {method}: {int((x != start).sum())}개소 이동, 예상 W {expected[1] / expected[0] - 1:+.2%} → {path.name}")

    def describe(partition: np.ndarray) -> dict:
        return {"capacity_kw": [float(instance.capacity[partition].sum()), float(instance.capacity[~partition].sum())],
                "imbalance": float(instance.ctil @ (2 * partition - 1.0))}

    record = {"key": key, **identity, "partition": str(path), "partition_sha": file_sha(path), "cohort_sites": int(len(instance.sites)),
              "cohort_capacity_kw": instance.total, "minimum_group_kw": minimum_kw, "risk_key": risk.key,
              "estimation_hours": [int(instance.est_hours.min()), int(instance.est_hours.max())],
              "stratified": describe(start), "search": {**describe(x), "moved_sites": int((x != start).sum()), "record": search},
              "feasibility": feasibility, "objective": objective,
              "expected_W_krw": {"stratified": float(expected[0]), "search": float(expected[1]),
                                 "rule": "independent evaluation scenarios; group median correction as in the search objective"}}
    write_json(record_path, record)
    return record
