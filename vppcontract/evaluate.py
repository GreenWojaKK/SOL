"""공통 평가기. 산출물 파일과 원자료(sites·generation·SMP)만 읽어 W, ΔW, 대상일 block bootstrap을 계산한다.

비교하는 산출물은 같은 기간·같은 발전소 집합을 담아야 한다. 정답 mask는 배정과 무관하게 그 발전소 집합 전체의 발전량이
유효한 시각이고, 모든 산출물·그룹에 같은 mask를 쓴다. 유효 발전량은 유한한 0 이상의 값이다.

robust fold test의 네 조건 판정(judge)과 package 간 fold 비교(fold_compare)도 여기에 둔다. 두 package가 같은 규칙으로 판정한다.
validation panel(docs/solarvpp2-design.md#p60-panel)은 고정 배정에 날짜별 구성원(entry_day ≤ 대상일)을 적용해 채점한다(panel_truth,
evaluate_panel, panel_compare).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .outputs import Output, read_json, read_output
from .settlement import daily, hourly_penalty, paired, run_bootstrap
from .timeaxis import (FOLD_PERIODS, GROUPS, N_ACTUAL, N_HOURS, SEASON_IDS, SEASONS, delivery_dates, hour_index, k_weight, period,
                       season_index)

ROBUST_SEEDS = (42, 43, 44, 45, 46)
TOTAL_GAIN, MIN_IMPROVED_FOLDS, MAX_SEASON_LOSS = 0.01, 4, 0.02


@dataclass
class Actuals:
    site_ids: np.ndarray
    capacity: np.ndarray
    energy: np.ndarray
    valid: np.ndarray
    smp: np.ndarray
    k: np.ndarray


def load_actuals(data_dir: Path) -> Actuals:
    data_dir = Path(data_dir)
    sites = pd.read_csv(data_dir / "sites.csv")
    ids = sites.site_id.astype(str).to_numpy(dtype=object)
    frame = pd.read_parquet(data_dir / "train" / "generation.parquet")
    h, on_hour = hour_index(frame.ts)
    site = pd.Index(ids).get_indexer(frame.site_id.astype(str).to_numpy())
    if (site < 0).any():
        raise ValueError("generation에 sites.csv에 없는 발전소가 있습니다.")
    inside = on_hour & (h >= 0) & (h < N_ACTUAL)
    key = h[inside] * len(ids) + site[inside]
    if len(np.unique(key)) != len(key):
        raise ValueError("generation에 중복 (site_id, ts) 키가 있습니다.")
    value = frame.energy.to_numpy(dtype=np.float64)[inside]
    ok = np.isfinite(value) & (value >= 0)
    energy = np.full((N_ACTUAL, len(ids)), np.nan)
    energy[h[inside][ok], site[inside][ok]] = value[ok]
    prices = pd.concat([pd.read_csv(data_dir / "train" / "smp.csv"), pd.read_csv(data_dir / "smp.csv")], ignore_index=True)
    ph, p_on = hour_index(pd.to_datetime(prices.ts, utc=True).dt.tz_convert("Asia/Seoul"))
    if not p_on.all() or len(np.unique(ph)) != N_HOURS or ph.min() != 0 or ph.max() != N_HOURS - 1:
        raise ValueError("SMP는 전역 시간축의 모든 시각을 한 번씩 덮어야 합니다.")
    smp = np.empty(N_HOURS)
    smp[ph] = prices.smp.to_numpy(dtype=np.float64)
    return Actuals(ids, sites.capacity_kw.to_numpy(dtype=np.float64), energy, np.isfinite(energy), smp, k_weight(np.arange(N_HOURS)))


def _period_hours(output: Output, period_name: str) -> np.ndarray:
    hours = period(period_name).hours
    if not np.isin(hours, output.hours).all():
        raise ValueError(f"{output.directory}에 {period_name} 기간의 시각이 모두 있지 않습니다.")
    return hours


def group_truth(actuals: Actuals, output: Output, hours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """그룹 실적 [시각, 그룹]과, 산출물 발전소 전체의 발전량이 유효한 시각 mask [시각]."""
    index = pd.Index(actuals.site_ids).get_indexer(output.site_ids.astype(str))
    if (index < 0).any():
        raise ValueError("배정에 sites.csv에 없는 발전소가 있습니다.")
    if hours.max() >= N_ACTUAL:
        raise ValueError("정답이 없는 기간은 채점하지 않습니다.")
    complete = actuals.valid[hours][:, index].all(axis=1)
    energy = np.nan_to_num(actuals.energy[hours][:, index])
    truth = np.stack([energy[:, output.group == g].sum(axis=1) for g in range(len(GROUPS))], axis=1)
    return truth, complete


def evaluate(outputs: list[Output], period_name: str, actuals: Actuals) -> list[dict]:
    """같은 발전소 집합을 담은 산출물들을 같은 정답 mask로 채점한다."""
    sets = {tuple(sorted(o.site_ids.astype(str))) for o in outputs}
    if len(sets) != 1:
        raise ValueError("비교하는 산출물의 발전소 집합이 같아야 합니다.")
    hours = _period_hours(outputs[0], period_name)
    parts = []
    for output in outputs:
        hours_o = _period_hours(output, period_name)
        truth, complete = group_truth(actuals, output, hours_o)
        forecast = output.forecast[np.searchsorted(output.hours, hours_o)]
        parts.append((truth, complete, forecast))
    common = np.logical_and.reduce([c for _, c, _ in parts])
    results = []
    for output, (truth, _, forecast) in zip(outputs, parts):
        mask = np.broadcast_to(common[:, None], truth.shape)
        penalty = hourly_penalty(truth, forecast, actuals.smp[hours], actuals.k[hours], mask)
        cap = [float(actuals.capacity[pd.Index(actuals.site_ids).get_indexer(output.members(g).astype(str))].sum()) for g in range(len(GROUPS))]
        results.append({"directory": str(output.directory), "period": period_name, "W_krw": float(penalty.sum()),
                        "W_by_group_krw": dict(zip(GROUPS, penalty.sum(axis=0).tolist())), "capacity_kw": dict(zip(GROUPS, cap)),
                        "scored_hours": int(common.sum()), "hours": len(hours), "daily": daily(hours, penalty)})
    return results


def entry_days(actuals: Actuals) -> np.ndarray:
    """발전소별로 24시간 실적이 모두 유효한 첫 대상일. 그런 날이 없으면 정답이 있는 대상일 수(어느 평가일보다 크다)다."""
    n_days = actuals.valid.shape[0] // 24
    full = actuals.valid[:n_days * 24].reshape(n_days, 24, -1).all(axis=1)
    return np.where(full.any(axis=0), np.argmax(full, axis=0), n_days)


def panel_truth(actuals: Actuals, site_ids, group, days) -> dict:
    """validation panel의 날짜별 구성원과 그룹 실적.

    구성원 G[g, d]는 배정의 그룹 g 가운데 entry_day가 d 이하인 발전소이고, 유효한 실적이 0인 발전소도 구성원이다. 그룹 실적은
    구성원 발전량의 합이고, 공통 시각 mask는 그날 두 그룹 구성원 전체의 실적이 유효한 시각이다. 시각은 대상일 순서의 24시간이다.
    반환: hours [H], truth [H, 2], complete [H], members [대상일, 2, 발전소](site_ids 순서).
    """
    days = np.asarray(days, dtype=np.int64)
    index = pd.Index(actuals.site_ids).get_indexer(np.asarray(site_ids).astype(str))
    if (index < 0).any():
        raise ValueError("배정에 sites.csv에 없는 발전소가 있습니다.")
    hours = (days[:, None] * 24 + np.arange(24)[None, :]).ravel()
    if hours.max() >= N_ACTUAL:
        raise ValueError("정답이 없는 대상일은 채점하지 않습니다.")
    group = np.asarray(group)
    entered = entry_days(actuals)[index][None, :] <= days[:, None]
    members = np.stack([entered & (group == g)[None, :] for g in range(len(GROUPS))], axis=1)
    if not members.any(axis=2).all():
        raise ValueError("구성원이 없는 그룹·대상일이 있습니다.")
    by_hour = np.repeat(members, 24, axis=0)                                   # [H, 2, 발전소]
    valid = actuals.valid[hours][:, index]
    energy = np.nan_to_num(actuals.energy[hours][:, index])
    truth = np.einsum("hn,hgn->hg", energy, by_hour.astype(np.float64))
    complete = ~(by_hour.any(axis=1) & ~valid).any(axis=1)
    return {"hours": hours, "truth": truth, "complete": complete, "members": members}


def panel_score(actuals: Actuals, truth: dict, forecast: np.ndarray, complete: np.ndarray | None = None) -> dict:
    """panel_truth의 결과와 제출값 [H, 2]로 W와 일별 벌점 [대상일, 2]을 계산한다. complete를 주면 그 공통 mask를 쓴다."""
    hours, mask = truth["hours"], truth["complete"] if complete is None else complete
    penalty = hourly_penalty(truth["truth"], forecast, actuals.smp[hours], actuals.k[hours], np.broadcast_to(mask[:, None], truth["truth"].shape))
    return {"W_krw": float(penalty.sum()), "W_by_group_krw": dict(zip(GROUPS, penalty.sum(axis=0).tolist())), "scored_hours": int(mask.sum()),
            "hours": len(hours), "daily": penalty.reshape(-1, 24, len(GROUPS)).sum(axis=1)}


def panel_days_of(output: Output) -> np.ndarray:
    """validation panel 산출물의 평가일(manifest의 panel.days)."""
    panel = output.manifest.get("panel") or {}
    if "days" not in panel:
        raise ValueError(f"{output.directory}는 validation panel 산출물이 아닙니다.")
    return np.asarray(panel["days"], dtype=np.int64)


def evaluate_panel(outputs: list[Output], actuals: Actuals) -> list[dict]:
    """같은 panel·배정의 산출물들을 날짜별 구성원과 공통 시각 mask로 채점한다."""
    days = panel_days_of(outputs[0])
    if any(not np.array_equal(panel_days_of(o), days) for o in outputs):
        raise ValueError("비교하는 산출물의 평가일이 같아야 합니다.")
    if len({_partition(o) for o in outputs}) != 1:
        raise ValueError("비교하는 산출물의 배정이 같아야 합니다.")
    truths = [panel_truth(actuals, o.site_ids, o.group, days) for o in outputs]
    common = np.logical_and.reduce([t["complete"] for t in truths])
    results = []
    for output, truth in zip(outputs, truths):
        if not np.isin(truth["hours"], output.hours).all():
            raise ValueError(f"{output.directory}에 평가일의 시각이 모두 있지 않습니다.")
        forecast = output.forecast[np.searchsorted(output.hours, truth["hours"])]
        results.append({"directory": str(output.directory), **panel_score(actuals, truth, forecast, common)})
    return results


def panel_compare(candidate: Output, baseline: Output, actuals: Actuals, replicates: int = 2000, seed: int = 42) -> dict:
    """validation panel 비교. W 변화율(후보 W / 기준 W − 1, 음수가 개선)과 ΔW, 연속 평가일 묶음 bootstrap 구간, 월·계절별 W,
    run이 기록한 seed별 W를 낸다. run이 기록한 결합 예측 W는 이 평가기의 W와 같아야 한다."""
    scored = dict(zip(("candidate", "baseline"), evaluate_panel([candidate, baseline], actuals)))
    days = panel_days_of(candidate)
    daily_w = {side: r["daily"].sum(axis=1) for side, r in scored.items()}

    def split(labels: np.ndarray, names) -> dict:
        out = {}
        for name in names:
            m = labels == name
            c, b = float(daily_w["candidate"][m].sum()), float(daily_w["baseline"][m].sum())
            out[str(name)] = {"days": int(m.sum()), "candidate_W_krw": c, "baseline_W_krw": b, "W_change": c / b - 1}
        return out

    months = delivery_dates(days).strftime("%Y-%m").to_numpy()
    seasons = np.array([SEASON_IDS[i] for i in season_index(days)])
    recorded = {"candidate": recorded_scores(candidate), "baseline": recorded_scores(baseline)}
    for side, r in scored.items():
        if abs(recorded[side]["W"] - r["W_krw"]) > 1.0:
            raise ValueError(f"{r['directory']}의 기록 W {recorded[side]['W']:,.2f}와 공통 평가기 W {r['W_krw']:,.2f}가 다릅니다.")
    seeds = sorted(set(recorded["candidate"]["seeds"]) & set(recorded["baseline"]["seeds"]), key=int)
    strip = lambda r: {k: v for k, v in r.items() if k != "daily"}
    return {"days": len(days), "candidate": strip(scored["candidate"]), "baseline": strip(scored["baseline"]),
            **run_bootstrap(scored["candidate"]["daily"], scored["baseline"]["daily"], days, replicates, seed),
            "by_month": split(months, sorted(set(months))), "by_season": split(seasons, [s for s in SEASON_IDS if s in set(seasons)]),
            "seed_W_krw": {side: recorded[side]["seeds"] for side in recorded},
            "seed_W_change": {s: recorded["candidate"]["seeds"][s] / recorded["baseline"]["seeds"][s] - 1 for s in seeds}}


def compare(a: Output, b: Output, period_name: str, actuals: Actuals, blocks=(7, 14), replicates: int = 2000, seed: int = 42) -> dict:
    """ΔW = W_a − W_b. 음수이면 a의 벌점이 작다."""
    ra, rb = evaluate([a, b], period_name, actuals)
    result = paired(ra["daily"], rb["daily"], blocks, replicates, seed)
    strip = lambda r: {k: v for k, v in r.items() if k != "daily"}
    return {"period": period_name, "a": strip(ra), "b": strip(rb), **result}


def compare_directories(a: Path, b: Path, period_name: str, data_dir: Path, **kwargs) -> dict:
    return compare(read_output(a), read_output(b), period_name, load_actuals(data_dir), **kwargs)


def judge(base: dict, cand: dict) -> dict:
    """fold별·seed별·계절별 W(원)로 네 조건을 판정한다.

    base·cand는 {"folds": {fold: W}, "seeds": {seed: {fold: W}}, "seasons": {계절: W}}이다. Fold W는 seed 결합 예측의 W이고,
    seed W는 그 seed 하나의 예측으로 구한 W이다. robust는 다섯 fold(FOLD_PERIODS)가 모두 있고 두 쪽의 seed가 ROBUST_SEEDS 다섯 개일
    때(full_test)만 네 조건으로 정한다. fold가 빠졌거나 seed가 다른 비교(screening 포함)는 조건 값을 참고로 남기고 robust는 False이다.
    """
    folds = list(base["folds"])
    w0 = np.array([base["folds"][f] for f in folds])
    w1 = np.array([cand["folds"][f] for f in folds])
    per_fold = w1 / w0 - 1
    total = float(w1.sum() / w0.sum() - 1)
    per_seed = {s: float(sum(cand["seeds"][s][f] for f in folds) / sum(base["seeds"][s][f] for f in folds) - 1) for s in base["seeds"]}
    seasons = {k: float(cand["seasons"][k] / base["seasons"][k] - 1) for k in base["seasons"]}
    improved = int((per_fold < 0).sum())
    checks = {"total": total <= -TOTAL_GAIN, "folds": improved >= MIN_IMPROVED_FOLDS,
              "seeds": all(v < 0 for v in per_seed.values()), "seasons": max(seasons.values()) <= MAX_SEASON_LOSS}
    robust_seeds = sorted(str(s) for s in ROBUST_SEEDS)
    full_test = (sorted(folds) == sorted(FOLD_PERIODS) and sorted(cand["folds"]) == sorted(folds)
                 and sorted(base["seeds"]) == robust_seeds and sorted(cand["seeds"]) == robust_seeds)
    return {"total": total, "per_fold": {f: float(v) for f, v in zip(folds, per_fold)}, "improved_folds": improved,
            "per_seed": per_seed, "seasons": seasons, "checks": checks, "full_test": full_test,
            "robust": full_test and all(checks.values())}


def recorded_scores(output: Output) -> dict:
    """산출물을 만든 run이 기록한 결합 예측 W와 seed별 W(같은 seed의 XGB·DL 결합 제출값의 W).

    solarvpp run 디렉터리는 report.json의 components[cell].W_cohort_krw와 per_seed_W_krw[seed].blend를 읽는다(seed 하나인 run은
    결합 예측 W). solarvpp2 forecast 단계의 산출물(<run>/outputs/<배정>)은 run의 instance.json의 evaluation.W_krw[배정]과
    seed_W_krw[배정]을 읽는다.
    """
    directory = Path(output.directory)
    if (directory / "report.json").exists():
        report = read_json(directory / "report.json")
        total = float(report["components"][output.manifest["cell"]]["W_cohort_krw"])
        if "per_seed_W_krw" in report:
            return {"W": total, "seeds": {str(s): float(v["blend"]) for s, v in report["per_seed_W_krw"].items()}}
        return {"W": total, "seeds": {str(s): total for s in output.manifest["seeds"]}}
    run = directory.parent.parent
    if output.manifest.get("package") == "solarvpp2" and (run / "instance.json").exists():
        evaluation = read_json(run / "instance.json").get("evaluation") or {}
        part = output.manifest["partition"]
        if "seed_W_krw" not in evaluation:
            raise ValueError(f"{run}에 seed별 W 기록이 없습니다.")
        return {"W": float(evaluation["W_krw"][part]), "seeds": {str(s): float(v) for s, v in evaluation["seed_W_krw"][part].items()}}
    raise ValueError(f"{directory}의 seed별 W 기록을 찾지 못했습니다.")


def _partition(output: Output) -> frozenset:
    """그룹 이름과 무관한 배정(발전소 집합 두 개)."""
    ids = output.site_ids.astype(str)
    return frozenset(frozenset(ids[output.group == g]) for g in range(len(GROUPS)))


def fold_compare(candidates: dict, baselines: dict, actuals: Actuals) -> dict:
    """robust fold test: fold마다 후보·기준 산출물을 같은 정답 mask로 채점하고 네 조건을 판정한다.

    candidates·baselines는 fold 이름 → 산출물 디렉터리(계약 형식 또는 solarvpp run)이다. 두 산출물은 발전소 집합과 배정이 같아야 한다.
    fold W(seed 결합 예측의 W)와 계절 W(결합 예측 일별 벌점의 계절 합)는 이 평가기로 계산한다. seed W는 각 run이 기록한 값이며,
    run이 기록한 결합 예측 W가 이 평가기의 W와 같을 때만 쓴다(같은 채점 규칙의 확인). 모든 fold와 두 쪽의 seed가 같아야 한다.
    """
    if set(candidates) != set(baselines):
        raise ValueError("후보와 기준의 fold가 같아야 합니다.")
    folds = [f for f in FOLD_PERIODS if f in candidates] + sorted(set(candidates) - set(FOLD_PERIODS))
    measures = {"candidate": {}, "baseline": {}}
    for f in folds:
        outputs = {"candidate": read_output(candidates[f]), "baseline": read_output(baselines[f])}
        if _partition(outputs["candidate"]) != _partition(outputs["baseline"]):
            raise ValueError(f"{f}: 후보와 기준의 발전소 집합 또는 배정이 다릅니다.")
        scored = dict(zip(outputs, evaluate(list(outputs.values()), f, actuals)))
        season = season_index(period(f).days)
        for side, output in outputs.items():
            recorded, r = recorded_scores(output), scored[side]
            if abs(recorded["W"] - r["W_krw"]) > 1.0:
                raise ValueError(f"{f}: {output.directory}의 기록 W {recorded['W']:,.2f}와 공통 평가기 W {r['W_krw']:,.2f}가 다릅니다.")
            measures[side][f] = {"W": r["W_krw"], "seeds": recorded["seeds"],
                                 "seasons": {k: float(r["daily"].sum(axis=1)[season == i].sum()) for i, k in enumerate(SEASONS)}}
    seed_sets = {tuple(sorted(m["seeds"], key=int)) for side in measures.values() for m in side.values()}
    if len(seed_sets) != 1:
        raise ValueError(f"모든 fold와 두 쪽의 seed가 같아야 합니다: {sorted(seed_sets)}")
    seeds = list(seed_sets.pop())

    def collect(side: dict) -> dict:
        return {"folds": {f: side[f]["W"] for f in folds}, "seeds": {s: {f: side[f]["seeds"][s] for f in folds} for s in seeds},
                "seasons": {k: sum(side[f]["seasons"][k] for f in folds) for k in SEASONS}}

    verdict = judge(collect(measures["baseline"]), collect(measures["candidate"]))
    return {"folds": folds, "seeds": [int(s) for s in seeds],
            "W_krw": {side: {f: measures[side][f]["W"] for f in folds} for side in measures},
            "runs": {f: {"candidate": str(candidates[f]), "baseline": str(baselines[f])} for f in folds}, **verdict}
