"""Instance 실행.

참조: docs/solarvpp2-design.md#instances, docs/solarvpp2-design.md#evaluation.

Instance마다 결정 cutoff의 forecaster 하나가 R 학습, 배정 결정, 제출·평가를 모두 맡는다. 정답이 있는 instance는 배정마다
같은 forecaster로 평가 기간을 예측해 실제 W를 계산한다. 산출물은 공통 계약층 형식(배정 CSV, 그룹 예측 CSV, manifest)으로 쓴다.

실행 단계는 셋이다. full은 그래프 전체(forecaster → R → 배정 탐색 → 산출물 → 평가)를 실행한다. forecast는 forecaster만 학습해
고정 배정(original·stratified)을 예측·평가하고 R 학습과 배정 탐색을 생략한다. 추천은 full 단계 run만 쓴다.
forecast 단계는 정답이 있으면 seed 결합 예측의 W와 함께, 같은 seed의 XGB·DL을 결합한 seed별 제출값의 W와 일별 벌점을 기록한다.
panel은 validation panel(docs/solarvpp2-design.md#p60-panel)의 평가일과 앞뒤 완충을 학습에서 뺀 forecaster로 고정 배정(stratified)의
날짜별 구성원 그룹 예측을 만들고 같은 구성원으로 채점한다.
"""
from __future__ import annotations

import dataclasses
import time
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vppcontract.evaluate import Actuals, evaluate, evaluate_panel, load_actuals, panel_score, panel_truth
from vppcontract.outputs import read_output, validate_submission, write_output
from vppcontract.settlement import daily, hourly_penalty, paired
from vppcontract.timeaxis import SEASON_IDS, delivery_dates, period, season_index

from .config import Spec, with_overrides
from .data import load_data
from .forecast.forecaster import Forecaster
from .store import evaluation_payload, forecast_payload, full_payload, panel_payload, prediction_payload
from .evaluation import evaluate_saved, finish_prediction, save_artifacts
from .partition import (
    Objective,
    SearchState,
    capacity_ok,
    partition_key,
    repair,
    search,
    starting_points,
)
from .risk import RiskModel, calibration_table
from .sampler import balanced_split, buffered_days, panel_days, stratified_split
from .store import Store, digest, environment, read_json, source_sha, versions, write_json

INSTANCES = ("V", "fold-2309", "fold-2312", "fold-2403", "H", "B", "test")
STAGES = ("full", "forecast", "panel")
SEASONS = {"spring": (3, 4, 5), "autumn": (9, 10, 11)}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def search_settings(spec: Spec, model: dict, smoke: bool) -> dict:
    settings = dict(model["partition"])
    if smoke:
        smoke_cfg = spec.config["smoke"]
        settings.update({"max_iterations": smoke_cfg["search_iterations"], "patience": smoke_cfg["search_iterations"],
                         "moves_per_iteration": smoke_cfg["search_moves"], "starts": {"original": True, "stratified": True, "random_halves": 1}})
    return settings


def fused_cache_fits(forecaster: Forecaster, days: int) -> bool:
    size = days * len(forecaster.data.sites) * 24 * forecaster.spec.hyperparameters["dl_fusion_width"] * 4 * len(forecaster.seeds)
    return size <= forecaster.spec.config["runtime"]["fused_cache_max_gib"] * 1024 ** 3


def group_members(index: np.ndarray, partitions: np.ndarray, n_pv: int) -> np.ndarray:
    """배정 [P, n_instance] → 그룹 membership [2P, PV]. 짝수 행이 G1, 홀수 행이 G2이다."""
    partitions = np.asarray(partitions, dtype=bool)
    groups = np.zeros((2 * len(partitions), n_pv), dtype=bool)
    groups[0::2, index] = partitions
    groups[1::2, index] = ~partitions
    return groups


def predict_partitions(forecaster: Forecaster, index: np.ndarray, partitions: np.ndarray, days: np.ndarray, chunk_days: int = 16,
                       experts: bool = False, per_seed: bool = False):
    """배정마다 G1·G2 제출값 [P, 시각, 2](kWh). 시각은 days의 24시간을 이어 붙인 순서이다.

    experts=True이면 XGB·DL 각각의 중앙값(같은 후처리)을 {"xgb", "dl"}로, per_seed=True이면 seed별 결합 제출값
    [S, P, 시각, 2](seed 순서는 forecaster.seeds)를 {"seeds"}로 함께 돌려준다.
    """
    groups = group_members(index, partitions, len(forecaster.data.sites))
    names = ("median", "xgb_median", "dl_median") if experts else ("median",)
    out = {n: np.zeros((len(partitions), len(days) * 24, 2)) for n in names}
    n_seed = len(forecaster.seeds)
    seeds = np.zeros((n_seed, len(partitions), len(days) * 24, 2)) if per_seed else None
    for start in range(0, len(days), chunk_days):
        chunk = days[start:start + chunk_days]
        pred = forecaster.predict(np.repeat(chunk, len(groups)), np.tile(groups, (len(chunk), 1)), chunk_days=chunk_days, per_seed=per_seed)
        for n in names:
            value = getattr(pred, n).reshape(len(chunk), len(partitions), 2, 24)
            out[n][:, start * 24:(start + len(chunk)) * 24] = value.transpose(1, 0, 3, 2).reshape(len(partitions), -1, 2)
        if per_seed:
            value = pred.seed_median.reshape(len(chunk), len(partitions), 2, n_seed, 24)
            seeds[:, :, start * 24:(start + len(chunk)) * 24] = value.transpose(3, 1, 0, 4, 2).reshape(n_seed, len(partitions), -1, 2)
    if experts or per_seed:
        extra = {"xgb": out["xgb_median"], "dl": out["dl_median"]} if experts else {}
        if per_seed:
            extra["seeds"] = seeds
        return out["median"], extra
    return out["median"]


def realized(forecaster: Forecaster, index: np.ndarray, x: np.ndarray, days: np.ndarray, forecast: np.ndarray) -> dict:
    """배정 x의 실제 벌점. 정답 mask는 배정과 무관하게 instance 발전소 전체의 발전량이 유효한 시각이다(공통 평가기와 같은 정의)."""
    hours = (days[:, None] * 24 + np.arange(24)[None, :]).ravel()
    base = forecaster.data.base
    valid = np.asarray(base.status)[hours][:, index] == 1
    energy = np.where(valid, np.asarray(base.energy)[hours][:, index], 0.0)
    truth = np.stack([energy[:, x].sum(axis=1), energy[:, ~x].sum(axis=1)], axis=1)
    complete = valid.all(axis=1)
    penalty = hourly_penalty(truth, forecast, np.asarray(base.smp)[hours], np.asarray(base.k)[hours], np.repeat(complete[:, None], 2, axis=1))
    return {"W_krw": float(penalty.sum()), "W_by_group_krw": penalty.sum(axis=0).tolist(), "scored_hours": int(complete.sum()),
            "daily": daily(hours, penalty)}


def risk_calibration(forecaster: Forecaster, risk: RiskModel, index: np.ndarray, partitions: dict, days: np.ndarray, bins: int) -> list[dict]:
    """평가 기간의 (대상일, 그룹, 시각)마다 전체 head의 R과 실현 절대오차를 C_A로 나눠 예측 R 구간별로 비교한다."""
    base = forecaster.data.base
    groups = group_members(index, np.stack(list(partitions.values())), len(forecaster.data.sites))
    preds, reals, weights = [], [], []
    for start in range(0, len(days), 16):
        chunk = days[start:start + 16]
        q_days, members = np.repeat(chunk, len(groups)), np.tile(groups, (len(chunk), 1))
        pred = forecaster.predict(q_days, members, hidden=True)
        r = risk.predict(pred, q_days)
        hours = q_days[:, None] * 24 + np.arange(24)[None, :]
        status, energy = np.asarray(base.status)[hours], np.nan_to_num(np.asarray(base.energy)[hours])
        complete = (status[..., index] == 1).all(axis=-1)
        truth = np.einsum("qtp,qp->qt", energy, members.astype(np.float64))
        q, s = np.nonzero(complete & pred.daylight)
        preds.append(r[q, s] / pred.cap[q])
        reals.append(np.abs(truth[q, s] - pred.median[q, s]) / pred.cap[q])
        weights.append(pred.cap[q])
    return calibration_table(np.concatenate(preds), np.concatenate(reals), np.concatenate(weights), bins)


def regime_bias(spec: Spec, forecaster: Forecaster, index: np.ndarray, partitions: dict, forecasts: np.ndarray, days: np.ndarray) -> dict:
    """배정마다 계절(봄·가을·그 외) × 평일·휴일의 대상일 수, W, k=3 시각 편향(Σ제출값 / Σ실적 − 1).

    휴일은 토·일·공휴일·근로자의 날이다. 날씨는 요일과 무관하므로 휴일에만 나타나는 편향은 수요·계통 쪽 상태의 흔적이다.
    """
    base = forecaster.data.base
    hours = (days[:, None] * 24 + np.arange(24)[None, :]).ravel()
    valid = np.asarray(base.status)[hours][:, index] == 1
    energy = np.where(valid, np.asarray(base.energy)[hours][:, index], 0.0)
    complete = valid.all(axis=1)
    k = np.asarray(base.k)[hours]
    k3 = (k == 3) & complete
    dates = [t.date() for t in delivery_dates(days)]
    off = np.array([spec.calendar.is_off(d, d - timedelta(days=1)) for d in dates])
    month = np.array([d.month for d in dates])
    season = np.full(len(days), "other", dtype=object)
    for label, months in SEASONS.items():
        season[np.isin(month, months)] = label
    out = {}
    for (label, x), forecast in zip(partitions.items(), forecasts):
        truth = np.stack([energy[:, x].sum(axis=1), energy[:, ~x].sum(axis=1)], axis=1)
        penalty = hourly_penalty(truth, forecast, np.asarray(base.smp)[hours], k, np.repeat(complete[:, None], 2, axis=1))
        day_w = daily(hours, penalty).sum(axis=1)
        error = ((forecast - truth).sum(axis=1) * k3).reshape(len(days), 24).sum(axis=1)
        actual = (truth.sum(axis=1) * k3).reshape(len(days), 24).sum(axis=1)
        rows = []
        for s in ("spring", "autumn", "other"):
            for kind, flag in (("workday", False), ("off", True)):
                m = (season == s) & (off == flag)
                if m.any():
                    rows.append({"season": s, "day": kind, "days": int(m.sum()), "W_krw": float(day_w[m].sum()),
                                 "k3_bias": float(error[m].sum() / max(actual[m].sum(), 1e-9))})
        out[label] = rows
    return out


def contract_check(spec: Spec, outputs: dict, scores: dict, name: str) -> dict:
    """공통 평가기로 같은 산출물의 W를 다시 계산해 instance 계산과 대조한다."""
    labels = list(outputs)
    recomputed = evaluate([read_output(outputs[k]) for k in labels], name, load_actuals(spec.data_dir))
    values = {k: r["W_krw"] for k, r in zip(labels, recomputed)}
    mismatch = {k: abs(values[k] - scores[k]["W_krw"]) for k in labels}
    if max(mismatch.values()) > 1e-3:
        raise RuntimeError(f"공통 평가기와 instance의 W가 다릅니다: {mismatch}")
    return values


def completed_run(report_path: Path, need_models: bool = False) -> bool:
    """완료된 기록을 그대로 쓸 수 있는지. 모델 파일이 필요한데 저장하지 않은 실행이면 다시 학습하도록 False이다."""
    if not report_path.exists():
        return False
    report = read_json(report_path)
    return report.get("status") == "completed" and not (need_models and not report.get("cache_stored", True))


def run_forecast_stage(spec: Spec, name: str, model_name: str, model: dict, per, eval_days: np.ndarray, forecaster: Forecaster,
                       smoke: bool, overrides: dict, need_models: bool = False) -> Path:
    """forecaster만 학습해 고정 배정(original·stratified)의 평가 기간 제출값과, 정답이 있으면 평가·regime 편향을 낸다."""
    sites = forecaster.data.sites
    index = np.flatnonzero(forecaster.cohort)
    capacity, share = sites.capacity[index], forecaster.share_low
    evaluation = evaluation_payload(spec.config["evaluation"], calendar=spec.calendar if per.has_labels else None)
    calculation = forecast_payload(name, forecaster.identity(), eval_days.tolist(), evaluation, versions("instance"))
    calculation = prediction_payload(calculation)
    run_key = digest(calculation)
    run_dir = spec.artifact_dir / "instances" / name / f"{model_name}-forecast{'-smoke' if smoke else ''}-{run_key[:10]}"
    report_path = run_dir / "instance.json"
    if completed_run(report_path, need_models):
        finish_prediction(spec, run_dir, forecaster.data)
        log(f"완료된 instance 재사용: {run_dir}")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {"package": "solarvpp2", "instance": name, "model": model_name, "definition": model, "parameters": overrides,
              "stage": "forecast", "smoke": smoke, "run_key": run_key, "calculation": calculation,
              "prediction_calculation": calculation, "evaluation_settings": evaluation,
              "cache_stored": Store.temporary is None, "status": "running", "decision_cutoff": per.cutoff, "cutoff_h": per.cutoff_h, "sites": len(index),
              "capacity_kw": float(capacity.sum()), "minimum_share": share, "heldout_days": len(forecaster.heldout_days),
              "train_days": len(forecaster.train_days), "train_queries": len(forecaster.train_q), "versions": versions(),
              "environment": environment(), "source_sha": source_sha(), "started": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    write_json(report_path, report)
    try:
        report["forecaster"] = forecaster.train()
        if int(forecaster.train_q.day.max()) > forecaster.cutoff_day:
            raise RuntimeError("학습에 cutoff 이후 대상일이 들어갔습니다.")
        partitions = {"stratified": repair(stratified_split(index, sites), capacity, share)}
        if per.has_labels:
            partitions = {"original": sites.group[index] == 0, **partitions}
        report["partition"] = {"capacity_kw": {k: [float(capacity[x].sum()), float(capacity[~x].sum())] for k, x in partitions.items()},
                               "feasible": {k: capacity_ok(x, capacity, share) for k, x in partitions.items()}}
        hours = (eval_days[:, None] * 24 + np.arange(24)[None, :]).ravel()
        forecasts, expert_forecasts = predict_partitions(forecaster, index, np.stack(list(partitions.values())), eval_days, experts=True,
                                                         per_seed=per.has_labels)
        outputs = {}
        for (part, x), forecast in zip(partitions.items(), forecasts):
            manifest = {"package": "solarvpp2", "model": model_name, "definition": model, "parameters": overrides, "instance": name,
                        "stage": "forecast", "partition": part, "decision_cutoff": per.cutoff, "smoke": smoke, "versions": versions(),
                        "forecaster": forecaster.identity(), "environment": report["environment"], "source_sha": report["source_sha"]}
            outputs[part] = write_output(run_dir / "outputs" / part, sites.ids[index], np.where(x, 0, 1), hours, forecast, manifest)
        report["outputs"] = {k: str(v) for k, v in outputs.items()}
        if per.has_labels:
            cfg = spec.config["evaluation"]
            blocks, reps, seed = tuple(cfg["bootstrap_blocks"]), int(cfg["bootstrap_replicates"]), int(cfg["bootstrap_seed"])
            scores = {part: realized(forecaster, index, x, eval_days, f) for (part, x), f in zip(partitions.items(), forecasts)}
            experts = {part: {e: realized(forecaster, index, x, eval_days, expert_forecasts[e][j])["W_krw"] for e in ("xgb", "dl")}
                       for j, (part, x) in enumerate(partitions.items())}
            # seed별 W: 같은 seed의 XGB·DL 결합 제출값으로 채점한다. 결합 예측의 W는 seed별 W의 평균이 아니다.
            seed_scores = {part: {str(s): realized(forecaster, index, x, eval_days, expert_forecasts["seeds"][k][j])
                                  for k, s in enumerate(forecaster.seeds)} for j, (part, x) in enumerate(partitions.items())}
            evaluation = {"W_krw": {k: v["W_krw"] for k, v in scores.items()}, "expert_W_krw": experts,
                          "seeds": [int(s) for s in forecaster.seeds],
                          "seed_W_krw": {part: {s: v["W_krw"] for s, v in by_seed.items()} for part, by_seed in seed_scores.items()},
                          "scored_hours": scores["stratified"]["scored_hours"],
                          "stratified_vs_original": paired(scores["stratified"]["daily"], scores["original"]["daily"], blocks, reps, seed),
                          "regime_bias": regime_bias(spec, forecaster, index, partitions, forecasts, eval_days)}
            # 일별 벌점 [대상일, 그룹]: "<배정>"은 결합 예측, "<배정>__s<seed>"는 seed별 결합 제출값이다.
            np.savez(run_dir / "daily_penalty.npz", days=eval_days, **{part: v["daily"] for part, v in scores.items()},
                     **{f"{part}__s{s}": v["daily"] for part, by_seed in seed_scores.items() for s, v in by_seed.items()})
            report["daily_penalty"] = str(run_dir / "daily_penalty.npz")
            if not smoke:
                evaluation["contract_recomputed_W_krw"] = contract_check(spec, outputs, scores, name)
            report["evaluation"] = evaluation
        report["status"] = "completed"
    except BaseException:
        report["status"] = "failed"
        (run_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(report_path, report)
        raise
    report["finished"] = pd.Timestamp.now(tz="Asia/Seoul").isoformat()
    write_json(report_path, report)
    save_artifacts(spec, run_dir, forecaster)
    finish_prediction(spec, run_dir, forecaster.data)
    log(f"완료: {run_dir}")
    return run_dir


def actuals_of(data) -> Actuals:
    """forecaster 자료의 발전량·유효성·SMP·k(공통 평가기와 같은 정의: 유효 = 유한하고 0 이상)."""
    base = data.base
    return Actuals(data.sites.ids.astype(str), data.sites.capacity, np.asarray(base.energy), np.asarray(base.status) == 1,
                   np.asarray(base.smp), np.asarray(base.k))


def run_panel_stage(spec: Spec, name: str, model_name: str, model: dict, per, panel_name: str, panel: dict, days: np.ndarray,
                    forecaster: Forecaster, smoke: bool, overrides: dict, need_models: bool = False) -> Path:
    """validation panel: 평가일과 앞뒤 완충을 학습에서 뺀 forecaster로 고정 배정(stratified)의 날짜별 구성원 그룹 예측을 만들고
    같은 구성원·공통 시각 mask로 채점한다. 구성원은 배정의 그룹 가운데 24시간 실적이 모두 유효한 첫 대상일이 평가일 이하인 발전소다.

    참조: docs/solarvpp2-design.md#p60-panel.
    """
    sites = forecaster.data.sites
    index = np.flatnonzero(forecaster.cohort)
    capacity, share = sites.capacity[index], forecaster.share_low
    x = repair(stratified_split(index, sites), capacity, share)
    group = np.where(x, 0, 1)
    record = {"name": panel_name, **panel, "days": days.tolist(), "membership": "entry_day"}
    evaluation_settings = evaluation_payload(spec.config["evaluation"])
    calculation = prediction_payload(panel_payload(name, forecaster.identity(), record, evaluation_settings, versions("instance")))
    run_key = digest(calculation)
    run_dir = spec.artifact_dir / "instances" / name / f"{model_name}-panel-{panel_name}{'-smoke' if smoke else ''}-{run_key[:10]}"
    report_path = run_dir / "instance.json"
    if completed_run(report_path, need_models):
        log(f"완료된 panel run 재사용: {run_dir}")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    excluded = forecaster.excluded_days
    report = {"package": "solarvpp2", "instance": name, "model": model_name, "definition": model, "parameters": overrides,
              "stage": "panel", "panel": record, "smoke": smoke, "run_key": run_key, "calculation": calculation,
              "prediction_calculation": calculation, "evaluation_settings": evaluation_settings, "cache_stored": Store.temporary is None, "status": "running",
              "decision_cutoff": per.cutoff, "cutoff_h": per.cutoff_h, "sites": len(index), "capacity_kw": float(capacity.sum()),
              "minimum_share": share, "heldout_days": len(forecaster.heldout_days), "excluded_days": len(excluded),
              "excluded_outside_heldout": int(np.setdiff1d(excluded, forecaster.heldout_days).size),
              "train_days": len(forecaster.train_days), "train_queries": len(forecaster.train_q), "versions": versions(),
              "environment": environment(), "source_sha": source_sha(), "started": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    write_json(report_path, report)
    try:
        report["forecaster"] = forecaster.train()
        trained = np.unique(forecaster.train_q.day)
        if np.intersect1d(trained, excluded).size or trained.max() > forecaster.cutoff_day:
            raise RuntimeError("학습에 panel 학습 제외 대상일이나 cutoff 이후 대상일이 들어갔습니다.")
        actuals = actuals_of(forecaster.data)
        truth = panel_truth(actuals, sites.ids[index], group, days)
        members = np.zeros((len(days), 2, len(sites)), dtype=bool)
        members[:, :, index] = truth["members"]
        pred = forecaster.predict(np.repeat(days, 2), members.reshape(-1, len(sites)), per_seed=True)

        def by_hour(values: np.ndarray) -> np.ndarray:            # [대상일·그룹, 24] → [시각, 그룹]
            return values.reshape(len(days), 2, 24).transpose(0, 2, 1).reshape(-1, 2)

        forecast, experts = by_hour(pred.median), {"xgb": by_hour(pred.xgb_median), "dl": by_hour(pred.dl_median)}
        seed_forecasts = {str(s): by_hour(pred.seed_median[:, k]) for k, s in enumerate(forecaster.seeds)}
        manifest = {"package": "solarvpp2", "model": model_name, "definition": model, "parameters": overrides, "instance": name,
                    "stage": "panel", "partition": "stratified", "panel": record, "decision_cutoff": per.cutoff, "smoke": smoke,
                    "versions": versions(), "forecaster": forecaster.identity(), "environment": report["environment"],
                    "source_sha": report["source_sha"]}
        output = write_output(run_dir / "outputs" / "stratified", sites.ids[index], group, truth["hours"], forecast, manifest)
        report["outputs"] = {"stratified": str(output)}
        np.savez(run_dir / "panel_forecasts.npz", days=days, hours=truth["hours"], forecast=forecast, **experts,
                 seeds=np.array(forecaster.seeds), seed_forecast=np.stack(list(seed_forecasts.values())))
        scores = panel_score(actuals, truth, forecast)
        seed_scores = {s: panel_score(actuals, truth, f) for s, f in seed_forecasts.items()}
        day_w = scores["daily"].sum(axis=1)
        months = delivery_dates(days).strftime("%Y-%m").to_numpy()
        seasons = season_index(days)
        member_cap = (truth["members"] * capacity[None, None, :]).sum(axis=(1, 2))
        evaluation = {"W_krw": {"stratified": scores["W_krw"]}, "W_by_group_krw": {"stratified": scores["W_by_group_krw"]},
                      "expert_W_krw": {"stratified": {e: panel_score(actuals, truth, f)["W_krw"] for e, f in experts.items()}},
                      "seeds": [int(s) for s in forecaster.seeds],
                      "seed_W_krw": {"stratified": {s: v["W_krw"] for s, v in seed_scores.items()}},
                      "scored_hours": scores["scored_hours"], "hours": scores["hours"],
                      "W_by_month_krw": {m: float(day_w[months == m].sum()) for m in sorted(set(months))},
                      "W_by_season_krw": {SEASON_IDS[i]: float(day_w[seasons == i].sum()) for i in sorted(set(seasons.tolist()))},
                      "membership": {"mean_capacity_share": float(member_cap.mean() / sites.capacity.sum()),
                                     "sites_ever": int(truth["members"].any(axis=(0, 1)).sum()),
                                     "unscored_hours": int((~truth["complete"]).sum())}}
        # 일별 벌점 [대상일, 그룹]: "stratified"는 결합 예측, "stratified__s<seed>"는 seed별 결합 제출값이다.
        np.savez(run_dir / "daily_penalty.npz", days=days, stratified=scores["daily"],
                 **{f"stratified__s{s}": v["daily"] for s, v in seed_scores.items()})
        report["daily_penalty"] = str(run_dir / "daily_penalty.npz")
        if not smoke:
            recomputed = evaluate_panel([read_output(output)], load_actuals(spec.data_dir))[0]["W_krw"]
            if abs(recomputed - scores["W_krw"]) > 1e-3:
                raise RuntimeError(f"공통 평가기와 panel 단계의 W가 다릅니다: {recomputed} ≠ {scores['W_krw']}")
            evaluation["contract_recomputed_W_krw"] = {"stratified": recomputed}
        report["evaluation"] = evaluation
        report["status"] = "completed"
    except BaseException:
        report["status"] = "failed"
        (run_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(report_path, report)
        raise
    report["finished"] = pd.Timestamp.now(tz="Asia/Seoul").isoformat()
    write_json(report_path, report)
    save_artifacts(spec, run_dir, forecaster)
    log(f"완료: {run_dir}")
    return run_dir


def run_instance(spec: Spec, name: str, model_name: str | None = None, smoke: bool = False, stage: str = "full",
                 overrides: dict | None = None, panel: str | None = None, need_models: bool = False) -> Path:
    """overrides({"xgb.input": …, "dl.input": …, "seeds": […]})를 주면 등록 정의의 parameter를 바꿔 실행한다.

    run 이름은 MK를 그대로 쓰고, 바꾼 값은 report의 parameters에 적는다. panel 단계는 panel(validation panel 이름)이 필요하다.
    need_models이면 완료된 기록이 있어도 모델 파일을 저장하지 않은 실행은 다시 학습한다.
    """
    if name not in INSTANCES:
        raise ValueError(f"instance는 {INSTANCES} 중 하나입니다.")
    if stage not in STAGES:
        raise ValueError(f"실행 단계는 {STAGES} 중 하나입니다.")
    if (stage == "panel") != (panel is not None):
        raise ValueError("validation panel 이름은 panel 단계에서만, 그리고 반드시 줍니다.")
    model_name = model_name or spec.default_model
    overrides = dict(overrides or {})
    if overrides:
        spec = with_overrides(spec, model_name, overrides)
    model = spec.model(model_name)
    per = period(name)
    if stage == "panel":
        definition = spec.panel(panel)
        if definition["instance"] != name:
            raise ValueError(f"validation panel {panel}은 {definition['instance']} instance에서 실행합니다.")
        days = panel_days(definition["blocks"])
        if days.max() > per.cutoff_h // 24 - int(definition["buffer_days"]):
            raise ValueError("validation panel의 평가일과 완충은 결정 cutoff 이전이어야 합니다.")
        data = dataclasses.replace(load_data(spec), spec=spec)
        forecaster = Forecaster(data, model_name, per.cutoff_h, days_needed=days, smoke=smoke,
                                excluded_days=buffered_days(days, int(definition["buffer_days"])))
        return run_panel_stage(spec, name, model_name, model, per, panel, definition, days, forecaster, smoke, overrides, need_models)
    eval_days = per.days[:spec.config["smoke"]["eval_days"]] if smoke else per.days
    data = dataclasses.replace(load_data(spec), spec=spec)
    sites = data.sites
    forecaster = Forecaster(data, model_name, per.cutoff_h, days_needed=eval_days, smoke=smoke)
    if stage == "forecast":
        return run_forecast_stage(spec, name, model_name, model, per, eval_days, forecaster, smoke, overrides, need_models)
    settings = search_settings(spec, model, smoke)
    objective_days = forecaster.heldout_days[-spec.config["smoke"]["objective_days"]:] if smoke else forecaster.heldout_days
    index = np.flatnonzero(forecaster.cohort)
    capacity = sites.capacity[index]
    share = forecaster.share_low
    risk = RiskModel(forecaster, smoke)
    evaluation = evaluation_payload(spec.config["evaluation"], calendar=spec.calendar if per.has_labels else None,
                                    candidate_pool=spec.config["smoke"]["candidate_pool"] if smoke and per.has_labels else None)
    calculation = full_payload(name, partition_key(risk, index, objective_days, settings), eval_days.tolist(),
                               evaluation, versions("instance"))
    calculation = prediction_payload(calculation)
    run_key = digest(calculation)
    run_dir = spec.artifact_dir / "instances" / name / f"{model_name}{'-smoke' if smoke else ''}-{run_key[:10]}"
    report_path = run_dir / "instance.json"
    if completed_run(report_path, need_models):
        finish_prediction(spec, run_dir, data)
        log(f"완료된 instance 재사용: {run_dir}")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {"package": "solarvpp2", "instance": name, "model": model_name, "definition": model, "parameters": overrides,
              "stage": "full", "smoke": smoke, "run_key": run_key, "calculation": calculation,
              "prediction_calculation": calculation, "evaluation_settings": evaluation,
              "cache_stored": Store.temporary is None, "status": "running", "decision_cutoff": per.cutoff, "cutoff_h": per.cutoff_h, "sites": len(index),
              "capacity_kw": float(capacity.sum()), "minimum_share": share, "objective_days": len(objective_days),
              "heldout_days": len(forecaster.heldout_days), "train_days": len(forecaster.train_days), "train_queries": len(forecaster.train_q),
              "versions": versions(), "environment": environment(), "source_sha": source_sha(),
              "started": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    write_json(report_path, report)
    try:
        report["forecaster"] = forecaster.train()
        report["risk"] = risk.fit()
        lineage = {"objective_last_day": int(objective_days.max()), "train_last_day": int(forecaster.train_q.day.max()),
                   "risk_last_day": int(forecaster.heldout_days.max()), "cutoff_day": forecaster.cutoff_day}
        if max(lineage["objective_last_day"], lineage["train_last_day"], lineage["risk_last_day"]) > forecaster.cutoff_day:
            raise RuntimeError(f"결정에 cutoff 이후 대상일이 들어갔습니다: {lineage}")
        report["lineage"] = lineage
        objective = Objective(forecaster, risk, index, objective_days, settings)
        state = SearchState(objective, spec.config["runtime"]["search_state_max_gib"])
        starts = starting_points(capacity, index, sites, share, settings)
        log(f"{name} 배정 탐색: 시작점 {list(starts)}, 목적 대상일 {len(objective_days)}일, 기여 저장 {state.store_device.type}")
        found = search(state, starts, capacity, share, settings)
        state_record = state.describe()
        del state
        forecaster.release_caches()
        partitions = {"stratified": repair(stratified_split(index, sites), capacity, share), "searched": found["x"]}
        if per.has_labels:
            partitions = {"original": sites.group[index] == 0, **partitions}
        if fused_cache_fits(forecaster, len(objective_days)):
            forecaster.enable_fused_cache()
        values = objective(np.stack(list(partitions.values())))
        report["partition"] = {"best_start": found["best_start"], "starts": found["starts"], "search_state": state_record,
                               "evaluations": objective.evaluations,
                               "objective_per_day_krw": dict(zip(partitions, values.tolist())),
                               "capacity_kw": {k: [float(capacity[x].sum()), float(capacity[~x].sum())] for k, x in partitions.items()},
                               "feasible": {k: capacity_ok(x, capacity, share) for k, x in partitions.items()}}
        hours = (eval_days[:, None] * 24 + np.arange(24)[None, :]).ravel()
        forecaster.release_caches()
        forecasts = predict_partitions(forecaster, index, np.stack(list(partitions.values())), eval_days)
        outputs = {}
        for (label, x), forecast in zip(partitions.items(), forecasts):
            manifest = {"package": "solarvpp2", "model": model_name, "definition": model, "instance": name, "partition": label,
                        "decision_cutoff": per.cutoff, "smoke": smoke, "versions": versions(), "forecaster": forecaster.identity(),
                        "risk_key": risk.key(), "objective_per_day_krw": report["partition"]["objective_per_day_krw"][label],
                        "environment": report["environment"], "source_sha": report["source_sha"]}
            outputs[label] = write_output(run_dir / "outputs" / label, sites.ids[index], np.where(x, 0, 1), hours, forecast, manifest)
        report["outputs"] = {k: str(v) for k, v in outputs.items()}
        if per.has_labels:
            report["evaluation"] = evaluate_instance(spec, forecaster, risk, objective, index, capacity, share, partitions, forecasts,
                                                     outputs, eval_days, name, smoke)
        report["status"] = "completed"
    except BaseException:
        report["status"] = "failed"
        (run_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(report_path, report)
        raise
    report["finished"] = pd.Timestamp.now(tz="Asia/Seoul").isoformat()
    write_json(report_path, report)
    save_artifacts(spec, run_dir, forecaster)
    finish_prediction(spec, run_dir, data)
    log(f"완료: {run_dir}")
    return run_dir


def evaluate_instance(spec: Spec, forecaster: Forecaster, risk: RiskModel, objective: Objective, index: np.ndarray, capacity: np.ndarray,
                      share: float, partitions: dict, forecasts: np.ndarray, outputs: dict, eval_days: np.ndarray, name: str,
                      smoke: bool) -> dict:
    """비교 2·3, 후보 배정 순위 상관, R calibration. 전체 기간이면 공통 평가기로 같은 W를 다시 계산해 대조한다."""
    cfg = spec.config["evaluation"]
    blocks, reps, seed = tuple(cfg["bootstrap_blocks"]), int(cfg["bootstrap_replicates"]), int(cfg["bootstrap_seed"])
    scores = {label: realized(forecaster, index, x, eval_days, forecast) for (label, x), forecast in zip(partitions.items(), forecasts)}
    result = {"W_krw": {k: v["W_krw"] for k, v in scores.items()}, "scored_hours": scores["searched"]["scored_hours"],
              "searched_vs_stratified": paired(scores["searched"]["daily"], scores["stratified"]["daily"], blocks, reps, seed)}
    if "original" in scores:
        result["stratified_vs_original"] = paired(scores["stratified"]["daily"], scores["original"]["daily"], blocks, reps, seed)
    result["regime_bias"] = regime_bias(spec, forecaster, index, partitions, forecasts, eval_days)
    if not smoke:
        result["contract_recomputed_W_krw"] = contract_check(spec, outputs, scores, name)
    pool_size = spec.config["smoke"]["candidate_pool"] if smoke else int(cfg["candidate_pool_size"])
    rng = np.random.default_rng(int(cfg["candidate_pool_seed"]))
    pool = np.stack([repair(balanced_split(capacity, rng, share), capacity, share) for _ in range(pool_size)] + list(partitions.values()))
    if fused_cache_fits(forecaster, len(objective.days)):
        forecaster.enable_fused_cache()
    predicted = objective(pool)
    forecaster.release_caches()
    pool_forecasts = predict_partitions(forecaster, index, pool, eval_days)
    realized_w = np.array([realized(forecaster, index, x, eval_days, f)["W_krw"] for x, f in zip(pool, pool_forecasts)])
    result["candidate_pool"] = {"random_halves": pool_size, "labels": [f"random{i}" for i in range(pool_size)] + list(partitions),
                                "spearman": float(spearmanr(predicted, realized_w).statistic),
                                "predicted_objective_per_day_krw": predicted.tolist(), "realized_W_krw": realized_w.tolist(),
                                "searched_share_of_random_with_lower_W": float((realized_w[:pool_size] < result["W_krw"]["searched"]).mean())}
    table = risk_calibration(forecaster, risk, index, partitions, eval_days, int(cfg["calibration_bins"]))
    result["risk_calibration_by_predicted_decile"] = table
    result["high_predicted_risk_mean_calibration"] = table[-1] if table else None
    return result


def select_run(spec: Spec, model_name: str, name: str) -> tuple[Path, dict, dict]:
    """같은 MK의 완료 run이 여러 개이면 현재 등록 정의와 component version이 같은 run 가운데 가장 늦게 끝난 run을 고른다.

    smoke run과 forecast 단계 run은 고르지 않는다. 고른 run과 함께 후보 수, 조건이 맞지 않아 뺀 run을 기록으로 돌려준다.
    """
    definition, current = spec.model(model_name), versions()
    runs = []
    for path in sorted((spec.artifact_dir / "instances" / name).glob(f"{model_name}-*/instance.json")):
        report = read_json(path)
        if (report.get("status") == "completed" and not report.get("smoke") and report.get("model") == model_name
                and report.get("stage", "full") == "full"):
            runs.append((path.parent, report))
    matching = [(d, r) for d, r in runs if r.get("definition") == definition and r.get("versions") == current]
    if not matching:
        raise ValueError(f"현재 정의·component version과 같은 완료된 {model_name} {name} instance가 없습니다(완료 {len(runs)}개).")
    directory, report = max(matching, key=lambda item: item[1].get("finished", ""))
    record = {"selected": str(directory), "finished": report.get("finished"), "completed_runs": len(runs),
              "excluded_by_definition_or_versions": [str(d) for d, r in runs if (d, r) not in matching]}
    return directory, report, record


def recommend(spec: Spec, model_name: str | None = None) -> dict:
    """H·B instance의 ΔW(R 배정 − 관측소 층화 배정) 합이 음수이면 R 배정을, 아니면 관측소 층화 배정을 test 배정으로 추천한다."""
    model_name = model_name or spec.default_model
    rule = spec.model(model_name)["recommendation"]
    reports, selection = {}, {}
    for name in [*rule["instances"], "test"]:
        directory, report, selection[name] = select_run(spec, model_name, name)
        if period(name).has_labels:
            evaluation_dir = evaluate_saved(spec, directory)
            report = {**report, "evaluation": read_json(evaluation_dir / "evaluation.json")["evaluation"]}
            selection[name]["evaluation"] = str(evaluation_dir)
        reports[name] = (directory, report)
    delta = {name: reports[name][1]["evaluation"]["searched_vs_stratified"]["delta_W_krw"] for name in rule["instances"]}
    choice = "searched" if sum(delta.values()) < 0 else rule["baseline"]
    chosen = Path(reports["test"][1]["outputs"][choice])
    record = {"model": model_name, "rule": "H·B 합산 ΔW(R − stratified)가 음수이면 R, 아니면 stratified", "delta_W_krw": delta,
              "sum_delta_W_krw": float(sum(delta.values())), "choice": choice, "submission": str(chosen),
              "validation": validate_submission(chosen, spec.data_dir), "instances": {k: str(v[0]) for k, v in reports.items()},
              "selection": selection,
              "created": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    out = spec.artifact_dir / "recommendation" / f"{model_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, record)
    return record
