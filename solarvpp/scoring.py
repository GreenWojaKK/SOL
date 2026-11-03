"""정산 계산: 그룹 예측과 배정(발전소 잔차의 그룹 합)의 S·W, 분포·중심 지표, 기여 분해, 후보 비교, paired moving-block bootstrap."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .data import GROUPS, clock_hour
from .forecast.queries import EvalPanel

if TYPE_CHECKING:
    from .bundling.partition import Instance

PENALTY_RATE = 0.10


def settlement(actual: np.ndarray, forecast: np.ndarray, smp: np.ndarray, k: np.ndarray, mask: np.ndarray) -> dict:
    """S = Σ M·SMP·Y − W, W = 0.10·Σ M·SMP·k·|a − Y|."""
    if actual.shape != forecast.shape or mask.shape != actual.shape:
        raise ValueError("실적·예측·mask의 shape이 같아야 합니다.")
    if not np.isfinite(forecast).all() or (forecast < 0).any():
        raise ValueError("예측은 유한하고 0 이상이어야 합니다.")
    if not np.isfinite(actual[mask]).all():
        raise ValueError("정답 mask가 1인 시각의 실적은 유한해야 합니다.")
    price = np.broadcast_to(np.asarray(smp, dtype=np.float64)[:, None], actual.shape)
    weight = np.broadcast_to(np.asarray(k, dtype=np.float64)[:, None], actual.shape)
    revenue = float((price * np.where(mask, actual, 0.0)).sum())
    penalty = float((PENALTY_RATE * price * weight * np.where(mask, np.abs(forecast - np.nan_to_num(actual)), 0.0)).sum())
    return {"revenue_krw": revenue, "W_krw": penalty, "S_krw": revenue - penalty, "group_hours": int(mask.sum())}


def _hour_penalty(panel: EvalPanel, forecast: np.ndarray) -> np.ndarray:
    error = np.where(panel.m, np.abs(forecast - np.nan_to_num(panel.y)), 0.0)
    return PENALTY_RATE * panel.smp[:, None] * panel.k[:, None] * error


def _distribution(panel: EvalPanel, q: np.ndarray, levels: np.ndarray, select: np.ndarray) -> dict:
    if not select.any():
        return {"rows": 0}
    y = panel.y[select][:, None]
    qs = q[select]
    error = y - qs
    pinball = np.maximum(levels * error, (levels - 1) * error)
    median = qs[:, int(np.flatnonzero(np.isclose(levels, 0.5))[0])]
    residual = median - y[:, 0]
    cap = np.broadcast_to(panel.cap[None, :], panel.m.shape)[select]
    return {"rows": int(select.sum()), "mean_pinball_kwh": float(pinball.mean()),
            "coverage_05_95": float(((y[:, 0] >= qs[:, 0]) & (y[:, 0] <= qs[:, -1])).mean()),
            "interval_width_kwh": float((qs[:, -1] - qs[:, 0]).mean()), "median_mae_kwh": float(np.abs(residual).mean()),
            "nrmse_capacity": float(np.sqrt(np.mean((residual / cap) ** 2)))}


def _ramp(panel: EvalPanel, forecast: np.ndarray, daylight_only: bool) -> float | None:
    pairs = panel.m[1:] & panel.m[:-1]
    if daylight_only:
        pairs &= panel.daylight[1:] | panel.daylight[:-1]
    if not pairs.any():
        return None
    error = np.diff(forecast, axis=0) - np.diff(np.nan_to_num(panel.y), axis=0)
    return float(np.abs(error[pairs]).mean())


def score(panel: EvalPanel, q: np.ndarray, forecast: np.ndarray, levels: np.ndarray) -> tuple[dict, np.ndarray]:
    """panel 정답으로 채점한다. 반환: 보고서와 일별 벌점 [대상일, 그룹]."""
    if not panel.has_labels:
        raise ValueError("정답이 없는 기간은 채점하지 않습니다.")
    levels = np.asarray(levels, dtype=np.float64)
    total = settlement(panel.y, forecast, panel.smp, panel.k, panel.m)
    report = {"panel_hash": panel.panel_hash, "cohort_id": panel.cohort_id, "period": panel.period,
              "cohort": {g: {"pv": int(panel.n_pv[i]), "capacity_kw": float(panel.cap[i])} for i, g in enumerate(GROUPS)},
              "S_cohort_krw": total["S_krw"], "W_cohort_krw": total["W_krw"], "revenue_krw": total["revenue_krw"],
              "group_hours": total["group_hours"], "by_group": {}}
    for i, g in enumerate(GROUPS):
        mask = np.zeros_like(panel.m)
        mask[:, i] = panel.m[:, i]
        part = settlement(panel.y, forecast, panel.smp, panel.k, mask)
        part["W_per_kw_krw"] = part["W_krw"] / panel.cap[i]
        report["by_group"][g] = part
    penalty = _hour_penalty(panel, forecast)
    revenue = panel.smp[:, None] * np.where(panel.m, np.nan_to_num(panel.y), 0.0)
    frame = pd.DataFrame({"month": np.repeat(panel.month, len(GROUPS)), "group": np.tile(GROUPS, len(panel.hours)),
                          "k": np.repeat(panel.k, len(GROUPS)), "smp_bin": np.repeat(panel.smp_bin, len(GROUPS)),
                          "hour": np.repeat(clock_hour(panel.hours), len(GROUPS)), "W_krw": penalty.ravel(), "revenue_krw": revenue.ravel(),
                          "label": panel.m.ravel()})
    report["breakdown"] = {
        "month_group": frame.groupby(["month", "group"])[["W_krw", "revenue_krw", "label"]].sum().reset_index().to_dict("records"),
        "k": frame.groupby("k")[["W_krw", "revenue_krw"]].sum().reset_index().to_dict("records"),
        "smp_quintile": frame.groupby("smp_bin")[["W_krw", "revenue_krw"]].sum().reset_index().to_dict("records"),
        "hour": frame.groupby("hour")[["W_krw", "revenue_krw"]].sum().reset_index().to_dict("records"),
    }
    report["label_coverage"] = {g: pd.Series(panel.m[:, i]).groupby(panel.month).sum().astype(int).to_dict() for i, g in enumerate(GROUPS)}
    for name, select in (("all", panel.m), ("daylight", panel.m & panel.daylight)):
        report[name] = _distribution(panel, q, levels, select)
        report[name]["ramp_mae_kwh"] = _ramp(panel, forecast, name == "daylight")
    daily = penalty.reshape(-1, 24, len(GROUPS)).sum(axis=1)
    return report, daily


def block_bootstrap(delta: np.ndarray, block: int, replicates: int, seed: int, level: float = 0.95) -> dict:
    """대상일별 차이의 합에 대한 moving-block bootstrap 구간."""
    delta = np.asarray(delta, dtype=np.float64)
    n = len(delta)
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    n_blocks = int(np.ceil(n / block))
    sums = np.empty(replicates)
    for r in range(replicates):
        picks = rng.choice(starts, size=n_blocks, replace=True)
        sample = np.concatenate([delta[s:s + block] for s in picks])[:n]
        sums[r] = sample.sum()
    alpha = (1 - level) / 2
    return {"block_days": block, "replicates": replicates, "seed": seed, "total": float(delta.sum()),
            "lower": float(np.quantile(sums, alpha)), "upper": float(np.quantile(sums, 1 - alpha))}


def compare(candidate: dict, baseline: dict, candidate_daily: np.ndarray, baseline_daily: np.ndarray, seed: int = 42) -> dict:
    """같은 evaluation panel에서만 비교한다. ΔS = −ΔW이다."""
    if candidate["panel_hash"] != baseline["panel_hash"]:
        raise ValueError("evaluation panel hash가 달라 비교할 수 없습니다.")
    if not np.isclose(candidate["revenue_krw"], baseline["revenue_krw"], rtol=0, atol=1e-6):
        raise ValueError("수익항이 달라 비교할 수 없습니다.")
    delta_daily = -(candidate_daily - baseline_daily).sum(axis=1)
    result = {"panel_hash": candidate["panel_hash"], "cohort_id": candidate["cohort_id"],
              "delta_S_krw": candidate["S_cohort_krw"] - baseline["S_cohort_krw"],
              "delta_W_krw": candidate["W_cohort_krw"] - baseline["W_cohort_krw"], "by_group": {}}
    for i, g in enumerate(GROUPS):
        cap = candidate["cohort"][g]["capacity_kw"]
        dw = candidate["by_group"][g]["W_krw"] - baseline["by_group"][g]["W_krw"]
        result["by_group"][g] = {"delta_W_krw": dw, "delta_W_per_kw_krw": dw / cap,
                                 "bootstrap_7d": block_bootstrap(-(candidate_daily[:, i] - baseline_daily[:, i]), 7, 2000, seed)}
    result["bootstrap"] = {f"{b}d": block_bootstrap(delta_daily, b, 2000, seed) for b in (7, 14)}
    return result


def score_partition(inst: Instance, x: np.ndarray) -> tuple[dict, np.ndarray]:
    """평가 기간의 정확한 bottom-up 정산 벌점. 일별 벌점은 bootstrap에 쓴다."""
    E, w = inst.eval_error, inst.eval_w
    R1 = E[:, x].sum(axis=1)
    total = E.sum(axis=1)
    R2 = total - R1
    hourly = w * (np.abs(R1) + np.abs(R2))
    W_split, W_merge = float(hourly.sum()), float((w * np.abs(total)).sum())
    day = inst.eval_hours // 24
    daily = np.bincount(day - day.min(), weights=hourly, minlength=int(day.max() - day.min() + 1))
    active = np.abs(R1) + np.abs(R2) > 0
    cap = [float(inst.capacity[x].sum()), float(inst.capacity[~x].sum())]
    b = (cap[0] - cap[1]) / inst.total
    return {"W_krw": W_split, "W_merge_krw": W_merge, "lost_krw": W_split - W_merge, "lost_ratio": (W_split - W_merge) / W_split,
            "group_error_corr": float(np.corrcoef(R1[active], R2[active])[0, 1]), "imbalance": b, "B": b * b,
            "capacity_kw": dict(zip(GROUPS, cap)), "sites": dict(zip(GROUPS, [int(x.sum()), int((~x).sum())]))}, daily
