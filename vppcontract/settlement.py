"""정산식과 paired 비교. W = 0.10·Σ SMP·k·|a − Y|이고 수익항은 배정·제출값과 무관하다."""
from __future__ import annotations

import numpy as np

from .timeaxis import SEASON_IDS, season_index

PENALTY_RATE = 0.10


def hourly_penalty(actual: np.ndarray, forecast: np.ndarray, smp: np.ndarray, k: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """그룹·시각별 정산 벌점 [시각, 그룹]. mask가 0인 칸은 0이다."""
    actual, forecast, mask = np.asarray(actual, dtype=np.float64), np.asarray(forecast, dtype=np.float64), np.asarray(mask, dtype=bool)
    if actual.shape != forecast.shape or mask.shape != actual.shape:
        raise ValueError("실적·예측·mask의 shape이 같아야 합니다.")
    if not np.isfinite(forecast).all() or (forecast < 0).any():
        raise ValueError("예측은 유한하고 0 이상이어야 합니다.")
    if not np.isfinite(actual[mask]).all():
        raise ValueError("정답 mask가 1인 칸의 실적은 유한해야 합니다.")
    weight = PENALTY_RATE * np.asarray(smp, dtype=np.float64) * np.asarray(k, dtype=np.float64)
    return weight[:, None] * np.where(mask, np.abs(forecast - np.nan_to_num(actual)), 0.0)


def daily(hours: np.ndarray, values: np.ndarray) -> np.ndarray:
    """시각별 값 [시각, ...]을 대상일별 합 [대상일, ...]으로 묶는다. hours는 연속한 전체 대상일이어야 한다."""
    hours = np.asarray(hours)
    if len(hours) % 24 or hours[0] % 24 or not np.array_equal(hours, np.arange(hours[0], hours[0] + len(hours))):
        raise ValueError("대상일 단위로 연속한 시간축이어야 합니다.")
    return np.asarray(values, dtype=np.float64).reshape(len(hours) // 24, 24, *np.shape(values)[1:]).sum(axis=1)


def season_totals(days: np.ndarray, daily_values: np.ndarray) -> dict:
    """대상일별 값 [대상일, ...]의 계절별 합(SEASON_IDS 이름 → 합). 계절은 대상일의 월로 정한다."""
    index = season_index(np.asarray(days))
    values = np.asarray(daily_values, dtype=np.float64).reshape(len(index), -1).sum(axis=1)
    return {name: float(values[index == i].sum()) for i, name in enumerate(SEASON_IDS)}


def block_bootstrap(delta: np.ndarray, block: int, replicates: int, seed: int, level: float = 0.95) -> dict:
    """대상일별 차이의 합에 대한 moving-block bootstrap 구간. 대상일 수가 블록보다 적으면 전체를 한 블록으로 쓴다."""
    delta = np.asarray(delta, dtype=np.float64)
    n = len(delta)
    used = min(int(block), n)
    rng = np.random.default_rng(seed)
    starts = np.arange(n - used + 1)
    n_blocks = int(np.ceil(n / used))
    sums = np.empty(replicates)
    for r in range(replicates):
        picks = rng.choice(starts, size=n_blocks, replace=True)
        sums[r] = np.concatenate([delta[s:s + used] for s in picks])[:n].sum()
    alpha = (1 - level) / 2
    return {"block_days": block, "effective_block_days": used, "replicates": replicates, "seed": seed, "total": float(delta.sum()),
            "lower": float(np.quantile(sums, alpha)), "upper": float(np.quantile(sums, 1 - alpha))}


def day_runs(days: np.ndarray) -> list[np.ndarray]:
    """연속한 대상일 묶음마다의 위치. 대상일은 겹치지 않는 오름차순이어야 한다."""
    days = np.asarray(days, dtype=np.int64)
    if np.any(np.diff(days) <= 0):
        raise ValueError("대상일은 겹치지 않는 오름차순이어야 합니다.")
    return np.split(np.arange(len(days)), np.flatnonzero(np.diff(days) > 1) + 1)


def run_bootstrap(daily_a: np.ndarray, daily_b: np.ndarray, days: np.ndarray, replicates: int = 2000, seed: int = 42,
                  level: float = 0.95) -> dict:
    """연속한 대상일 묶음을 단위로 다시 뽑는 bootstrap. ΔW = W_a − W_b와 W 변화율 W_a / W_b − 1의 구간을 낸다.

    입력은 [대상일] 또는 [대상일, 그룹]이고, 음수이면 a의 벌점이 작다. validation panel처럼 떨어진 묶음으로 평가할 때 쓴다.
    """
    a, b = (np.asarray(v, dtype=np.float64) for v in (daily_a, daily_b))
    a, b = (v.sum(axis=1) if v.ndim == 2 else v for v in (a, b))
    runs = day_runs(days)
    sums_a, sums_b = np.array([a[r].sum() for r in runs]), np.array([b[r].sum() for r in runs])
    picks = np.random.default_rng(seed).integers(0, len(runs), size=(replicates, len(runs)))
    total_a, total_b = sums_a[picks].sum(axis=1), sums_b[picks].sum(axis=1)
    alpha = (1 - level) / 2
    delta, change = total_a - total_b, total_a / total_b - 1
    return {"blocks": len(runs), "replicates": replicates, "seed": seed, "level": level,
            "delta_W_krw": float(a.sum() - b.sum()), "W_change": float(a.sum() / b.sum() - 1),
            "delta_W_krw_interval": [float(np.quantile(delta, alpha)), float(np.quantile(delta, 1 - alpha))],
            "W_change_interval": [float(np.quantile(change, alpha)), float(np.quantile(change, 1 - alpha))]}


def paired(daily_a: np.ndarray, daily_b: np.ndarray, blocks=(7, 14), replicates: int = 2000, seed: int = 42) -> dict:
    """ΔW = W_a − W_b와 대상일 block bootstrap 구간. 입력은 [대상일] 또는 [대상일, 그룹]이고, 음수이면 a의 벌점이 작다."""
    a, b = (np.asarray(v, dtype=np.float64) for v in (daily_a, daily_b))
    delta = (a.sum(axis=1) if a.ndim == 2 else a) - (b.sum(axis=1) if b.ndim == 2 else b)
    return {"delta_W_krw": float(delta.sum()), "bootstrap": {f"{b}d": block_bootstrap(delta, b, replicates, seed) for b in blocks}}
