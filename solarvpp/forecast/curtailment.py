"""출력제어 후처리 module.

출력제어는 계통 운영 정책에 따라 시기마다 빈도가 달라지는 사건으로 본다. MK4의 XGB·DL·blend는 그대로 두고, blend의 중앙값에
출력제어로 더 필요한 보정을 더한다. MK4는 출력제어 날을 포함한 실제 발전량으로 학습했으므로, 출력제어가 없을 때의 잔차 분포는 같은
pipeline의 그룹별 OOT 예측 잔차에서 구한다. 출력제어가 걸렸을 때 얼마나 줄어드는지는 발전소별로 정해 그룹으로 모은다.

1. 표시: cutoff 이전 발전량에서 맑은 날(발전소 WB 관측소의 표시 10~16시 관측 운량 평균이 clear_cloud_pct 미만) 낮(표시 11~15시)만
   발전소 자신의 맑은 날 기준값보다 크게 떨어지고, 아침(8~9시)과 저녁(16~17시)은 모두 실적이 있고 정상인 발전소-날을 찾는다.
   아침이나 저녁 실적이 없는 날은 고장·통신 단절과 구분할 수 없으므로 표시하지 않는다. 그날 표시 9~16시 가운데 기준값 대비 비율이
   cut_ratio 아래인 시각을 꺼진 시각으로 두고, 같은 시각에 min_plants곳 이상, 합계 min_cut_kw 이상이 함께 꺼진 시각을 출력제어
   시각으로 둔다. 기준값은 발전소·달력 월·시각별 맑은 날 이용률의 reference_quantile 분위수이다.
2. 확률(전역): 출력제어는 지시 한 번에 여러 발전소가 함께 꺼지는 사건이므로 시각마다 확률 하나를 둔다. 기본 확률은 예보 일사량·
   봄가을·토요일·일요일과 공휴일로 맞춘 L2 logistic이다(학습 기간 전체, 같은 가중). 최근 보정 b는 cutoff 시점에서 반감기 가중
   2^(−경과일/H)로 기본 확률의 logit에 더할 상수를 맞춘다. 평가 기간에는 실적이 들어오지 않으므로 기간 전체에 cutoff 시점의 b를 쓴다.
3. 발전소별로 잃는 양: 출력제어 시각에 발전소마다 판정할 수 있었던 횟수 대비 꺼진 횟수(참여)와 꺼졌을 때의 깊이(1 − 기준값 대비
   비율)를 같은 반감기 가중으로 센다. prior_strength가 0보다 크면 용량 구분(LARGE_KW 이상·미만)의 평균 참여·깊이를 그만큼의 가중
   표본으로 섞는다(한 번도 꺼지지 않았거나 판정 기록이 없는 발전소도 구분 평균에서 출발한다). 그룹이 시각 h에 잃는 비율 φ_g(h)는
   발전소별 (참여 × 깊이)를 그 시각 맑은 날 기대 발전량(용량 × 기준 이용률)으로 가중 평균한 값이다.
4. 잔차 분포(그룹): 같은 모델(module 제외)·배정·seed로 cutoff 이전에 학습한 OOT run(source)의 그룹별 중앙값 예측 m과 실적 y로 상대
   잔차 (y − m) / m을 cutoff까지 모은다. G0는 출력제어가 없는 시각 가운데 예보상 맑은 낮(표시 9~16시, 예보 맑음 지수
   forecast_clearness 이상)의 잔차이고, m이 그룹 용량의 min_forecast_share 이상인 시각만 쓰며 같은 반감기 가중을 준다. 꺼진 발전소는
   날씨와 상관없이 자기 몫을 잃으므로, 출력제어가 걸렸을 때의 분포 G1은 G0 표본을 (1 + r)(1 − φ) − 1로 옮긴 분포이다.
5. 추가 보정: 출력제어 확률 p와 φ에서 (1 − p)·G0 + p·G1 혼합 분포의 가중 중앙값과 G0 가중 중앙값의 차이 Δ(p, φ)를 추가 보정으로
   두고, 중앙값을 m·(1 + Δ)로 옮긴다(p = 0이나 φ = 0이면 Δ = 0). 다른 분위수도 같은 양만큼 옮긴 뒤 [0, 그룹 용량]으로 자르고 정렬한다.

cutoff 이전에 출력제어 시각이 없으면 module은 꺼지고 입력 분위수를 그대로 돌려준다.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from vppcontract.timeaxis import HolidayCalendar, calendar_month, delivery_dates

from ..data import PROVIDERS, ROUNDS, Base, Sites, Weather
from .features import KASTEN_CZEPLAK, PHYSICS_SOURCES, clear_sky_ghi
from .queries import actual_view

MID = slice(10, 15)              # 표시 11~15시
MORNING = slice(7, 9)            # 표시 8~9시
EVENING = slice(15, 17)          # 표시 16~17시
WINDOW = (8, 16)                 # 꺼진 시각을 찾는 slot: 표시 9~16시
MIN_REFERENCE = 0.05             # 기준 이용률이 이보다 작은 칸은 비율을 계산하지 않는다
SHOULDER_MONTHS = (3, 4, 5, 9, 10, 11)
FEATURES = ("forecast_ghi", "shoulder_season", "saturday", "rest_day")
MISSING_CLOUD = 50.0             # 두 제공사 예보가 모두 없는 발전소-시각의 운량(%)
OFFSET_PRIOR = 10.0              # 최근 보정 b의 사전 표준편차(logit)
GRID = np.linspace(0.0, 1.0, 101)  # 추가 보정 Δ를 미리 계산하는 출력제어 확률 격자
PHI = np.linspace(0.0, 0.5, 51)    # 추가 보정 Δ를 미리 계산하는 그룹 감소 비율 격자
LARGE_KW = 500.0                   # 용량 사전값의 큰 발전소 기준(kW)


@dataclass
class Labels:
    ratio: np.ndarray            # 발전소 기준값 대비 이용률 [대상일, 24, 발전소]. 계산할 수 없으면 NaN
    clear: np.ndarray            # 발전소의 맑은 날 [대상일, 발전소]
    cut: np.ndarray              # 꺼진 시각 [대상일, 24, 발전소]
    event: np.ndarray            # 출력제어 시각 [대상일, 24]
    reference: np.ndarray        # 발전소·달력 월·시각별 맑은 날 기준 이용률 [12, 24, 발전소]


@dataclass
class Source:
    """보정 잔차를 주는 OOT run의 그룹별 중앙값 예측과 실적."""
    run: str
    run_key: str
    cutoff_h: int                # source run의 학습 cutoff
    hours: np.ndarray            # [시각]
    median: np.ndarray           # 중앙값 예측 [시각, 그룹] kWh
    actual: np.ndarray           # 실적 [시각, 그룹] kWh (채점 mask 밖은 NaN)
    capacity: np.ndarray         # 그룹 용량 [그룹] kW


def _nan_mean(x: np.ndarray, axis: int) -> np.ndarray:
    count = np.isfinite(x).sum(axis=axis)
    return np.where(count > 0, np.nansum(x, axis=axis) / np.maximum(count, 1), np.nan)


def _nan_max(x: np.ndarray, axis: int) -> np.ndarray:
    return np.where(np.isfinite(x).any(axis=axis), np.nanmax(np.where(np.isfinite(x), x, -np.inf), axis=axis), np.nan)


def curtailment_labels(energy: np.ndarray, capacity: np.ndarray, wb_station: np.ndarray, cloud: np.ndarray, settings: dict) -> Labels:
    """cutoff까지의 하루 단위 발전량으로 꺼진 시각과 출력제어 시각을 표시한다.

    energy는 [대상일 × 24, 발전소] kWh(유효하지 않거나 cutoff 이후면 NaN), cloud는 같은 시각의 WB 관측소 관측 운량 [시각, 관측소](%)이다.
    """
    n_days = len(energy) // 24
    u = (energy[:n_days * 24] / capacity).reshape(n_days, 24, -1)
    station_cloud = _nan_mean(cloud[:n_days * 24].reshape(n_days, 24, -1)[:, 9:16], axis=1)       # 표시 10~16시
    clear = np.nan_to_num(station_cloud[:, wb_station], nan=np.inf) < settings["clear_cloud_pct"]
    month = calendar_month(np.arange(n_days))
    reference = np.full((12, 24, u.shape[2]), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                   # 맑은 날이 없는 칸은 기준값이 NaN이다
        for m in range(1, 13):
            rows = month == m
            if rows.any():
                reference[m - 1] = np.nanquantile(np.where(clear[rows][:, None, :], u[rows], np.nan), settings["reference_quantile"], axis=0)
    ref = reference[month - 1]
    ratio = u / np.where(ref > MIN_REFERENCE, ref, np.nan)
    sides = np.stack([_nan_max(ratio[:, MORNING], axis=1), _nan_max(ratio[:, EVENING], axis=1)])
    shoulders = (np.nan_to_num(sides, nan=-np.inf) >= settings["shoulder_ratio"]).all(axis=0)
    cut_day = clear & (_nan_mean(ratio[:, MID], axis=1) < settings["cut_ratio"]) & shoulders
    cut = cut_day[:, None, :] & (ratio < settings["cut_ratio"])
    cut[:, :WINDOW[0]] = False
    cut[:, WINDOW[1]:] = False
    event = (cut.sum(axis=2) >= settings["min_plants"]) & ((cut * capacity).sum(axis=2) >= settings["min_cut_kw"])
    return Labels(ratio, clear, cut, event, reference)


def irradiance(weather: Weather, sites: Sites, solar: np.ndarray, hours: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """시각별 (예보 일사량, clear-sky 일사량)(kW/m², 전 발전소 용량 가중 평균). 예보 일사량은 round 17 두 제공사의 예보 운량으로
    Haurwitz clear-sky 일사량을 보정해 평균한 값이다."""
    weight = sites.capacity / sites.capacity.sum()
    index = {p: weather.channel_names(p, "forecast").index(PHYSICS_SOURCES[p][0]) for p in PROVIDERS}
    forecast, clear = np.empty(len(hours)), np.empty(len(hours))
    for start in range(0, len(hours), chunk):
        h = np.asarray(hours[start:start + chunk])
        g_cs = clear_sky_ghi(np.asarray(solar[h][..., 1], dtype=np.float64))
        clouds = np.stack([np.asarray(weather.values[f"forecast_{p}"][ROUNDS.index(17), h, :, index[p]], dtype=np.float64)[:, sites.station_index[p]]
                           for p in PROVIDERS])
        cloud = np.nan_to_num(_nan_mean(clouds, axis=0), nan=MISSING_CLOUD)
        g = g_cs * (1.0 - KASTEN_CZEPLAK[0] * np.clip(cloud / 100.0, 0.0, 1.0) ** KASTEN_CZEPLAK[1])
        forecast[start:start + chunk] = g @ weight / 1000.0
        clear[start:start + chunk] = g_cs @ weight / 1000.0
    return forecast, clear


def design(calendar: HolidayCalendar, hours: np.ndarray, ghi: np.ndarray) -> np.ndarray:
    """확률 모델 입력 [시각, FEATURES]: 예보 일사량, 봄·가을, 토요일, 일요일·공휴일·근로자의 날."""
    days = np.asarray(hours) // 24
    unique, inverse = np.unique(days, return_inverse=True)
    saturday, rest = calendar.day_kinds(unique)
    shoulder = np.isin(calendar_month(unique), SHOULDER_MONTHS)
    return np.column_stack([ghi, shoulder[inverse], saturday[inverse], rest[inverse]]).astype(np.float64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _log_likelihood(eta: np.ndarray, y: np.ndarray, weight: np.ndarray | float = 1.0) -> float:
    return float((weight * (y * eta - np.logaddexp(0.0, eta))).sum())


def fit_logistic(x: np.ndarray, y: np.ndarray, ridge: float) -> dict:
    """표준화한 입력의 L2 logistic(절편은 벌점 없음)을 Newton 반복으로 맞춘다. 목적값이 늘지 않는 단계는 반으로 줄인다."""
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    z = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    penalty = np.full(z.shape[1], ridge)
    penalty[0] = 0.0
    beta = np.zeros(z.shape[1])
    beta[0] = np.log((y.sum() + 0.5) / (len(y) - y.sum() + 0.5))          # 사건 비율에서 시작한다

    def objective(b: np.ndarray) -> float:
        return _log_likelihood(z @ b, y) - 0.5 * (penalty * b * b).sum()

    current = objective(beta)
    for _ in range(200):
        p = _sigmoid(z @ beta)
        gradient = z.T @ (y - p) - penalty * beta
        hessian = (z * (p * (1 - p))[:, None]).T @ z + np.diag(penalty) + 1e-9 * np.eye(len(beta))
        step = np.linalg.solve(hessian, gradient)
        for _ in range(60):
            candidate = objective(beta + step)
            if candidate >= current - 1e-12:
                break
            step = step / 2
        beta, current = beta + step, candidate
        if np.abs(step).max() < 1e-10:
            break
    return {"features": list(FEATURES), "mean": mean, "scale": scale, "intercept": float(beta[0]), "coef": beta[1:]}


def logit(model: dict, x: np.ndarray) -> np.ndarray:
    return model["intercept"] + ((x - model["mean"]) / model["scale"]) @ model["coef"]


def recent_offset(eta: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    """반감기 가중 log-likelihood와 사전 분포 N(0, OFFSET_PRIOR²)의 합을 최대로 하는 logit 상수 b.
    목적의 기울기가 b에 대해 줄어들기만 하므로 기울기가 0인 점을 구간 이분법으로 찾는다."""
    def slope(b: float) -> float:
        return float((weight * (y - _sigmoid(eta + b))).sum() - b / OFFSET_PRIOR ** 2)

    low, high = -100.0, 100.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if slope(middle) > 0:
            low = middle
        else:
            high = middle
        if high - low < 1e-12:
            break
    return 0.5 * (low + high)


def recency(days: np.ndarray, cutoff_day: int, half_life: float) -> np.ndarray:
    return np.power(2.0, -(cutoff_day - np.asarray(days, dtype=np.float64)) / half_life)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """가중 중앙값. 가중치가 0인 표본을 빼고, 정렬한 표본의 가중치 중심 사이를 선형 보간한다."""
    values, weights = np.asarray(values, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    positive = weights > 0
    order = np.argsort(values[positive], kind="stable")
    v, w = values[positive][order], weights[positive][order]
    cumulative = np.cumsum(w)
    return float(np.interp(0.5 * cumulative[-1], cumulative - 0.5 * w, v))


def site_loss(labels: Labels, capacity: np.ndarray, cutoff_day: int, settings: dict) -> dict:
    """출력제어 시각에 발전소별 참여(판정할 수 있었던 가중 횟수 대비 꺼진 가중 횟수)와 깊이(꺼졌을 때 1 − 기준값 대비 비율)를 센다.
    prior_strength가 0보다 크면 용량 구분(LARGE_KW 이상·미만)의 평균 참여·깊이를 그만큼의 가중 표본으로 섞는다."""
    weight = recency(np.arange(cutoff_day + 1), cutoff_day, settings["half_life_days"])[:, None, None] * labels.event[..., None]
    eligible = labels.clear[:, None, :] & np.isfinite(labels.ratio)
    cut = labels.cut & eligible
    seen, hits = (weight * eligible).sum(axis=(0, 1)), (weight * cut).sum(axis=(0, 1))
    depth = (weight * cut * (1.0 - np.clip(np.nan_to_num(labels.ratio, nan=1.0), 0.0, 1.0))).sum(axis=(0, 1))
    strength = float(settings["prior_strength"])
    prior_participation, prior_depth = np.zeros(len(capacity)), np.zeros(len(capacity))
    if strength > 0:
        for members in (capacity >= LARGE_KW, capacity < LARGE_KW):
            if seen[members].sum() > 0:
                prior_participation[members] = hits[members].sum() / seen[members].sum()
            if hits[members].sum() > 0:
                prior_depth[members] = depth[members].sum() / hits[members].sum()
    participation = (hits + strength * prior_participation) / np.where(seen + strength > 0, seen + strength, 1.0)
    mean_depth = (depth + strength * prior_depth) / np.where(hits + strength > 0, hits + strength, 1.0)
    return {"participation": participation, "depth": mean_depth, "loss": participation * mean_depth, "seen": seen, "hits": hits}


def expected_output(reference: np.ndarray, capacity: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """[시각, 발전소] 맑은 날 기대 발전량(kWh) = 용량 × 발전소·달력 월·시각별 맑은 날 기준 이용률. 기준값이 없는 발전소는 같은 칸의
    발전소 중앙값을 쓴다."""
    hours = np.asarray(hours)
    month, slot = calendar_month(hours // 24) - 1, hours % 24
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                   # 모든 발전소의 기준값이 없는 칸(야간)은 0으로 둔다
        fill = np.nanmedian(reference, axis=2)[month, slot]
    ratio = reference[month, slot]
    ratio = np.where(np.isfinite(ratio), ratio, np.nan_to_num(fill)[:, None])
    return np.nan_to_num(ratio) * capacity


def group_share(loss: np.ndarray, output: np.ndarray, members: np.ndarray) -> np.ndarray:
    """[시각, 그룹] 출력제어가 걸렸을 때 그룹이 잃는 비율 φ: 발전소별 (참여 × 깊이)를 맑은 날 기대 발전량으로 가중 평균한다."""
    members = np.asarray(members, dtype=np.float64)
    total = output @ members.T
    return np.where(total > 0, (output * loss) @ members.T / np.where(total > 0, total, 1.0), 0.0)


def correction_surface(r0: np.ndarray, w0: np.ndarray) -> np.ndarray:
    """[GRID, PHI] 출력제어 확률 p와 그룹 감소 비율 φ마다 (1 − p)·G0 + p·G1 혼합 분포의 중앙값에서 G0 중앙값을 뺀 추가 보정.

    G0의 분포 F0는 가중 표본의 가중치 중심을 잇는 연속 누적분포(weighted_median과 같은 정의)이고, G1은 G0 표본을 (1 + r)(1 − φ) − 1로
    옮긴 분포라 F1(x) = F0((1 + x)/(1 − φ) − 1)이다. 혼합 분포의 중앙값은 (1 − p)·F0(x) + p·F1(x) = 0.5의 근을 이분법으로 구하므로
    p와 φ가 커질수록 보정이 줄지 않고, p = 0이나 φ = 0이면 0이다."""
    order = np.argsort(r0, kind="stable")
    values, weights = np.asarray(r0, dtype=np.float64)[order], np.asarray(w0, dtype=np.float64)[order] / np.sum(w0)
    centers = np.cumsum(weights) - 0.5 * weights
    base = float(np.interp(0.5, centers, values))
    surface = np.zeros((len(GRID), len(PHI)))
    p = GRID[1:]
    for j, phi in enumerate(PHI[1:], start=1):
        low = np.full(len(p), (1.0 + values[0]) * (1.0 - phi) - 1.0 - 1e-9)
        high = np.full(len(p), values[-1] + 1e-9)
        for _ in range(60):
            middle = 0.5 * (low + high)
            mixed = ((1.0 - p) * np.interp(middle, values, centers, left=0.0, right=1.0)
                     + p * np.interp((1.0 + middle) / (1.0 - phi) - 1.0, values, centers, left=0.0, right=1.0))
            below = mixed < 0.5
            low, high = np.where(below, middle, low), np.where(below, high, middle)
        surface[1:, j] = 0.5 * (low + high) - base
    return surface


def _bilinear(surface: np.ndarray, p: np.ndarray, phi: np.ndarray) -> np.ndarray:
    p, phi = np.clip(p, GRID[0], GRID[-1]), np.clip(phi, PHI[0], PHI[-1])
    i = np.clip(np.searchsorted(GRID, p, side="right") - 1, 0, len(GRID) - 2)
    j = np.clip(np.searchsorted(PHI, phi, side="right") - 1, 0, len(PHI) - 2)
    a, b = (p - GRID[i]) / (GRID[i + 1] - GRID[i]), (phi - PHI[j]) / (PHI[j + 1] - PHI[j])
    return ((1 - a) * (1 - b) * surface[i, j] + a * (1 - b) * surface[i + 1, j]
            + (1 - a) * b * surface[i, j + 1] + a * b * surface[i + 1, j + 1])


@dataclass
class CurtailmentModule:
    settings: dict
    active: bool
    probability: np.ndarray      # 평가 시각별 출력제어 확률 [시각]
    share: np.ndarray            # 출력제어가 걸렸을 때 그룹이 잃는 비율 φ [시각, 그룹]
    surface: np.ndarray          # 그룹별 추가 보정 Δ(p, φ)(중앙값 대비 비율) [그룹, GRID, PHI]
    summary: dict

    def shift(self, median: np.ndarray) -> np.ndarray:
        """[시각, 그룹] 중앙값에 더할 kWh."""
        ratio = np.stack([_bilinear(self.surface[g], self.probability, self.share[:, g]) for g in range(len(self.surface))], axis=1)
        return ratio * median

    def apply(self, quantiles: np.ndarray, levels: np.ndarray, capacity: np.ndarray) -> np.ndarray:
        """[시각, 그룹, 분위수] kWh 분위수를 중앙값의 추가 보정만큼 옮긴다. module이 꺼져 있으면 입력을 그대로 돌려준다."""
        if not self.active:
            return quantiles
        median = quantiles[..., int(np.flatnonzero(np.isclose(levels, 0.5))[0])]
        moved = np.asarray(quantiles, dtype=np.float64) + self.shift(median)[..., None]
        return np.sort(np.clip(moved, 0.0, np.asarray(capacity, dtype=np.float64)[None, :, None]), axis=-1)


def residual_samples(source: Source, labels: Labels, cutoff_h: int, forecast_clearness: np.ndarray, settings: dict) -> dict:
    """source의 cutoff 이전 평가 시각에서 그룹별 G0 상대 잔차와 반감기 가중, 출력제어 시각의 실제 상대 잔차(대조용)를 모은다.
    forecast_clearness는 source 시각별 예보 맑음 지수이다."""
    cutoff_day = cutoff_h // 24
    hours = np.asarray(source.hours)
    keep = hours <= cutoff_h
    hours, median, actual, clearness = hours[keep], source.median[keep], source.actual[keep], forecast_clearness[keep]
    event = labels.event.reshape(-1)[hours]
    window = np.isin(hours % 24, np.arange(*WINDOW))
    weight = recency(hours // 24, cutoff_day, settings["half_life_days"])
    out = {}
    for g in range(median.shape[1]):
        usable = np.isfinite(actual[:, g]) & (median[:, g] >= settings["min_forecast_share"] * source.capacity[g])
        ratio = np.where(usable, (actual[:, g] - median[:, g]) / np.where(usable, median[:, g], 1.0), np.nan)
        g0 = usable & ~event & window & (clearness >= settings["forecast_clearness"])
        g1 = usable & event
        out[g] = {"r0": ratio[g0], "w0": weight[g0], "r1": ratio[g1], "hours1": hours[g1]}
    return out


def fit_curtailment(settings: dict, calendar: HolidayCalendar, base: Base, weather: Weather, sites: Sites, cutoff_h: int,
                    members: np.ndarray, hours: np.ndarray, source: Source | None) -> CurtailmentModule:
    """cutoff 이전 실적으로 출력제어를 표시하고 전역 확률과 발전소별로 잃는 양을 맞추며, source의 OOT 잔차로 그룹별 추가 보정
    Δ(p, φ)를 만들어 그룹 구성 members [그룹, 발전소]와 평가 시각 hours에 적용할 module을 만든다. 출력제어 시각이 있는데 source가
    없거나 source에 비교할 맑은 낮 잔차가 없으면 보정을 만들 수 없어 멈춘다."""
    cutoff_day = cutoff_h // 24
    view = actual_view(np.asarray(base.energy), np.asarray(base.status), cutoff_h)
    cloud_index = weather.channel_names("wb", "observed").index("wb_cloud")
    cloud = view.observed(np.asarray(weather.values["observed_wb"][..., cloud_index]))
    span = (cutoff_day + 1) * 24
    labels = curtailment_labels(view.energy[:span], sites.capacity, sites.station_index["wb"], cloud[:span], settings)
    events = np.argwhere(labels.event)
    record = {"active": bool(len(events)), "cutoff_h": int(cutoff_h), "event_hours": int(len(events)),
              "event_days": [_event_day(labels, sites, d) for d in np.unique(events[:, 0])] if len(events) else []}
    n_groups = len(members)
    probability, share = np.zeros(len(hours)), np.zeros((len(hours), n_groups))
    if not len(events):
        return CurtailmentModule(settings, False, probability, share, np.zeros((n_groups, len(GRID), len(PHI))), record)
    if source is None:
        raise ValueError("cutoff 이전에 출력제어 시각이 있지만, 보정 잔차를 줄 같은 모델·배정·seed의 OOT run이 없습니다.")

    # 전역 확률: 기본 확률(학습 기간 전체, 같은 가중)과 cutoff 시점의 최근 보정
    solar = np.asarray(base.solar)
    train_hours = np.arange(span)
    sun = clear_sky_ghi(np.asarray(solar[train_hours][..., 1], dtype=np.float64)).mean(axis=1) > 0
    first_day = int(np.flatnonzero(view.valid[:span].any(axis=1)).min()) // 24
    rows = train_hours[sun & (train_hours // 24 >= first_day)]
    x = design(calendar, rows, irradiance(weather, sites, solar, rows)[0])
    y = labels.event.reshape(-1)[rows].astype(np.float64)
    model = fit_logistic(x, y, settings["ridge"])
    offset = recent_offset(logit(model, x), y, recency(rows // 24, cutoff_day, settings["half_life_days"]))

    # 발전소별로 잃는 양과 그룹 감소 비율 φ
    loss = site_loss(labels, sites.capacity, cutoff_day, settings)
    hours = np.asarray(hours)
    share = group_share(loss["loss"], expected_output(labels.reference, sites.capacity, hours), members)

    # 그룹 잔차 분포 G0와 추가 보정 표면
    source_ghi, source_clear = irradiance(weather, sites, solar, source.hours)
    clearness = np.where(source_clear > 0, source_ghi / np.where(source_clear > 0, source_clear, 1.0), 0.0)
    samples = residual_samples(source, labels, cutoff_h, clearness, settings)
    if any(len(s["r0"]) == 0 for s in samples.values()):
        raise ValueError(f"source {source.run}의 cutoff 이전 평가 시각에 비교할 맑은 낮 잔차가 없어 보정을 만들 수 없습니다.")
    surface = np.stack([correction_surface(s["r0"], s["w0"]) for s in samples.values()])

    target_ghi, target_clear = irradiance(weather, sites, solar, hours)
    target_sun = target_clear > 0
    probability[target_sun] = _sigmoid(logit(model, design(calendar, hours[target_sun], target_ghi[target_sun])) + offset)
    high = np.unique(hours[probability >= 0.5] // 24)
    risky = probability >= 0.2
    record.update({
        "base": {"features": model["features"], "mean": model["mean"].tolist(), "scale": model["scale"].tolist(),
                 "intercept": model["intercept"], "coef": model["coef"].tolist(), "rows": int(len(rows)), "events": int(y.sum())},
        "recent_offset": offset, "half_life_days": settings["half_life_days"], "prior_strength": settings["prior_strength"],
        "source": {"run": source.run, "run_key": source.run_key, "cutoff_h": int(source.cutoff_h),
                   "first_h": int(np.asarray(source.hours).min()), "last_h_used": int(min(np.asarray(source.hours).max(), cutoff_h))},
        "plants": [{"site_id": str(sites.ids[i]), "capacity_kw": float(sites.capacity[i]), "participation": float(loss["participation"][i]),
                    "depth": float(loss["depth"][i]), "seen": float(loss["seen"][i]), "hits": float(loss["hits"][i])}
                   for i in np.argsort(-loss["loss"] * sites.capacity) if loss["loss"][i] > 0][:30],
        "groups": [_group_record(g, samples[g], surface[g], share, risky, loss["loss"], labels, sites, members) for g in range(n_groups)],
        "probability": {"max": float(probability.max()), "hours_at_least_half": int((probability >= 0.5).sum()),
                        "days_at_least_half": [d.date().isoformat() for d in delivery_dates(high)]},
    })
    return CurtailmentModule(settings, True, probability, share, surface, record)


def _group_record(g: int, sample: dict, surface: np.ndarray, share: np.ndarray, risky: np.ndarray, loss: np.ndarray, labels: Labels,
                  sites: Sites, members: np.ndarray) -> dict:
    """그룹의 G0 요약, 출력제어 확률이 0.2 이상인 시각의 φ, 추가 보정 표면 일부, source 출력제어 시각의 실제 잔차와 발전소로 만든 값의
    대조(같은 사건으로 맞춘 값이라 표본 안 대조이다)."""
    median0 = weighted_median(sample["r0"], sample["w0"])
    marks = {f"p={p:g},phi={f:g}": float(_bilinear(surface, np.array([p]), np.array([f]))[0]) for p in (0.2, 0.5) for f in (0.05, 0.1, 0.15)}
    compare = []
    if len(sample["hours1"]):
        phi = group_share(loss, expected_output(labels.reference, sites.capacity, sample["hours1"]), members)[:, g]
        compare = [{"h": int(h), "actual": float(r), "site_built": float((1 + median0) * (1 - f) - 1)}
                   for h, r, f in zip(sample["hours1"], sample["r1"], phi)]
    return {"g0_hours": int(len(sample["r0"])), "g0_median": median0,
            "phi_when_probability_at_least_0.2": {"mean": float(share[risky, g].mean()) if risky.any() else 0.0,
                                                  "max": float(share[risky, g].max()) if risky.any() else 0.0},
            "correction": marks, "event_hours_compare": compare}


def _event_day(labels: Labels, sites: Sites, day: int) -> dict:
    slots = np.flatnonzero(labels.event[day])
    plants = np.flatnonzero(labels.cut[day][slots].any(axis=0))
    return {"date": delivery_dates([day])[0].date().isoformat(), "hours": [int(s) + 1 for s in slots],
            "plants": [str(sites.ids[i]) for i in plants], "cut_kw": float(sites.capacity[plants].sum())}
