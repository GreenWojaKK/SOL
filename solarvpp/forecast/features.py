"""입력 구성과 feature: 입력 역할·입력 구성별 기상 입력 선택, PV 단위 feature, PV physics proxy, 하루 맥락 집계,
XGB query-시각 행렬, feature manifest, 학습 구간에서만 fit하는 전처리(대체·표준화·ZCA, power curve).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import InputConfig, input_key
from ..data import PROVIDERS, ROUNDS, Sites, Weather, day_of_year
from .queries import QuerySet


class BranchScaler:
    """중앙값 대체 → 표준화 → ZCA. fit 입력은 [행, channel]이고 결측은 NaN이다."""

    def __init__(self, epsilon: float = 0.01):
        self.epsilon = epsilon

    def fit(self, x: np.ndarray) -> "BranchScaler":
        x = np.asarray(x, dtype=np.float64)
        median = np.nanmedian(x, axis=0)
        self.median = np.where(np.isfinite(median), median, 0.0)
        clean = np.where(np.isfinite(x), x, self.median)
        self.mean = clean.mean(axis=0)
        scale = clean.std(axis=0)
        self.scale = np.where(scale > 1e-8, scale, 1.0)
        z = (clean - self.mean) / self.scale
        eig, vectors = np.linalg.eigh(z.T @ z / max(len(z) - 1, 1))
        self.matrix = (vectors * (1.0 / np.sqrt(np.maximum(eig, 0.0) + self.epsilon))) @ vectors.T
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        clean = np.where(np.isfinite(x), x, self.median)
        return (((clean - self.mean) / self.scale) @ self.matrix).astype(np.float32)

    def state(self) -> dict:
        return {"median": self.median, "mean": self.mean, "scale": self.scale, "matrix": self.matrix, "epsilon": np.float64(self.epsilon)}

    @classmethod
    def from_state(cls, state: dict) -> "BranchScaler":
        scaler = cls(float(state["epsilon"]))
        for name in ("median", "mean", "scale", "matrix"):
            setattr(scaler, name, np.asarray(state[name], dtype=np.float64))
        return scaler


class Standardizer:
    def fit(self, x: np.ndarray) -> "Standardizer":
        x = np.asarray(x, dtype=np.float64)
        self.mean = np.nanmean(x, axis=0)
        scale = np.nanstd(x, axis=0)
        self.mean = np.where(np.isfinite(self.mean), self.mean, 0.0)
        self.scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        return np.where(np.isfinite(z), z, 0.0).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean, "scale": self.scale}

    @classmethod
    def from_state(cls, state: dict) -> "Standardizer":
        out = cls()
        out.mean, out.scale = np.asarray(state["mean"]), np.asarray(state["scale"])
        return out


class PowerCurve:
    """정규화 발전량 z를 query 평균 max(cos 천정각, 0)의 1~3차 다항식으로 맞춘다. 절편이 없어 야간은 0이다."""

    def __init__(self, degree: int = 3, ridge: float = 0.1):
        self.degree, self.ridge = degree, ridge

    def basis(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.stack([x ** p for p in range(1, self.degree + 1)], axis=-1)

    def fit(self, x: np.ndarray, z: np.ndarray, weight: np.ndarray) -> "PowerCurve":
        ok = np.isfinite(x) & np.isfinite(z) & (weight > 0)
        if ok.sum() < 100:
            raise ValueError(f"power curve를 맞출 유효 표본이 부족합니다: {ok.sum()}")
        w = np.sqrt(weight[ok] / weight[ok].mean())
        a = self.basis(x[ok]) * w[:, None]
        self.coef = np.linalg.solve(a.T @ a + self.ridge * np.eye(self.degree), a.T @ (z[ok] * w))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.nan_to_num(self.basis(x) @ self.coef), 0.0, 1.0)

    def state(self) -> dict:
        return {"coef": self.coef.tolist(), "degree": self.degree, "ridge": self.ridge}


CROSS_PROVIDER = {
    "temperature": ("wa_temp", "wb_temp"),
    "relative_humidity": ("wa_humidity", "wb_rel_hum"),
    "cloud": ("wa_cloud", "wb_cloud"),
    "wind_speed_ms": ("wa_wind_speed", "wb_wind_speed"),
    "wind_speed_times_sin_direction": ("wa_wind_vec_sin", "wb_wind_vec_sin"),
    "wind_speed_times_cos_direction": ("wa_wind_vec_cos", "wb_wind_vec_cos"),
}


STATIC_CHANNELS = ("lat", "lon", "capacity_kw", "wa_dist", "wb_dist")
SOLAR_FEATURES = ("solar_elev", "solar_cosz_pos", "solar_azim_sin", "solar_azim_cos")
TIME_FEATURES = ("time_slot_sin", "time_slot_cos", "time_doy_sin", "time_doy_cos")


# PV physics proxy의 원천 channel. 모두 canonical 단위(%, °C)이다.
# 참조: docs/solarvpp-reference.md#features.
PHYSICS_SOURCES = {"wa": ("wa_cloud", "wa_temp", "wa_humidity"), "wb": ("wb_cloud", "wb_temp", "wb_dew_point")}


HAURWITZ = (1098.0, 0.057)
KASTEN_CZEPLAK = (0.75, 3.4)
MAGNUS = (17.62, 243.12)
MODULE_HEATING = 0.031
DERATE = (0.004, 25.0)


@dataclass
class ProviderInput:
    provider: str
    active: bool
    values: np.ndarray
    present: np.ndarray
    channels: list
    sources: list

    @property
    def names(self) -> list[str]:
        return [c["name"] for c in self.channels]


@dataclass
class InputSet:
    role: str
    input_config: InputConfig
    providers: dict
    excluded: tuple

    @property
    def active(self) -> list[str]:
        return [p for p in PROVIDERS if self.providers[p].active]

    @property
    def disagreement(self) -> bool:
        return self.input_config.disagreement and all(self.providers[p].active for p in PROVIDERS)


def resolve_inputs(weather: Weather, input_config: InputConfig, excluded: tuple[str, ...], schema: dict) -> InputSet:
    """입력 구성의 forecast 입력을 고른다. 원래 각도 풍향과 제외 목록의 열은 모델 입력에서 뺀다."""
    if input_config.input_role != "forecast":
        raise NotImplementedError(f"input_role={input_config.input_role}은 O-COMMON·O-FULL 단계에서 구현합니다.")
    if list(schema["cross_provider_variables"]) != list(CROSS_PROVIDER):
        raise ValueError("schema의 cross_provider_variables와 제공사 간 대응표가 다릅니다.")
    providers = {}
    for p in PROVIDERS:
        specs = weather.channels[f"forecast_{p}"]
        used = [i for i, c in enumerate(specs) if c["name"] not in excluded and not c["name"].endswith("_wind_dir")]
        channels = [specs[i] for i in used]
        sources = sorted({s for c in channels for s in c["source"]}, key=[c["name"] for c in specs].index)
        values = np.asarray(weather.values[f"forecast_{p}"])[..., used]
        providers[p] = ProviderInput(p, p in input_config.providers, values, np.asarray(weather.present[f"forecast_{p}"]), channels, sources)
    return InputSet(input_config.input_role, input_config, providers, tuple(excluded))


def clear_sky_ghi(cosz: np.ndarray) -> np.ndarray:
    """Haurwitz clear-sky 일사량(W/m²). 태양이 지평선 아래이면 0이다."""
    a, b = HAURWITZ
    safe = np.where(cosz > 0, cosz, 1.0)
    return np.where(cosz > 0, a * safe * np.exp(-b / safe), 0.0)


def dew_point(temp: np.ndarray, humidity: np.ndarray) -> np.ndarray:
    """기온(°C)·상대습도(%)에서 Magnus식 이슬점(°C). 습도는 [1, 100]%로 자른다."""
    a, b = MAGNUS
    gamma = np.log(np.clip(humidity, 1.0, 100.0) / 100.0) + a * temp / (b + temp)
    return b * gamma / (a - gamma)


def physics_block(inputs: InputSet, sites: Sites, solar: np.ndarray, hours) -> tuple[np.ndarray, list[str]]:
    """PV physics proxy [시각, PV, 13]: clear-sky 일사량 1개와 제공사·round별 운량 보정 일사량·이슬점 차·온도 감률."""
    g_cs = clear_sky_ghi(solar[hours][..., 1].astype(np.float64))
    blocks, names = [g_cs], ["phys_ghi_clear_sky"]
    for p in PROVIDERS:
        pi = inputs.providers[p]
        station = sites.station_index[p]
        cloud_i, temp_i, third_i = (pi.names.index(n) for n in PHYSICS_SOURCES[p])
        for r_i, r in enumerate(ROUNDS):
            cloud = pi.values[r_i, hours, :, cloud_i][:, station].astype(np.float64)
            temp = pi.values[r_i, hours, :, temp_i][:, station].astype(np.float64)
            third = pi.values[r_i, hours, :, third_i][:, station].astype(np.float64)
            dew = dew_point(temp, third) if p == "wa" else third
            g = g_cs * (1.0 - KASTEN_CZEPLAK[0] * np.clip(cloud / 100.0, 0.0, 1.0) ** KASTEN_CZEPLAK[1])
            blocks += [g, temp - dew, 1.0 - DERATE[0] * (temp + MODULE_HEATING * g - DERATE[1])]
            names += [f"phys_ghi_cloud_{p}_r{r}", f"phys_dewpoint_spread_{p}_r{r}", f"phys_temp_derate_{p}_r{r}"]
    return np.stack(blocks, axis=-1), names


def time_features(days: np.ndarray) -> np.ndarray:
    """[대상일, 24, 4]: 구간 중앙 시각 sin·cos와 대상일 연중일 sin·cos."""
    slot = 2 * np.pi * (np.arange(24) + 0.5) / 24
    doy = 2 * np.pi * day_of_year(days) / 365.25
    return np.stack(np.broadcast_arrays(np.sin(slot)[None, :], np.cos(slot)[None, :], np.sin(doy)[:, None], np.cos(doy)[:, None]), axis=-1)


def pv_block(inputs: InputSet, sites: Sites, solar: np.ndarray, day: int) -> tuple[np.ndarray, list[str]]:
    """대상일 하루의 PV 단위 연속 feature [24, PV, F]. 입력 구성에 따라 태양 위치·disagreement·revision·physics를 고른다."""
    hours = slice(day * 24, (day + 1) * 24)
    blocks, names = [], []
    rounds = {}
    for p in inputs.active:
        pi = inputs.providers[p]
        station = sites.station_index[p]
        rounds[p] = [pi.values[r, hours][:, station, :].astype(np.float64) for r in range(len(ROUNDS))]
        for r, value in zip(ROUNDS, rounds[p]):
            blocks.append(value)
            names += [f"{n}_r{r}" for n in pi.names]
        if inputs.input_config.revision:
            blocks.append(rounds[p][1] - rounds[p][0])
            names += [f"{n}_rev" for n in pi.names]
    if inputs.disagreement:
        index = {p: inputs.providers[p].names for p in PROVIDERS}
        for r_i, r in enumerate(ROUNDS):
            for key, (a, b) in CROSS_PROVIDER.items():
                blocks.append((rounds["wa"][r_i][..., index["wa"].index(a)] - rounds["wb"][r_i][..., index["wb"].index(b)])[..., None])
                names.append(f"dis_{key}_r{r}")
    if inputs.input_config.solar_geometry:
        sol = solar[hours].astype(np.float64)
        blocks.append(np.stack([sol[..., 0], np.maximum(sol[..., 1], 0), sol[..., 2], sol[..., 3]], axis=-1))
        names += list(SOLAR_FEATURES)
    if inputs.input_config.physics_proxy:
        physics, physics_names = physics_block(inputs, sites, solar, hours)
        blocks.append(physics)
        names += physics_names
    static = np.column_stack([sites.lat, sites.lon, sites.capacity, sites.distance["wa"], sites.distance["wb"]])
    blocks.append(np.broadcast_to(static[None], (24, *static.shape)))
    names += list(STATIC_CHANNELS)
    return np.concatenate(blocks, axis=-1), names


def weighted_moments(x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x [24, PV, F], weights [Q, PV] → 값이 있는 PV로 다시 정규화한 가중 평균·표준편차 [Q, 24, F]."""
    finite = np.isfinite(x)
    x0 = np.where(finite, x, 0.0)
    den = np.einsum("qi,tif->qtf", weights, finite.astype(np.float64))
    num = np.einsum("qi,tif->qtf", weights, x0)
    sq = np.einsum("qi,tif->qtf", weights, x0 * x0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = num / den
        var = np.maximum(sq / den - mean * mean, 0.0)
    mean[den <= 0] = np.nan
    var[den <= 0] = np.nan
    return mean, np.sqrt(var)


QUERY_FEATURES = ("query_capacity_kw", "query_pv_count", "query_hhi", "is_G2")
DAY_CONTEXT_TOKENS = ("cloud", "phys_ghi", "precip_prob", "rain_prob", "uv_idx", "vis", "ceiling")
DAY_CONTEXT_STATS = ("daymean", "daymin", "daymax")


def day_context(mean: np.ndarray, names: list[str], sun: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """query-day 하루의 하늘 상태 요약 [Q, 24, 3K]. 운량·운량 보정 일사·강수 확률·UV·시정·운고 채널을 주간에서 집계한다.

    mean [Q, 24, F]는 query의 시각별 용량 가중 평균, sun [Q, 24]는 query 평균 max(cos 천정각, 0)이다. 일평균은 sun 가중 평균,
    최솟값·최댓값은 sun > 0인 시각에서 계산하며 같은 값을 24시간에 둔다. 값이 없는 채널은 NaN으로 두어 XGB가 결측으로 처리한다.
    """
    select = [i for i, n in enumerate(names) if any(token in n for token in DAY_CONTEXT_TOKENS)]
    v = mean[..., select]
    day = (sun > 0)[..., None] & np.isfinite(v)
    w = np.where(day, sun[..., None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = (np.where(day, v, 0.0) * w).sum(axis=1) / w.sum(axis=1)
    low = np.where(day, v, np.inf).min(axis=1)
    high = np.where(day, v, -np.inf).max(axis=1)
    empty = ~day.any(axis=1)
    stats = [np.where(empty, np.nan, s) for s in (avg, low, high)]
    out = np.concatenate([np.broadcast_to(s[:, None, :], (len(mean), mean.shape[1], len(select))) for s in stats], axis=-1)
    return out, [f"day_{names[i]}_{stat}" for stat in DAY_CONTEXT_STATS for i in select]


def xgb_matrix(queries: QuerySet, inputs: InputSet, sites: Sites, solar: np.ndarray, sunup: np.ndarray, rows: str):
    """query-시각 feature 행렬. rows='target'은 정답 mask·주간 시각, rows='all'은 24시간 전체이다.

    반환: (행렬, (query, slot) index, 열 이름, power curve 입력). power curve 입력은 query 용량 가중 평균 max(cos 천정각, 0)이며
    입력 구성과 무관하게 target 변환에 쓴다. Power-curve target 변환은 모든 구성에서 고정한다.
    참조: docs/solarvpp-reference.md#xgb.
    """
    if rows not in ("target", "all"):
        raise ValueError(rows)
    input_config = inputs.input_config
    select = queries.target_mask if rows == "target" else np.ones((len(queries), 24), dtype=bool)
    out, index, curve_x, names = [], [], [], None
    for day in np.unique(queries.day):
        qi = np.flatnonzero(queries.day == day)
        qi = qi[select[qi].any(axis=1)]
        if len(qi) == 0:
            continue
        block, pv_names = pv_block(inputs, sites, solar, int(day))
        weights = queries.members[qi] * sites.capacity[None, :]
        mean, std = weighted_moments(block, weights)
        hours = slice(int(day) * 24, (int(day) + 1) * 24)
        cap = weights.sum(axis=1)
        cosz_bar = np.einsum("qi,ti->qt", weights, np.maximum(solar[hours][..., 1].astype(np.float64), 0.0)) / cap[:, None]
        extra = []
        if input_config.solar_geometry:
            extra.append(np.einsum("qi,ti->qt", weights, sunup[hours].astype(np.float64)) / cap[:, None])
        if input_config.time_features:
            extra += list(np.moveaxis(np.broadcast_to(time_features(np.array([day]))[0], (len(qi), 24, 4)), -1, 0))
        missing = []
        for p in inputs.active:
            present = inputs.providers[p].present[:, hours][:, :, sites.station_index[p]]
            for r in range(len(ROUNDS)):
                missing.append(1 - np.einsum("qi,ti->qt", weights, present[r].astype(np.float64)) / cap[:, None])
        qinfo = np.stack(np.broadcast_arrays(
            cap[:, None], queries.n_pv[qi, None].astype(np.float64), queries.hhi[qi, None], queries.group[qi, None].astype(np.float64),
            *extra, *missing), axis=-1)
        feats = [mean, std, qinfo]
        if input_config.day_context:
            context, context_names = day_context(mean, [f"{n}__mean" for n in pv_names], cosz_bar)
            feats.append(context)
        feats = np.concatenate(feats, axis=-1)
        for j, q in enumerate(qi):
            slots = np.flatnonzero(select[q])
            out.append(feats[j, slots])
            curve_x.append(cosz_bar[j, slots])
            index.append(np.column_stack([np.full(len(slots), q), slots]))
        if names is None:
            names = ([f"{n}__mean" for n in pv_names] + [f"{n}__std" for n in pv_names] + list(QUERY_FEATURES)
                     + (["sunup_capacity_share"] if input_config.solar_geometry else []) + (list(TIME_FEATURES) if input_config.time_features else [])
                     + [f"missing_capacity_share_{p}_r{r}" for p in inputs.active for r in ROUNDS]
                     + (context_names if input_config.day_context else []))
    return (np.concatenate(out).astype(np.float32), np.concatenate(index).astype(np.int64), names,
            np.concatenate(curve_x).astype(np.float64))


def feature_manifest(inputs: InputSet, names: list[str], deviations: list[dict]) -> dict:
    """XGB 입력 열마다 원본 열·round·변환·집계를 기록한다."""
    specs = {c["name"]: (p, c) for p in PROVIDERS for c in inputs.providers[p].channels}
    columns = []
    for i, name in enumerate(names):
        base, _, agg = name.partition("__")
        entry = {"index": i, "name": name, "aggregation": f"capacity-weighted {agg} over query members (NaN-aware)" if agg else "query-level"}
        stem, _, suffix = base.rpartition("_")
        if base.startswith("dis_"):
            key, r = base[4:].rsplit("_r", 1)
            entry.update(block="disagreement", round=int(r), source=list(CROSS_PROVIDER[key]), transform="wa - wb at the PV's paired stations")
        elif suffix in ("r10", "r17", "rev") and stem in specs:
            p, c = specs[stem]
            entry.update(block=f"{p}_{'revision' if suffix == 'rev' else 'round'}", provider=p, source=list(c["source"]), unit=c["unit"],
                         round=None if suffix == "rev" else int(suffix[1:]),
                         transform=f"r17 - r10 of ({c['transform']})" if suffix == "rev" else c["transform"])
        elif base in SOLAR_FEATURES:
            entry.update(block="solar", transform="NOAA approximation at PV coordinates and interval midpoint")
        elif base.startswith("phys_"):
            entry.update(block="physics", transform="fixed PV physics proxy (docs/solarvpp-design.md#features): Haurwitz clear-sky, Kasten-Czeplak cloud "
                                                    "adjustment, Magnus dew point, module temperature derating")
        elif base in TIME_FEATURES:
            entry.update(block="time", transform="interval-midpoint clock hour and delivery-day day-of-year sin/cos")
        elif base in ("sunup_capacity_share", "power_curve"):
            entry.update(block="solar", transform="solar-geometry derived query feature")
        elif base in STATIC_CHANNELS:
            entry.update(block="static", source=["sites.csv"])
        else:
            entry.update(block="query")
        columns.append(entry)
    return {"role": inputs.role, "input": input_key(inputs.input_config), "columns": columns,
            "excluded_inputs": [{"inputs": d["inputs"], "reason": d["reason"], "id": d["id"]} for d in deviations],
            "providers": {p: {"active": inputs.providers[p].active, "channels": inputs.providers[p].names,
                              "raw_sources": inputs.providers[p].sources} for p in PROVIDERS}}
