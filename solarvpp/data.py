"""데이터 계약: 전역 시간축, PV 좌표의 태양 위치와 야간 규칙, 원자료의 키·품질·단위 규칙, base cache(generation·SMP·태양 위치)와
weather cache.

전역 시간축의 h=0은 2021-01-01 01:00 KST이고, 시각은 우측닫힘 1시간 구간의 끝이다.
대상일은 date(ts−1h)이므로 대상일 index는 h//24, slot은 h%24(slot 0 = 01시, slot 23 = 다음날 00시)이다.
base·weather cache는 seed와 실험 간에 공유한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .store import Store, digest, file_sha, versions

if TYPE_CHECKING:
    from .config import Spec

TZ = "Asia/Seoul"
T0 = pd.Timestamp("2021-01-01 01:00", tz=TZ)
HOUR_NS = 3_600_000_000_000
N_DAYS = 1_977
N_HOURS = N_DAYS * 24
N_ACTUAL = 38_688  # generation·observed가 있는 2025-06-01 00:00까지
K_PEAK_HOURS = (11, 12, 13, 14, 15)


def hour_index(ts) -> tuple[np.ndarray, np.ndarray]:
    """timestamp를 h로 바꾼다. 단위(ms·us·ns)와 무관하며, 정시 여부를 함께 돌려준다."""
    values = pd.DatetimeIndex(ts)
    if values.tz is None:
        raise ValueError("timestamp에는 timezone이 있어야 합니다.")
    delta = values.as_unit("ns").asi8 - T0.value
    return np.floor_divide(delta, HOUR_NS), delta % HOUR_NS == 0


def to_h(timestamp) -> int:
    h, on_hour = hour_index([pd.Timestamp(timestamp)])
    if not on_hour[0]:
        raise ValueError(f"정시가 아닌 시각입니다: {timestamp}")
    return int(h[0])


def timestamps(h) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(T0 + pd.to_timedelta(np.asarray(h, dtype=np.int64), unit="h"))


def clock_hour(h) -> np.ndarray:
    return (np.asarray(h) + 1) % 24


def k_weight(h) -> np.ndarray:
    return np.where(np.isin(clock_hour(h), K_PEAK_HOURS), 3, 1).astype(np.int8)


def day_of(h):
    return np.asarray(h) // 24


def delivery_dates(days) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(T0.normalize().tz_localize(None) + pd.to_timedelta(np.asarray(days, dtype=np.int64), unit="D"))


def delivery_month(h) -> np.ndarray:
    return delivery_dates(day_of(h)).strftime("%Y-%m").to_numpy()


def day_of_year(days) -> np.ndarray:
    return delivery_dates(days).dayofyear.to_numpy()


CHANNELS = ("solar_elev", "solar_cosz", "solar_azim_sin", "solar_azim_cos")


def solar_position(instants: pd.DatetimeIndex, lat: np.ndarray, lon: np.ndarray):
    """NOAA 근사식. instants는 계산할 시각(KST)이고 결과는 [시각, 지점]의 고도(도)·방위(rad)·cos 천정각이다."""
    hour = (instants.hour.to_numpy() + instants.minute.to_numpy() / 60)[:, None]
    doy = instants.dayofyear.to_numpy()[:, None]
    gamma = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eq = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma) - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) - 0.006758 * np.cos(2 * gamma)
            + 0.000907 * np.sin(2 * gamma) - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    ha = np.radians((hour * 60 + eq + 4 * (np.asarray(lon)[None, :] - 135)) / 4 - 180)
    la = np.radians(np.asarray(lat)[None, :])
    cosz = np.clip(np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha), -1, 1)
    elev = 90 - np.degrees(np.arccos(cosz))
    azimuth = np.arctan2(-np.sin(ha), np.tan(decl) * np.cos(la) - np.sin(la) * np.cos(ha))
    return elev, azimuth, cosz


def build_solar(h: np.ndarray, lat: np.ndarray, lon: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """구간 중앙의 [고도, cos 천정각, 방위 sin, 방위 cos]와, 구간 시작·중앙·끝 중 태양이 떠 있는지 여부."""
    features = np.empty((len(h), len(lat), len(CHANNELS)), dtype=np.float32)
    sunup = np.zeros((len(h), len(lat)), dtype=bool)
    for start in range(0, len(h), chunk):
        end_ts = timestamps(h[start:start + chunk])
        for offset in (60, 30, 0):
            elev, azimuth, cosz = solar_position(end_ts - pd.Timedelta(minutes=offset), lat, lon)
            sunup[start:start + chunk] |= elev > 0
            if offset == 30:
                features[start:start + chunk] = np.stack([elev, cosz, np.sin(azimuth), np.cos(azimuth)], axis=-1)
    return features, sunup


GROUPS = ("G1", "G2")
original_MAPPING = {"A": 0, "B": 1}
ROUNDS = (10, 17)
PROVIDERS = ("wa", "wb")
DATA_PACKAGES = ("numpy", "pandas", "pyarrow")
SPLIT_HOURS = {"train": (0, N_ACTUAL), "test": (N_ACTUAL, N_HOURS)}


def package_versions(names=DATA_PACKAGES) -> dict:
    return {name: version(name) for name in names}


@dataclass
class Sites:
    ids: np.ndarray
    group: np.ndarray
    capacity: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    stations: dict
    station_index: dict
    distance: dict
    original_hash: str

    def __len__(self) -> int:
        return len(self.ids)

    def group_capacity(self) -> np.ndarray:
        return np.array([self.capacity[self.group == g].sum() for g in range(len(GROUPS))])


def load_sites(spec: Spec) -> Sites:
    frame = pd.read_csv(spec.data_dir / "sites.csv")
    if not frame.site_id.is_unique or frame.current_group.map(original_MAPPING).isna().any():
        raise ValueError("sites.csv의 site_id는 유일하고 current_group은 A 또는 B여야 합니다.")
    stations, index, distance = {}, {}, {}
    for provider in PROVIDERS:
        names = np.array(sorted(frame[f"{provider}_station"].unique()))
        stations[provider] = names
        index[provider] = np.searchsorted(names, frame[f"{provider}_station"].to_numpy())
        distance[provider] = frame[f"{provider}_dist"].to_numpy(dtype=np.float64)
    group = frame.current_group.map(original_MAPPING).to_numpy(dtype=np.int8)
    original_hash = digest(list(zip(frame.site_id.tolist(), [GROUPS[g] for g in group])))
    return Sites(frame.site_id.to_numpy(dtype=object), group, frame.capacity_kw.to_numpy(dtype=np.float64),
                 frame.lat.to_numpy(dtype=np.float64), frame.lon.to_numpy(dtype=np.float64), stations, index, distance, original_hash)


def load_generation(spec: Spec, sites: Sites) -> tuple[np.ndarray, np.ndarray, dict]:
    """energy [N_ACTUAL, PV]와 상태(0 결측, 1 유효, 2 무효)를 만든다. 유효 = 유한하고 0 이상."""
    frame = pd.read_parquet(spec.data_dir / "train" / "generation.parquet")
    h, on_hour = hour_index(frame.ts)
    site = pd.Index(sites.ids).get_indexer(frame.site_id.to_numpy())
    if (site < 0).any():
        raise ValueError("generation에 sites.csv에 없는 발전소가 있습니다.")
    inside = on_hour & (h >= 0) & (h < N_ACTUAL)
    key = h[inside] * len(sites) + site[inside]
    if len(np.unique(key)) != len(key):
        raise ValueError("generation에 중복 (site_id, ts) 키가 있습니다.")
    value = frame.energy.to_numpy(dtype=np.float64)[inside]
    valid = np.isfinite(value) & (value >= 0)
    energy = np.full((N_ACTUAL, len(sites)), np.nan)
    status = np.zeros((N_ACTUAL, len(sites)), dtype=np.int8)
    energy[h[inside][valid], site[inside][valid]] = value[valid]
    status[h[inside], site[inside]] = np.where(valid, 1, 2)
    audit = {"rows": len(frame), "off_hour_rows": int((~on_hour).sum()), "outside_actual_rows": int((on_hour & ~inside).sum()),
             "invalid_rows": int((~valid).sum()), "negative_rows": int((value < 0).sum()), "zero_rows": int((value == 0).sum()),
             "valid_rows": int(valid.sum())}
    return energy, status, audit


def load_smp(spec: Spec) -> tuple[np.ndarray, np.ndarray, dict]:
    train = pd.read_csv(spec.data_dir / "train" / "smp.csv")
    test = pd.read_csv(spec.data_dir / "smp.csv")
    frame = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)
    h, on_hour = hour_index(pd.to_datetime(frame.ts, utc=True).dt.tz_convert("Asia/Seoul"))
    if not on_hour.all() or len(np.unique(h)) != len(h) or h.min() != 0 or h.max() != N_HOURS - 1 or len(h) != N_HOURS:
        raise ValueError("SMP는 전역 시간축의 모든 시각을 한 번씩 덮어야 합니다.")
    smp = np.empty(N_HOURS)
    smp[h] = frame.smp.to_numpy(dtype=np.float64)
    if not np.isfinite(smp).all() or (smp < 0).any():
        raise ValueError("SMP는 유한하고 0 이상이어야 합니다.")
    k = k_weight(np.arange(N_HOURS))
    test_h = h[frame.split.to_numpy() == "test"]
    if not np.array_equal(k[test_h], frame.loc[frame.split == "test", "k"].to_numpy(dtype=np.int8)):
        raise ValueError("smp.csv의 k가 시가 11~15시 규칙과 다릅니다.")
    audit = {"rows_train": len(train), "rows_test": len(test), "zero_price_hours": int((smp == 0).sum())}
    return smp, k, audit


@dataclass(frozen=True)
class Channel:
    name: str
    source: tuple[str, ...]
    unit: str
    transform: str
    in_forecast: bool
    in_observed: bool
    shared_target: bool


def channels(schema: dict, provider: str, kind: str) -> list[Channel]:
    """제공사·자료 종류별 canonical channel 순서. 풍향은 원래 각도, sin, cos, 풍속×sin, 풍속×cos로 둔다."""
    fields = [f for f in schema["providers"][provider]["weather_fields"] if f[f"in_{kind}"]]
    names = {f["name"] for f in fields}
    result = []
    for f in fields:
        flags = (f["in_forecast"], f["in_observed"], f["forecast_to_observed_target"])
        scale = "" if f["scale_factor"] == 1 else f" * {f['scale_factor']:.10g}"
        result.append(Channel(f["name"], (f["name"],), f["canonical_unit"], f"{f['name']} [{f['source_unit']}]{scale}", *flags))
        if f["name"].endswith("_wind_dir"):
            for fn in ("sin", "cos"):
                result.append(Channel(f"{f['name']}_{fn}", (f["name"],), "1", f"{fn}(radians({f['name']}))", *flags))
            speed = f"{provider}_wind_speed"
            if speed in names:
                for fn in ("sin", "cos"):
                    result.append(Channel(f"{provider}_wind_vec_{fn}", (speed, f["name"]), "m/s",
                                          f"{speed}[m/s] * {fn}(radians({f['name']}))", *flags))
    return result


def parse_quality_rules(schema: dict) -> dict:
    rules = {}
    for name, text in schema["observation_quality"]["invalid_targets"].items():
        if m := re.match(r"\s*outside\s*\[\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*\]", text):
            rules[name] = (float(m.group(1)), float(m.group(2)))
        elif m := re.match(r"\s*>\s*([-0-9.]+)", text):
            rules[name] = (-np.inf, float(m.group(1)))
        elif m := re.match(r"\s*<\s*([-0-9.]+)", text):
            rules[name] = (float(m.group(1)), np.inf)
        else:
            raise ValueError(f"해석할 수 없는 품질 규칙입니다: {name}: {text}")
    return rules


def resolve_duplicates(frame: pd.DataFrame, key: list[str], fields: list[str]) -> tuple[pd.DataFrame, dict]:
    """같은 키의 필드 값이 모두 같으면 유지하고, 다르면 그 키·필드를 결측으로 둔다."""
    duplicated = frame.duplicated(key, keep=False)
    if not duplicated.any():
        return frame, {"duplicate_keys": 0, "conflicting_fields": {}}
    groups = frame[duplicated].groupby(key, sort=False)
    resolved = groups[fields].first().where(groups[fields].nunique(dropna=False) == 1)
    conflicts = (groups[fields].nunique(dropna=False) > 1).sum()
    frame = pd.concat([frame[~duplicated], resolved.reset_index()], ignore_index=True)
    return frame, {"duplicate_keys": len(resolved), "conflicting_fields": {k: int(v) for k, v in conflicts.items() if v}}


def read_weather(spec: Spec, sites: Sites, provider: str, kind: str, split: str) -> tuple[pd.DataFrame, dict]:
    """원본을 읽어 정시 외 행을 제외하고 중복 키를 정리한 뒤, 관측 품질 규칙을 적용한다."""
    path = spec.data_dir / split / f"{kind}_{provider}.parquet"
    frame = pd.read_parquet(path)
    station_col = f"{provider}_station"
    fields = [f.name for f in channels(spec.schema, provider, kind) if f.source == (f.name,) and not f.name.endswith(("_sin", "_cos"))]
    h, on_hour = hour_index(frame.ts)
    audit = {"file": f"{split}/{kind}_{provider}.parquet", "rows": len(frame), "off_hour_rows": int((~on_hour).sum()),
             "off_hour_examples": frame.loc[~on_hour, "ts"].astype(str).drop_duplicates().head(5).tolist(),
             "nan_rows_by_field": {c: int(v) for c, v in frame[fields].isna().sum().items() if v}}
    station = pd.Index(sites.stations[provider]).get_indexer(frame[station_col].to_numpy())
    if (station < 0).any():
        raise ValueError(f"sites.csv에 없는 기상지점이 있습니다: {path}")
    lo, hi = SPLIT_HOURS[split]
    if ((h[on_hour] < lo) | (h[on_hour] >= hi)).any():
        raise ValueError(f"{path}의 시각이 {split} 구간 밖에 있습니다.")
    frame = frame.assign(h=h, station=station)[on_hour]
    key = ["station", "h"] + (["round"] if kind == "forecast" else [])
    if kind == "forecast" and not frame["round"].isin(ROUNDS).all():
        raise ValueError(f"{path}에 10·17 외의 round가 있습니다.")
    frame, dup_audit = resolve_duplicates(frame[key + fields], key, fields)
    audit.update(dup_audit)
    if kind == "observed":
        violations = {}
        for name, (low, high) in parse_quality_rules(spec.schema).items():
            if name in frame:
                bad = (frame[name] < low) | (frame[name] > high)
                violations[name] = int(bad.sum())
                frame.loc[bad, name] = np.nan
        audit["quality_violations"] = violations
    n_rounds = len(ROUNDS) if kind == "forecast" else 1
    audit["missing_keys"] = int(len(sites.stations[provider]) * n_rounds * (hi - lo) - len(frame))
    return frame.reset_index(drop=True), audit


def canonicalize(schema: dict, provider: str, kind: str, frame: pd.DataFrame) -> np.ndarray:
    """원본 단위 값을 schema의 canonical 단위와 channel 순서로 바꾼다."""
    scale = {f["name"]: f["scale_factor"] for f in schema["providers"][provider]["weather_fields"]}
    out = []
    for channel in channels(schema, provider, kind):
        if channel.name.endswith("_wind_dir_sin") or channel.name.endswith("_wind_dir_cos"):
            angle = np.radians(frame[channel.source[0]].to_numpy(dtype=np.float64))
            out.append(np.sin(angle) if channel.name.endswith("_sin") else np.cos(angle))
        elif "_wind_vec_" in channel.name:
            speed, direction = channel.source
            angle = np.radians(frame[direction].to_numpy(dtype=np.float64))
            value = frame[speed].to_numpy(dtype=np.float64) * scale[speed]
            out.append(value * (np.sin(angle) if channel.name.endswith("_sin") else np.cos(angle)))
        else:
            out.append(frame[channel.name].to_numpy(dtype=np.float64) * scale[channel.name])
    return np.column_stack(out).astype(np.float32)


def base_key(spec: Spec) -> str:
    sources = [spec.data_dir / p for p in ("sites.csv", "train/generation.parquet", "train/smp.csv", "smp.csv")]
    return digest({"sources": {p.name + str(i): file_sha(p) for i, p in enumerate(sources)},
                   "versions": versions("base"), "packages": package_versions()})


KINDS = ("forecast", "observed")


@dataclass
class Base:
    energy: np.ndarray
    status: np.ndarray
    smp: np.ndarray
    k: np.ndarray
    solar: np.ndarray
    sunup: np.ndarray
    audit: dict
    key: str


@dataclass
class Weather:
    values: dict
    present: dict
    channels: dict
    audit: dict
    key: str

    def channel_names(self, provider: str, kind: str) -> list[str]:
        return [c["name"] for c in self.channels[f"{kind}_{provider}"]]


def load_base(spec: Spec, sites: Sites, store: Store) -> Base:
    key = base_key(spec)
    if not store.exists("base", key):
        print("[base] generation·SMP·태양 위치 cache 생성", flush=True)
        energy, status, gen_audit = load_generation(spec, sites)
        smp, k, smp_audit = load_smp(spec)
        solar, sunup = build_solar(np.arange(N_HOURS), sites.lat, sites.lon)
        store.write("base", key, {"energy": energy, "status": status, "smp": smp, "k": k, "solar": solar, "sunup": sunup},
                    {"audit": {"generation": gen_audit, "smp": smp_audit}, "original_hash": sites.original_hash})
    arrays, meta = store.load("base", key)
    return Base(arrays["energy"], arrays["status"], arrays["smp"], arrays["k"], arrays["solar"], arrays["sunup"], meta["audit"], key)


def weather_key(spec: Spec) -> str:
    sources = {f"{split}/{kind}_{p}": file_sha(spec.data_dir / split / f"{kind}_{p}.parquet")
               for p in PROVIDERS for kind, splits in (("forecast", ("train", "test")), ("observed", ("train",))) for split in splits}
    return digest({"sources": sources, "sites": file_sha(spec.data_dir / "sites.csv"), "schema": spec.hashes["schema"],
                   "schema_version": spec.schema["schema_version"], "versions": versions("weather"),
                   "packages": package_versions()})


def build_weather(spec: Spec, sites: Sites) -> tuple[dict, dict]:
    arrays, meta = {}, {"channels": {}, "audit": {}}
    for provider in PROVIDERS:
        n_station = len(sites.stations[provider])
        for kind in KINDS:
            names = channels(spec.schema, provider, kind)
            meta["channels"][f"{kind}_{provider}"] = [c.__dict__ for c in names]
            if kind == "forecast":
                values = np.full((len(ROUNDS), N_HOURS, n_station, len(names)), np.nan, dtype=np.float32)
                present = np.zeros((len(ROUNDS), N_HOURS, n_station), dtype=bool)
                splits = ("train", "test")
            else:
                values = np.full((N_ACTUAL, n_station, len(names)), np.nan, dtype=np.float32)
                present = np.zeros((N_ACTUAL, n_station), dtype=bool)
                splits = ("train",)
            for split in splits:
                print(f"[weather] {split}/{kind}_{provider}", flush=True)
                frame, audit = read_weather(spec, sites, provider, kind, split)
                meta["audit"][audit["file"]] = audit
                canonical = canonicalize(spec.schema, provider, kind, frame)
                h, station = frame.h.to_numpy(), frame.station.to_numpy()
                if kind == "forecast":
                    r = np.searchsorted(ROUNDS, frame["round"].to_numpy())
                    values[r, h, station] = canonical
                    present[r, h, station] = True
                else:
                    values[h, station] = canonical
                    present[h, station] = True
            arrays[f"{kind}_{provider}"] = values
            arrays[f"present_{kind}_{provider}"] = present
    return arrays, meta


def load_weather(spec: Spec, sites: Sites, store: Store) -> Weather:
    key = weather_key(spec)
    if not store.exists("weather", key):
        arrays, meta = build_weather(spec, sites)
        store.write("weather", key, arrays, meta)
    arrays, meta = store.load("weather", key)
    values = {name: arrays[name] for name in arrays if not name.startswith("present_")}
    present = {name.removeprefix("present_"): arrays[name] for name in arrays if name.startswith("present_")}
    return Weather(values, present, meta["channels"], meta["audit"], key)
