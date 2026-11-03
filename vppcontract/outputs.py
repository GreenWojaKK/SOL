"""산출물 형식.

기간마다 한 디렉터리에 배정 `composition.csv`(site_id, group), 그룹 예측 `forecast.csv`(group, ts, forecast_kwh),
`manifest.json`을 둔다. 정답이 있는 기간의 배정은 그 기간 cohort의 발전소만 담고, test 배정은 253개소 전부를 담는다.
solarvpp run 디렉터리(validation-forecast.csv 또는 forecast.csv와 manifest의 cohort)도 같은 형식으로 읽는다.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .timeaxis import GROUPS, MINIMUM_GROUP_CAPACITY_KW, hour_index, timestamps

PARTITION_FILE = "composition.csv"
FORECAST = "forecast.csv"
MANIFEST = "manifest.json"


@dataclass
class Output:
    directory: Path
    site_ids: np.ndarray
    group: np.ndarray
    hours: np.ndarray
    forecast: np.ndarray
    manifest: dict = field(default_factory=dict)

    def members(self, g: int) -> np.ndarray:
        return self.site_ids[self.group == g]


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_json(path: Path, value) -> None:
    Path(path).write_text(json.dumps(_plain(value), ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def text_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_output(directory: Path, site_ids, group, hours, forecast, manifest: dict) -> Path:
    """배정·그룹 예측·manifest를 쓴다. group은 0(G1)·1(G2), forecast는 [시각, 그룹] kWh이다."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    group, forecast, hours = np.asarray(group), np.asarray(forecast, dtype=np.float64), np.asarray(hours)
    if forecast.shape != (len(hours), len(GROUPS)):
        raise ValueError("forecast는 [시각, 그룹] 형태여야 합니다.")
    if not np.isfinite(forecast).all() or (forecast < 0).any():
        raise ValueError("forecast는 유한하고 0 이상이어야 합니다.")
    pd.DataFrame({"site_id": np.asarray(site_ids).astype(str), "group": [GROUPS[int(g)] for g in group]}).to_csv(
        directory / PARTITION_FILE, index=False, lineterminator="\n")
    stamps = [t.isoformat(sep=" ") for t in timestamps(hours)]
    pd.DataFrame({"group": np.repeat(GROUPS, len(hours)), "ts": np.tile(stamps, len(GROUPS)),
                  "forecast_kwh": np.concatenate([forecast[:, g] for g in range(len(GROUPS))])}).to_csv(
        directory / FORECAST, index=False, lineterminator="\n")
    files = {name: text_sha(directory / name) for name in (PARTITION_FILE, FORECAST)}
    write_json(directory / MANIFEST, {**manifest, "files": files})
    return directory


def _forecast_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["group", "ts", "forecast_kwh"]:
        raise ValueError(f"{path}의 열은 group, ts, forecast_kwh여야 합니다.")
    return _forecast_table_from(frame, path)


def read_output(directory: Path) -> Output:
    """계약 형식 디렉터리 또는 solarvpp run 디렉터리를 읽는다."""
    directory = Path(directory)
    manifest = read_json(directory / MANIFEST) if (directory / MANIFEST).exists() else {}
    if (directory / PARTITION_FILE).exists():
        frame = pd.read_csv(directory / PARTITION_FILE, dtype={"site_id": str, "group": str})
        if list(frame.columns) != ["site_id", "group"] or not frame.site_id.is_unique or not frame.group.isin(GROUPS).all():
            raise ValueError(f"{directory / PARTITION_FILE}은 유일한 site_id와 G1·G2 group을 담아야 합니다.")
        site_ids, group = frame.site_id.to_numpy(dtype=object), np.array([GROUPS.index(g) for g in frame.group], dtype=np.int8)
        hours, forecast = _forecast_table(directory / FORECAST)
    elif "cohort" in manifest and "site_ids" in manifest["cohort"]:
        site_map = manifest["cohort"]["site_ids"]
        site_ids = np.array([s for g in GROUPS for s in site_map[g]], dtype=object)
        group = np.array([i for i, g in enumerate(GROUPS) for _ in site_map[g]], dtype=np.int8)
        name = "validation-forecast.csv" if (directory / "validation-forecast.csv").exists() else FORECAST
        frame = pd.read_csv(directory / name)
        hours, forecast = _forecast_table_from(frame, directory / name)
    else:
        raise ValueError(f"{directory}는 산출물 형식도 solarvpp run 형식도 아닙니다.")
    return Output(directory, site_ids, group, hours, forecast, manifest)


def _forecast_table_from(frame: pd.DataFrame, path: Path) -> tuple[np.ndarray, np.ndarray]:
    """그룹 예측 표를 [시각, 그룹] 배열로 바꾼다. solarvpp run 예측 CSV도 group·ts·forecast_kwh 열을 담는다."""
    tmp = frame[["group", "ts", "forecast_kwh"]]
    if not tmp.group.isin(GROUPS).all():
        raise ValueError(f"{path}의 group은 G1·G2여야 합니다.")
    h, on_hour = hour_index(pd.to_datetime(tmp.ts))
    if not on_hour.all() or tmp.duplicated(["group", "ts"]).any():
        raise ValueError(f"{path}의 시각이 정시가 아니거나 중복되었습니다.")
    hours = np.unique(h)
    out = np.full((len(hours), len(GROUPS)), np.nan)
    out[np.searchsorted(hours, h), [GROUPS.index(g) for g in tmp.group]] = tmp.forecast_kwh.to_numpy(dtype=np.float64)
    if np.isnan(out).any():
        raise ValueError(f"{path}에 그룹마다 같은 시각이 모두 있어야 합니다.")
    return hours, out


def validate_submission(directory: Path, data_dir: Path) -> dict:
    """제출 형식 검사: 253개소 배정, 그룹당 20,000 kW 이상, smp.csv 8,760시각, timezone, 0 이상의 유한값."""
    directory, data_dir = Path(directory), Path(data_dir)
    sites = pd.read_csv(data_dir / "sites.csv")
    partition = pd.read_csv(directory / PARTITION_FILE, dtype={"site_id": str, "group": str})
    forecast = pd.read_csv(directory / FORECAST)
    smp = pd.read_csv(data_dir / "smp.csv")
    lines = {name: len((directory / name).read_text(encoding="utf-8").splitlines()) for name in (PARTITION_FILE, FORECAST)}
    expected_ts = set(pd.to_datetime(smp.ts).map(lambda t: t.isoformat(sep=" ")))
    cap = partition.merge(sites[["site_id", "capacity_kw"]], on="site_id").groupby("group").capacity_kw.sum()
    checks = {
        "partition_columns": list(partition.columns) == ["site_id", "group"],
        "partition_lines": lines[PARTITION_FILE] == len(sites) + 1,
        "partition_sites": sorted(partition.site_id) == sorted(sites.site_id.astype(str)) and partition.site_id.is_unique,
        "partition_groups": set(partition.group) == set(GROUPS),
        "group_capacity": bool((cap.reindex(list(GROUPS)).fillna(0.0) >= MINIMUM_GROUP_CAPACITY_KW).all()),
        "forecast_columns": list(forecast.columns) == ["group", "ts", "forecast_kwh"],
        "forecast_lines": lines[FORECAST] == len(GROUPS) * len(smp) + 1,
        "forecast_keys_unique": not forecast.duplicated(["group", "ts"]).any(),
        "forecast_ts_match_smp": all(set(forecast.loc[forecast.group == g, "ts"]) == expected_ts for g in GROUPS),
        "forecast_ts_timezone": bool(forecast.ts.astype(str).str.endswith("+09:00").all()),
        "forecast_finite_nonnegative": bool(np.isfinite(forecast.forecast_kwh).all() and (forecast.forecast_kwh >= 0).all()),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"제출 파일 검사 실패: {failed}")
    return {"checks": checks, "lines": lines, "group_capacity_kw": {g: float(cap[g]) for g in GROUPS},
            "sha": {name: text_sha(directory / name) for name in (PARTITION_FILE, FORECAST)}}
