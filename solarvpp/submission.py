"""test 입력 분포 점검과 제출 파일 작성·검사."""
from __future__ import annotations

import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .config import Spec
from .data import GROUPS, N_ACTUAL, N_HOURS, PROVIDERS, ROUNDS, Sites, Weather, delivery_dates, timestamps, to_h
from .store import digest, read_json, versions, write_json

MINIMUM_GROUP_CAPACITY_KW = 20_000


def check_test_inputs(spec: Spec, weather: Weather) -> dict:
    """test forecast를 같은 달의 기준 1년과 provider×station×round×channel 단위로 비교한다."""
    rule = spec.config["test_input_check"]
    ref_lo = to_h(pd.Timestamp(rule["reference_start"] + " 01:00", tz="Asia/Seoul"))
    ref_hi = to_h(pd.Timestamp(rule["reference_end"] + " 01:00", tz="Asia/Seoul") + pd.Timedelta(hours=23))
    ref_h, test_h = np.arange(ref_lo, ref_hi + 1), np.arange(N_ACTUAL, N_HOURS)
    month = {"ref": delivery_dates(ref_h // 24).month.to_numpy(), "test": delivery_dates(test_h // 24).month.to_numpy()}
    flags, details = {}, []
    for p in PROVIDERS:
        values = np.asarray(weather.values[f"forecast_{p}"])
        for c, name in enumerate(weather.channel_names(p, "forecast")):
            # 드문 사건만 값이 있는 channel(예: 결빙 확률)은 지점 표준편차가 0에 가까우므로 channel 전체 표준편차를 하한으로 둔다.
            channel_scale = float(np.nanstd(values[:, ref_h, :, c]))
            for s in range(values.shape[2]):
                rev = {}
                for key, hours in (("ref", ref_h), ("test", test_h)):
                    a, b = values[0, hours, s, c], values[1, hours, s, c]
                    both = np.isfinite(a) & np.isfinite(b)
                    rev[key] = float((np.abs(b[both] - a[both]) > 1e-9).mean()) if both.any() else np.nan
                for r, round_name in enumerate(ROUNDS):
                    ref, test = values[r, ref_h, s, c], values[r, test_h, s, c]
                    fr, ft = np.isfinite(ref), np.isfinite(test)
                    if fr.sum() < 100 or ft.sum() < 100:
                        continue
                    scale = max(float(np.std(ref[fr])), channel_scale, 1e-6)
                    shifts = []
                    for m in range(1, 13):
                        a = ref[fr & (month["ref"] == m)]
                        b = test[ft & (month["test"] == m)]
                        if len(a) and len(b):
                            shifts.append((b.mean() - a.mean()) / scale)
                    shifts = np.abs(np.array(shifts))
                    lo, hi = np.quantile(ref[fr], [0.001, 0.999])
                    item = {"provider": p, "channel": name, "station": int(s), "round": round_name,
                            "max_month_shift": float(shifts.max()), "months_over": int((shifts > rule["mean_shift"]).sum()),
                            "ks": float(ks_2samp(ref[fr], test[ft]).statistic),
                            "out_of_range": float(((test[ft] < lo) | (test[ft] > hi)).mean()),
                            "revision_rate_ref": rev["ref"], "revision_rate_test": rev["test"]}
                    kinds = []
                    if item["months_over"] >= rule["mean_shift_months"]:
                        kinds.append("mean_shift")
                    if item["ks"] > rule["ks"]:
                        kinds.append("ks")
                    if item["out_of_range"] > rule["out_of_range"]:
                        kinds.append("out_of_range")
                    if np.isfinite(rev["ref"]) and np.isfinite(rev["test"]) and abs(rev["test"] - rev["ref"]) > rule["revision_rate_change"]:
                        kinds.append("revision_rate")
                    for kind in kinds:
                        flag = flags.setdefault(f"{name}:{kind}", {"provider": p, "channel": name, "kind": kind, "station_rounds": 0, "worst": 0.0})
                        flag["station_rounds"] += 1
                        value = {"mean_shift": item["max_month_shift"], "ks": item["ks"], "out_of_range": item["out_of_range"],
                                 "revision_rate": abs(rev["test"] - rev["ref"])}[kind]
                        flag["worst"] = max(flag["worst"], float(value))
                    if kinds:
                        details.append(item)
    return {"weather_key": weather.key, "rule": rule, "reference_hours": [int(ref_lo), int(ref_hi)],
            "flags": dict(sorted(flags.items())), "flagged_station_rounds": details}


def gate(report: dict, used_channels: set[str], acknowledged: list[str], strict: bool = False) -> dict:
    """Distribution flag는 advisory로 기록한다. strict=True를 선택한 경우에만 acknowledgement를 요구한다."""
    pending, accepted, unused = [], [], []
    for flag_id, flag in report["flags"].items():
        if flag["channel"] not in used_channels:
            unused.append(flag_id)
        elif any(fnmatch.fnmatch(flag_id, pattern) for pattern in acknowledged):
            accepted.append(flag_id)
        else:
            pending.append(flag_id)
    return {"blocking": pending if strict else [], "warnings": pending, "strict": strict,
            "acknowledged": accepted, "not_used_as_input": unused, "patterns": list(acknowledged)}


def write_submission(sites: Sites, smp_path: Path, hours: np.ndarray, forecast: np.ndarray, out_dir: Path) -> dict:
    """composition.csv·forecast.csv를 쓰고 다시 읽어 형식을 검사한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    partition = pd.DataFrame({"site_id": sites.ids.astype(str), "group": [GROUPS[g] for g in sites.group]})
    stamps = [t.isoformat(sep=" ") for t in timestamps(hours)]
    frame = pd.DataFrame({"group": np.repeat(GROUPS, len(hours)), "ts": np.tile(stamps, len(GROUPS)),
                          "forecast_kwh": np.concatenate([forecast[:, g] for g in range(len(GROUPS))])})
    partition.to_csv(out_dir / "composition.csv", index=False, lineterminator="\n")
    frame.to_csv(out_dir / "forecast.csv", index=False, lineterminator="\n")
    return validate_submission(sites, smp_path, out_dir)


def validate_submission(sites: Sites, smp_path: Path, out_dir: Path) -> dict:
    partition = pd.read_csv(out_dir / "composition.csv")
    forecast = pd.read_csv(out_dir / "forecast.csv")
    lines = {name: len((out_dir / name).read_text(encoding="utf-8").splitlines()) for name in ("composition.csv", "forecast.csv")}
    smp = pd.read_csv(smp_path)
    expected_ts = set(pd.to_datetime(smp.ts).map(lambda t: t.isoformat(sep=" ")))
    cap = partition.merge(pd.DataFrame({"site_id": sites.ids.astype(str), "capacity_kw": sites.capacity}), on="site_id").groupby("group").capacity_kw.sum()
    checks = {
        "partition_lines": lines["composition.csv"] == len(sites) + 1,
        "partition_sites": sorted(partition.site_id) == sorted(sites.ids.astype(str)) and partition.site_id.is_unique,
        "partition_groups": set(partition.group) == set(GROUPS),
        "group_capacity": bool((cap.reindex(list(GROUPS)) >= MINIMUM_GROUP_CAPACITY_KW).all()),
        "forecast_lines": lines["forecast.csv"] == len(GROUPS) * len(smp) + 1,
        "forecast_keys_unique": not forecast.duplicated(["group", "ts"]).any(),
        "forecast_ts_match_smp": all(set(forecast.loc[forecast.group == g, "ts"]) == expected_ts for g in GROUPS),
        "forecast_ts_timezone": bool(forecast.ts.str.endswith("+09:00").all()),
        "forecast_finite_nonnegative": bool(np.isfinite(forecast.forecast_kwh).all() and (forecast.forecast_kwh >= 0).all()),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"제출 파일 검사 실패: {failed}")
    capacity_factor = {g: float(forecast.loc[forecast.group == g, "forecast_kwh"].sum() / (cap[g] * len(smp))) for g in GROUPS}
    return {"checks": checks, "lines": lines, "group_capacity_kw": {g: float(cap[g]) for g in GROUPS}, "annual_capacity_factor": capacity_factor,
            "sha": {name: digest((out_dir / name).read_text(encoding="utf-8")) for name in ("composition.csv", "forecast.csv")}}


def report_path(spec: Spec, weather: Weather, reports_dir: Path) -> Path:
    key = digest({"weather": weather.key, "rule": spec.config["test_input_check"], "versions": versions("test_input_check")})
    return reports_dir / f"test-inputs-{key[:16]}.json"


def load_or_check(spec: Spec, weather: Weather, reports_dir: Path) -> dict:
    path = report_path(spec, weather, reports_dir)
    if path.exists():
        return read_json(path)
    report = check_test_inputs(spec, weather)
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(path, report)
    return report
