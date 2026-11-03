"""expert 분위수의 확률예측 진단: 채점, 계절·시간대·예보 조건 표, 데스크톱 viewer용 데이터.

package는 checkpoint에서 분위수를 꺼내 아래 파일을 한 출력 폴더에 쓰고, 이 module은 그 파일만 읽어 진단을 만든다.
결과 폴더는 repo의 `viewer/`(Rust 데스크톱 앱)가 연다. 이 module은 표와 viewer 데이터만 쓰고 그림은 그리지 않는다.

- `common.npz`: hours, dates, ts, groups, alphas, truth [시각, 그룹](결측 NaN), complete, daylight, common_daylight,
  capacity_kw [그룹], seeds, smp, k, cloud·disagreement [시각, 그룹]
- `<expert>_quantiles.npz`: quantiles·raw [시각, 그룹, 분위수], seed_quantiles·seed_raw [seed, 시각, 그룹, 분위수]
- `manifest.json`: contract_version, instance, experts(ID → label), 선택 항목 regime_edges·provenance.
  cloud·disagreement는 선택적인 예보 조건이다. 이전 v1 artifact의 xgb·dl 이름을 읽는다.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .identity import digest, file_sha
from .outputs import read_json, write_json
from .timeaxis import GROUPS, SEASON_IDS, delivery_dates, season_index, timestamps

VERSION = 3
CONTRACT_VERSION = 1
LEVELS = ((50, .25, .75), (80, .1, .9), (90, .05, .95))
# 데스크톱 viewer 실행 파일. VPP_PROBABILISTIC_VIEWER 환경 변수로 다른 경로를 지정할 수 있다.
VIEWER = Path(__file__).resolve().parents[1] / "viewer" / "target" / "release" / "probabilistic-viewer.exe"
VIEWER_DATA = "explorer-data.json"


def expert_definitions(manifest):
    """ID·표시 이름은 artifact가 정한다. 초기 v1 artifact의 두 expert 이름도 읽는다."""
    items = manifest.get("experts", {"xgb": {"label": "Tree-only"}, "dl": {"label": "DL-only"}})
    if not isinstance(items, dict) or not items:
        raise ValueError("manifest.experts는 비어 있지 않은 ID → metadata 객체여야 합니다.")
    for name, metadata in items.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not isinstance(metadata, dict):
            raise ValueError("expert ID 또는 metadata 형식이 올바르지 않습니다.")
        if not isinstance(metadata.get("label", name), str):
            raise ValueError("expert label은 문자열이어야 합니다.")
    return {name: {"label": meta.get("label", name)} for name, meta in items.items()}


def validate_artifact(directory, manifest, common, experts):
    """시간·그룹·분위수·mask 계약을 검사한다. 원 package, 모델 파일, 원자료 경로를 조회하지 않는다."""
    if manifest.get("contract_version", 1) != CONTRACT_VERSION:
        raise ValueError("지원하지 않는 probabilistic contract version입니다.")
    required = {"hours", "dates", "ts", "groups", "alphas", "truth", "complete", "daylight", "common_daylight", "capacity_kw", "seeds", "smp", "k"}
    if required - common.keys():
        raise ValueError(f"common.npz 필수 항목 누락: {sorted(required - common.keys())}")
    h, a, capacity = common["hours"], common["alphas"], common["capacity_kw"]
    n, g, q, s = len(h), len(GROUPS), len(a), len(common["seeds"])
    if not n or h.dtype.kind not in "iu" or (h < 0).any() or (np.diff(h) <= 0).any():
        raise ValueError("hours는 오름차순의 중복 없는 전역 시간 index여야 합니다.")
    days = np.unique(h // 24)
    if not np.array_equal(h, (days[:, None] * 24 + np.arange(24)).ravel()):
        raise ValueError("각 대상일은 24개 시각을 모두 포함해야 합니다.")
    if list(common["groups"].astype(str)) != list(GROUPS):
        raise ValueError("groups 순서는 G1, G2여야 합니다.")
    if not np.array_equal(common["dates"].astype(str), delivery_dates(days).strftime("%Y-%m-%d").to_numpy()):
        raise ValueError("dates와 hours가 일치하지 않습니다.")
    if not np.array_equal(common["ts"].astype(str), timestamps(h).astype(str).to_numpy()):
        raise ValueError("ts와 hours가 일치하지 않습니다.")
    if a.ndim != 1 or not np.isfinite(a).all() or (a <= 0).any() or (a >= 1).any() or (np.diff(a) <= 0).any():
        raise ValueError("alphas는 (0,1) 범위의 오름차순 분위수여야 합니다.")
    for value in (.05, .1, .25, .5, .75, .9, .95):
        qindex(a, value)
    if capacity.shape != (g,) or not np.isfinite(capacity).all() or (capacity <= 0).any():
        raise ValueError("capacity_kw는 그룹별 양의 용량이어야 합니다.")
    for name, shape in {"truth": (n, g), "complete": (n,), "daylight": (n, g), "common_daylight": (n,), "smp": (n,), "k": (n,)}.items():
        if common[name].shape != shape:
            raise ValueError(f"{name} shape은 {shape}여야 합니다.")
    for name in ("complete", "daylight", "common_daylight"):
        if common[name].dtype.kind != "b":
            raise ValueError(f"{name}은 boolean mask여야 합니다.")
    if not np.array_equal(common["common_daylight"], common["daylight"].all(axis=1)):
        raise ValueError("common_daylight는 두 그룹 daylight의 교집합이어야 합니다.")
    y = common["truth"][common["complete"]]
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("complete 시각의 실적은 유한한 0 이상 값이어야 합니다.")
    for name in ("smp", "k"):
        if not np.isfinite(common[name]).all() or (common[name] < 0).any():
            raise ValueError(f"{name}은 유한한 0 이상 값이어야 합니다.")
    if len(np.unique(common["seeds"])) != s:
        raise ValueError("seed가 중복되었습니다.")
    for name, arrays in experts.items():
        expected = {"quantiles": (n, g, q), "raw": (n, g, q), "seed_quantiles": (s, n, g, q), "seed_raw": (s, n, g, q)}
        if {"quantiles", "raw", "seed_quantiles", "seed_raw"} - arrays.keys():
            raise ValueError(f"{name}의 분위수 필드가 누락되었습니다.")
        for field, shape in expected.items():
            v = arrays[field]
            if v.shape != shape or not np.isfinite(v).all():
                raise ValueError(f"{name}.{field} shape 또는 유한값 조건이 맞지 않습니다.")
            if field.endswith("quantiles") and ((np.diff(v, axis=-1) < -1e-9).any() or (v < 0).any() or (v > capacity[None, :, None] + 1e-6).any()):
                raise ValueError(f"{name}.{field}의 단조성·물리 범위가 맞지 않습니다.")
    for axis in ("cloud", "disagreement"):
        if axis in common:
            edges = np.asarray(manifest.get("regime_edges", {}).get(axis, []))
            if common[axis].shape != (n, g) or not np.isfinite(edges).all() or (np.diff(edges) <= 0).any():
                raise ValueError(f"{axis} 예보 조건의 shape·경계가 올바르지 않습니다.")
    for name, sha in manifest.get("artifact_hashes", {}).items():
        path = (Path(directory) / name).resolve()
        if path.parent != Path(directory).resolve() or not path.is_file() or file_sha(path) != sha:
            raise ValueError(f"artifact hash 또는 경로가 맞지 않습니다: {name}")


def qindex(alphas, value):
    matches = np.flatnonzero(np.isclose(alphas, value))
    if len(matches) != 1:
        raise ValueError(f"분위수 {value}가 정확히 한 번 필요합니다.")
    return int(matches[0])


def score_quantiles(y, q, alphas, capacity, raw=None):
    """동일 mask로 고른 한 그룹·조건의 주간 표본. 오차 부호는 예측−실제이다."""
    y, q, alphas = np.asarray(y), np.asarray(q), np.asarray(alphas)
    if not len(y):
        raise ValueError("분위수 채점에는 유효 표본이 필요합니다.")
    if q.shape != (len(y), len(alphas)) or not np.isfinite(q).all() or not np.isfinite(y).all():
        raise ValueError("정답·분위수의 shape 또는 유한값 조건이 맞지 않습니다.")
    if np.any(np.diff(q, axis=-1) < -1e-9):
        raise ValueError("최종 분위수는 단조 증가해야 합니다.")
    error = q[:, qindex(alphas, .5)] - y
    residual = y[:, None] - q
    pinball = np.maximum(alphas * residual, (alphas - 1) * residual)
    summary = {"n": len(y), "median_bias_kwh": float(np.median(error)), "mean_bias_kwh": float(np.mean(error)),
               "median_bias_norm": float(np.median(error) / capacity), "mae_kwh": float(np.mean(abs(error))),
               "pinball_kwh": float(pinball.mean()), "pinball_norm": float(pinball.mean() / capacity),
               "actual_zero_rate": float(np.mean(y == 0))}
    if raw is not None:
        summary.update(crossing_rate=float(np.mean(np.any(np.diff(raw, axis=-1) < 0, axis=-1))),
                       clipping_rate=float(np.mean((raw < 0) | (raw > capacity))))
    intervals = []
    for level, low, high in LEVELS:
        lo, hi = q[:, qindex(alphas, low)], q[:, qindex(alphas, high)]
        below, above = y < lo, y > hi
        intervals.append({"level": level, "nominal": level / 100, "coverage": float(np.mean(~(below | above))),
                          "below": float(np.mean(below)), "above": float(np.mean(above)),
                          "width_kwh": float(np.mean(hi - lo)), "width_norm": float(np.mean(hi - lo) / capacity),
                          "zero_width_rate": float(np.mean(hi == lo)),
                          "endpoint_tie_rate": float(np.mean((y == lo) | (y == hi)))})
    reliability = [{"alpha": float(a), "strict": float(np.mean(y < q[:, j])), "inclusive": float(np.mean(y <= q[:, j])),
                    "pinball_kwh": float(pinball[:, j].mean())} for j, a in enumerate(alphas)]
    return summary, intervals, reliability


def load_arrays(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    with np.load(directory / "common.npz", allow_pickle=False) as f:
        common = {k: f[k] for k in f.files}
    experts = {}
    for expert in expert_definitions(manifest):
        with np.load(directory / f"{expert}_quantiles.npz", allow_pickle=False) as f:
            experts[expert] = {k: f[k] for k in f.files}
    validate_artifact(directory, manifest, common, experts)
    return common, experts


def diagnose(directory, output_dir=None):
    """저장 NPZ만으로 표·그림 입력을 재계산한다. 실현값이 없는 시각은 모든 지표에서 제외한다."""
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    c, experts = load_arrays(directory)
    destination = Path(output_dir) if output_dir else directory
    destination.mkdir(parents=True, exist_ok=True)
    seasons = np.array(SEASON_IDS)[season_index(c["hours"] // 24)]
    mask = c["complete"] & c["common_daylight"]
    summaries, intervals, reliability, hourly, regimes = [], [], [], [], []
    for expert, arrays in experts.items():
        variants = [("ensemble", arrays["quantiles"], arrays["raw"])]
        variants += [(str(s), arrays["seed_quantiles"][j], arrays["seed_raw"][j]) for j, s in enumerate(c["seeds"])]
        for seed, q, raw in variants:
            for g, group in enumerate(GROUPS):
                for season in ("all", *SEASON_IDS):
                    in_season = np.ones(len(mask), dtype=bool) if season == "all" else seasons == season
                    use = mask & in_season
                    if not use.any():
                        continue
                    summary, ints, rel = score_quantiles(c["truth"][use, g], q[use, g], c["alphas"], c["capacity_kw"][g], raw[use, g])
                    keys = {"expert": expert, "seed": seed, "group": group, "season": season}
                    error = q[:, g, qindex(c["alphas"], .5)] - c["truth"][:, g]
                    valid_all = c["complete"] & in_season
                    summary.update(W_krw=float(np.sum(.1 * c["smp"][valid_all] * c["k"][valid_all] * abs(error[valid_all]))),
                                   days=int(len(np.unique(c["hours"][use] // 24))), all_hours=int(valid_all.sum()))
                    summaries.append({**keys, **summary})
                    intervals.extend({**keys, "n": int(use.sum()), **r} for r in ints)
                    reliability.extend({**keys, "n": int(use.sum()), **r} for r in rel)
                    if seed == "ensemble" and season != "all":
                        for slot in range(24):
                            u = use & (c["hours"] % 24 == slot)
                            hourly.append({**keys, "hour": slot + 1, "n": int(u.sum()),
                                           "median_bias_norm": float(np.median(error[u]) / c["capacity_kw"][g]) if u.any() else None})
                if seed == "ensemble":
                    for axis in ("cloud", "disagreement"):
                        if axis not in c:
                            continue
                        values = c[axis][:, g]
                        bins = np.searchsorted(manifest["regime_edges"][axis], values, side="right")
                        for b in range(len(manifest["regime_edges"][axis]) + 1):
                            use = mask & np.isfinite(values) & (bins == b)
                            if use.any():
                                sm, ints, _ = score_quantiles(c["truth"][use, g], q[use, g], c["alphas"], c["capacity_kw"][g])
                                mid = next(r for r in ints if r["level"] == 80)
                                regimes.append({"expert": expert, "group": group, "axis": axis, "bin": b, **sm,
                                                "coverage80": mid["coverage"], "width80_norm": mid["width_norm"]})
    tables = {"seasonal_summary": summaries, "intervals": intervals, "reliability": reliability, "hourly_bias": hourly, "weather_regimes": regimes}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(destination / f"{name}.csv", index=False, encoding="utf-8-sig")
    write_json(destination / "diagnostics.json", {"version": VERSION, "mask": "complete cohort and both groups daylight; W uses all complete hours", **tables})
    return c, experts, tables


def explorer_data(c, experts, tables, manifest):
    levels = [.05, .1, .25, .5, .75, .9, .95]
    selected = [qindex(c["alphas"], a) for a in levels]
    rows = []
    for d, date in enumerate(c["dates"]):
        sl = slice(d * 24, (d + 1) * 24)
        for g, group in enumerate(GROUPS):
            truth = c["truth"][sl, g]
            row = {"date": str(date), "group": group, "capacity_mw": float(c["capacity_kw"][g] / 1000),
                   "actual": [round(float(y) / 1000, 6) if np.isfinite(y) else None for y in truth],
                   "valid": c["complete"][sl].tolist(), "daylight": c["common_daylight"][sl].tolist()}
            for expert in experts:
                row[expert] = np.round(experts[expert]["quantiles"][sl, g][:, selected] / 1000, 6).tolist()
            rows.append(row)
    primary = {key: [r for r in value if r.get("seed", "ensemble") == "ensemble"] for key, value in tables.items()}
    available = {r["season"] for r in primary["seasonal_summary"]}
    return {"schema_version": VERSION, "manifest": manifest, "experts": expert_definitions(manifest),
            "seasons": [s for s in SEASON_IDS if s in available], "levels": levels, "rows": rows, "tables": primary}


def report(directory, output_dir=None):
    """공통 artifact 검증 → 평가표·viewer 데이터. 입력 폴더와 지정한 결과 폴더만 사용한다."""
    directory = Path(directory).resolve()
    output = Path(output_dir).resolve() if output_dir else directory / "report"
    manifest = read_json(directory / "manifest.json")
    definitions = expert_definitions(manifest)
    c, experts, tables = diagnose(directory, output)
    payload = explorer_data(c, experts, tables, manifest)
    write_json(output / VIEWER_DATA, payload)
    inputs = {name: file_sha(directory / name) for name in
              ("manifest.json", "common.npz", *(f"{e}_quantiles.npz" for e in definitions))}
    generated = ["diagnostics.json", VIEWER_DATA, *(f"{t}.csv" for t in tables)]
    receipt = {"artifact_directory": str(directory), "directory": str(output), "viewer_data": str(output / VIEWER_DATA),
               "diagnostic_version": VERSION, "contract_version": CONTRACT_VERSION, "input_hashes": inputs,
               "seasons": payload["seasons"], "experts": definitions,
               "generated_hashes": {name: file_sha(output / name) for name in generated}}
    receipt["key"] = digest({"inputs": inputs, "diagnostic_version": VERSION})
    write_json(output / "report.json", receipt)
    return receipt


def viewer_path(viewer=None):
    """데스크톱 viewer 실행 파일. 인자, VPP_PROBABILISTIC_VIEWER, repo 빌드 위치 순서로 찾는다."""
    path = Path(viewer or os.environ.get("VPP_PROBABILISTIC_VIEWER") or VIEWER)
    if not path.is_file():
        raise FileNotFoundError(
            f"확률예측 viewer 실행 파일이 없습니다: {path}\n"
            "repo의 viewer 폴더에서 `cargo build --release`로 빌드하거나, "
            "VPP_PROBABILISTIC_VIEWER에 실행 파일 경로를 지정하세요.")
    return path


def open_viewer(report_dir, viewer=None, launch=subprocess.Popen):
    """결과 폴더를 데스크톱 viewer로 연다. viewer는 별도 프로세스로 실행하고 기다리지 않는다."""
    report_dir = Path(report_dir).resolve()
    if not (report_dir / VIEWER_DATA).is_file():
        raise FileNotFoundError(f"{report_dir}에 {VIEWER_DATA}가 없습니다. `vppcontract probabilistic`으로 먼저 결과를 만드세요.")
    path = viewer_path(viewer)
    launch([str(path), str(report_dir)])
    return path
