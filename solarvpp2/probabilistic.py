"""저장 checkpoint의 expert 분위수 추출. 채점·표·viewer 데이터는 공통 계약층(`vppcontract.probabilistic`)이 만든다.

학습과 배정 탐색을 호출하지 않는다. 추론 원본은 model key로, 진단은 저장 NPZ와 version으로 식별한다.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from vppcontract import probabilistic as diagnostics
from vppcontract.evaluate import Actuals, group_truth
from vppcontract.identity import digest, file_sha
from vppcontract.outputs import read_output
from vppcontract.timeaxis import GROUPS, delivery_dates, period, timestamps
from .store import Store, environment, read_json, source_sha, write_json

VERSION = 1
EXPERTS = ("xgb", "dl")


def weather_context(data, members, hours):
    """17시 발행 운량의 그룹 용량 가중 평균. 제공사 하나라도 결측이면 그 그룹·시각을 NaN으로 둔다."""
    clouds = []
    for provider in ("wa", "wb"):
        weather = data.weather
        j = weather.channel_names(provider, "forecast").index(provider + "_cloud")
        values = np.asarray(weather.values["forecast_" + provider])[1, hours, :, j]
        pv = values[:, data.sites.station_index[provider]]
        columns = []
        for membership in members:
            columns.append(np.sum(pv[:, membership] * data.sites.capacity[membership], axis=1) /
                           data.sites.capacity[membership].sum())
        clouds.append(np.stack(columns, axis=1))
    return (clouds[0] + clouds[1]) / 2, abs(clouds[0] - clouds[1])


def extract(spec, instance_dir, partition, output_dir):
    """지정 run의 모델·전처리와 고정 배정으로 expert 출력을 저장한다. 실패 시 학습으로 대체하지 않는다.

    정답이 있는 기간의 완료된 고정 cohort instance(forecast·full 단계)를 받는다. panel 단계는 날짜별 구성원을 쓰므로 받지 않는다.
    """
    from .data import load_data
    from .evaluation import inference_spec
    from .forecast.forecaster import Forecaster

    instance_dir, output_dir = Path(instance_dir).resolve(), Path(output_dir).resolve()
    source = read_json(instance_dir / "instance.json")
    saved = read_json(instance_dir / "artifacts.json")["inference"]
    if source.get("stage") == "panel" or source.get("smoke") or source.get("status") != "completed":
        raise ValueError("완료된 고정 cohort의 정규 instance를 지정하세요.")
    if not source.get("cache_stored", True) and Store.temporary is None:
        raise ValueError("이 실행은 모델 파일을 저장하지 않았습니다. 같은 설정에 --extract-quantiles를 붙여 다시 실행하세요.")
    per = period(source["instance"])
    if not per.has_labels:
        raise ValueError("정답이 있는 기간이 필요합니다.")
    original = read_output(instance_dir / "outputs" / partition)
    if not np.array_equal(original.hours, per.hours):
        raise ValueError("저장된 예측은 평가 기간 전체를 포함해야 합니다.")
    identity = {"source": str(instance_dir), "inference": saved, "models": source["forecaster"],
                "partition": file_sha(original.directory / "composition.csv"), "version": VERSION}
    key = digest(identity)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing["key"] != key:
            raise ValueError("출력 폴더에 다른 추론 identity가 있습니다. 새 폴더를 지정하세요.")
        for name, sha in existing["artifact_hashes"].items():
            if file_sha(output_dir / name) != sha:
                raise ValueError(f"저장 artifact가 변경되었습니다: {name}")
        print("[probabilistic] 저장 분위수 재사용", flush=True)
        return existing
    resolved = inference_spec(spec, source, saved)
    print("[probabilistic] 원 run 데이터·전처리·checkpoint 로딩", flush=True)
    data = load_data(resolved)
    if (data.base.key, data.weather.key) != (saved["base_key"], saved["weather_key"]):
        raise ValueError("원 run과 base·weather identity가 다릅니다.")
    f = Forecaster(data, source["model"], source["cutoff_h"], per.days)
    f.load_cached(source["forecaster"], original.manifest["forecaster"], saved["power_curve"])
    index = pd.Index(data.sites.ids.astype(str)).get_indexer(original.site_ids.astype(str))
    if (index < 0).any() or set(index) != set(np.flatnonzero(f.cohort)):
        raise ValueError("저장 배정과 checkpoint의 cohort가 다릅니다.")
    members = np.zeros((2, len(data.sites.ids)), dtype=bool)
    for g in range(2):
        members[g, index[original.group == g]] = True
    capacities = members @ data.sites.capacity
    actuals = Actuals(data.sites.ids, data.sites.capacity, np.asarray(data.base.energy), np.asarray(data.base.status) == 1,
                      np.asarray(data.base.smp), np.asarray(data.base.k))
    truth, complete = group_truth(actuals, original, per.hours)
    parts = []
    for start in range(0, len(per.days), 16):
        chunk = per.days[start:start + 16]
        parts.append(f.predict(np.repeat(chunk, 2), np.tile(members, (len(chunk), 1)), expert_quantiles=True))
        print(f"[probabilistic] 추론 {min(start + 16, len(per.days))}/{len(per.days)}일", flush=True)
    from .forecast.forecaster import Prediction
    prediction = Prediction.concat(parts)

    def hourly(a):
        # query 순서 = 대상일, 그룹. seed 축은 export에서 맨 앞으로 옮긴다.
        if a.ndim == 2:
            return a.reshape(-1, 2, 24).transpose(0, 2, 1).reshape(-1, 2)
        if a.ndim == 3:
            return a.reshape(-1, 2, 24, a.shape[-1]).transpose(0, 2, 1, 3).reshape(-1, 2, a.shape[-1])
        return a.reshape(-1, 2, len(f.seeds), 24, a.shape[-1]).transpose(2, 0, 3, 1, 4).reshape(len(f.seeds), -1, 2, a.shape[-1])

    blend_median = hourly(prediction.median)
    max_diff = float(np.max(abs(blend_median - original.forecast)))
    if not np.allclose(blend_median, original.forecast, rtol=2e-6, atol=.02):
        raise ValueError(f"원 run의 median 재현 실패: 최대 차이 {max_diff} kWh")
    output_dir.mkdir(parents=True, exist_ok=True)
    cloud, disagreement = weather_context(data, members, per.hours)
    train_hours = (f.train_days[:, None] * 24 + np.arange(24)).ravel()
    train_cloud, train_disagreement = weather_context(data, members, train_hours)
    train_sun = np.stack([np.any(f.sunup[train_hours][:, m], axis=1) for m in members], axis=1)
    regime_edges = {}
    for name, value in (("cloud", train_cloud), ("disagreement", train_disagreement)):
        selected = value[train_sun & np.isfinite(value)]
        regime_edges[name] = np.unique(np.quantile(selected, [1 / 3, 2 / 3])).tolist()
    daylight = hourly(prediction.daylight)
    # 두 그룹의 주간 비교에도 동일 시각을 사용한다.
    common_daylight = daylight.all(axis=1)
    np.savez_compressed(output_dir / "common.npz", hours=per.hours, dates=delivery_dates(per.days).strftime("%Y-%m-%d").to_numpy(dtype=str),
                        ts=timestamps(per.hours).astype(str).to_numpy(dtype=str), groups=np.array(GROUPS), alphas=resolved.quantiles,
                        truth=np.where(complete[:, None], truth, np.nan), complete=complete, daylight=daylight,
                        common_daylight=common_daylight, capacity_kw=capacities, seeds=f.seeds,
                        smp=actuals.smp[per.hours], k=actuals.k[per.hours], cloud=cloud, disagreement=disagreement)
    validation = {"blend_median_max_abs_diff_kwh": max_diff, "expert_W": {}}
    for expert in EXPERTS:
        q = hourly(getattr(prediction, expert + "_quantiles"))
        median = q[..., diagnostics.qindex(resolved.quantiles, .5)]
        expected_median = hourly(getattr(prediction, expert + "_median"))
        np.testing.assert_array_equal(median, expected_median)
        W = float((.1 * actuals.smp[per.hours, None] * actuals.k[per.hours, None] *
                   abs(median - truth) * complete[:, None]).sum())
        # expert별 W는 forecast 단계 run만 기록한다. 기록이 없으면 blend median 재현만으로 원 run과 대조한다.
        old_W = source.get("evaluation", {}).get("expert_W_krw", {}).get(partition, {}).get(expert)
        if old_W is not None and not np.isclose(W, old_W, rtol=2e-6, atol=.1):
            raise ValueError(f"{expert} W 재현 실패: {W} vs {old_W}")
        validation["expert_W"][expert] = {"recomputed": W, "original": old_W, "difference": None if old_W is None else W - old_W}
        np.savez_compressed(output_dir / f"{expert}_quantiles.npz", quantiles=q, raw=hourly(getattr(prediction, expert + "_raw")),
                            seed_quantiles=hourly(getattr(prediction, expert + "_seed_quantiles")),
                            seed_raw=hourly(getattr(prediction, expert + "_seed_raw")))
    shutil.copy2(original.directory / "composition.csv", output_dir / "composition.csv")
    checkpoints = {}
    for expert, filename in (("xgb", "model.ubj"), ("dl", "model.pt")):
        for seed in f.seeds:
            model_key = source["forecaster"][expert]["seeds"][str(seed)]["model_key"]
            path = data.store.path("models", model_key) / filename
            checkpoints[f"{expert}:{seed}"] = {"key": model_key, "path": str(path), "sha256": file_sha(path)}
    manifest = {"version": VERSION, "contract_version": diagnostics.CONTRACT_VERSION,
                "experts": {"xgb": {"label": "Tree-only"}, "dl": {"label": f"DL-only (site_aux_ratio={f.site_aux_ratio:g})"}},
                "key": key, "source_instance": str(instance_dir), "instance": per.name,
                "partition": partition, "cohort_size": len(index), "capacity_kw": capacities.tolist(), "seeds": f.seeds,
                "cutoff": per.cutoff, "first_ts": str(timestamps(per.hours)[0]), "last_ts": str(timestamps(per.hours)[-1]),
                "days": len(per.days), "hours": len(per.hours), "valid_hours": int(complete.sum()),
                "valid_daylight_hours": int((complete & common_daylight).sum()), "site_aux_ratio": f.site_aux_ratio,
                "model": source["definition"], "checkpoints": checkpoints, "training_performed": False,
                "axis": {"quantiles": ["time", "group", "alpha"], "seed_quantiles": ["seed", "time", "group", "alpha"]},
                "raw_stage": "kWh; seed mean before clipping, night mask and sorting; XGB night inputs are skipped",
                "regime_edges": regime_edges, "regime_edge_source": "cutoff 이전 학습 날짜의 두 그룹 주간 예보",
                "validation": validation, "environment": environment(), "source_sha": source_sha()}
    manifest["artifact_hashes"] = {name: file_sha(output_dir / name) for name in ("common.npz", "xgb_quantiles.npz", "dl_quantiles.npz", "composition.csv")}
    write_json(manifest_path, manifest)
    f.release_caches()
    return manifest



def run(spec, instance_dir, partition="stratified", output_dir=None):
    """checkpoint에서 공통 분위수 artifact를 추출한다. 평가·viewer 생성은 vppcontract CLI가 맡는다."""
    directory = Path(output_dir) if output_dir else spec.artifact_dir / "reports" / "probabilistic" / Path(instance_dir).name / partition
    manifest = extract(spec, instance_dir, partition, directory)
    return {"directory": str(directory.resolve()), "manifest": str((directory / "manifest.json").resolve()), "key": manifest["key"]}
