"""고정 H 배정 decoder 적응(transfer)과 H 이전 R0의 조건부 보정(recalibrate) 실험. 제출 경로가 없어 MK로 등록하지 않는다.

입력·전처리·encoder는 parent checkpoint를 유지한다. 학습/보정을 봉인한 뒤 H artifact의 정답을 평가기에 전달한다.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import copy
import hashlib
import time

import numpy as np
import pandas as pd
import torch

from vppcontract.identity import digest, file_sha
from vppcontract.outputs import read_output
from vppcontract.timeaxis import SEASON_IDS, delivery_dates, period, season_index, to_h
from .sampler import QuerySet, actual_view, joined_days, labels_of, daylight_of, describe
from .store import environment, read_json, write_json, source_sha, transfer_payload, recalibration_payload, array_hash
from .forecast import dl as dlm
from .forecast.forecaster import Forecaster, finalize

VERSION = 1
CALIBRATION_VERSION = 2


def state_hash(state, names=None):
    h = hashlib.sha256()
    for name in sorted(state if names is None else names):
        value = state[name].detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(value.tobytes())
    return h.hexdigest()


def hide_future(base, cutoff):
    """학습 객체에 전달할 generation을 cutoff에서 차단하며 원본 배열은 보존한다."""
    visible = np.arange(len(base.status))[:, None] <= cutoff
    return replace(base, energy=np.where(visible, np.asarray(base.energy), np.nan),
                   status=np.where(visible, np.asarray(base.status), 0))


def projected_queries(view, sites, sunup, days, groups):
    ds = np.repeat(np.asarray(days, dtype=np.int64), len(groups))
    members = np.tile(groups, (len(days), 1)) & (joined_days(view)[None, :] <= ds[:, None])
    keep = members.any(axis=1)
    ds, members = ds[keep], members[keep]
    mask, y = labels_of(members, ds, view)
    cap, count, hhi = describe(members, sites.capacity)
    q = QuerySet(ds, np.zeros(len(ds), dtype=np.int8), members, mask, daylight_of(members, ds, sunup), y, cap, count, hhi)
    return q.subset(np.flatnonzero(q.target_mask.any(axis=1)))


def concatenate_queries(parts):
    names = ("day", "kind", "members", "mask", "daylight", "y_kwh", "cap", "n_pv", "hhi")
    return QuerySet(**{name: np.concatenate([getattr(q, name) for q in parts]) for name in names})


def learning_guard(target, replay, calibration, cutoff, heldout_days):
    for name, q in (("target", target), ("replay", replay), ("calibration", calibration)):
        if not len(q) or np.any(q.day > cutoff // 24):
            raise ValueError(f"{name}: cutoff 이후 또는 빈 query입니다.")
        valid_hours = (q.day[:, None] * 24 + np.arange(24))[q.target_mask]
        if np.any(valid_hours > cutoff):
            raise ValueError(f"{name}: cutoff 이후 정답이 있습니다.")
        held = np.isin(q.day, heldout_days)
        if (name == "calibration" and not held.all()) or (name != "calibration" and held.any()):
            raise ValueError(f"{name}: R0 분리가 맞지 않습니다.")
    if np.intersect1d(np.r_[target.day, replay.day], calibration.day).size:
        raise ValueError("학습·보정 날짜가 겹칩니다.")
    return {name: {"queries": len(q), "days": len(np.unique(q.day)), "group_hours": int(q.target_mask.sum()),
                   "first_day": int(q.day.min()), "last_day": int(q.day.max()), "query_hash": q.hash,
                   "dates": delivery_dates(np.unique(q.day)).strftime("%Y-%m-%d").tolist()}
            for name, q in (("target", target), ("replay", replay), ("calibration", calibration))}


@torch.no_grad()
def pooled_inputs(f, queries):
    """고정된 encoder의 query별 pooled 표현. 입력 통계 fit과 정답 참조를 하지 않는다."""
    result = {s: {} for s in f.seeds}
    query = np.empty((len(queries), 3), np.float32)
    days = np.unique(queries.day)
    for start in range(0, len(days), 16):
        idx = np.flatnonzero(np.isin(queries.day, days[start:start + 16]))
        agg = f.aggregate(queries.day[idx], queries.members[idx])
        query[idx] = f.batcher.query(queries.cap[idx], queries.n_pv[idx], queries.hhi[idx]).cpu().numpy()
        for seed in f.seeds:
            mean, std = dlm.pooled_from_sums(*agg.dl[seed], agg.xgb["cap"])
            for name, values in (("mean", mean), ("std", std)):
                if name not in result[seed]: result[seed][name] = np.empty((len(queries), *values.shape[1:]), np.float32)
                result[seed][name][idx] = values.float().cpu().numpy()
        print(f"[transfer] encoder 고정 표현 {min(start + 16, len(days))}/{len(days)}일", flush=True)
    for arrays in result.values(): arrays["query"] = query
    return result


def mixture_normalizer(queries, n_target, n_replay):
    probability = np.r_[np.full(n_target, .5 / n_target), np.full(n_replay, .5 / n_replay)]
    cells = queries.target_mask.sum(axis=1)
    return float(np.sum(probability * cells * queries.cap) / np.sum(probability * cells))


def train_decoder(net, arrays, queries, n_target, n_replay, alphas, settings, seed, device):
    """encoder를 eval·freeze하고 decoder/head에만 1:1 query mixture pinball을 학습한다."""
    if len(queries) != n_target + n_replay:
        raise ValueError("학습 query 길이가 target/replay 길이와 다릅니다.")
    net.eval()
    prefixes = (net.decoder_name + ".", "head.")
    trainable, frozen = [], []
    for name, param in net.named_parameters():
        param.requires_grad_(name.startswith(prefixes))
        (trainable if param.requires_grad else frozen).append(name)
    state = net.state_dict()
    frozen = [name for name in state if not name.startswith(prefixes)]
    before_frozen, before_trainable = state_hash(state, frozen), state_hash(state, trainable)
    getattr(net, net.decoder_name).train()
    net.head.train()
    if any(module.training for name, module in net.named_children() if name not in (net.decoder_name, "head")):
        raise RuntimeError("고정 모듈이 training mode입니다.")
    tensors = {name: torch.as_tensor(value, device=device) for name, value in arrays.items()}
    z = torch.as_tensor(np.nan_to_num(queries.z), dtype=torch.float32, device=device)
    mask = torch.as_tensor(queries.target_mask, device=device)
    cap = torch.as_tensor(queries.cap, dtype=torch.float32, device=device)
    alpha = torch.as_tensor(alphas, dtype=torch.float32, device=device)
    normalizer = mixture_normalizer(queries, n_target, n_replay)
    seeds = dlm._seeds(seed)
    rng = np.random.default_rng(seeds["batch"])
    steps, half = int(settings["steps"]), int(settings["batch_queries"]) // 2
    schedule = np.concatenate([dlm.batch_schedule(n_target, steps, half, rng),
                               n_target + dlm.batch_schedule(n_replay, steps, half, rng)], axis=1)
    torch.manual_seed(seeds["dropout"])
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    history, started = [], time.perf_counter()
    for step, indices in enumerate(schedule):
        ix = torch.as_tensor(indices, device=device)
        q, _ = net.decode(*(tensors[name].index_select(0, ix) for name in ("mean", "std", "query")))
        loss = dlm.l0_loss(q, z[ix], mask[ix], cap[ix], alpha, normalizer)
        if not torch.isfinite(loss): raise RuntimeError("Transfer loss가 유한하지 않습니다.")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, settings["gradient_clip"])
        optimizer.step(); scheduler.step()
        if (step + 1) % 50 == 0 or step == 0:
            row = {"step": step + 1, "loss": float(loss.detach()), "lr": scheduler.get_last_lr()[0]}
            history.append(row)
            print(f"[transfer] seed={seed} {step + 1}/{steps} pinball={row['loss']:.6f}", flush=True)
    net.eval()
    after_frozen, after_trainable = state_hash(net.state_dict(), frozen), state_hash(net.state_dict(), trainable)
    if before_frozen != after_frozen: raise RuntimeError("고정 encoder 가중치가 변경됐습니다.")
    if before_trainable == after_trainable: raise RuntimeError("decoder·head 가중치가 갱신되지 않았습니다.")
    record = {"seed": seed, "steps": steps, "loss": "19-quantile pinball", "normalizer": normalizer,
              "trainable_parameters": trainable, "frozen_sha_before": before_frozen, "frozen_sha_after": after_frozen,
              "trainable_sha_before": before_trainable, "trainable_sha_after": after_trainable,
              "target_draws": steps * half, "replay_draws": steps * half, "schedule_sha": digest(schedule.tolist()),
              "history": history, "seconds": time.perf_counter() - started}
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, record, schedule


@torch.no_grad()
def decode_queries(net, arrays, device):
    net.eval(); parts = []
    for start in range(0, len(arrays["query"]), 128):
        q, _ = net.decode(*(torch.as_tensor(arrays[k][start:start + 128], device=device) for k in ("mean", "std", "query")))
        parts.append(q.float().cpu().numpy())
    return np.concatenate(parts)


def positive_scale(y, q, alphas, weights, lower, center=None):
    """주어진 중심에서 pinball을 최소화하는 양수 폭 배율의 weighted breakpoint 해."""
    mid = q[:, int(np.flatnonzero(np.isclose(alphas, .5))[0])][:, None]
    choose = alphas < .5 if lower else alphas > .5
    delta = q[:, choose] - mid
    tau = 1 - alphas[choose] if lower else alphas[choose]
    mass = weights[:, None] * abs(delta)
    ok = mass > 1e-12
    if not ok.any(): return 1.0
    center = mid if center is None else np.asarray(center)[:, None]
    ratios = np.broadcast_to(y[:, None] - center, delta.shape)[ok] / delta[ok]
    w = mass[ok]
    threshold = float(np.sum((mass * tau[None, :])[ok]))
    order = np.argsort(ratios, kind="stable")
    index = min(int(np.searchsorted(np.cumsum(w[order]), threshold, side="left")), len(order) - 1)
    return max(1e-6, float(ratios[order[index]]))


def fit_calibration(q, queries, alphas, shrinkage_days):
    """R0의 과거 query만으로 전체·계절별 하단/상단 배율을 fit한다. 각 날짜의 총 weight는 1이다."""
    mask = queries.target_mask
    eligible = mask.any(axis=1)
    counts = {int(d): int(np.sum((queries.day == d) & eligible)) for d in np.unique(queries.day[eligible])}
    query_weight = np.array([1 / counts[int(d)] if use else 0 for d, use in zip(queries.day, eligible)])
    weights = np.broadcast_to((query_weight / np.maximum(mask.sum(axis=1), 1))[:, None], mask.shape)[mask]
    pred = (q / queries.cap[:, None, None])[mask]
    y = queries.z[mask]
    days = np.broadcast_to(queries.day[:, None], mask.shape)[mask]
    season = np.array(SEASON_IDS)[season_index(days)]
    global_scale = [positive_scale(y, pred, alphas, weights, lo) for lo in (True, False)]
    record = {"global": global_scale, "seasons": {}, "shrinkage_days": float(shrinkage_days),
              "dates": delivery_dates(np.unique(days)).strftime("%Y-%m-%d").tolist(), "group_hours": int(mask.sum()),
              "query_hash": queries.hash, "median_fixed": True,
              "objective": "capacity-normalized pinball before final clipping; each date has total weight 1"}
    for s in SEASON_IDS:
        take = season == s
        n = len(np.unique(days[take])); rho = n / (n + shrinkage_days)
        fitted = [positive_scale(y[take], pred[take], alphas, weights[take], lo) for lo in (True, False)] if n else global_scale
        record["seasons"][s] = {"days": n, "rho": rho, "fitted": fitted,
                                  "scales": ((1 - rho) * np.array(global_scale) + rho * np.array(fitted)).tolist()}
    return record


def weighted_median(values, weights):
    """유한한 비음수 weight의 weighted absolute-error 최소점. 총 weight가 0이면 shift 0."""
    values, weights = np.asarray(values), np.asarray(weights)
    if values.shape != weights.shape or not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Weighted median의 값·가중치가 올바르지 않습니다.")
    keep = weights > 0
    if not keep.any(): return 0.0
    values, weights = values[keep], weights[keep]
    order = np.argsort(values, kind="stable")
    i = min(np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2, side="left"), len(order) - 1)
    return float(values[order[i]])


def fit_location_scale(q, truth, mask, capacity, days, alphas, settlement_weight, shrinkage_days, query_hash):
    """R0의 W-weighted 중앙값 shift를 정한 뒤, 이동한 중심에서 양쪽 pinball 배율을 fit한다."""
    mask = np.asarray(mask, dtype=bool)
    if (q.shape != (*mask.shape, len(alphas)) or truth.shape != mask.shape or settlement_weight.shape != mask.shape
            or capacity.shape != (len(mask),) or days.shape != (len(mask),)
            or not np.isfinite(capacity).all() or (capacity <= 0).any() or not mask.any()
            or not np.isfinite(q).all() or not np.isfinite(truth[mask]).all()
            or not np.isfinite(settlement_weight).all() or (settlement_weight < 0).any()):
        raise ValueError("Location-scale calibration 배열이 올바르지 않습니다.")
    j = int(np.flatnonzero(np.isclose(alphas, .5))[0])
    pred = (q / capacity[:, None, None])[mask]
    y = (truth / capacity[:, None])[mask]
    cell_days = np.broadcast_to(days[:, None], mask.shape)[mask]
    seasons = np.array(SEASON_IDS)[season_index(cell_days)]
    # 정규화 residual에 capacity를 곱한 weight를 쓰면 kWh 단위 W와 같은 목적이다.
    w = (settlement_weight * capacity[:, None])[mask]
    residual = y - pred[:, j]
    global_shift = weighted_median(residual, w)
    eligible = mask.any(axis=1)
    counts = {int(d): int(np.sum((days == d) & eligible)) for d in np.unique(days[eligible])}
    query_weight = np.array([1 / counts[int(d)] if use else 0 for d, use in zip(days, eligible)])
    width_weight = np.broadcast_to((query_weight / np.maximum(mask.sum(axis=1), 1))[:, None], mask.shape)[mask]
    global_scale = [positive_scale(y, pred, alphas, width_weight, lo, pred[:, j] + global_shift) for lo in (True, False)]
    fitted = {"kind": "seasonal-location-scale-W", "median_fixed": False, "global_shift": global_shift,
              "global": global_scale, "seasons": {}, "shrinkage_days": float(shrinkage_days),
              "dates": delivery_dates(np.unique(cell_days)).strftime("%Y-%m-%d").tolist(),
              "group_hours": int(mask.sum()), "query_hash": query_hash,
              "location_objective": "sum(0.1*SMP*k*capacity*abs(y/capacity-q50/capacity-shift)); before clipping",
              "width_objective": "capacity-normalized pinball at the shifted center; each date has total weight 1; before clipping"}
    for s in SEASON_IDS:
        take = seasons == s
        n = len(np.unique(cell_days[take])); rho = n / (n + shrinkage_days)
        has_weight = bool(w[take].sum() > 0)
        raw_shift = weighted_median(residual[take], w[take]) if has_weight else global_shift
        shift = (1 - rho) * global_shift + rho * raw_shift
        scales = [positive_scale(y[take], pred[take], alphas, width_weight[take], lo, pred[take, j] + shift)
                  for lo in (True, False)] if n else global_scale
        fitted["seasons"][s] = {"days": n, "rho": rho, "fitted_shift": raw_shift, "shift": shift,
                                "positive_settlement_weight": has_weight, "fitted": scales,
                                "scales": ((1 - rho) * np.array(global_scale) + rho * np.array(scales)).tolist()}
    return fitted


def apply_calibration(q, alphas, seasons, capacity, fitted, daylight=None):
    """계절별 중앙값 shift·양쪽 폭을 적용한 뒤 물리 범위와 야간 0을 적용한다."""
    q = np.asarray(q, dtype=np.float64)
    seasons = np.broadcast_to(seasons, q.shape[:-1]); capacity = np.broadcast_to(capacity, q.shape[:-1])
    scales = np.array([fitted["seasons"][s]["scales"] for s in seasons.ravel()]).reshape(*seasons.shape, 2)
    j = int(np.flatnonzero(np.isclose(alphas, .5))[0]); median = q[..., j:j + 1]
    multiplier = np.where(alphas < .5, scales[..., :1], scales[..., 1:])
    shifts = np.array([fitted["seasons"][s].get("shift", 0.0) for s in seasons.ravel()]).reshape(seasons.shape)
    out = np.clip(median + (shifts * capacity)[..., None] + multiplier * (q - median), 0, capacity[..., None])
    if fitted["median_fixed"]:
        out[..., j] = q[..., j]
    elif daylight is None:
        raise ValueError("중앙값 보정에는 daylight mask가 필요합니다.")
    if daylight is not None:
        out = np.where(np.broadcast_to(daylight, q.shape[:-1])[..., None], out, 0.)
    if (np.diff(out, axis=-1) < -1e-9).any(): raise RuntimeError("보정 후 분위수가 교차합니다.")
    return out


def run(spec, instance_dir, baseline_artifact, partition="stratified"):
    from .data import load_data
    from .evaluation import inference_spec
    from vppcontract.probabilistic import report

    instance_dir, baseline_artifact = Path(instance_dir).resolve(), Path(baseline_artifact).resolve()
    source = read_json(instance_dir / "instance.json"); saved = read_json(instance_dir / "artifacts.json")["inference"]
    if not source.get("cache_stored", True): raise ValueError("parent 실행은 모델 파일을 저장하지 않았습니다. 같은 설정에 --use-cache를 붙여 다시 실행하세요.")
    settings = spec.experiment("transfer")
    if source["instance"] != "H" or source["model"] != settings["parent_model"] or source["definition"]["dl"].get("site_aux_ratio") != 30:
        raise ValueError("H의 MK5(site_aux_ratio=30) parent를 지정하세요.")
    if source.get("smoke") or source.get("status") != "completed" or source["definition"].get("season"):
        raise ValueError("완료된 Global parent가 필요합니다.")
    if source["definition"]["seeds"] != [42, 43, 44]: raise ValueError("이번 실험의 seed는 42·43·44입니다.")
    baseline = read_json(baseline_artifact / "manifest.json")
    if Path(baseline["source_instance"]).resolve() != instance_dir or baseline["partition"] != partition:
        raise ValueError("Baseline artifact가 parent·배정과 다릅니다.")
    for name, sha in baseline["artifact_hashes"].items():
        if file_sha(baseline_artifact / name) != sha: raise ValueError(f"Baseline hash 불일치: {name}")
    partition_file = instance_dir / "outputs" / partition / "composition.csv"
    partition_sha = file_sha(partition_file)
    if partition_sha != file_sha(baseline_artifact / "composition.csv"): raise ValueError("배정이 다릅니다.")
    per = period("H")
    if source["cutoff_h"] != per.cutoff_h or pd.Timestamp(baseline["cutoff"]) != pd.Timestamp(per.cutoff):
        raise ValueError("Parent와 baseline은 H 시작 전 cutoff를 사용해야 합니다.")
    resolved = inference_spec(spec, source, saved)
    print("[transfer] parent 입력·전처리·checkpoint 로딩", flush=True)
    data = load_data(resolved)
    if (data.base.key, data.weather.key) != (saved["base_key"], saved["weather_key"]): raise ValueError("Parent 데이터 identity가 다릅니다.")
    # 이 시점부터 trainer가 보유한 generation/status에는 H 정답이 없다.
    data = replace(data, base=hide_future(data.base, per.cutoff_h))
    f = Forecaster(data, source["model"], per.cutoff_h, per.days)
    if f.device.type != "cuda": raise RuntimeError("실행에는 CUDA GPU가 필요합니다.")
    if f.feature_key() != source["forecaster"]["xgb"]["feature_key"]: raise ValueError("원 학습 query/입력 재현 실패")
    original = read_output(instance_dir / "outputs" / partition)
    f.load_cached(source["forecaster"], original.manifest["forecaster"], saved["power_curve"])
    if f.decoder is not None or f.provider_context is not None or f.day_scale is not None or f.plant_routing is not None or f.group_routing is not None:
        raise ValueError("MK5 BiGRU parent를 사용합니다.")
    sites = data.sites
    positions = pd.Index(sites.ids.astype(str)).get_indexer(original.site_ids.astype(str))
    groups = np.zeros((2, len(sites.ids)), bool)
    for g in range(2): groups[g, positions[original.group == g]] = True
    if not np.array_equal(groups.any(axis=0), f.cohort): raise ValueError("Parent cohort와 배정이 다릅니다.")
    first = to_h((pd.Timestamp(per.start) - pd.DateOffset(months=settings["history_months"])).isoformat()) // 24
    train_days = f.train_days[f.train_days >= first]; cal_days = f.heldout_days[f.heldout_days >= first]
    target = projected_queries(f.view, sites, f.sunup, train_days, groups)
    replay = f.global_q.subset(np.flatnonzero((f.global_q.day >= first) & f.global_q.target_mask.any(axis=1)))
    cal = projected_queries(f.view, sites, f.sunup, cal_days, groups)
    audit = learning_guard(target, replay, cal, per.cutoff_h, f.heldout_days)
    if np.isin(f.global_q.day, f.heldout_days).any(): raise RuntimeError("Parent 학습 query에 R0가 포함됐습니다.")
    parents = {str(s): source["forecaster"]["dl"]["seeds"][str(s)]["model_key"] for s in f.seeds}
    payload = transfer_payload({"run": str(instance_dir), "keys": parents, "prep": source["forecaster"]["dl"]["prep_key"]},
                               partition_sha, {k: v["query_hash"] for k, v in audit.items()}, settings, VERSION)
    key = digest(payload); directory = spec.artifact_dir / "experiments" / f"transfer-H-{key[:12]}"
    directory.mkdir(parents=True, exist_ok=True)
    record = {"experiment": "transfer", "parent_model": source["model"], "key": key, "payload": payload, "settings": settings,
              "cohort": int(f.cohort.sum()), "cutoff": per.cutoff, "seeds": f.seeds, "data_split": audit,
              "status": "running", "environment": environment(), "source_sha": source_sha(), "training": {},
              "no_H_training_labels": bool(np.isnan(data.base.energy[per.start_h:]).all()),
              "no_R0_training_queries": True, "started": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    write_json(directory / "run.json", record)
    print(f"[transfer] run={directory} target={len(target)} replay={len(replay)} calibration={len(cal)}", flush=True)
    combined = concatenate_queries([target, replay, cal]); n_train = len(target) + len(replay)
    pool = pooled_inputs(f, combined)
    learning = combined.subset(np.arange(n_train)); baseline_cal, transfer_cal, new_keys = {}, {}, {}
    for seed in f.seeds:
        net = f.nets[seed]
        parent_path = data.store.path("models", parents[str(seed)]) / "model.pt"
        parent_sha = file_sha(parent_path)
        calibration_inputs = {k: a[n_train:] for k, a in pool[seed].items()}
        baseline_cal[seed] = decode_queries(net, calibration_inputs, f.device)
        model_key = digest({"transfer": key, "seed": seed, "parent_sha": parent_sha})
        if data.store.exists("models", model_key):
            details = data.store.load("models", model_key, names=[])[1]["details"]
            net.load_state_dict(torch.load(data.store.path("models", model_key) / "model.pt", map_location=f.device)["state_dict"])
            print(f"[transfer] seed={seed} 완료된 적응 checkpoint 재사용", flush=True)
        else:
            state, details, schedule = train_decoder(net, {k: a[:n_train] for k, a in pool[seed].items()}, learning,
                                                     len(target), len(replay), resolved.quantiles, settings, seed, f.device)
            data.store.write("models", model_key, {"schedule": schedule},
                             {"experiment": "transfer", "parent": parents[str(seed)], "parent_sha": parent_sha,
                              "details": details, "environment": record["environment"], "payload": payload},
                             {"model.pt": lambda path, state=state: torch.save({"state_dict": state}, path)})
        if file_sha(parent_path) != parent_sha: raise RuntimeError("Parent checkpoint가 변경됐습니다.")
        transfer_cal[seed] = decode_queries(net, calibration_inputs, f.device)
        record["training"][str(seed)] = {"key": model_key, "parent_sha": parent_sha, **details}
        new_keys[str(seed)] = model_key
        write_json(directory / "run.json", record)
    del pool
    cal_models = {}
    for name, values in (("B", baseline_cal), ("T", transfer_cal)):
        q = finalize(np.mean([values[s] for s in f.seeds], axis=0), cal.cap, cal.daylight)
        cal_models[name] = fit_calibration(q, cal, resolved.quantiles, settings["shrinkage_days"])
        np.savez_compressed(directory / f"calibration_{name}.npz", quantiles=q, truth=cal.y_kwh, mask=cal.target_mask,
                            capacity=cal.cap, days=cal.day)
    calibration_path = directory / "calibration.json"
    write_json(calibration_path, {"version": VERSION, "cutoff": per.cutoff, "coefficients": cal_models})
    record.update(calibration_sha=file_sha(calibration_path), stage="learning_sealed", sealed_at=pd.Timestamp.now(tz="Asia/Seoul").isoformat())
    write_json(directory / "run.json", record)
    # 학습·보정이 끝난 뒤 H 입력에 대한 추론과 평가 산출물 조립을 시작한다.
    print("[transfer] 학습·보정 봉인 완료. H 추론 시작", flush=True)
    pred = f.predict(np.repeat(per.days, 2), np.tile(groups, (len(per.days), 1)), expert_quantiles=True)
    def hourly(a):
        if a.ndim == 3: return a.reshape(-1, 2, 24, a.shape[-1]).transpose(0, 2, 1, 3).reshape(-1, 2, a.shape[-1])
        return a.reshape(-1, 2, len(f.seeds), 24, a.shape[-1]).transpose(2, 0, 3, 1, 4).reshape(len(f.seeds), -1, 2, a.shape[-1])
    t_arrays = {"quantiles": hourly(pred.dl_quantiles), "raw": hourly(pred.dl_raw),
                "seed_quantiles": hourly(pred.dl_seed_quantiles), "seed_raw": hourly(pred.dl_seed_raw)}
    comparison = directory / "comparison"; comparison.mkdir(exist_ok=True)
    import shutil
    shutil.copy2(baseline_artifact / "common.npz", comparison / "common.npz")
    shutil.copy2(partition_file, comparison / "composition.csv")
    with np.load(comparison / "common.npz", allow_pickle=False) as z:
        c = dict(z)
    if not np.array_equal(c["hours"], per.hours) or c["seeds"].tolist() != f.seeds: raise ValueError("H artifact 시간축·seed가 다릅니다.")
    input_configs = {}
    for label, old in (("Tree", "xgb"), ("B", "dl")):
        with np.load(baseline_artifact / f"{old}_quantiles.npz", allow_pickle=False) as z: input_configs[label] = dict(z)
    input_configs["T"] = t_arrays
    seasons = np.array(SEASON_IDS)[season_index(c["hours"] // 24)][:, None]
    for name in ("B", "T"):
        value = input_configs[name]
        input_configs[name + "_C"] = {**value,
            "quantiles": apply_calibration(value["quantiles"], c["alphas"], seasons, c["capacity_kw"], cal_models[name]),
            "seed_quantiles": apply_calibration(value["seed_quantiles"], c["alphas"], seasons, c["capacity_kw"], cal_models[name])}
        j = int(np.flatnonzero(np.isclose(c["alphas"], .5))[0])
        np.testing.assert_array_equal(value["quantiles"][..., j], input_configs[name + "_C"]["quantiles"][..., j])
    labels = {"Tree": "Tree · 기존 XGB", "B": "B · 기존 DL", "B_C": "B+C · 기존 DL + 보정", "T": "T · Transfer DL", "T_C": "T+C · Transfer DL + 보정"}
    for name in labels: np.savez_compressed(comparison / f"{name}_quantiles.npz", **input_configs[name])
    manifest = copy.deepcopy(baseline)
    expert_meta = {k: {"label": v, "parent_model": source["model"], "transfer": k in ("T", "T_C"),
                       "calibration": k.endswith("_C"), "model": source["model"]}
                   for k, v in labels.items()}
    manifest.update(contract_version=1, key=key, model=source["definition"], experiment="transfer",
                    experts=expert_meta, training_performed=True,
                    environment=record["environment"], source_instance=str(instance_dir), transfer_run=str(directory),
                    calibration={"sha": record["calibration_sha"], "coefficients": cal_models, "median_fixed": True},
                    transfer_checkpoints=new_keys, source_sha=source_sha(),
                    raw_stage="underlying neural output before clipping/night/sorting; calibrated input_configs retain the same raw output")
    manifest["artifact_hashes"] = {name: file_sha(comparison / name) for name in ["common.npz", "composition.csv", *(k + "_quantiles.npz" for k in labels)]}
    write_json(comparison / "manifest.json", manifest)
    if file_sha(calibration_path) != record["calibration_sha"]: raise RuntimeError("H 평가 전 보정 계수가 바뀌었습니다.")
    result = report(comparison)
    summary = pd.read_csv(Path(result["directory"]) / "seasonal_summary.csv")
    for left, right in (("B", "B_C"), ("T", "T_C")):
        x = summary[(summary.expert == left) & (summary.seed == "ensemble")].sort_values(["group", "season"])
        y = summary[(summary.expert == right) & (summary.seed == "ensemble")].sort_values(["group", "season"])
        np.testing.assert_array_equal(x.W_krw.to_numpy(), y.W_krw.to_numpy())
    record.update(status="completed", stage="evaluated", report_directory=result["directory"],
                  median_W_invariant=True, finished=pd.Timestamp.now(tz="Asia/Seoul").isoformat())
    write_json(directory / "run.json", record)
    f.release_caches()
    return {"experiment": "transfer", "run_directory": str(directory), "comparison_artifact": str(comparison), "report_directory": result["directory"]}


def calibration_guard(arrays, audit, cutoff, training_days):
    """저장된 R0의 날짜·유효 정답 시각을 확인한 뒤 fitting을 허용한다."""
    days, mask = arrays["days"], arrays["mask"]
    if days.ndim != 1 or days.dtype.kind not in "iu" or mask.dtype.kind != "b" or mask.shape != (len(days), 24):
        raise ValueError("저장된 calibration 날짜·mask가 올바르지 않습니다.")
    hours = days[:, None] * 24 + np.arange(24)
    dates = delivery_dates(np.unique(days)).strftime("%Y-%m-%d").tolist()
    if (not mask.any() or days.min() < 0 or (days > cutoff // 24).any() or (hours[mask] > cutoff).any()):
        raise ValueError("Calibration 정답이 cutoff를 넘거나 비어 있습니다.")
    if (dates != audit["dates"] or len(days) != audit["queries"] or int(mask.sum()) != audit["group_hours"]
            or set(dates) & set(training_days)):
        raise ValueError("저장된 R0 분리·query 수가 원 학습 기록과 다릅니다.")
    return hours


def recalibrate(spec, transfer_run):
    """transfer 실험의 B·T checkpoint/예측을 재사용하고 R0 중앙값·구간 보정을 봉인한 뒤 H를 평가한다."""
    import shutil
    from .data import load_smp
    from vppcontract.probabilistic import load_arrays, report

    transfer_run = Path(transfer_run).resolve()
    source = read_json(transfer_run / "run.json")
    settings = spec.experiment("recalibration")
    per = period("H")
    if (source["status"] != "completed"
            or pd.Timestamp(source["cutoff"]) != pd.Timestamp(per.cutoff)
            or not source["no_H_training_labels"] or not source["no_R0_training_queries"]):
        raise ValueError("H 이전 학습·R0 분리를 확인한 완료된 transfer run이 필요합니다.")
    # MK 이름으로 실행한 기록은 transfer 설정을 definition.adaptation에 담는다.
    if (source.get("settings") or source.get("definition", {}).get("adaptation")) != spec.experiment("transfer"):
        raise ValueError("transfer 설정은 저장된 원 run과 같아야 합니다.")
    source_artifact = transfer_run / "comparison"
    old_manifest = read_json(source_artifact / "manifest.json")
    if (old_manifest["instance"] != "H" or old_manifest["key"] != source["key"]
            or old_manifest["cohort_size"] != source["cohort"] or old_manifest["seeds"] != source["seeds"]
            or old_manifest["artifact_hashes"]["composition.csv"] != source["payload"]["partition_sha"]):
        raise ValueError("원 run과 H 비교 artifact의 identity가 다릅니다.")
    if file_sha(transfer_run / "calibration.json") != source["calibration_sha"]:
        raise ValueError("원 run의 봉인된 calibration hash가 다릅니다.")
    paths = ["run.json", "calibration.json", "calibration_B.npz", "calibration_T.npz", "comparison/manifest.json"]
    source_hashes = {name: file_sha(transfer_run / name) for name in paths}
    audit = source["data_split"]["calibration"]
    training_dates = source["data_split"]["target"]["dates"] + source["data_split"]["replay"]["dates"]
    saved_cal, price_hashes = {}, {}
    smp, k, _ = load_smp(spec)
    for name in ("B", "T"):
        with np.load(transfer_run / f"calibration_{name}.npz", allow_pickle=False) as z:
            a = dict(z)
        hours = calibration_guard(a, audit, per.cutoff_h, training_dates)
        # cutoff 이후 시각은 mask 밖이어도 보정 입력 weight를 0으로 둔다.
        a["settlement_weight"] = np.where(a["mask"], .1 * smp[hours] * k[hours], 0.)
        price_hashes[name] = array_hash(a["settlement_weight"])
        saved_cal[name] = a
    for field in ("truth", "mask", "capacity", "days"):
        np.testing.assert_array_equal(saved_cal["B"][field], saved_cal["T"][field])
    payload = recalibration_payload(source["key"], source_hashes, digest(price_hashes), settings, CALIBRATION_VERSION)
    key = digest(payload)
    directory = spec.artifact_dir / "experiments" / f"recalibration-H-{key[:12]}"
    directory.mkdir(parents=True, exist_ok=True)
    record = {"experiment": "recalibration", "parent_model": source["parent_model"], "key": key, "payload": payload,
              "settings": settings, "source_transfer_run": str(transfer_run), "cohort": source["cohort"],
              "cutoff": per.cutoff, "seeds": source["seeds"], "data_split": source["data_split"],
              "training": source["training"], "checkpoint_reused": True, "training_performed": False,
              "no_H_training_labels": True, "no_R0_training_queries": True, "environment": environment(),
              "source_sha": source_sha(), "status": "running", "stage": "calibrating",
              "started": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    write_json(directory / "run.json", record)
    coefficients = {}
    alphas = spec.quantiles
    for name, a in saved_cal.items():
        coefficients[name] = fit_location_scale(a["quantiles"], a["truth"], a["mask"], a["capacity"], a["days"],
            alphas, a["settlement_weight"], settings["shrinkage_days"], audit["query_hash"])
        np.savez_compressed(directory / f"calibration_{name}.npz", **a)
        print(f"[recalibrate] {name} R0 {len(coefficients[name]['dates'])}일 중앙값·구간 보정 완료", flush=True)
    cal_path = directory / "calibration.json"
    write_json(cal_path, {"version": CALIBRATION_VERSION, "cutoff": per.cutoff, "alphas": alphas, "coefficients": coefficients})
    record.update(calibration_sha=file_sha(cal_path), stage="learning_sealed", sealed_at=pd.Timestamp.now(tz="Asia/Seoul").isoformat())
    write_json(directory / "run.json", record)
    print("[recalibrate] R0 계수 봉인 완료. 저장된 H artifact 평가 시작", flush=True)
    # H 정답은 이 지점부터 공통 평가기로 전달한다.
    c, input_configs = load_arrays(source_artifact)
    if not np.array_equal(c["hours"], per.hours) or not np.array_equal(c["alphas"], alphas):
        raise ValueError("H 시간축 또는 분위수가 원 설정과 다릅니다.")
    seasons = np.array(SEASON_IDS)[season_index(c["hours"] // 24)][:, None]
    median_change = {}
    j = int(np.flatnonzero(np.isclose(alphas, .5))[0])
    for name in ("B", "T"):
        base = input_configs[name]
        input_configs[name + "_C"] = {**base, **{field: apply_calibration(base[field], alphas, seasons, c["capacity_kw"],
            coefficients[name], c["daylight"]) for field in ("quantiles", "seed_quantiles")}}
        delta = input_configs[name + "_C"]["quantiles"][..., j] - base["quantiles"][..., j]
        median_change[name] = {"changed_group_hours": int(np.count_nonzero(delta)), "max_abs_kwh": float(abs(delta).max())}
    comparison = directory / "comparison"; comparison.mkdir(exist_ok=True)
    for filename in ("common.npz", "composition.csv", "Tree_quantiles.npz", "B_quantiles.npz", "T_quantiles.npz"):
        shutil.copy2(source_artifact / filename, comparison / filename)
    for name in ("B_C", "T_C"):
        np.savez_compressed(comparison / f"{name}_quantiles.npz", **input_configs[name])
    manifest = copy.deepcopy(old_manifest)
    manifest.pop("derived_model", None)
    manifest.update(key=key, experiment="recalibration", training_performed=False,
                    calibration_run=str(directory), environment=record["environment"], source_sha=source_sha(),
                    calibration={"sha": record["calibration_sha"], "coefficients": coefficients, "median_fixed": False})
    manifest["source_validation"] = manifest.pop("validation", {})
    for name in ("B_C", "T_C"):
        manifest["experts"][name].update(calibration_kind=settings["kind"],
            label=("B+C · 기존 DL + 중앙값·구간 보정" if name == "B_C" else "T+C · Transfer DL + 중앙값·구간 보정"))
    for meta in manifest["experts"].values():
        meta["model"] = source["parent_model"]
    manifest["artifact_hashes"] = {name: file_sha(comparison / name) for name in old_manifest["artifact_hashes"]}
    write_json(comparison / "manifest.json", manifest)
    result = report(comparison)
    if file_sha(cal_path) != record["calibration_sha"]:
        raise RuntimeError("H 평가 중 보정 계수가 변경됐습니다.")
    if any(file_sha(transfer_run / name) != sha for name, sha in source_hashes.items()):
        raise RuntimeError("재사용한 원 run의 파일이 실행 중 변경됐습니다.")
    for filename in ("common.npz", "composition.csv", "Tree_quantiles.npz", "B_quantiles.npz", "T_quantiles.npz"):
        if manifest["artifact_hashes"][filename] != old_manifest["artifact_hashes"][filename]:
            raise RuntimeError("재사용한 비교 기준 또는 고정 H 조건이 변경됐습니다.")
    record.update(status="completed", stage="evaluated", report_directory=result["directory"],
                  median_fixed=False, median_change=median_change, finished=pd.Timestamp.now(tz="Asia/Seoul").isoformat())
    write_json(directory / "run.json", record)
    return {"experiment": "recalibration", "run_directory": str(directory), "comparison_artifact": str(comparison), "report_directory": result["directory"]}
