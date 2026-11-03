"""실험 실행: cutoff → query → seed별 XGB·DL 학습·예측 → 후처리 → 평가 → 산출물·manifest.

입력×손실 grid의 실행과 판정도 여기에 둔다. grid 분석은 저장된 run 보고서만 읽는다.
참조: docs/solarvpp-reference.md#initial-grid, docs/solarvpp-reference.md#noise-floor.

계절 학습(run의 season)은 Global 학습 query를 먼저 만든 뒤 대상일 계절로 고르거나 가중한다. 입력 전처리 state와 observed target
표준화는 모든 전략이 Global 것을 쓴다. ft는 같은 seed의 Global XGB booster·DL weights(cache)에서 이어 학습하고, only는 계절 query로
처음부터 학습하며(XGB 행렬·power curve도 계절 행), weighted는 전체 query에 계절 대상일 가중을 준다.
참조: solarvpp2 산출물 reports/seasonal-training-2026-10-21/plan.md.
"""
from __future__ import annotations

import contextlib
import os
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from vppcontract.timeaxis import season_mask

from .bundling.partition import apply_partition
from .config import (
    GRID_LOSSES,
    INPUTS,
    MODEL_FIELDS,
    SEASON_DEFAULTS,
    SEASON_OPTIONS,
    LossConfig,
    Period,
    Spec,
    InputConfig,
    curtailment_settings,
    season_settings,
    settlement_slot_weight,
    input_key,
)
from .data import GROUPS, PROVIDERS, Base, Sites, Weather, load_base, load_sites, load_weather, timestamps
from .store import calculation_identity, find_run as find_calculated_run, identity_key, index_runs
from .forecast.features import InputSet, PowerCurve, feature_manifest, resolve_inputs, xgb_matrix
from .forecast.models import blend, crossing_stats, decision, finalize, seed_mean, to_panel
from .forecast.queries import (
    COHORT_POLICY,
    QUERY_POLICY,
    EvalPanel,
    QuerySet,
    actual_view,
    check_evidence,
    eval_panel,
    freeze_cohort,
    period_queries,
    training_queries,
)
from .scoring import compare as compare_reports
from .scoring import score
from .store import (Store, close_temporary_cache, digest, environment, experiment_cache_enabled, file_sha, open_temporary_cache, package_versions,
                    read_json, source_sha, versions, write_json)
from .submission import gate, load_or_check, write_submission
from .validation import selection_context

LEVELS = {
    "smoke": (None, "screening_seeds"),
    "diagnostic": (None, "screening_seeds"),
    "noise_floor": (None, "noise_floor"),
    "screening": (None, "screening_seeds"),
    "confirmation": (None, "confirmation_seeds"),
    "retrospective": (None, "confirmation_seeds"),
    "bundling": (None, "confirmation_seeds"),
    "submission": (("test",), "confirmation_seeds"),
}


EXPERTS = ("xgb", "dl")


@dataclass
class Data:
    spec: Spec
    sites: Sites
    store: Store
    base: Base
    weather: Weather
    partition: dict | None = None


def load_data(spec: Spec, partition: str | Path | None = None) -> Data:
    """partition을 주면 그룹 배정만 바꾼 Sites를 쓴다. base·weather cache는 배정과 무관하다."""
    sites = load_sites(spec)
    record = None
    if partition is not None:
        sites, record = apply_partition(sites, partition)
    store = Store(spec.artifact_dir / "cache")
    return Data(spec, sites, store, load_base(spec, sites, store), load_weather(spec, sites, store), record)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def cell_id(input_name: str, loss: LossConfig) -> str:
    """입력×손실 칸 이름. 보조손실 칸은 세기를 붙인다(예: forecast-physics__ramp-aux0.1)."""
    return f"{input_name}__{loss.cell_suffix}"


def seeds_label(seeds: list[int]) -> str:
    if len(seeds) > 3 and seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}to{seeds[-1]}"
    return "-".join(map(str, seeds))


def level_seeds(spec: Spec, level: str) -> list[int]:
    key = LEVELS[level][1]
    return list(spec.protocol["noise_floor"]["seeds"]) if key == "noise_floor" else list(spec.hyperparameters[key])


@contextlib.contextmanager
def cuda_lock(path: Path):
    """protocol의 parallel_cuda_training_jobs=1: CUDA 학습은 한 번에 하나만 실행한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"다른 CUDA 학습이 실행 중입니다. 실행 중인 작업이 없으면 {path}를 지우세요.") from None
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def with_labels(queries: QuerySet, panel: EvalPanel) -> QuerySet:
    """평가 query에 panel 정답을 붙인 사본(진단용 손실 계산에만 쓴다)."""
    out = queries.subset(np.arange(len(queries)))
    start_day = panel.hours[0] // 24
    for qi in range(len(out)):
        rows = slice((out.day[qi] - start_day) * 24, (out.day[qi] - start_day + 1) * 24)
        out.mask[qi] = panel.m[rows, out.group[qi]]
        out.y_kwh[qi] = panel.y[rows, out.group[qi]]
    return out


def settings_hash(spec: Spec) -> str:
    return digest({"hyperparameters": spec.hyperparameters, "quantiles": spec.quantiles.tolist(), "blending": spec.protocol["blending"],
                   "decisions": spec.recorded_decisions})


def h_guard(spec: Spec, period: Period, setup: ForecastSetup, selection: str | None, level: str = "retrospective", experts=EXPERTS) -> dict:
    """선택적인 frozen comparison 검증과 실행 맥락 기록."""
    return selection_context(period, setup_signature(setup), settings_hash(spec), selection, level, experts)


def _state_arrays(states: dict) -> dict:
    return {f"{name}__{k}": np.asarray(v) for name, state in states.items() for k, v in state.items()}


def _state_dict(arrays: dict) -> dict:
    states = {}
    for key, value in arrays.items():
        name, _, k = key.partition("__")
        states.setdefault(name, {})[k] = np.asarray(value)
    return states


def feature_cache_key(data: Data, inputs: InputSet, keys: dict) -> str:
    """XGB feature 행렬 cache key."""
    return digest({"queries": keys["queries"], "eval": keys["eval"], "weather": data.weather.key, "base": data.base.key,
                   "input": input_key(inputs.input_config), "excluded": inputs.excluded, "decisions": keys["decisions"],
                   "versions": versions("queries", "features")})


def prep_cache_key(data: Data, inputs: InputSet, keys: dict, epsilon: float) -> str:
    """DL 입력 tensor·전처리 cache key. physics proxy·시간 표현·scaler는 forecast.features를 같이 쓴다."""
    return digest({"queries": keys["queries"], "eval": keys["eval"], "weather": data.weather.key, "base": data.base.key,
                   "input": input_key(inputs.input_config), "excluded": inputs.excluded, "epsilon": epsilon, "decisions": keys["decisions"],
                   "versions": versions("queries", "features", "dl_prep")})


def xgb_model_key(feature_key: str, spec: Spec, rounds: int, seed: int, slot_weight=None, season: dict | None = None) -> str:
    """기존 XGB model cache의 계산 설정을 한곳에서 구성한다. 계절 학습 설정(season_part)은 Global이 아닐 때만 넣는다."""
    identity = {"features": feature_key, "hyperparameters": {k: v for k, v in spec.hyperparameters.items() if k.startswith("xgb")},
                "rounds": rounds, "seed": seed, "quantiles": spec.quantiles.tolist(),
                "packages": package_versions(("numpy", "xgboost")), "versions": versions("xgb")}
    if slot_weight is not None:
        identity["slot_weight"] = [float(v) for v in slot_weight]
    if season is not None:
        identity["season"] = season
    return digest(identity)


def dl_model_key(prep_key: str, spec: Spec, steps: int, seed: int, loss: LossConfig, season: dict | None = None) -> str:
    """기존 DL model cache의 계산 설정을 한곳에서 구성한다. 계절 학습 설정(season_part)은 Global이 아닐 때만 넣는다."""
    identity = {"prep": prep_key, "hyperparameters": {k: v for k, v in spec.hyperparameters.items() if k.startswith(("dl_", "zca"))},
                "steps": steps, "seed": seed, "quantiles": spec.quantiles.tolist(), "loss": loss.__dict__,
                "packages": package_versions(("numpy", "torch")), "versions": versions("dl")}
    if season is not None:
        identity["season"] = season
    return digest(identity)


def in_season(days: np.ndarray, season: dict) -> np.ndarray:
    """계절 학습에 쓰는 대상일(학습 창): 대상 계절에 margin_months만큼 앞뒤 월을 더한다. 평가 계절과는 별개이다."""
    return season_mask(days, season["name"], season.get("margin_months", SEASON_OPTIONS["margin_months"]))


def expert_season(season_run: dict | None, kind: str) -> dict | None:
    """expert(kind: xgb·dl)에 적용할 계절 학습. 적용하지 않는 expert는 None(Global 모델·key)이다."""
    if season_run is None or kind not in season_run["settings"].get("experts", SEASON_OPTIONS["experts"]):
        return None
    return season_run


def season_budget(season: dict, queries: QuerySet, steps: int, warmup_steps: int) -> dict:
    """계절 학습의 DL step 예산. N_G·N_s는 daylight target이 있는 Global·계절 학습 query 수, T_s = max(1, round(steps·N_s/N_G))이다.
    only의 warm-up은 round(warm-up·T_s/steps)로 줄이되 calibration 뒤 학습 step이 남도록 T_s − 1 이하로 둔다. N_s = 0이면 학습할 수 없다.
    N_s는 학습 창(in_season) 기준이고, steps = full이면 T_s와 warm-up을 Global 예산으로 둔다."""
    used = queries.target_mask.any(axis=1)
    n_g, n_s = int(used.sum()), int((used & in_season(queries.day, season)).sum())
    if n_s == 0:
        raise ValueError(f"{season['name']} 계절의 학습 query가 없어 학습할 수 없습니다(N_s = 0).")
    if season.get("steps", SEASON_OPTIONS["steps"]) == "full":
        return {"N_G": n_g, "N_s": n_s, "T_s": steps, "warmup": min(warmup_steps, steps - 1)}
    t_s = max(1, int(round(steps * n_s / n_g)))
    return {"N_G": n_g, "N_s": n_s, "T_s": t_s, "warmup": min(int(round(warmup_steps * t_s / steps)), t_s - 1)}


def season_record(season: dict, queries: QuerySet) -> dict:
    """DL 학습 query의 계절 표본 가중치 비중(가중 전·후): query 칸(유효 칸 × 용량), query(용량), unique 대상일.
    표본 가중치의 비중이며 손실·gradient 기여율이 아니다."""
    used = queries.subset(np.flatnonzero(queries.target_mask.any(axis=1)))
    weight = season["weight"] if season["strategy"] == "weighted" else 1.0
    in_q = in_season(used.day, season)
    days = np.unique(used.day)
    in_d = in_season(days, season)

    def share(values, mask):
        a = np.where(mask, weight, 1.0)
        return {"unweighted": float(values[mask].sum() / values.sum()), "weighted": float((values * a)[mask].sum() / (values * a).sum())}
    return {"name": season["name"], "strategy": season["strategy"],
            "sample_weight_share": {"query_cell": share(used.target_mask.sum(axis=1) * used.cap, in_q), "query": share(used.cap, in_q),
                                    "unique_day": share(np.ones(len(days)), in_d)}}


def season_keys(keys: dict, season_run: dict | None) -> dict:
    """XGB 행렬 key의 입력. only는 계절 query로 행렬·power curve를 만들므로 계절 query를 넣고, 다른 전략은 Global 그대로 쓴다."""
    if season_run is None or season_run["settings"]["strategy"] != "only":
        return keys
    return {**keys, "queries": digest({"global": keys["queries"], "season": season_run["settings"]["name"],
                                       "hash": season_run["queries"].hash})}


def season_part(season_run: dict | None, kind: str, parent: str | None = None) -> dict | None:
    """model key에 넣을 계절 학습 설정(kind: xgb·dl). Global이면 None이다. ft는 같은 seed의 Global 모델 key를 parent로 넣는다."""
    if season_run is None:
        return None
    season = season_run["settings"]
    part = {"name": season["name"], "strategy": season["strategy"]}
    if season.get("margin_months"):
        part["margin_months"] = int(season["margin_months"])
    if kind == "dl" and season.get("steps", "scaled") != "scaled" and season["strategy"] != "weighted":
        part["steps_budget"] = season["steps"]
    if season["strategy"] == "weighted":
        part["weight"] = season["weight"]
    elif kind == "xgb" and season["strategy"] == "ft":
        part.update(rounds=season["xgb_ft_rounds"], learning_rate=season["xgb_ft_learning_rate"], parent=parent)
    elif kind == "dl":
        part.update(steps=season_run["budget"]["T_s"], queries=season_run["queries"].hash)
        if season["strategy"] == "ft":
            part.update(learning_rate=season["ft_lr"], parent=parent)
        else:
            part["warmup"] = season_run["budget"]["warmup"]
    return part


def xgb_key(feature_key: str, spec: Spec, rounds: int, seed: int, slot_weight, season_run: dict | None) -> str:
    """seed의 XGB model key. 계절 학습이면 계절 설정을 넣는다(ft의 parent는 같은 행렬·seed의 Global key)."""
    parent = xgb_model_key(feature_key, spec, rounds, seed, slot_weight)
    return parent if season_run is None else xgb_model_key(feature_key, spec, rounds, seed, slot_weight, season_part(season_run, "xgb", parent))


def dl_key(prep_key: str, spec: Spec, steps: int, seed: int, loss: LossConfig, season_run: dict | None) -> str:
    """seed의 DL model key. 계절 학습이면 계절 설정을 넣는다(ft의 parent는 같은 전처리·seed의 Global key)."""
    parent = dl_model_key(prep_key, spec, steps, seed, loss)
    return parent if season_run is None else dl_model_key(prep_key, spec, steps, seed, loss, season_part(season_run, "dl", parent))


def planned_models(data, inputs, xgb_inputs, keys, setup, experts, seeds, rounds, steps, season_run=None) -> dict:
    """학습 없이 이번 run이 사용할 seed별 model cache key를 계산한다."""
    details = {}
    if "xgb" in experts:
        xgb_run = expert_season(season_run, "xgb")
        feature_key = feature_cache_key(data, xgb_inputs, season_keys(keys, xgb_run))
        slot_weight = settlement_slot_weight() if setup.xgb_time_weight == "settlement" else None
        details["xgb"] = {"details": {str(s): {"model_key": xgb_key(feature_key, data.spec, rounds, s, slot_weight, xgb_run)}
                                     for s in seeds}}
    if "dl" in experts:
        prep_key = prep_cache_key(data, inputs, keys, data.spec.hyperparameters["zca_epsilon"])
        details["dl"] = {"details": {str(s): {"model_key": dl_key(prep_key, data.spec, steps, s, setup.loss, expert_season(season_run, "dl"))}
                                    for s in seeds}}
    return details


def run_xgb(data: Data, inputs: InputSet, train_q: QuerySet, eval_q: QuerySet, seeds: list[int], rounds: int, keys: dict,
            diagnostic: bool, slot_weight: np.ndarray | None = None, season_run: dict | None = None) -> dict:
    """train_q·keys는 Global이다. season_run이 only이면 계절 query로 행렬·power curve를 만들고, weighted는 계절 행 가중을 곱하며,
    ft는 같은 seed의 Global booster에 계절 행으로 tree를 더한다."""
    import xgboost as xgb

    from .forecast.models import fit_xgb, gain_share, predict_xgb
    spec, store, sites = data.spec, data.store, data.sites
    solar, sunup = np.asarray(data.base.solar), np.asarray(data.base.sunup)
    season = None if season_run is None else season_run["settings"]
    if season is not None and season["strategy"] == "only":
        train_q = season_run["queries"]
    feature_key = feature_cache_key(data, inputs, season_keys(keys, season_run))
    if not store.exists("features", feature_key):
        started = time.perf_counter()
        log("XGB feature 행렬 생성")
        x_train, i_train, names, curve_train = xgb_matrix(train_q, inputs, sites, solar, sunup, "target")
        x_pred, i_pred, _, curve_pred = xgb_matrix(eval_q, inputs, sites, solar, sunup, "all")
        z_train = train_q.z[i_train[:, 0], i_train[:, 1]]
        cap_train = train_q.cap[i_train[:, 0]]
        curve = PowerCurve().fit(curve_train, z_train, cap_train)
        pc_train, pc_pred = curve.predict(curve_train), curve.predict(curve_pred)
        if inputs.input_config.solar_geometry:
            x_train = np.column_stack([x_train, pc_train]).astype(np.float32)
            x_pred = np.column_stack([x_pred, pc_pred]).astype(np.float32)
            names = names + ["power_curve"]
        store.write("features", feature_key, {"x_train": x_train, "i_train": i_train, "z_train": z_train, "cap_train": cap_train,
                                              "x_pred": x_pred, "i_pred": i_pred, "pc_train": pc_train, "pc_pred": pc_pred},
                    {"names": names, "power_curve": curve.state(), "manifest": feature_manifest(inputs, names, spec.config["deviations"]),
                     "seconds": time.perf_counter() - started})
    arrays, meta = store.load("features", feature_key, mmap=False)
    names = meta["names"]
    x_train, x_pred, i_pred = arrays["x_train"], arrays["x_pred"], arrays["i_pred"]
    residual = arrays["z_train"] - arrays["pc_train"]
    # 정산 시간 가중은 용량 가중에 곱해 sample weight로 넣는다. fit_xgb가 평균 1로 다시 맞춘다.
    weight_basis = arrays["cap_train"] if slot_weight is None else arrays["cap_train"] * slot_weight[arrays["i_train"][:, 1]]
    out = {"feature_key": feature_key, "manifest": meta["manifest"], "power_curve": meta["power_curve"], "z": {}, "details": {},
           "rows": {"train": int(len(x_train)), "predict": int(len(x_pred)), "features": len(names)}}
    season_rows = None
    if season is not None:
        season_rows = in_season(train_q.day[arrays["i_train"][:, 0]], season)
        base_weight = weight_basis
        if season["strategy"] == "weighted":
            weight_basis = weight_basis * np.where(season_rows, season["weight"], 1.0)
        out["season"] = {"name": season["name"], "strategy": season["strategy"], "rows": int(len(season_rows)), "season_rows": int(season_rows.sum()),
                         "season_weight_share": {"unweighted": float(base_weight[season_rows].sum() / base_weight.sum()),
                                                 "weighted": float(weight_basis[season_rows].sum() / weight_basis.sum())}}
    for seed in seeds:
        model_key = xgb_key(feature_key, spec, rounds, seed, slot_weight, season_run)
        if not store.exists("models", model_key):
            if season is not None and season["strategy"] == "ft":
                parent_key = xgb_model_key(feature_key, spec, rounds, seed, slot_weight)
                if not store.exists("models", parent_key):
                    raise ValueError(f"Seasonal-FT의 Global XGB 모델(seed {seed})이 cache에 없습니다. Global을 먼저 실행하세요(설정을 바꾼 Global은 --use-cache로).")
                parent = xgb.Booster()
                parent.load_model(str(store.path("models", parent_key) / "model.ubj"))
                log(f"XGB 계절 FT seed={seed} rows={int(season_rows.sum())} rounds=+{season['xgb_ft_rounds']}")
                booster, details = fit_xgb(x_train[season_rows], residual[season_rows], weight_basis[season_rows], names, spec.hyperparameters,
                                           spec.quantiles, seed, season["xgb_ft_rounds"], init_booster=parent,
                                           learning_rate=season["xgb_ft_learning_rate"])
                details["parent_model_key"] = parent_key
            else:
                log(f"XGB 학습 seed={seed} rows={len(x_train)} features={len(names)} rounds={rounds}")
                booster, details = fit_xgb(x_train, residual, weight_basis, names, spec.hyperparameters, spec.quantiles, seed, rounds)
            if season is not None:
                details["season"] = {**season_part(season_run, "xgb", details.get("parent_model_key")), **out["season"]}
            started = time.perf_counter()
            raw = predict_xgb(booster, x_pred, names, spec.hyperparameters["xgb_threads"]) + arrays["pc_pred"][:, None]
            details["predict_seconds"] = time.perf_counter() - started
            z = np.zeros((len(eval_q), 24, len(spec.quantiles)), dtype=np.float32)
            z[i_pred[:, 0], i_pred[:, 1]] = raw
            store.write("models", model_key, {"z": z}, {"model": "xgb", "seed": seed, "details": details},
                        {"model.ubj": lambda path, b=booster: b.save_model(str(path))})
        model_arrays, model_meta = store.load("models", model_key, mmap=False)
        out["z"][seed] = model_arrays["z"]
        out["details"][seed] = {**model_meta["details"], "model_key": model_key, "model_bytes": (store.path("models", model_key) / "model.ubj").stat().st_size}
    if diagnostic:
        seed = seeds[0]
        booster = xgb.Booster()
        booster.load_model(str(store.path("models", out["details"][seed]["model_key"]) / "model.ubj"))
        flipped = x_pred.copy()
        column = names.index("is_G2")
        flipped[:, column] = 1.0 - flipped[:, column]
        median = int(np.flatnonzero(np.isclose(spec.quantiles, 0.5))[0])
        base_z = predict_xgb(booster, x_pred, names, spec.hyperparameters["xgb_threads"])[:, median]
        flip_z = predict_xgb(booster, flipped, names, spec.hyperparameters["xgb_threads"])[:, median]
        day = eval_q.daylight[i_pred[:, 0], i_pred[:, 1]]
        delta = np.abs(flip_z - base_z)[day]
        cap = eval_q.cap[i_pred[:, 0]][day]
        out["diagnostics"] = {"is_G2_flip": {"seed": seed, "mean_abs_delta_z50": float(delta.mean()),
                                             "mean_abs_delta_q50_kwh": float((delta * cap).mean()), "gain_share": gain_share(booster, "is_G2")}}
    return out


def run_dl(data: Data, inputs: InputSet, train_q: QuerySet, eval_q: QuerySet, labelled_eval: QuerySet | None, cutoff_h: int,
           seeds: list[int], steps: int, keys: dict, loss: LossConfig, season_run: dict | None = None) -> dict:
    """train_q·keys는 Global이다. 입력 전처리 state·observed target 표준화는 Global 학습 대상일로 정하고, season_run이 있으면
    그 전략의 query·step·가중·시작 weights로 학습한다."""
    import torch

    from .forecast.train import Batcher, build_net, build_prep, configure_determinism, fit_states, oot_monitor, predict_dl, train_dl
    spec, store, sites = data.spec, data.store, data.sites
    if not torch.cuda.is_available():
        raise RuntimeError("DL은 CUDA GPU에서 실행합니다. CPU fallback은 쓰지 않습니다.")
    configure_determinism(spec.config["runtime"]["torch_threads"])
    epsilon = spec.hyperparameters["zca_epsilon"]
    prep_key = prep_cache_key(data, inputs, keys, epsilon)
    if not store.exists("prep", prep_key):
        log("DL 전처리 fit(대체·표준화·ZCA)")
        states = fit_states(inputs, sites, cutoff_h, epsilon, train_q, np.asarray(data.base.solar))
        store.write("prep", prep_key, _state_arrays(states), {"fit_end_day": int(cutoff_h // 24)})
    states = _state_dict(store.load("prep", prep_key, mmap=False)[0])
    started = time.perf_counter()
    observed = None
    if inputs.input_config.day_state:
        observed = {"values": {p: np.asarray(data.weather.values[f"observed_{p}"]) for p in ("wa", "wb")},
                    "names": {p: data.weather.channel_names(p, "observed") for p in ("wa", "wb")}, "cutoff_h": int(cutoff_h)}
    site = None
    if inputs.input_config.site_aux_ratio > 0:
        view = actual_view(np.asarray(data.base.energy), np.asarray(data.base.status), cutoff_h)
        site = {"energy": view.energy, "sunup": np.asarray(data.base.sunup)}
    prep = build_prep(inputs, sites, np.asarray(data.base.solar), np.concatenate([train_q.day, eval_q.day]), states, observed, site)
    device = torch.device("cuda")
    batcher = Batcher(prep, sites, device, spec.config["runtime"]["dl_preload_max_gib"])
    out = {"prep_key": prep_key, "layout": prep.layout, "build_seconds": time.perf_counter() - started, "z": {}, "details": {}}
    season = None if season_run is None else season_run["settings"]
    if season is not None:
        out["season"] = {**season_run["record"], "budget": season_run["budget"]}
    for seed in seeds:
        model_key = dl_key(prep_key, spec, steps, seed, loss, season_run)
        if not store.exists("models", model_key):
            queries, n_steps, kwargs, parent_key = train_q, steps, {}, None
            if season is not None:
                queries, n_steps, kwargs = season_dl_args(season_run, prep, train_q, steps)
                if season["strategy"] == "ft":
                    parent_key = dl_model_key(prep_key, spec, steps, seed, loss)
                    kwargs.update(ft_parent(store, parent_key, seed))
            log(f"DL 학습 {inputs.input_config.name}x{loss.cell_suffix} seed={seed} steps={n_steps} queries={len(queries)}"
                + (f" ({season['name']}·{season['strategy']})" if season is not None else ""))
            monitor = oot_monitor(batcher, labelled_eval, spec.quantiles) if labelled_eval is not None else None
            with cuda_lock(spec.artifact_dir / ".cuda.lock"):
                state, details = train_dl(prep, batcher, queries, spec.hyperparameters, spec.quantiles, seed, n_steps, loss, monitor,
                                          site_aux_ratio=inputs.input_config.site_aux_ratio, **kwargs)
                net = build_net(prep, spec.hyperparameters, len(spec.quantiles), {p: bool(prep.flags[i]) for i, p in enumerate(PROVIDERS)}).to(device)
                net.load_state_dict(state)
                t0 = time.perf_counter()
                z = predict_dl(net, batcher, eval_q).astype(np.float32)
                details["predict_seconds"] = time.perf_counter() - t0
            if season is not None:
                details["season"] = {**season_part(season_run, "dl", parent_key), **out["season"]}
            store.write("models", model_key, {"z": z}, {"model": "dl", "seed": seed, "details": details},
                        {"model.pt": lambda path, s=state: torch.save({"state_dict": s, "layout": prep.layout, "seed": seed}, path)})
        model_arrays, model_meta = store.load("models", model_key, mmap=False)
        out["z"][seed] = model_arrays["z"]
        out["details"][seed] = {**model_meta["details"], "model_key": model_key}
    del batcher
    torch.cuda.empty_cache()
    return out


def season_dl_args(season_run: dict, prep, train_q: QuerySet, steps: int) -> tuple[QuerySet, int, dict]:
    """계절 DL 학습의 (query, step 수, train_dl 인자). weighted는 Global query·step에 대상일 가중을, only는 계절 query·T_s·줄인
    warm-up을, ft는 계절 query·T_s·FT 학습률을 쓴다(ft의 시작 weights·고정 λ는 ft_parent)."""
    season, budget = season_run["settings"], season_run["budget"]
    if season["strategy"] == "weighted":
        return train_q, steps, {"day_weight": np.where(in_season(prep.days, season), season["weight"], 1.0)}
    if season["strategy"] == "only":
        return season_run["queries"], budget["T_s"], {"warmup_steps": budget["warmup"]}
    return season_run["queries"], budget["T_s"], {"learning_rate": season["ft_lr"]}


def ft_parent(store: Store, parent_key: str, seed: int) -> dict:
    """Seasonal-FT의 시작 weights와 고정 λ: 같은 seed의 Global DL 모델(cache)과 그 학습 기록."""
    import torch
    if not store.exists("models", parent_key):
        raise ValueError(f"Seasonal-FT의 Global DL 모델(seed {seed})이 cache에 없습니다. Global을 먼저 실행하세요(설정을 바꾼 Global은 --use-cache로).")
    details = store.load("models", parent_key, names=[])[1]["details"]
    state = torch.load(store.path("models", parent_key) / "model.pt", map_location="cpu")["state_dict"]
    return {"init_state": state, "fixed": {"lambda": details.get("lambda"), "lambda_site": (details.get("site_aux") or {}).get("lambda")}}


def per_seed_scores(results: dict, seeds: list[int], eval_q: QuerySet, panel: EvalPanel, levels: np.ndarray, weight: float,
                    adjust=None) -> tuple[dict, dict, dict]:
    """seed마다 XGB·DL·블렌드(같은 seed끼리)의 W_cohort를 채점하고 seed 간 표준편차와 단일 seed 판정 문턱 2√2·σ를 계산한다.
    adjust가 있으면 블렌드에 같은 후처리 module을 적용한다. 셋째 값은 seed별 제출값(블렌드, expert 하나면 그 expert)의 일별 벌점이다."""
    per_seed, daily = {}, {}
    for seed in seeds:
        q = {name: to_panel(finalize(np.asarray(r["z"][seed], dtype=np.float64), eval_q.cap, eval_q.daylight), eval_q, panel)
             for name, r in results.items()}
        if len(q) == 2:
            q["blend"] = blend(q["xgb"], q["dl"], weight)
            if adjust is not None:
                q["blend"] = adjust(q["blend"])
        scored = {name: score(panel, v, decision(v, levels), levels) for name, v in q.items()}
        per_seed[str(seed)] = {name: value[0]["W_cohort_krw"] for name, value in scored.items()}
        daily[str(seed)] = scored["blend" if "blend" in scored else next(iter(scored))][1]
    spread = {}
    for name in per_seed[str(seeds[0])]:
        values = np.array([per_seed[str(s)][name] for s in seeds])
        sd = float(values.std(ddof=1))
        spread[name] = {"n": len(values), "mean_krw": float(values.mean()), "std_krw": sd, "single_seed_threshold_krw": float(2 * np.sqrt(2) * sd)}
    return per_seed, spread, daily


@dataclass(frozen=True)
class ForecastSetup:
    xgb_input_config: InputConfig
    dl_input_config: InputConfig
    loss: LossConfig
    xgb_time_weight: str
    xgb_weight: float
    name: str
    model: str | None = None
    curtailment: dict | None = None      # 출력제어 후처리 module 설정. None이면 쓰지 않는다


def _setup_name(xgb_input_config: InputConfig, dl_input_config: InputConfig, loss: LossConfig, xgb_time_weight: str, xgb_weight: float,
                default_weight: float, curtailment: dict | None = None) -> str:
    """run 이름. 블렌드 비율이 protocol 기본값과 다르면 `__weighted-quantile-w{비율}`을 붙여 비율만 다른 모델도 이름이 갈리게 한다.
    출력제어 module을 쓰면 `__curtailment-h{반감기 일수}`를 붙인다."""
    dl_name = dl_input_config.name
    if dl_input_config.site_aux_ratio > 0:
        dl_name = f"{dl_name}-site-aux{dl_input_config.site_aux_ratio:g}"
    name = cell_id(dl_name, loss)
    if xgb_input_config != dl_input_config or xgb_time_weight != loss.time_weight:
        name = f"XGB.{xgb_input_config.name}{'-settlement' if xgb_time_weight == 'settlement' else ''}__DL.{name}"
    if xgb_weight != default_weight:
        name = f"{name}__weighted-quantile-w{xgb_weight:g}"
    if curtailment:
        name = f"{name}__curtailment-h{curtailment['half_life_days']:g}"
    return name


def setup_signature(setup: ForecastSetup) -> dict:
    """모델 구성. 등록 모델 정의(MODEL_FIELDS와 쓰는 MODULE_FIELDS)와 같은 형태이며 run identity·선택 기록·audit이 이것으로 대조한다."""
    signature = {"xgb_input": setup.xgb_input_config.name, "xgb_time_weight": setup.xgb_time_weight, "dl_input": setup.dl_input_config.name,
                 "dl_loss": setup.loss.name, "aux_ratio": float(setup.loss.aux_ratio), "dl_time_weight": setup.loss.time_weight, "blend": "weighted-quantile",
                 "xgb_weight": float(setup.xgb_weight)}
    if setup.dl_input_config.site_aux_ratio > 0:
        signature["site_aux_ratio"] = float(setup.dl_input_config.site_aux_ratio)
    if setup.curtailment:
        signature["curtailment"] = dict(setup.curtailment)
    return signature


def model_signature(spec: Spec, name: str) -> dict:
    """등록 모델의 구성(setup_signature와 같은 형태)."""
    item = spec.model(name)
    signature = {field: float(item[field]) if field in ("aux_ratio", "xgb_weight") else item[field] for field in MODEL_FIELDS}
    if item.get("site_aux_ratio"):
        signature["site_aux_ratio"] = float(item["site_aux_ratio"])
    if item.get("curtailment"):
        signature["curtailment"] = curtailment_settings(item["curtailment"])
    return signature


def model_label(spec: Spec, setup: ForecastSetup) -> str | None:
    """구성이 같은 등록 모델의 MK 버전. 없으면 None이다(이름표는 run identity에 넣지 않는다)."""
    signature = setup_signature(setup)
    return next((name for name in spec.registry["models"] if model_signature(spec, name) == signature), None)


def model_parameters(spec: Spec, base: str | None, signature: dict, partition: dict | None) -> dict:
    """기준 MK 정의와 다른 parameter와, original이 아닌 배정(partition 파일·hash)."""
    out = {}
    if base is not None:
        registered = model_signature(spec, base)
        out = {field: value for field, value in signature.items() if registered.get(field) != value}
        if "site_aux_ratio" in registered and "site_aux_ratio" not in signature:
            out["site_aux_ratio"] = 0.0
    if partition and not partition.get("is_original", False):
        out["partition"] = {"file": Path(partition["path"]).name, "hash": partition["hash"]}
    return out


def _parameter_text(key: str, value) -> str:
    if key == "partition":
        return value["file"]
    if key == "curtailment":
        return f"half_life_days={value['half_life_days']:g}" if value else "off"
    if key == "season":
        changed = [f"{k}={v:g}" for k, v in value.items() if k in SEASON_DEFAULTS and v != SEASON_DEFAULTS[k]]
        changed += [f"{k}={'+'.join(v) if isinstance(v, list) else v}" for k, v in value.items() if k in SEASON_OPTIONS]
        return "-".join([value["name"], value["strategy"], *changed])
    return str(value)


def parameter_label(version: str | None, parameters: dict) -> str:
    """'MK2 (xgb_weight=0.2, partition=stratified-test.csv)' 형태의 표기."""
    items = ", ".join(f"{k}={_parameter_text(k, v)}" for k, v in parameters.items())
    return f"{version or '-'} ({items})" if items else (version or "-")


def run_parameters(spec: Spec, manifest: dict) -> tuple[str | None, dict]:
    """run manifest의 기준 MK와 parameter. 두 항목이 없는 manifest는 모델 구성과 기준 모델에서 다시 계산한다."""
    if "parameters" in manifest:
        return manifest.get("model_version"), manifest["parameters"]
    base = None if manifest.get("recipe") else (manifest.get("base_model") or manifest.get("model_version"))
    return base, model_parameters(spec, base, manifest["model"], manifest.get("partition"))


def resolve_forecast_setup(spec: Spec, level: str, input_name: str | None = None, loss_name: str | None = None,
                           aux_ratio: float | None = None, time_weight: str | None = None, xgb_input_name: str | None = None,
                           xgb_time_weight: str | None = None, recipe: str | None = None, model: str | None = None,
                           xgb_weight: float | None = None, dl_input_name: str | None = None,
                           curtailment: dict | None = None, site_aux_ratio: float | None = None) -> ForecastSetup:
    """MK·recipe를 base로 읽고 명시한 override를 적용한다. 등록 정의는 보존한다.
    curtailment를 주면 출력제어 module을 그 설정으로 쓰고, 주지 않으면 base MK에 적힌 설정(없으면 쓰지 않음)을 따른다.
    recipe·noise_floor 기준 칸·두 expert 입력 override(input_name)는 발전소 보조 head 없이 시작하고, dl_input_name은 base MK의 세기를 유지한다."""
    overrides = (input_name, loss_name, aux_ratio, time_weight, xgb_input_name, xgb_time_weight, xgb_weight, dl_input_name, site_aux_ratio)
    if recipe is not None and model is not None:
        raise ValueError("recipe와 model은 별도로 사용합니다.")
    item = spec.model(model or spec.champion)
    if recipe is not None:
        recipes = spec.protocol.get("forecast_recipes", {})
        if recipe not in recipes:
            raise ValueError(f"알 수 없는 forecast recipe입니다: {recipe}")
        item = {**item, "xgb_weight": spec.blend_weight, "site_aux_ratio": 0.0, **recipes[recipe]}
    elif level == "noise_floor" and all(v is None for v in overrides) and model is None:
        cell = spec.protocol["noise_floor"]["cell"]
        item.update(xgb_input=cell["input"], dl_input=cell["input"], dl_loss=cell["loss"], aux_ratio=0.0, site_aux_ratio=0.0,
                    xgb_time_weight="none", dl_time_weight="none", xgb_weight=spec.blend_weight)
    if input_name is not None:
        item.update(xgb_input=input_name, dl_input=input_name, site_aux_ratio=0.0)
    if time_weight is not None:
        item.update(xgb_time_weight=time_weight, dl_time_weight=time_weight)
    for field, value in (("xgb_input", xgb_input_name), ("dl_input", dl_input_name), ("dl_loss", loss_name),
                         ("aux_ratio", aux_ratio), ("xgb_time_weight", xgb_time_weight), ("xgb_weight", xgb_weight),
                         ("site_aux_ratio", site_aux_ratio)):
        if value is not None:
            item[field] = value
    if item["xgb_time_weight"] not in ("none", "settlement"):
        raise ValueError(f"알 수 없는 XGB time weight입니다: {item['xgb_time_weight']}")
    weight = float(item["xgb_weight"])
    if not 0.0 <= weight <= 1.0:
        raise ValueError("xgb_weight는 [0, 1]이어야 합니다.")
    if not np.isfinite(float(item["aux_ratio"])) or float(item["aux_ratio"]) < 0:
        raise ValueError("aux_ratio는 유한한 0 이상의 값이어야 합니다.")
    if not np.isfinite(float(item.get("site_aux_ratio", 0.0))) or float(item.get("site_aux_ratio", 0.0)) < 0:
        raise ValueError("site_aux_ratio는 유한한 0 이상의 값이어야 합니다.")
    dl_input_config, xgb_input_config = spec.dl_input_config(item), spec.input_config(item["xgb_input"])
    loss = spec.loss(item["dl_loss"], float(item["aux_ratio"]), level == "smoke", item["dl_time_weight"])
    xgb_time_weight = item["xgb_time_weight"]
    module = curtailment_settings(curtailment) if curtailment is not None else (
        curtailment_settings(item["curtailment"]) if item.get("curtailment") else None)
    name = _setup_name(xgb_input_config, dl_input_config, loss, xgb_time_weight, weight, spec.blend_weight, module)
    setup = ForecastSetup(xgb_input_config, dl_input_config, loss, xgb_time_weight, weight, name, curtailment=module)
    return ForecastSetup(**{**setup.__dict__, "model": model_label(spec, setup)})


def curtailment_source(spec: Spec, signature: dict, partition: dict | None, seeds: list[int], cutoff_h: int) -> Path | None:
    """출력제어 보정의 잔차를 줄 OOT run을 고른다.

    같은 모델(출력제어 module 제외)·배정·seed·학습 설정으로 XGB·DL을 모두 돌려 평가까지 끝낸 module·계절 학습 없는 run 가운데, 학습 cutoff가
    이번 cutoff보다 앞서고 평가 시각이 이번 cutoff 이전에 걸치는 run이다. 여럿이면 cutoff가 가장 늦은 run(같으면 가장 늦게 끝난 run)을 쓴다.
    """
    core = {k: v for k, v in signature.items() if k != "curtailment"}
    target = partition["hash"] if partition and not partition.get("is_original", False) else None
    found = []
    for path in spec.artifact_dir.glob("runs/*/*/manifest.json"):
        manifest = read_json(path)
        record = manifest.get("partition") or {}
        lineage = manifest.get("lineage") or {}
        if (manifest.get("execution_status") != "completed" or manifest.get("evaluation_status") != "evaluated"
                or manifest.get("level") in ("smoke", "diagnostic") or manifest.get("curtailment") or manifest.get("season")
                or manifest.get("model") != core or manifest.get("seeds") != list(seeds) or set(manifest.get("experts", [])) != set(EXPERTS)
                or manifest.get("settings_hash") != settings_hash(spec)
                or (record.get("hash") if record and not record.get("is_original", False) else None) != target):
            continue
        if lineage.get("outer_cutoff_h", cutoff_h) < cutoff_h and lineage.get("eval_min_h", cutoff_h + 1) <= cutoff_h:
            found.append((lineage["outer_cutoff_h"], manifest.get("finished", ""), path.parent))
    return max(found)[2] if found else None


def curtailment_residual_source(run_dir: Path, sites: Sites, energy: np.ndarray, status: np.ndarray):
    """OOT run의 그룹별 중앙값 예측과, 그 run의 cohort 구성원이 모두 유효한 시각의 그룹 실적."""
    from .forecast.curtailment import Source
    manifest, stored = read_json(run_dir / "manifest.json"), np.load(run_dir / "quantiles.npz")
    hours = stored["hours"]
    index = {str(s): i for i, s in enumerate(sites.ids)}
    actual = np.full((len(hours), len(GROUPS)), np.nan)
    for g, name in enumerate(GROUPS):
        members = np.array([index[s] for s in manifest["cohort"]["site_ids"][name]])
        valid = (status[hours][:, members] == 1).all(axis=1)
        actual[valid, g] = energy[hours][:, members][valid].sum(axis=1)
    return Source(str(run_dir), manifest["run_key"], int(manifest["lineage"]["outer_cutoff_h"]), hours,
                  stored["forecast"].astype(np.float64), actual, np.asarray(manifest["cohort"]["capacity_kw"], dtype=np.float64))


def is_parameter_experiment(manifest: dict) -> bool:
    """등록 정의를 벗어난 run인지. 배정만 바꾼 run은 같은 정의의 제출 경로이고 submission level은 제출에 쓰므로 실험으로 보지 않는다."""
    if manifest["level"] == "submission":
        return False
    return bool(manifest.get("recipe") or manifest.get("season") or set(manifest["parameters"]) - {"partition"})


def run(spec: Spec, period_name: str, level: str, input_name: str | None = None, loss_name: str | None = None, aux_ratio: float | None = None,
        seeds: list[int] | None = None, experts=EXPERTS, selection: str | None = None, partition: str | Path | None = None,
        time_weight: str | None = None, xgb_input_name: str | None = None, xgb_time_weight: str | None = None,
        recipe: str | None = None, model: str | None = None, xgb_weight: float | None = None,
        dl_input_name: str | None = None, curtailment: bool = False, curtailment_half_life: float | None = None,
        season: dict | None = None, site_aux_ratio: float | None = None, use_cache: bool = False) -> Path:
    """입력·손실을 생략하면 champion(MK 버전)을, noise_floor 수준은 grid 기준 칸을 실행한다(resolve_forecast_setup).
    curtailment이거나 반감기를 주면 blend 뒤에 출력제어 후처리 module을 붙인다(설정 기본값은 runtime config).
    season(config.season_settings)을 주면 그 계절 학습 전략으로 학습한다. 평가 기간·cohort·panel은 Global과 같다.
    등록 정의를 벗어난 run(is_parameter_experiment)은 새 학습 cache를 임시 폴더에만 쓴다. use_cache나 환경 변수 VPP_EXPERIMENT_CACHE=1이면 저장하고,
    저장하지 않은 완료 run을 다시 요청하면 다시 학습한다."""
    if level not in LEVELS:
        raise ValueError(f"알 수 없는 level입니다: {level}")
    allowed, _ = LEVELS[level]
    if allowed and period_name not in allowed:
        raise ValueError(f"level={level}은 {allowed} 기간에서 실행합니다.")
    seeds = [int(s) for s in (seeds or level_seeds(spec, level))]
    experts = tuple(m for m in EXPERTS if m in experts)
    if not experts:
        raise ValueError("실행할 expert가 없습니다.")
    period = spec.period(period_name)
    smoke = level == "smoke"
    module_settings = spec.curtailment(curtailment_half_life) if curtailment or curtailment_half_life is not None else None
    season = None if season is None else season_settings(**season)
    setup = resolve_forecast_setup(spec, level, input_name, loss_name, aux_ratio, time_weight, xgb_input_name, xgb_time_weight,
                                   recipe, model, xgb_weight, dl_input_name, module_settings, site_aux_ratio)
    input_config, loss, cell_name = setup.dl_input_config, setup.loss, setup.name
    guard = h_guard(spec, period, setup, selection, level, experts)
    if "dl" in experts:
        from .forecast.train import configure_determinism
        configure_determinism(spec.config["runtime"]["torch_threads"])
    data = load_data(spec, partition)
    sites, base = data.sites, data.base
    is_original = data.partition is None or data.partition["is_original"]
    energy, status, sunup, smp = (np.asarray(a) for a in (base.energy, base.status, base.sunup, base.smp))
    view = actual_view(energy, status, period.cutoff_h)
    cutoff_day = period.cutoff_h // 24
    first_day = cutoff_day - spec.config["smoke"]["train_days"] + 1 if smoke else 0
    train_q = training_queries(view, sites, sunup, first_day)
    days = period.days[:spec.config["smoke"]["eval_days"]] if smoke else period.days
    cohort = freeze_cohort(view, sites, period)
    eval_q = period_queries(cohort, sites, days, sunup)
    panel = eval_panel(period, cohort, sites, energy, status, smp, sunup, eval_q, days if smoke else None)
    evidence = label_evidence(panel, sites, spec.evidence, check=is_original) if period.has_labels and not smoke else None
    inputs = resolve_inputs(data.weather, input_config, spec.excluded_inputs, spec.schema)
    xgb_inputs = inputs if setup.xgb_input_config == input_config else resolve_inputs(data.weather, setup.xgb_input_config, spec.excluded_inputs, spec.schema)
    rounds = spec.config["smoke"]["xgb_rounds"] if smoke else spec.hyperparameters["xgb_rounds"]
    steps = spec.config["smoke"]["dl_steps"] if smoke else spec.hyperparameters["dl_optimizer_steps"]
    season_run = None
    if season is not None:      # Global 학습 query에서 대상일 계절로 고른다(날짜별 query 구성은 Global과 같다)
        season_q = train_q.subset(np.flatnonzero(in_season(train_q.day, season)))
        season_run = {"settings": season, "queries": season_q, "budget": season_budget(season, train_q, steps, loss.warmup_steps),
                      "record": season_record(season, train_q)}
    keys = {"queries": digest({"base": base.key, "original": sites.original_hash, "policy": QUERY_POLICY, "night_rule": "start-mid-end",
                               "cutoff_h": period.cutoff_h, "first_day": first_day, "hash": train_q.hash}),
            "eval": digest({"cohort": cohort.hash, "policy": COHORT_POLICY, "days": days.tolist(), "hash": eval_q.hash}),
            "decisions": digest(spec.recorded_decisions)}
    lineage = {"outer_cutoff": str(period.fit_cutoff), "outer_cutoff_h": period.cutoff_h, "inner_cutoff": None,
               "fit_actual_max_h": int(np.flatnonzero(view.valid.any(axis=1)).max()), "fit_query_max_day": int(train_q.day.max()),
               "fit_forecast_max_day": cutoff_day, "eval_min_h": int(panel.hours.min()), "eval_max_h": int(panel.hours.max()),
               "training_first_day": int(first_day)}
    lineage["checks"] = {"actual_before_cutoff": lineage["fit_actual_max_h"] <= period.cutoff_h,
                         "queries_before_cutoff_day": lineage["fit_query_max_day"] <= cutoff_day,
                         "eval_after_cutoff": lineage["eval_min_h"] > period.cutoff_h}
    if not all(lineage["checks"].values()):
        raise RuntimeError(f"cutoff lineage 검사 실패: {lineage['checks']}")
    identity = {"cell": cell_name, "model": setup_signature(setup), "input_config": input_key(input_config), "loss_config": loss.__dict__,
                "model_inputs": {"xgb": input_key(setup.xgb_input_config), "dl": input_key(input_config)},
                "model_time_weights": {"xgb": setup.xgb_time_weight, "dl": loss.time_weight},
                "input_role": input_config.input_role, "period": period_name, "level": level, "seeds": seeds,
                "experts": list(experts), "keys": keys, "weather_key": data.weather.key, "base_key": base.key, "panel_hash": panel.panel_hash,
                "original_hash": sites.original_hash, "settings_hash": settings_hash(spec), "spec_hashes": spec.hashes,
                "versions": versions("base", "weather", "queries", "features", "xgb", "dl_prep", "dl", "postprocess", "scoring",
                                     *(("curtailment",) if setup.curtailment else ())),
                "rounds": rounds, "steps": steps, "xgb_weight": setup.xgb_weight, "guard": guard,
                "partition": data.partition}
    if season is not None:
        identity["season"] = season
    source_dir = None
    if setup.curtailment:
        source_dir = curtailment_source(spec, setup_signature(setup), data.partition, seeds, period.cutoff_h)
        identity["curtailment"] = {"settings": setup.curtailment, "calendar": digest(spec.calendar.rules),
                                   "version": versions("curtailment")["curtailment"],
                                   "source": read_json(source_dir / "manifest.json")["run_key"] if source_dir else None}
    identity["lineage"] = lineage
    identity["experts_detail"] = planned_models(data, inputs, xgb_inputs, keys, setup, experts, seeds, rounds, steps, season_run)
    identity["effective_runtime"] = ({"torch_threads": int(spec.config["runtime"]["torch_threads"]),
                                      "cublas_workspace": spec.config["runtime"]["cublas_workspace"]} if "dl" in experts else {})
    calculation = calculation_identity(identity)
    run_key = identity_key(identity)
    existing = find_calculated_run(spec.artifact_dir, run_key)
    store_cache = use_cache or experiment_cache_enabled()
    if existing is not None and not (store_cache and not read_json(existing / "manifest.json").get("cache_stored", True)):
        log(f"동일 계산의 완료 run 재사용: {existing}")
        return existing
    base_model = None if recipe else (model or spec.champion)
    run_dir = spec.artifact_dir / "runs" / period_name / f"{base_model or recipe}-{level}-{run_key[:10]}"
    if ((run_dir / "manifest.json").exists() and read_json(run_dir / "manifest.json").get("execution_status") == "completed"
            and not (store_cache and not read_json(run_dir / "manifest.json").get("cache_stored", True))):
        if identity_key(read_json(run_dir / "manifest.json")) != run_key:
            raise ValueError("완료 run의 계산 identity가 요청과 다릅니다.")
        log(f"완료된 run 재사용: {run_dir}")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"run_key": run_key, **identity, "calculation": calculation,
                "provenance": {"spec_hashes": spec.hashes, "settings_source": spec.settings_source},
                "execution_status": "running", "evaluation_status": "pending",
                "selection_status": "unselected", "champion_at_execution": spec.champion,
                "model_version": base_model,
                "parameters": {**model_parameters(spec, base_model, identity["model"], data.partition), **({"season": season} if season else {})},
                "recipe": recipe,
                "cohort": {"id": panel.cohort_id, "pv": panel.n_pv.tolist(), "capacity_kw": panel.cap.tolist(),
                           "site_ids": {g: cohort.site_ids(sites, i) for i, g in enumerate(GROUPS)}},
                "evidence": evidence, "lineage": lineage, "decisions": spec.recorded_decisions,
                "queries": {"train": len(train_q), "train_by_group": {g: int((train_q.group == i).sum()) for i, g in enumerate(GROUPS)},
                            "train_target_hours": int(train_q.target_mask.sum()), "eval": len(eval_q), "eval_days": len(days)},
                "environment": environment(include_torch="dl" in experts), "source_sha": source_sha(), "started": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    temporary = not store_cache and is_parameter_experiment(manifest)
    manifest["cache_stored"] = not temporary
    write_json(run_dir / "manifest.json", manifest)
    timings = {}
    if temporary:
        open_temporary_cache(spec.artifact_dir)
    try:
        results = {}
        if "xgb" in experts:
            t0 = time.perf_counter()
            slot_weight = None
            if setup.xgb_time_weight == "settlement":
                slot_weight = settlement_slot_weight()
                # 정산 k의 시간 규칙이 base cache의 k와 같은지 확인한다.
                k_by_day = np.asarray(data.base.k).reshape(-1, 24)
                if not np.array_equal(k_by_day, np.broadcast_to(slot_weight, k_by_day.shape)):
                    raise ValueError("정산 k의 slot 규칙이 base cache와 다릅니다.")
            results["xgb"] = run_xgb(data, xgb_inputs, train_q, eval_q, seeds, rounds, keys, level == "diagnostic", slot_weight,
                                     expert_season(season_run, "xgb"))
            timings["xgb_seconds"] = time.perf_counter() - t0
            write_json(run_dir / "feature_manifest.json", results["xgb"]["manifest"])
        if "dl" in experts:
            t0 = time.perf_counter()
            labelled = with_labels(eval_q, panel) if level == "diagnostic" else None
            results["dl"] = run_dl(data, inputs, train_q, eval_q, labelled, period.cutoff_h, seeds, steps, keys, loss,
                                   expert_season(season_run, "dl"))
            timings["dl_seconds"] = time.perf_counter() - t0
        levels = spec.quantiles
        median = int(np.flatnonzero(np.isclose(levels, 0.5))[0])
        quantiles, crossing = {}, {}
        for name, result in results.items():
            z = seed_mean([result["z"][s] for s in seeds])
            crossing[name] = {"seed_mean": crossing_stats(z, eval_q.daylight, median),
                              "per_seed": {str(s): crossing_stats(np.asarray(result["z"][s], dtype=np.float64), eval_q.daylight, median) for s in seeds}}
            quantiles[name] = to_panel(finalize(z, eval_q.cap, eval_q.daylight), eval_q, panel)
        weight = setup.xgb_weight
        final = blend(quantiles["xgb"], quantiles["dl"], weight) if len(quantiles) == 2 else next(iter(quantiles.values()))
        module, extra = None, {}
        if setup.curtailment:
            from .forecast.curtailment import fit_curtailment
            t0 = time.perf_counter()
            source = curtailment_residual_source(source_dir, sites, energy, status) if source_dir else None
            module = fit_curtailment(setup.curtailment, spec.calendar, base, data.weather, sites, period.cutoff_h, cohort.members, panel.hours,
                                     source)
            quantiles["blend_before_curtailment"] = final
            final = module.apply(final, levels, panel.cap)
            extra["curtailment_probability"] = module.probability
            timings["curtailment_seconds"] = time.perf_counter() - t0
            log(f"출력제어 module: {'출력제어 시각 ' + str(module.summary['event_hours']) + '개로 적용' if module.active else '출력제어 기록이 없어 꺼짐'}")
        quantiles["blend"] = final
        forecast = decision(final, levels)
        np.savez_compressed(run_dir / "quantiles.npz", hours=panel.hours, levels=levels, forecast=forecast,
                            **{k: v.astype(np.float32) for k, v in quantiles.items()}, **extra)
        stamps = [t.isoformat(sep=" ") for t in timestamps(panel.hours)]
        pd.DataFrame({"group": np.repeat(GROUPS, len(stamps)), "ts": np.tile(stamps, len(GROUPS)),
                      "forecast_kwh": np.concatenate([forecast[:, g] for g in range(len(GROUPS))])}
                     ).to_csv(run_dir / ("forecast.csv" if period_name == "test" else "validation-forecast.csv"), index=False, lineterminator="\n")
        report = {"crossing_before_sort": crossing, "components": {}, "diagnostics": {}}
        if "xgb" in results and "diagnostics" in results["xgb"]:
            report["diagnostics"].update(results["xgb"]["diagnostics"])
        if "dl" in results and level == "diagnostic":
            report["diagnostics"]["dl_oot_l0_curve"] = {str(s): results["dl"]["details"][s].get("oot_curve") for s in seeds}
        if panel.has_labels:
            daily = {}
            for name, q in quantiles.items():
                label = cell_name if name == "blend" else f"{name}_component"
                report["components"][label], daily[label] = score(panel, q, decision(q, levels), levels)
            if len(seeds) > 1:      # seed별 제출값의 일별 벌점은 `<칸 이름>__s<seed>`로 함께 저장한다
                adjust = (lambda q: module.apply(q, levels, panel.cap)) if module is not None else None
                report["per_seed_W_krw"], report["seed_spread"], seed_daily = per_seed_scores(results, seeds, eval_q, panel, levels, weight,
                                                                                              adjust)
                daily.update({f"{cell_name}__s{s}": v for s, v in seed_daily.items()})
            np.savez_compressed(run_dir / "daily_penalty.npz", **daily)
        if module is not None:
            report["curtailment"] = module.summary
        if "dl" in results:
            report["dl_lambda"] = {str(s): {"lambda": d.get("lambda"), "calibration": d.get("calibration"),
                                            "warmup_state_sha": d.get("warmup_state_sha")} for s, d in results["dl"]["details"].items()}
        if season_run is not None:
            report["season"] = {"settings": season, "budget": season_run["budget"], "dl": season_run["record"],
                                "xgb": results["xgb"].get("season") if "xgb" in results else None}
        write_json(run_dir / "report.json", report)
        manifest["experts_detail"] = {name: {"details": {str(s): d for s, d in result["details"].items()},
                                            **{k: v for k, v in result.items() if k in ("feature_key", "prep_key", "rows", "power_curve", "layout", "build_seconds",
                                                                                         "season")}}
                                     for name, result in results.items()}
        manifest["timings"] = timings
        manifest["execution_status"] = "completed"
        manifest["evaluation_status"] = evaluation_status(manifest, panel, forecast, period, smoke, report)
        if period_name != "test" and not smoke:
            summary = report["components"][cell_name]
            log(f"{cell_name} {period_name} S_cohort={summary['S_cohort_krw']:,.0f} W_cohort={summary['W_cohort_krw']:,.0f} "
                f"({', '.join(f'{k}: {v['W_cohort_krw']:,.0f}' for k, v in report['components'].items() if k != cell_name)})")
    except BaseException:
        manifest["execution_status"] = "failed"
        (run_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        manifest["traceback"] = str(run_dir / "traceback.txt")
        write_json(run_dir / "manifest.json", manifest)
        raise
    finally:
        close_temporary_cache()
    manifest["finished"] = pd.Timestamp.now(tz="Asia/Seoul").isoformat()
    if identity_key(manifest) != run_key:
        raise RuntimeError("학습 결과의 model cache key가 실행 계획과 다릅니다.")
    write_json(run_dir / "manifest.json", manifest)
    index_runs(spec.artifact_dir, [run_dir], write=True)
    log(f"완료: {run_dir} ({manifest['evaluation_status']}, {parameter_label(base_model, manifest['parameters'])})")
    return run_dir


def label_evidence(panel: EvalPanel, sites: Sites, evidence: dict, check: bool = True) -> dict:
    """original 구성이고 cohort evidence가 있는 기간은 대조하고, 그 밖에는 cohort와 정답 mask 완전성을 기록한다."""
    if check and panel.period in evidence["periods"]:
        return check_evidence(panel, sites, evidence)
    groups = {g: {"sites": int(panel.n_pv[i]), "capacity_kw": float(panel.cap[i]), "complete_hours": int(panel.m[:, i].sum()),
                  "missing_label_hours": int((~panel.m[:, i]).sum())} for i, g in enumerate(GROUPS)}
    return {"period": panel.period, "status": "not_applicable", "labels_complete": bool(panel.m.all()), "groups": groups}


def evaluation_status(manifest: dict, panel: EvalPanel, forecast: np.ndarray, period: Period, smoke: bool, report: dict) -> str:
    if smoke:
        return "execution_check"
    if period.name == "test":
        return "not_applicable"
    ok = (forecast.shape == (period.expected_hours, len(GROUPS)) and np.isfinite(forecast).all() and (forecast >= 0).all()
          and _evidence_ok(manifest.get("evidence") or {})
          and report["components"].get(manifest["cell"], {}).get("panel_hash") == panel.panel_hash
          and all(manifest["lineage"]["checks"].values()))
    return "evaluated" if ok else "incomplete"


def _evidence_ok(evidence: dict) -> bool:
    return evidence.get("status") == "matched" or (evidence.get("status") == "not_applicable" and evidence.get("labels_complete") is True)


def _component(run_dir: Path, name: str | None = None) -> tuple[dict, np.ndarray, dict]:
    manifest = read_json(run_dir / "manifest.json")
    label = name or manifest["cell"]
    report = read_json(run_dir / "report.json")["components"][label]
    daily = np.load(run_dir / "daily_penalty.npz")[label]
    return report, daily, manifest


def compare_runs(spec: Spec, candidate: Path, baseline: Path) -> dict:
    c_report, c_daily, c_manifest = _component(candidate)
    b_report, b_daily, b_manifest = _component(baseline)
    for m in (c_manifest, b_manifest):
        if m["evaluation_status"] != "evaluated":
            raise ValueError(f"평가 완료되지 않은 run은 비교하지 않습니다: {m['run_key'][:10]} ({m['evaluation_status']})")
    result = compare_reports(c_report, b_report, c_daily, b_daily)
    result.update({"candidate": {"cell": c_manifest["cell"], "run_key": c_manifest["run_key"], "level": c_manifest["level"]},
                   "baseline": {"cell": b_manifest["cell"], "run_key": b_manifest["run_key"], "level": b_manifest["level"]},
                   "exposure_status": c_manifest.get("guard", {}).get("exposure_status")})
    out = spec.artifact_dir / "reports" / f"compare-{c_manifest['run_key'][:10]}-{b_manifest['run_key'][:10]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    return result


def select(spec: Spec, baseline: Path, finalist: Path, out: Path) -> dict:
    """같은 evaluation panel의 두 run을 명시적으로 고정한다. Champion 선택은 adopt 명령이 담당한다."""
    manifests = {role: read_json(Path(d) / "manifest.json") for role, d in (("baseline", baseline), ("finalist", finalist))}
    for role, m in manifests.items():
        if m["execution_status"] != "completed" or m["evaluation_status"] != "evaluated":
            raise ValueError(f"{role}는 평가 완료된 run이어야 합니다.")
    if manifests["baseline"]["panel_hash"] != manifests["finalist"]["panel_hash"]:
        raise ValueError("두 run의 evaluation panel이 다릅니다.")
    if any("model" not in m for m in manifests.values()):
        raise ValueError("run manifest에 모델 구성이 없습니다. 현재 코드로 다시 실행합니다.")
    record = {"frozen": True, "created": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
              "period": manifests["baseline"]["period"],
              "panel_hash": manifests["baseline"]["panel_hash"],
              **{role: {"cell": m["cell"], "model": m["model"], "model_version": m.get("model_version"), "run_key": m["run_key"],
                        "seeds": m["seeds"], "experts": m["experts"], "settings_hash": m["settings_hash"]} for role, m in manifests.items()}}
    record["sha"] = digest(record)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(out), record)
    return record


def reevaluate(spec: Spec, run_dir: Path) -> dict:
    """저장된 분위수를 다시 채점한다. panel hash가 run과 같아야 한다."""
    manifest = read_json(run_dir / "manifest.json")
    if manifest["level"] == "smoke" or manifest["period"] == "test":
        raise ValueError("연간 정답이 있는 run만 다시 채점합니다.")
    data = load_data(spec, (manifest.get("partition") or {}).get("path"))
    period = spec.period(manifest["period"])
    energy, status, sunup, smp = (np.asarray(a) for a in (data.base.energy, data.base.status, data.base.sunup, data.base.smp))
    cohort = freeze_cohort(actual_view(energy, status, period.cutoff_h), data.sites, period)
    eval_q = period_queries(cohort, data.sites, period.days, sunup)
    panel = eval_panel(period, cohort, data.sites, energy, status, smp, sunup, eval_q)
    if panel.panel_hash != manifest["panel_hash"]:
        raise ValueError("evaluation panel hash가 run과 다릅니다.")
    stored = np.load(run_dir / "quantiles.npz")
    levels = stored["levels"]
    out = {}
    for name in ("xgb", "dl", "blend"):
        if name in stored:
            report, _ = score(panel, stored[name].astype(np.float64), decision(stored[name].astype(np.float64), levels), levels)
            out[name] = {k: report[k] for k in ("S_cohort_krw", "W_cohort_krw", "revenue_krw", "panel_hash")}
    return out


def seed_daily(spec: Spec, run_dir: Path) -> dict:
    """run의 seed별 제출값(같은 seed의 XGB·DL을 run과 같은 후처리·결합)을 다시 채점한 일별 벌점 [대상일, 그룹].
    seed별 일별 벌점을 저장하지 않은 run도 model cache로 계산한다. panel hash가 run과 같아야 한다."""
    manifest = read_json(run_dir / "manifest.json")
    if manifest["level"] == "smoke" or manifest["period"] == "test" or manifest.get("curtailment"):
        raise ValueError("연간 정답이 있고 출력제어 module이 없는 run만 seed별로 다시 채점합니다.")
    data = load_data(spec, (manifest.get("partition") or {}).get("path"))
    period = spec.period(manifest["period"])
    energy, status, sunup, smp = (np.asarray(a) for a in (data.base.energy, data.base.status, data.base.sunup, data.base.smp))
    cohort = freeze_cohort(actual_view(energy, status, period.cutoff_h), data.sites, period)
    eval_q = period_queries(cohort, data.sites, period.days, sunup)
    panel = eval_panel(period, cohort, data.sites, energy, status, smp, sunup, eval_q)
    if panel.panel_hash != manifest["panel_hash"]:
        raise ValueError("evaluation panel hash가 run과 다릅니다.")
    levels, out = spec.quantiles, {}
    for seed in manifest["seeds"]:
        q = {}
        for name in manifest["experts"]:
            key = manifest["experts_detail"][name]["details"][str(seed)]["model_key"]
            z = np.asarray(data.store.load("models", key, names=["z"], mmap=False)[0]["z"], dtype=np.float64)
            q[name] = to_panel(finalize(z, eval_q.cap, eval_q.daylight), eval_q, panel)
        value = blend(q["xgb"], q["dl"], float(manifest["xgb_weight"])) if len(q) == 2 else next(iter(q.values()))
        out[str(seed)] = score(panel, value, decision(value, levels), levels)[1]
    return out


def gate_evidence(path: Path) -> dict:
    """제출 gate 근거: 공통 평가기 비교 보고서(`vppcontract compare`)의 W·ΔW·7일 block bootstrap 구간."""
    report = read_json(path)
    if not {"period", "a", "b", "delta_W_krw", "bootstrap"} <= set(report):
        raise ValueError(f"{path}는 공통 평가기 비교 보고서가 아닙니다.")
    interval = report["bootstrap"]["7d"]
    return {"path": str(path), "sha": file_sha(path), "period": report["period"], "candidate": report["a"]["directory"],
            "baseline": report["b"]["directory"], "W_candidate_krw": report["a"]["W_krw"], "W_baseline_krw": report["b"]["W_krw"],
            "delta_W_krw": report["delta_W_krw"], "bootstrap_7d_krw": [interval["lower"], interval["upper"]]}


def submit(spec: Spec, run_dir: Path, acknowledge: list[str], dry_run: bool, strict_distribution: bool = False,
           evidence: list[Path] | None = None) -> dict:
    """test run으로 제출 파일을 만든다. 기록에는 기준 MK, 바뀐 parameter·배정, gate 근거(비교 보고서)를 적는다."""
    manifest = read_json(run_dir / "manifest.json")
    if manifest["period"] != "test" or manifest["execution_status"] != "completed" or manifest["level"] == "smoke":
        raise ValueError("제출은 완료된 test run(smoke 제외)으로만 만듭니다.")
    if manifest["input_role"] not in spec.protocol["submission_input_roles"]:
        raise ValueError(f"input_role={manifest['input_role']}은 제출에 쓸 수 없습니다.")
    proof = [gate_evidence(Path(p)) for p in evidence or []]
    data = load_data(spec, (manifest.get("partition") or {}).get("path"))
    if manifest.get("partition") and data.partition["hash"] != manifest["partition"]["hash"]:
        raise ValueError("제출 partition이 forecast run의 구성과 다릅니다.")
    report = load_or_check(spec, data.weather, spec.artifact_dir / "reports")
    configs = manifest.get("model_inputs") or {"shared": manifest["input_config"]}
    used = set()
    excluded = tuple(sorted({name for item in manifest.get("decisions", {}).get("deviations", spec.config["deviations"])
                             for name in item["inputs"]}))
    for expert, config in configs.items():
        if expert != "shared" and expert not in manifest["experts"]:
            continue
        inputs = resolve_inputs(data.weather, InputConfig(**config), excluded, spec.schema)
        used.update(c for p in inputs.active for c in inputs.providers[p].names)
    decision_gate = gate(report, used, acknowledge, strict=strict_distribution)
    if decision_gate["blocking"]:
        raise ValueError(f"test 입력 분포 표시가 확인되지 않았습니다(--acknowledge 필요): {decision_gate['blocking']}")
    stored = np.load(run_dir / "quantiles.npz")
    if dry_run:
        with tempfile.TemporaryDirectory() as tmp:
            validation = write_submission(data.sites, spec.data_dir / "smp.csv", stored["hours"], stored["forecast"], Path(tmp))
        out_dir = None
    else:
        out_dir = spec.artifact_dir / "submissions" / run_dir.name
        validation = write_submission(data.sites, spec.data_dir / "smp.csv", stored["hours"], stored["forecast"], out_dir)
    version, parameters = run_parameters(spec, manifest)
    record = {"run_key": manifest["run_key"], "run_dir": str(run_dir), "model_version": version, "parameters": parameters,
              "model_label": parameter_label(version, parameters), "model": manifest["model"],
              "evidence": proof, "dry_run": dry_run, "gate": decision_gate,
              "validation": validation, "created": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
              "output_dir": str(out_dir) if out_dir else None}
    if out_dir:
        write_json(out_dir / "submission-manifest.json", record)
    return record


def audit(spec: Spec) -> dict:
    data = load_data(spec)
    report = {"base": data.base.audit, "weather": data.weather.audit, "base_key": data.base.key, "weather_key": data.weather.key}
    out = spec.artifact_dir / "reports" / f"audit-{data.weather.key[:8]}-{data.base.key[:8]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    return report


def cohorts(spec: Spec) -> dict:
    data = load_data(spec)
    energy, status, sunup, smp = (np.asarray(a) for a in (data.base.energy, data.base.status, data.base.sunup, data.base.smp))
    out = {}
    for name in ("D", "V", "H"):
        period = spec.period(name)
        cohort = freeze_cohort(actual_view(energy, status, period.cutoff_h), data.sites, period)
        eval_q = period_queries(cohort, data.sites, period.days, sunup)
        panel = eval_panel(period, cohort, data.sites, energy, status, smp, sunup, eval_q)
        out[name] = {**check_evidence(panel, data.sites, spec.evidence), "cohort_id": panel.cohort_id, "panel_hash": panel.panel_hash,
                     "pv": panel.n_pv.tolist(), "capacity_kw": panel.cap.round(3).tolist(), "complete_hours": panel.m.sum(axis=0).tolist()}
    return out


def query_counts(spec: Spec, period_name: str) -> dict:
    data = load_data(spec)
    period = spec.period(period_name)
    view = actual_view(np.asarray(data.base.energy), np.asarray(data.base.status), period.cutoff_h)
    queries = training_queries(view, data.sites, np.asarray(data.base.sunup))
    per = np.unique(np.stack([queries.day, queries.group], axis=1), axis=0, return_counts=True)[1]
    return {"period": period_name, "cutoff_h": period.cutoff_h, "queries": len(queries),
            "by_group": {g: int((queries.group == i).sum()) for i, g in enumerate(GROUPS)}, "max_per_day_group": int(per.max()),
            "xgb_training_rows": int(queries.target_mask.sum()), "query_hash": queries.hash}


COMPONENTS = {"blend": None, "dl": "dl_component", "xgb": "xgb_component"}


def reference_cell(spec) -> str:
    """입력×손실 grid의 기준 칸(protocol noise_floor.cell). noise floor·screening·confirmation 비교를 이 칸에 맞춘다."""
    cell = spec.protocol["noise_floor"]["cell"]
    return cell_id(cell["input"], spec.loss(cell["loss"]))


def run_grid(spec: Spec, period: str, level: str, inputs=INPUTS, losses=GRID_LOSSES, aux_ratio: float | None = None, seeds=None) -> list[Path]:
    """입력 구성마다 XGB를 한 번 학습하고(손실 칸끼리 공유), 손실 칸마다 DL을 학습한다. CUDA 학습은 순서대로 하나씩 돈다."""
    out = []
    for input_name in inputs:
        for loss_name in losses:
            out.append(run(spec, period, level, input_name, loss_name, aux_ratio, seeds))
    return out


def find_run(spec: Spec, period: str, cell: str, level: str, seeds: list[int]) -> Path | None:
    """Grid의 같은 칸·학습 설정으로 완료·평가된 두 expert run 중 가장 최근 것. 다른 배정이나 계절 학습 run은 뺀다."""
    found = []
    for manifest_path in (spec.artifact_dir / "runs" / period).glob("*/manifest.json"):
        directory = manifest_path.parent
        manifest = read_json(manifest_path)
        if (manifest.get("cell") == cell and manifest.get("level") == level and not manifest.get("season")
                and (manifest.get("partition") or {"is_original": True})["is_original"]
                and manifest.get("execution_status") == "completed" and manifest.get("evaluation_status") == "evaluated"
                and manifest.get("seeds") == [int(s) for s in seeds]
                and set(manifest.get("experts", [])) == set(EXPERTS) and manifest.get("settings_hash") == settings_hash(spec)):
            found.append((manifest.get("finished", ""), directory))
    return max(found)[1] if found else None


def cell_scores(spec: Spec, period: str, cell: str, level: str, seeds: list[int]) -> dict | None:
    directory = find_run(spec, period, cell, level, seeds)
    if directory is None:
        return None
    report = read_json(directory / "report.json")
    manifest = read_json(directory / "manifest.json")
    out = {"run": str(directory), "panel_hash": manifest["panel_hash"]}
    for name, label in COMPONENTS.items():
        out[name] = report["components"][label or manifest["cell"]]["W_cohort_krw"]
    out["lambda"] = {s: v.get("lambda") for s, v in report.get("dl_lambda", {}).items()}
    out["warmup_state_sha"] = {s: v.get("warmup_state_sha") for s, v in report.get("dl_lambda", {}).items()}
    return out


def noise_floor(spec: Spec, period: str = "V", required: bool = False) -> dict | None:
    seeds = level_seeds(spec, "noise_floor")
    directory = find_run(spec, period, reference_cell(spec), "noise_floor", seeds)
    if directory is None:
        if required:
            raise ValueError("noise floor run이 없습니다.")
        return None
    report = read_json(directory / "report.json")
    spread = report["seed_spread"]
    return {"run": str(directory), "seeds": seeds, "per_seed_W_krw": report["per_seed_W_krw"],
            **{name: spread[name] for name in COMPONENTS}}


def _main_effects(delta: dict, component: str) -> dict:
    """입력 효과: 손실 평균 ΔW로 forecast-solar 위 2×2(dr·phys) 주효과와 forecast-raw→forecast-solar 차이. 값이 음수면 W가 줄어 개선이다."""
    mean = {i: float(np.mean([delta[i][l][component] for l in delta[i]])) for i in delta if delta[i]}
    out = {"input_mean_delta_krw": mean}
    raw, sun, dr, phys, both = "forecast-raw", "forecast-solar", "forecast-solar-diff", "forecast-solar-physics", "forecast-physics"
    if all(i in mean for i in (raw, sun, dr, phys, both)):
        out["sun_minus_raw"] = mean[sun] - mean[raw]
        out["dr_main_effect"] = (mean[dr] + mean[both]) / 2 - (mean[sun] + mean[phys]) / 2
        out["phys_main_effect"] = (mean[phys] + mean[both]) / 2 - (mean[sun] + mean[dr]) / 2
        out["dr_phys_interaction"] = (mean[both] - mean[phys]) - (mean[dr] - mean[sun])
    return out


def analyze_screening(spec: Spec, period: str = "V", aux_ratio: float | None = None) -> dict:
    floor = noise_floor(spec, period)
    seeds = level_seeds(spec, "screening")
    threshold = {name: floor[name]["single_seed_threshold_krw"] if floor else None for name in COMPONENTS}
    reference = cell_scores(spec, period, reference_cell(spec), "screening", seeds)
    panel_hash = reference["panel_hash"] if reference else None
    cells, delta, missing = {}, {}, []
    for input_name in INPUTS:
        delta[input_name] = {}
        for loss_name in GRID_LOSSES:
            cell = cell_id(input_name, spec.loss(loss_name, aux_ratio))
            scores = cell_scores(spec, period, cell, "screening", seeds)
            if scores is None:
                missing.append(cell)
                continue
            if panel_hash is None:
                panel_hash = scores["panel_hash"]
            if scores["panel_hash"] != panel_hash:
                raise ValueError(f"{cell}의 evaluation panel이 기준 칸과 다릅니다.")
            d = {name: scores[name] - reference[name] if reference else None for name in COMPONENTS}
            cells[cell] = {**scores, "delta_krw": d,
                           "beyond_noise": {name: bool(d[name] < -threshold[name]) if reference and floor else None for name in COMPONENTS}}
            if reference:
                delta[input_name][loss_name] = d
    loss_effect = {}
    for loss_name in GRID_LOSSES:
        if loss_name == "pinball":
            continue
        pairs = [(delta[i][loss_name], delta[i]["pinball"]) for i in INPUTS if loss_name in delta[i] and "pinball" in delta[i]]
        if pairs:
            loss_effect[loss_name] = {name: float(np.mean([a[name] - b[name] for a, b in pairs])) for name in ("dl", "blend")}
    candidates = sorted((c for c in cells if c != reference_cell(spec)), key=lambda c: cells[c]["blend"])
    ranking = [{"cell": c, "W_krw": cells[c]["blend"], "delta_W_krw": cells[c]["delta_krw"]["blend"],
                "beyond_noise": cells[c]["beyond_noise"]} for c in candidates]
    return {"period": period, "noise_floor": floor, "threshold_krw": threshold, "reference": reference, "cells": cells,
            "missing_cells": missing, "input_effect": {name: _main_effects(delta, name) for name in COMPONENTS},
            "loss_effect_vs_pinball_krw": loss_effect, "candidates": candidates, "ranking": ranking}


def analyze_confirmation(spec: Spec, cells: list[str], period: str = "V") -> dict:
    seeds = level_seeds(spec, "confirmation")
    reference = cell_scores(spec, period, reference_cell(spec), "confirmation", seeds)
    out = {"reference": reference, "cells": {}}
    panel_hash = reference["panel_hash"] if reference else None
    for cell in cells:
        scores = cell_scores(spec, period, cell, "confirmation", seeds)
        if scores is not None:
            if panel_hash is None:
                panel_hash = scores["panel_hash"]
            if scores["panel_hash"] != panel_hash:
                raise ValueError("confirmation run의 evaluation panel이 다릅니다.")
            out["cells"][cell] = {**scores, "delta_krw": {name: scores[name] - reference[name] if reference else None for name in COMPONENTS}}
    return out


def write_report(spec: Spec, name: str, value: dict) -> Path:
    path = spec.artifact_dir / "reports" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)
    log(f"보고서: {path}")
    return path
