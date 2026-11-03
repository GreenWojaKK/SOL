"""발전소 단위 잔차: primitive 모델(MK 버전)의 두 expert를 발전소-일 query(구성원 1개)에 적용한 예측과
잔차 e = Y − ŷ_blend.
참조: docs/solarvpp-reference.md#site-primitive.

XGB는 모델의 `xgb_input`·`xgb_time_weight`, DL은 `dl_input`·`dl_loss`·`aux_ratio`·`dl_time_weight`를 쓰고, 두 중앙값을 모델의
`xgb_weight`로 섞는다. 그룹 모델과 같은 feature·power curve·학습 함수(`fit_xgb`, `run_dl`)를 쓴다.
같은 분위수끼리 섞는 블렌드의 중앙값은 두 모델 중앙값의 가중합이므로, XGB는 중앙값만 학습한다.
발전소 단위 XGB 행렬에는 query-day 하늘 상태 요약(`ctx`)이 없으므로, `xgb_input`에 `ctx`가 있는 모델은 쓰지 않는다.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from ..config import Spec, settlement_slot_weight, input_key
from ..data import ROUNDS, Sites
from ..experiment import Data, log, run_dl
from ..forecast.features import TIME_FEATURES, InputSet, PowerCurve, pv_block, resolve_inputs, time_features
from ..forecast.models import finalize, fit_xgb, predict_xgb, seed_mean
from ..forecast.queries import QuerySet, actual_view
from ..store import digest, package_versions, versions

POLICY = "one query per site-day; mask = valid generation hours at or before the cutoff; daylight = the site's own sunup"
MEDIAN = np.array([0.5])


def site_matrix(inputs: InputSet, sites: Sites, solar: np.ndarray, select: np.ndarray, first_day: int, last_day: int):
    """선택한 (시각, 발전소) 행의 feature. select는 전역 시간축 [h, PV] bool이다.

    열은 `pv_block`의 PV 열(그룹 모델 `__mean` 열의 구성원 1개 값), 시간 표현, 대응 관측소의 round별 결측 indicator이다.
    반환: (행렬, 전역 시각 h, 발전소 index, 열 이름, power curve 입력 max(cos 천정각, 0)).
    """
    rows, hs, idx, curve, names = [], [], [], [], None
    for day in range(first_day, last_day + 1):
        hours = slice(day * 24, (day + 1) * 24)
        slot, site = np.nonzero(select[hours])
        if len(slot) == 0:
            continue
        block, pv_names = pv_block(inputs, sites, solar, day)
        parts = [block[slot, site]]
        if inputs.input_config.time_features:
            parts.append(time_features(np.array([day]))[0][slot])
        for p in inputs.active:
            present = inputs.providers[p].present[:, hours][:, :, sites.station_index[p]]
            parts.append(np.stack([1.0 - present[r][slot, site] for r in range(len(ROUNDS))], axis=-1))
        rows.append(np.concatenate(parts, axis=-1).astype(np.float32))
        hs.append(day * 24 + slot)
        idx.append(site)
        curve.append(np.maximum(solar[hours][slot, site, 1].astype(np.float64), 0.0))
        if names is None:
            names = (list(pv_names) + (list(TIME_FEATURES) if inputs.input_config.time_features else [])
                     + [f"missing_{p}_r{r}" for p in inputs.active for r in ROUNDS])
    if not rows:
        raise ValueError("선택한 발전소-시각 행이 없습니다.")
    return np.concatenate(rows), np.concatenate(hs).astype(np.int64), np.concatenate(idx).astype(np.int64), names, np.concatenate(curve)


def training_select(energy: np.ndarray, status: np.ndarray, sunup: np.ndarray, cutoff_h: int) -> tuple[np.ndarray, np.ndarray]:
    """cutoff까지 정답이 있고 해가 떠 있는 발전소-시각. 단일 발전소 query의 야간 규칙은 그 발전소의 sunup이다."""
    view = actual_view(energy, status, cutoff_h)
    select = np.zeros(sunup.shape, dtype=bool)
    select[:len(view.valid)] = view.valid & sunup[:len(view.valid)]
    return select, view.energy


def site_queries(sites, sunup: np.ndarray, days, index, mask: np.ndarray | None = None, energy: np.ndarray | None = None) -> QuerySet:
    """발전소 하나짜리 query 묶음. mask가 없으면 정답 없는 예측용 query이다."""
    days = np.asarray(days, dtype=np.int32)
    index = np.asarray(index, dtype=np.int64)
    count = len(days)
    members = np.zeros((count, len(sites.ids)), dtype=bool)
    members[np.arange(count), index] = True
    hours = days[:, None].astype(np.int64) * 24 + np.arange(24)[None, :]
    daylight = np.asarray(sunup)[hours, index[:, None]]
    mask = np.zeros((count, 24), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    y = np.full((count, 24), np.nan) if energy is None else np.where(mask, np.asarray(energy)[hours, index[:, None]], np.nan)
    return QuerySet(days, np.zeros(count, dtype=np.int8), members, mask, daylight, y, sites.capacity[index].astype(np.float64),
                    np.ones(count, dtype=np.int16), np.ones(count), {"policy": POLICY})


def training_site_queries(sites, energy: np.ndarray, status: np.ndarray, sunup: np.ndarray, cutoff_h: int) -> QuerySet:
    """cutoff까지 정답이 한 시각 이상 있는 발전소-일. cutoff 이후 실적은 `actual_view`가 먼저 지운다."""
    view = actual_view(energy, status, cutoff_h)
    days_total = view.valid.shape[0] // 24
    valid = view.valid[:days_total * 24].reshape(days_total, 24, -1)
    day, site = np.nonzero(valid.any(axis=1))
    return site_queries(sites, sunup, day, site, valid[day, :, site], view.energy)


def primitive_inputs(spec: Spec, data: Data, cfg: dict) -> tuple[dict, InputSet, InputSet]:
    """primitive 설정({"model": MK 버전, "seeds"})의 모델 정의와 두 expert의 입력."""
    model = spec.model(cfg["model"])
    xgb_inputs = resolve_inputs(data.weather, spec.input_config(model["xgb_input"]), spec.excluded_inputs, spec.schema)
    dl_inputs = resolve_inputs(data.weather, spec.dl_input_config(model), spec.excluded_inputs, spec.schema)
    if xgb_inputs.input_config.day_context:
        raise NotImplementedError(f"{cfg['model']}의 xgb_input {model['xgb_input']}은 ctx를 써서 발전소 단위 XGB로 만들 수 없습니다.")
    return model, xgb_inputs, dl_inputs


def primitive_key(spec: Spec, data: Data, cfg: dict, period, xgb_inputs: InputSet, dl_inputs: InputSet) -> str:
    """잔차 cache key. 모델 버전 이름이 아니라 모델 정의와 두 expert의 입력 구성을 넣는다."""
    return digest({"base": data.base.key, "weather": data.weather.key, "model": spec.model(cfg["model"]),
                   "seeds": [int(s) for s in cfg["seeds"]],
                   "inputs": {"xgb": input_key(xgb_inputs.input_config), "dl": input_key(dl_inputs.input_config)},
                   "excluded": {"xgb": xgb_inputs.excluded, "dl": dl_inputs.excluded}, "cutoff_h": period.cutoff_h,
                   "hours": [period.start_h, period.end_h], "hyperparameters": spec.hyperparameters, "quantiles": spec.quantiles.tolist(),
                   "decisions": digest(spec.recorded_decisions), "packages": package_versions(("numpy", "xgboost", "torch")),
                    "versions": versions("queries", "features", "xgb", "dl_prep", "dl", "postprocess", "residuals")})


def primitive_descriptor(spec: Spec, data: Data, period_name: str, cfg: dict) -> dict:
    """학습 없이 period의 residual cache identity를 계산한다."""
    model = spec.model(cfg["model"])
    inputs = [SimpleNamespace(input_config=config, excluded=spec.excluded_inputs)
              for config in (spec.input_config(model["xgb_input"]), spec.dl_input_config(model))]
    return {"kind": "residuals", "key": primitive_key(spec, data, cfg, spec.period(period_name), *inputs)}


def _xgb_median(spec: Spec, data: Data, inputs, period, seeds, slot_weight: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """발전소 단위 XGB 중앙값. slot_weight가 있으면 그룹 XGB처럼 용량 가중에 정산 시간 가중 k를 곱한다."""
    sites, base = data.sites, data.base
    energy, status, sunup, solar = (np.asarray(a) for a in (base.energy, base.status, base.sunup, base.solar))
    select, masked = training_select(energy, status, sunup, period.cutoff_h)
    x, h, i, names, curve_x = site_matrix(inputs, sites, solar, select, 0, period.cutoff_h // 24)
    cap = sites.capacity[i]
    z = masked[h, i] / cap
    curve = PowerCurve().fit(curve_x, z, cap)
    pc = curve.predict(curve_x)
    hours = period.hours
    pred_select = np.zeros(sunup.shape, dtype=bool)
    pred_select[hours] = sunup[hours]
    xp, hp, ip, _, curve_p = site_matrix(inputs, sites, solar, pred_select, period.start_h // 24, period.end_h // 24)
    pcp = curve.predict(curve_p)
    if inputs.input_config.solar_geometry:          # 그룹 XGB와 같이 power curve 값을 열로 더한다
        x = np.column_stack([x, pc]).astype(np.float32)
        xp = np.column_stack([xp, pcp]).astype(np.float32)
        names = names + ["power_curve"]
    weight = cap if slot_weight is None else cap * slot_weight[h % 24]
    raw, details = [], {}
    for seed in seeds:
        log(f"발전소 primitive XGB {period.name} seed={seed} rows={len(x)} features={len(names)}")
        booster, info = fit_xgb(x, z - pc, weight, names, spec.hyperparameters, MEDIAN, seed, int(spec.hyperparameters["xgb_rounds"]))
        raw.append(predict_xgb(booster, xp, names, spec.hyperparameters["xgb_threads"])[:, 0])
        details[str(seed)] = info
    yhat = np.zeros((len(hours), len(sites.ids)), dtype=np.float64)
    yhat[hp - period.start_h, ip] = np.clip(pcp + np.mean(raw, axis=0), 0.0, 1.0) * sites.capacity[ip]
    return yhat, {"rows": int(len(x)), "features": len(names), "seeds": details, "power_curve": curve.state(),
                  "slot_weight": None if slot_weight is None else [float(v) for v in slot_weight]}


def _dl_median(spec: Spec, data: Data, inputs, period, seeds, loss) -> tuple[np.ndarray, dict]:
    sites, base = data.sites, data.base
    energy, status, sunup = (np.asarray(a) for a in (base.energy, base.status, base.sunup))
    train_q = training_site_queries(sites, energy, status, sunup, period.cutoff_h)
    n = len(sites.ids)
    day = np.repeat(period.days, n)
    eval_q = site_queries(sites, sunup, day, np.tile(np.arange(n), len(period.days)))
    keys = {"queries": digest({"base": base.key, "policy": POLICY, "cutoff_h": period.cutoff_h, "hash": train_q.hash}),
            "eval": digest({"policy": POLICY, "days": period.days.tolist(), "hash": eval_q.hash}),
            "decisions": digest(spec.recorded_decisions)}
    log(f"발전소 primitive DL {period.name} train_queries={len(train_q)} eval_queries={len(eval_q)}")
    result = run_dl(data, inputs, train_q, eval_q, None, period.cutoff_h, list(seeds), int(spec.hyperparameters["dl_optimizer_steps"]), keys, loss)
    q = finalize(seed_mean([result["z"][s] for s in seeds]), eval_q.cap, eval_q.daylight)
    median = q[..., int(np.flatnonzero(np.isclose(spec.quantiles, 0.5))[0])]
    yhat = np.zeros((len(period.days), 24, n), dtype=np.float64)
    yhat[eval_q.day - period.days[0], :, np.tile(np.arange(n), len(period.days))] = median
    details = {"train_queries": len(train_q), "eval_queries": len(eval_q), "prep_key": result["prep_key"],
               "models": {str(s): {k: v for k, v in d.items() if k in ("model_key", "lambda", "warmup_state_sha", "seconds")}
                          for s, d in result["details"].items()}}
    return yhat.reshape(-1, n), details


def site_primitive(spec: Spec, data: Data, period_name: str, cfg: dict) -> dict:
    """기간 하나의 발전소 블렌드 중앙값 ŷ와 잔차 e = Y − ŷ(정답 없으면 NaN). cache에 없으면 학습·예측한다."""
    period = spec.period(period_name)
    model, xgb_inputs, dl_inputs = primitive_inputs(spec, data, cfg)
    key = primitive_key(spec, data, cfg, period, xgb_inputs, dl_inputs)
    store = data.store
    if not store.exists("residuals", key):
        started = time.perf_counter()
        seeds = [int(s) for s in cfg["seeds"]]
        slot_weight = settlement_slot_weight() if model["xgb_time_weight"] == "settlement" else None
        yhat_xgb, xgb_info = _xgb_median(spec, data, xgb_inputs, period, seeds, slot_weight)
        loss = spec.loss(model["dl_loss"], float(model["aux_ratio"]), time_weight=model["dl_time_weight"])
        yhat_dl, dl_info = _dl_median(spec, data, dl_inputs, period, seeds, loss)
        weight = float(model["xgb_weight"])
        yhat = weight * yhat_xgb + (1.0 - weight) * yhat_dl
        hours = period.hours
        status, energy = np.asarray(data.base.status), np.asarray(data.base.energy)
        error = np.where(status[hours] == 1, energy[hours] - yhat, np.nan) if period.has_labels else np.full(yhat.shape, np.nan)
        store.write("residuals", key, {"yhat": yhat.astype(np.float32), "yhat_xgb": yhat_xgb.astype(np.float32),
                                            "yhat_dl": yhat_dl.astype(np.float32), "error": error.astype(np.float32), "hours": hours},
                    {"period": period_name, "cutoff_h": period.cutoff_h, "primitive": cfg, "model": model, "xgb_weight": weight, "xgb": xgb_info,
                     "dl": dl_info, "seconds": time.perf_counter() - started})
    arrays, meta = store.load("residuals", key, mmap=False)
    return {"key": key, "kind": "residuals", "period": period_name, "cutoff_h": int(meta["cutoff_h"]),
            "hours": np.asarray(arrays["hours"]), "yhat": np.asarray(arrays["yhat"]), "error": np.asarray(arrays["error"]), "meta": meta}
