"""XGB 집합 expert: 19분위수 모델. target은 집합 정규화 발전량에서 집합 power curve를 뺀 잔차이다.

참조: docs/solarvpp2-design.md#xgb.
"""
from __future__ import annotations

import time

import numpy as np
import xgboost as xgb

from ..features import PowerCurve, SetFeatures


def xgb_params(hyperparameters: dict, quantiles: np.ndarray, seed: int) -> dict:
    return {"objective": "reg:quantileerror", "quantile_alpha": np.asarray(quantiles), "tree_method": "hist",
            "max_depth": hyperparameters["xgb_max_depth"], "eta": hyperparameters["xgb_learning_rate"], "subsample": hyperparameters["xgb_subsample"],
            "colsample_bytree": hyperparameters["xgb_colsample_bytree"], "min_child_weight": hyperparameters["xgb_min_child_weight"],
            "reg_lambda": hyperparameters["xgb_reg_lambda"], "nthread": hyperparameters["xgb_threads"], "device": hyperparameters["xgb_device"], "seed": seed}


def normalized_weights(weight: np.ndarray) -> np.ndarray:
    """용량 비례를 유지하면서 min_child_weight·lambda의 의미를 보존하도록 평균 1로 맞춘다."""
    weight = np.asarray(weight, dtype=np.float64)
    return weight / weight.mean()


def fit_xgb(x: np.ndarray, residual: np.ndarray, weight: np.ndarray, names: list[str], hyperparameters: dict, quantiles: np.ndarray,
            seed: int, rounds: int, init_booster: xgb.Booster | None = None, learning_rate: float | None = None) -> tuple[xgb.Booster, dict]:
    """init_booster를 주면 그 모델에 tree를 rounds만큼 더한다(training continuation). learning_rate는 이번 학습의 학습률이다."""
    started = time.perf_counter()
    data = xgb.DMatrix(x, label=residual, weight=normalized_weights(weight), feature_names=names, nthread=hyperparameters["xgb_threads"])
    params = xgb_params(hyperparameters, quantiles, seed)
    if learning_rate is not None:
        params["eta"] = float(learning_rate)
    booster = xgb.train(params, data, num_boost_round=rounds, xgb_model=init_booster)
    details = {"rows": len(x), "features": int(x.shape[1]), "rounds": rounds,
               "weight_normalizer": float(np.mean(weight)), "seconds": time.perf_counter() - started}
    if init_booster is not None or learning_rate is not None:
        details.update({"initial_rounds": 0 if init_booster is None else int(init_booster.num_boosted_rounds()),
                        "total_rounds": int(booster.num_boosted_rounds()), "learning_rate": float(params["eta"])})
    return booster, details


def predict_xgb(booster: xgb.Booster, x: np.ndarray, names: list[str], threads: int) -> np.ndarray:
    out = booster.predict(xgb.DMatrix(x, feature_names=names, nthread=threads))
    return np.asarray(out, dtype=np.float32).reshape(len(x), -1)


def with_power_curve(features: SetFeatures, rows: np.ndarray, curve_x: np.ndarray, curve: PowerCurve) -> tuple[np.ndarray, np.ndarray]:
    """태양 위치를 쓰는 입력 구성이면 power curve 값을 열로 붙인다. 반환은 (행렬, power curve 값)이다."""
    pc = curve.predict(curve_x)
    if features.inputs.input_config.solar_geometry:
        rows = np.concatenate([rows, pc.reshape(*rows.shape[:-1], 1).astype(np.float32)], axis=-1)
    return rows, pc


def predict_sets(boosters: dict, features: SetFeatures, curve: PowerCurve, block: dict, members: np.ndarray, threads: int) -> dict:
    """대상일 하루, membership [G, PV]의 seed별 정규화 분위수 z [G, 24, Q]."""
    rows, curve_x = features.rows(block, members)
    rows, pc = with_power_curve(features, rows, curve_x, curve)
    names = feature_names(features)
    flat = rows.reshape(-1, rows.shape[-1])
    return {seed: (predict_xgb(b, flat, names, threads) + pc.reshape(-1)[:, None]).reshape(len(members), 24, -1)
            for seed, b in boosters.items()}


def feature_names(features: SetFeatures) -> list[str]:
    return features.names + (["power_curve"] if features.inputs.input_config.solar_geometry else [])
