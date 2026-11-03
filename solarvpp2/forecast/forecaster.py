"""집합 forecaster: 결정 cutoff 하나의 XGB·DL 집합 expert를 학습하고, 임의의 (대상일, 집합)에 대해
결합 분위수·제출값과 R 입력용 집합 표현을 낸다.

참조: docs/solarvpp2-design.md#forecaster.

학습은 결정 cutoff 이전 대상일 중 held-out 블록과 학습 제외 대상일(validation panel)이 아닌 날의 sampler query로 한다. 후처리는 expert별 seed 평균 →
용량 곱·[0, C_A] clip·야간 0·정렬 → weighted-quantile 결합 → 정렬 → 0.5 분위수이다. 요청하면 같은 seed의 XGB·DL을 같은 후처리로 결합한
seed별 제출값도 낸다(seed별 채점용, 결합 예측은 seed 평균 경로 그대로이다).

계절 학습(모델 정의의 season)은 Global 학습 query를 먼저 만든 뒤 대상일 계절로 고르거나 가중한다. 입력 전처리 state는 모든 전략이
Global 것을 쓴다. ft는 같은 seed의 Global XGB booster·DL weights(cache)에서 이어 학습하고, only는 계절 query로 처음부터 학습하며,
weighted는 전체 query에 계절 대상일 가중을 준다. 참조: reports/seasonal-training-2026-10-21/plan.md.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, fields

import numpy as np
import torch

from ..config import day_scale_settings, decoder_settings, group_routing_settings, plant_routing_settings, provider_context_settings, season_settings
from ..config import site_aux_ratio as site_aux_ratio_of
from ..config import slot_weight
from ..data import PROVIDERS, Data
from ..features import (
    PowerCurve,
    SetFeatures,
    daylight_of,
    resolve_inputs,
    training_matrix,
)
from vppcontract.partition import minimum_share
from vppcontract.timeaxis import season_mask

from ..sampler import (
    QuerySet,
    actual_view,
    build_queries,
    cohort_mask,
    heldout_mask,
)
from ..store import digest, package_versions, versions
from . import contexts, decoders, routing
from . import dl as dlm
from .xgb import feature_names, fit_xgb, with_power_curve


def finalize(z: np.ndarray, capacity: np.ndarray, daylight: np.ndarray) -> np.ndarray:
    """정규화 분위수에 집합 용량을 곱하고 [0, C_A]로 자른 뒤 야간을 0으로 두고 정렬한다."""
    cap = np.asarray(capacity, dtype=np.float64)[:, None, None]
    kwh = np.clip(np.asarray(z, dtype=np.float64) * cap, 0.0, cap)
    return np.sort(np.where(daylight[..., None], kwh, 0.0), axis=-1)


def blend(first: np.ndarray, second: np.ndarray, weight: float) -> np.ndarray:
    """weighted-quantile: 같은 분위수끼리 weight·first + (1-weight)·second를 결합하고 다시 정렬한다."""
    return np.sort(weight * first + (1.0 - weight) * second, axis=-1)


@dataclass
class Prediction:
    quantiles: np.ndarray       # [B, 24, Q] 결합 분위수(kWh)
    median: np.ndarray          # [B, 24] 제출값 a_θ
    xgb_median: np.ndarray      # [B, 24]
    dl_median: np.ndarray       # [B, 24]
    hidden: np.ndarray | None   # [B, 24, R] 시각별 집합 표현(seed별 표현을 이어 붙인다)
    cap: np.ndarray             # [B]
    n_pv: np.ndarray
    hhi: np.ndarray
    daylight: np.ndarray        # [B, 24]
    station_hhi: np.ndarray | None = None   # [B, 2] WA·WB 관측소별 용량 비율의 제곱합
    seed_median: np.ndarray | None = None   # [B, S, 24] 같은 seed의 XGB·DL 결합 제출값(seed 순서는 Forecaster.seeds)
    xgb_quantiles: np.ndarray | None = None  # [B, 24, Q] expert별 후처리 분위수(kWh)
    dl_quantiles: np.ndarray | None = None
    xgb_raw: np.ndarray | None = None        # [B, 24, Q] seed 평균 후, clip·야간 처리·정렬 전(kWh)
    dl_raw: np.ndarray | None = None
    xgb_seed_quantiles: np.ndarray | None = None  # [B, S, 24, Q]
    dl_seed_quantiles: np.ndarray | None = None
    xgb_seed_raw: np.ndarray | None = None
    dl_seed_raw: np.ndarray | None = None

    @staticmethod
    def concat(parts: list[Prediction]) -> Prediction:
        def join(name):
            values = [getattr(p, name) for p in parts]
            return None if values[0] is None else np.concatenate(values)
        return Prediction(**{f.name: join(f.name) for f in fields(Prediction)})

    def take(self, index: np.ndarray) -> Prediction:
        return Prediction(**{f.name: None if getattr(self, f.name) is None else getattr(self, f.name)[index] for f in fields(Prediction)})


@dataclass
class Aggregates:
    """(대상일, 집합)마다의 용량 가중합. XGB 집합 feature와 DL pooled 값은 이 값에서만 계산한다.

    xgb는 SetFeatures.sums와 같은 키(cap·n_pv·csq·den·num·sq·cosz·sunup·present)의 torch [B, ...],
    dl은 seed → (S⁽¹⁾, S⁽²⁾) torch [B, 24, H], station은 provider → 관측소별 용량 합 numpy [B, S]이다.
    """
    days: np.ndarray
    xgb: dict
    dl: dict
    station: dict


class Forecaster:
    """결정 cutoff의 집합 forecaster. days_needed는 학습 외에 예측할 대상일(평가·test 기간)이다.

    excluded_days는 risk용 held-out 블록과 별도로 학습에서 빼는 대상일이다(validation panel의 평가일과 앞뒤 완충). 학습 query,
    보조 target, 입력 전처리 fit에서 빠지고, risk 학습과 배정 목적의 held-out 블록은 그대로이다. 비어 있으면 key·cache가 그대로이다.
    """

    def __init__(self, data: Data, model_name: str, cutoff_h: int, days_needed: np.ndarray | None = None, smoke: bool = False,
                 device: str | None = None, excluded_days: np.ndarray | None = None):
        self.data, self.spec, self.smoke = data, data.spec, smoke
        self.model_name, self.model = model_name, data.spec.model(model_name)
        if self.model.get("adaptation"):
            raise ValueError("adaptation 모델은 parent checkpoint를 지정한 transfer 명령으로 실행합니다.")
        self.cutoff_h, self.cutoff_day = int(cutoff_h), int(cutoff_h) // 24
        spec, base = self.spec, data.base
        self.energy, self.status, self.sunup, self.solar = (np.asarray(a) for a in (base.energy, base.status, base.sunup, base.solar))
        self.view = actual_view(self.energy, self.status, self.cutoff_h)
        self.cohort = cohort_mask(self.view)
        held = self.model["heldout"]
        first_day = self.cutoff_day - spec.config["smoke"]["train_days"] + 1 if smoke else 0
        days = np.arange(max(first_day, 0), self.cutoff_day + 1)
        self.heldout_days = days[heldout_mask(days, held["block_days"], held["cycle_blocks"], held["offset"])]
        self.excluded_days = np.unique(np.asarray(excluded_days if excluded_days is not None else [], dtype=np.int64))
        self.train_days = np.setdiff1d(days, np.union1d(self.heldout_days, self.excluded_days))
        self.share_low = minimum_share(float(data.sites.capacity.sum()))
        self.global_q = build_queries(self.view, data.sites, self.sunup, self.train_days, self.cohort, self.model["sampler"], "train",
                                      self.share_low)
        if (np.intersect1d(self.global_q.day, np.union1d(self.heldout_days, self.excluded_days)).size
                or self.global_q.day.max() > self.cutoff_day):
            raise RuntimeError("학습 query에 held-out 블록, 학습 제외 대상일, cutoff 이후 대상일이 들어갔습니다.")
        identity = {"base": base.key, "cutoff_h": self.cutoff_h, "first_day": int(days[0]), "heldout": held,
                    "sampler": self.model["sampler"], "share_low": self.share_low, "hash": self.global_q.hash, "versions": versions("sampler")}
        if len(self.excluded_days):
            identity["excluded_days"] = self.excluded_days.tolist()
        self.queries_key = digest(identity)
        # 계절 학습: Global query에서 대상일로 고른다(날짜별 query membership이 Global과 같다). only는 이 query로 학습한다.
        self.train_q = self.global_q
        if self.season is not None:
            self.season_q = self.global_q.subset(np.flatnonzero(self.in_season(self.global_q.day)))
            if self.season["strategy"] == "only" and self.season_applies("xgb"):
                self.train_q = self.season_q
        self.prep_days = np.union1d(days, np.asarray(days_needed if days_needed is not None else [], dtype=np.int64))
        cuda = torch.cuda.is_available()
        self.device = torch.device(device or ("cuda" if cuda else "cpu"))
        self.feature_device = self.device
        self.xgb_inputs = resolve_inputs(data.weather, spec.input_config(self.model["xgb"]["input"]), spec.excluded_inputs, spec.schema)
        self.dl_inputs = resolve_inputs(data.weather, spec.input_config(self.model["dl"]["input"]), spec.excluded_inputs, spec.schema)
        self.features = SetFeatures(self.xgb_inputs, data.sites, self.solar, self.sunup, self.feature_device)
        self.seeds = [int(s) for s in (spec.config["smoke"]["seeds"] if smoke else self.model["seeds"])]
        self.rounds = spec.config["smoke"]["xgb_rounds"] if smoke else spec.hyperparameters["xgb_rounds"]
        self.steps = spec.config["smoke"]["dl_steps"] if smoke else spec.hyperparameters["dl_optimizer_steps"]
        self.loss = spec.loss(self.model["dl"]["loss"], self.model["dl"]["aux_ratio"], smoke, self.model["dl"]["time_weight"])
        self.median_index = int(np.flatnonzero(np.isclose(spec.quantiles, 0.5))[0])
        self.details: dict = {}
        self.boosters, self.curve, self.nets, self.batcher = {}, None, {}, None
        self.fused_cache: dict | None = None

    @property
    def site_aux_ratio(self) -> float:
        """발전소 이용률 보조 head의 세기 비율(모델 정의의 dl.site_aux_ratio, 없으면 0)."""
        return site_aux_ratio_of(self.model)

    @property
    def decoder(self) -> dict | None:
        """pooled 시퀀스 decoder 설정(기본값을 채운 사본). 정의에 없거나 BiGRU이면 None이다."""
        return decoder_settings(self.model)

    @property
    def provider_context(self) -> dict | None:
        """provider 표현 context 설정(기본값을 채운 사본). 정의에 없으면 None이다."""
        return provider_context_settings(self.model)

    @property
    def day_scale(self) -> dict | None:
        """decoder 뒤에 붙는 집합별 하루 배율 head 설정. 정의에 없으면 None이다."""
        return day_scale_settings(self.model)

    @property
    def pv_width(self) -> int:
        """PV 표현의 폭. group routing의 neutral 표현을 이어 붙이면 fusion 폭의 두 배이다."""
        width = self.spec.hyperparameters["dl_fusion_width"]
        return 2 * width if self.group_routing is not None and self.group_routing["context"] == "neutral" else width

    @property
    def plant_routing(self) -> dict | None:
        """집합 pooling 전 provider 표현의 temporal routing 설정(기본값을 채운 사본). 정의에 없으면 None이다."""
        return plant_routing_settings(self.model)

    @property
    def group_routing(self) -> dict | None:
        """집합 pooling 뒤 sequence의 temporal routing 설정(기본값을 채운 사본). 정의에 없으면 None이다."""
        return group_routing_settings(self.model)

    @property
    def season(self) -> dict | None:
        """계절 학습 설정(기본값을 채운 사본). 정의에 season이 없으면 None(Global)이다."""
        return season_settings(self.model)

    def in_season(self, days: np.ndarray) -> np.ndarray:
        """계절 학습에 쓰는 대상일(학습 창): 대상 계절에 margin_months만큼 앞뒤 월을 더한다. 평가 계절과는 별개이다."""
        return season_mask(days, self.season["name"], self.season["margin_months"])

    def season_applies(self, kind: str) -> bool:
        """expert(kind: xgb·dl)에 계절 학습을 적용하는지. 적용하지 않는 expert는 Global 모델·key를 그대로 쓴다."""
        return self.season is not None and kind in self.season["experts"]

    def season_budget(self) -> dict:
        """ft·only의 DL step 예산. N_G·N_s는 daylight target이 있는 Global·계절 학습 query 수(학습 창 기준), T_s = max(1, round(steps·N_s/N_G))이다.
        only의 warm-up은 round(warm-up·T_s/steps)로 줄이되 calibration 뒤 학습 step이 남도록 T_s − 1 이하로 둔다.
        steps = full이면 T_s와 warm-up을 Global 예산으로 둔다."""
        name = self.season["name"]
        used = self.global_q.target_mask.any(axis=1)
        n_g, n_s = int(used.sum()), int((used & self.in_season(self.global_q.day)).sum())
        if n_s == 0:
            raise ValueError(f"{name} 계절의 학습 query가 없어 학습할 수 없습니다(N_s = 0).")
        if self.season["steps"] == "full":
            return {"N_G": n_g, "N_s": n_s, "T_s": self.steps, "warmup": min(self.loss.warmup_steps, self.steps - 1)}
        steps = max(1, int(round(self.steps * n_s / n_g)))
        warmup = min(int(round(self.loss.warmup_steps * steps / self.steps)), steps - 1)
        return {"N_G": n_g, "N_s": n_s, "T_s": steps, "warmup": warmup}

    def _season_key(self, kind: str, seed: int) -> dict:
        """model key에 넣을 계절 학습 설정(kind: xgb·dl). ft는 같은 seed의 Global 모델 key를 parent로 넣는다."""
        season = self.season
        out = {"name": season["name"], "strategy": season["strategy"]}
        if season["margin_months"]:
            out["margin_months"] = int(season["margin_months"])
        if kind == "dl" and season["steps"] != "scaled" and season["strategy"] != "weighted":
            out["steps_budget"] = season["steps"]
        if season["strategy"] == "weighted":
            out["weight"] = float(season["weight"])
        elif kind == "xgb" and season["strategy"] == "ft":
            out.update(rounds=int(season["xgb_ft_rounds"]), learning_rate=float(season["xgb_ft_learning_rate"]),
                       parent=self.xgb_model_key(seed, season=False))
        elif kind == "dl":
            budget = self.season_budget()
            out.update(steps=budget["T_s"])
            if season["strategy"] == "ft":
                out.update(learning_rate=float(season["ft_lr"]), parent=self.dl_model_key(seed, season=False))
            else:
                out.update(warmup=budget["warmup"])
        return out

    # ---- identity ----
    def feature_key(self) -> str:
        """XGB 학습 행렬 key. only는 계절 query로 행렬·power curve를 만들므로 계절을 넣고, 다른 전략은 Global 행렬을 쓴다."""
        payload = {"queries": self.queries_key, "weather": self.data.weather.key, "cell": self.xgb_inputs.input_config.__dict__,
                   "excluded": self.xgb_inputs.excluded, "versions": versions("sampler", "features")}
        season = self.season
        if season is not None and season["strategy"] == "only" and self.season_applies("xgb"):
            payload["season"] = {"name": season["name"], "strategy": "only"}
            if season["margin_months"]:
                payload["season"]["margin_months"] = int(season["margin_months"])
        return digest(payload)

    def prep_key(self) -> str:
        return digest({"queries": self.queries_key, "weather": self.data.weather.key, "cell": self.dl_inputs.input_config.__dict__,
                       "excluded": self.dl_inputs.excluded, "epsilon": self.spec.hyperparameters["zca_epsilon"],
                       "versions": versions("sampler", "features", "dl_prep")})

    def xgb_model_key(self, seed: int, season: bool = True) -> str:
        """season=False이면 같은 정의의 Global key이다(계절 설정은 Global이 아닐 때만 key에 넣는다)."""
        payload = {"features": self.feature_key(), "hyperparameters": {k: v for k, v in self.spec.hyperparameters.items() if k.startswith("xgb")},
                   "rounds": self.rounds, "seed": seed, "quantiles": self.spec.quantiles.tolist(), "time_weight": self.model["xgb"]["time_weight"],
                   "packages": package_versions(("numpy", "xgboost")), "versions": versions("xgb")}
        if season and self.season_applies("xgb"):
            payload["season"] = self._season_key("xgb", seed)
        return digest(payload)

    def dl_model_key(self, seed: int, season: bool = True) -> str:
        """site_aux_ratio은 0보다 클 때만, decoder는 BiGRU가 아닐 때만, provider context·temporal routing은 있을 때만(설정과 구현 version), 계절 설정은 Global이
        아닐 때만 넣는다. 보조 head가 없는 정의(MK3 등), BiGRU·context 없는 정의(MK5 등)와 Global의 key·cache가 그대로이다.
        season=False이면 같은 정의의 Global key이다."""
        payload = {"prep": self.prep_key(), "hyperparameters": {k: v for k, v in self.spec.hyperparameters.items() if k.startswith(("dl_", "zca"))},
                   "steps": self.steps, "seed": seed, "quantiles": self.spec.quantiles.tolist(), "loss": self.loss.__dict__,
                   "packages": package_versions(("numpy", "torch")), "versions": versions("dl")}
        if self.site_aux_ratio > 0:
            payload["site_aux_ratio"] = self.site_aux_ratio
        if self.decoder is not None:
            payload["decoder"] = decoders.identity(self.decoder)
        if self.provider_context is not None:
            payload["provider_context"] = contexts.identity(self.provider_context)
        if self.day_scale is not None:
            payload["day_scale"] = {"config": self.day_scale, "version": dlm.DAY_SCALE_VERSION}
        if self.plant_routing is not None:
            payload["plant_routing"] = routing.plant_identity(self.plant_routing)
        if self.group_routing is not None:
            payload["group_routing"] = routing.group_identity(self.group_routing)
        if season and self.season_applies("dl"):
            payload["season"] = self._season_key("dl", seed)
        return digest(payload)

    def identity(self) -> dict:
        """forecaster를 정하는 key 묶음. risk·partition key의 상위 key로 쓴다."""
        if hasattr(self, "cached_identity"):
            return self.cached_identity
        from ..store import forecaster_payload
        return forecaster_payload(self.model_name, self.model, self.cutoff_h,
                                  [self.xgb_model_key(s) for s in self.seeds], [self.dl_model_key(s) for s in self.seeds],
                                  versions("postprocess"), self.smoke)


    def inference_spec(self) -> dict:
        """Weights와 함께 보존할 입력·구조·후처리 설정. 실행 중인 Spec의 resolved 값을 기록한다."""
        from copy import deepcopy
        keys = ("quantiles", "hyperparameters", "input_configs", "excluded_inputs", "smoke", "aux_lambda", "runtime")
        return {"config": {k: deepcopy(self.spec.config[k]) for k in keys},
                "schema": deepcopy(self.spec.schema), "schema_sha": self.spec.hashes["schema"],
                "base_key": self.data.base.key, "weather_key": self.data.weather.key,
                "artifact_root": str(self.data.store.root.parent), "power_curve": self.curve.state()}

    def load_cached(self, details: dict, identity: dict, power_curve: dict) -> dict:
        """지정된 산출물의 feature·prep·모델 cache만 읽는다. Cache miss는 학습으로 대체하지 않는다."""
        import xgboost as xgb
        from copy import deepcopy

        spec, store = self.spec, self.data.store
        dlm.configure_determinism(spec.config["runtime"]["torch_threads"])
        self.seeds = [int(s) for s in details["xgb"]["seeds"]]
        self.cached_identity = deepcopy(identity)
        self.details = deepcopy(details)
        self.curve = PowerCurve(degree=power_curve["degree"], ridge=power_curve["ridge"])
        self.curve.coef = np.asarray(power_curve["coef"])
        for seed in self.seeds:
            key = details["xgb"]["seeds"][str(seed)]["model_key"]
            booster = xgb.Booster()
            booster.load_model(str(store.path("models", key) / "model.ubj"))
            if self.device.type == "cuda":
                booster.set_param({"device": "cuda"})
            self.boosters[seed] = booster
        states = {}
        for name, value in store.load("prep", details["dl"]["prep_key"], mmap=False)[0].items():
            group, _, field = name.partition("__")
            states.setdefault(group, {})[field] = value
        prep = dlm.build_prep(self.dl_inputs, self.data.sites, self.solar, self.prep_days, states, self._observed())
        self.batcher = dlm.Batcher(prep, self.data.sites, self.device, spec.config["runtime"]["dl_preload_max_gib"])
        active = {p: bool(prep.flags[i]) for i, p in enumerate(PROVIDERS)}
        for seed in self.seeds:
            key = details["dl"]["seeds"][str(seed)]["model_key"]
            net = dlm.build_net(prep, spec.hyperparameters, len(spec.quantiles), active, site_head=self.site_aux_ratio > 0,
                                decoder=self.decoder, provider_context=self.provider_context, day_scale=self.day_scale,
                                plant_routing=self.plant_routing, group_routing=self.group_routing).to(self.device)
            net.load_state_dict(torch.load(store.path("models", key) / "model.pt", map_location=self.device)["state_dict"])
            net.eval()
            self.nets[seed] = net
        return self.details

    # ---- training ----
    def train(self) -> dict:
        self._train_xgb()
        self._train_dl()
        return self.details

    def _train_xgb(self) -> None:
        import xgboost as xgb
        store, spec = self.data.store, self.spec
        key = self.feature_key()
        if not store.exists("features", key):
            started = time.perf_counter()
            print(f"[{time.strftime('%H:%M:%S')}] XGB 집합 feature 행렬 생성 ({len(self.train_q)} query)", flush=True)
            x, index, curve_x = training_matrix(self.features, self.train_q.day, self.train_q.members, self.train_q.target_mask)
            z = self.train_q.z[index[:, 0], index[:, 1]]
            cap = self.train_q.cap[index[:, 0]]
            curve = PowerCurve().fit(curve_x, z, cap)
            x, pc = with_power_curve(self.features, x, curve_x, curve)
            store.write("features", key, {"x": x, "index": index, "z": z, "cap": cap, "pc": pc},
                        {"names": feature_names(self.features), "power_curve": curve.state(), "seconds": time.perf_counter() - started})
        arrays, meta = store.load("features", key, mmap=False)
        self.curve = PowerCurve()
        self.curve.coef = np.asarray(meta["power_curve"]["coef"])
        names = meta["names"]
        weight = arrays["cap"]
        slot = slot_weight(self.model["xgb"]["time_weight"])
        if slot is not None:
            weight = weight * slot[arrays["index"][:, 1]]
        season = self.season if self.season_applies("xgb") else None
        in_season = None if season is None else self.in_season(self.train_q.day[arrays["index"][:, 0]])
        season_record = None
        if season is not None:
            base_weight = weight
            if season["strategy"] == "weighted":
                weight = weight * np.where(in_season, float(season["weight"]), 1.0)
            season_record = {"name": season["name"], "strategy": season["strategy"], "rows": int(len(in_season)),
                             "season_rows": int(in_season.sum()),
                             "season_weight_share": {"unweighted": float(base_weight[in_season].sum() / base_weight.sum()),
                                                     "weighted": float(weight[in_season].sum() / weight.sum())}}
        self.details["xgb"] = {"feature_key": key, "rows": len(arrays["x"]), "features": len(names), "seeds": {}}
        if season_record is not None:
            self.details["xgb"]["season"] = season_record
        for seed in self.seeds:
            model_key = self.xgb_model_key(seed)
            if not store.exists("models", model_key):
                if season is not None and season["strategy"] == "ft":
                    parent_key = self.xgb_model_key(seed, season=False)
                    if not store.exists("models", parent_key):
                        raise ValueError(f"Seasonal-FT의 Global XGB 모델(seed {seed})이 cache에 없습니다. Global을 먼저 실행하세요(설정을 바꾼 Global은 --use-cache로).")
                    parent = xgb.Booster()
                    parent.load_model(str(store.path("models", parent_key) / "model.ubj"))
                    print(f"[{time.strftime('%H:%M:%S')}] XGB 계절 FT seed={seed} rows={int(in_season.sum())} rounds=+{season['xgb_ft_rounds']}", flush=True)
                    booster, details = fit_xgb(arrays["x"][in_season], (arrays["z"] - arrays["pc"])[in_season], weight[in_season], names,
                                               spec.hyperparameters, spec.quantiles, seed, int(season["xgb_ft_rounds"]), init_booster=parent,
                                               learning_rate=float(season["xgb_ft_learning_rate"]))
                    details["parent_model_key"] = parent_key
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] XGB 학습 seed={seed} rows={len(arrays['x'])} rounds={self.rounds}", flush=True)
                    booster, details = fit_xgb(arrays["x"], arrays["z"] - arrays["pc"], weight, names, spec.hyperparameters, spec.quantiles, seed, self.rounds)
                if season is not None:
                    details["season"] = {**self._season_key("xgb", seed), **season_record}
                store.write("models", model_key, None, {"model": "xgb", "seed": seed, "details": details},
                            {"model.ubj": lambda path, b=booster: b.save_model(str(path))})
            booster = xgb.Booster()
            booster.load_model(str(store.path("models", model_key) / "model.ubj"))
            if self.device.type == "cuda":
                booster.set_param({"device": "cuda"})   # 예측만 GPU에서 한다. 학습 설정과 model key는 바뀌지 않는다.
            self.boosters[seed] = booster
            self.details["xgb"]["seeds"][str(seed)] = {"model_key": model_key, **store.load("models", model_key, names=[])[1]["details"]}

    def _observed(self) -> dict | None:
        if not self.dl_inputs.input_config.day_state:
            return None
        weather = self.data.weather
        return {"values": {p: np.asarray(weather.values[f"observed_{p}"]) for p in PROVIDERS},
                "names": {p: weather.channel_names(p, "observed") for p in PROVIDERS}, "cutoff_h": self.cutoff_h,
                "excluded_days": self.not_trained_days}

    def _site(self) -> dict | None:
        """발전소 이용률 보조 정답의 원자료. cutoff 이후·무효 status는 view에서 NaN이고, held-out 블록과 학습 제외 대상일은 뺀다."""
        if self.site_aux_ratio <= 0:
            return None
        return {"energy": self.view.energy, "sunup": self.sunup, "excluded_days": self.not_trained_days}

    @property
    def not_trained_days(self) -> np.ndarray:
        """보조 target에서 빼는 대상일: risk용 held-out 블록 ∪ 학습 제외 대상일."""
        return np.union1d(self.heldout_days, self.excluded_days)

    def _train_dl(self) -> None:
        spec, store = self.spec, self.data.store
        if self.device.type != "cuda" and not self.smoke:
            raise RuntimeError("DL은 CUDA GPU에서 실행합니다. CPU fallback은 쓰지 않습니다.")
        dlm.configure_determinism(spec.config["runtime"]["torch_threads"])
        key = self.prep_key()
        if not store.exists("prep", key):           # 입력 전처리는 모든 전략이 Global query 기준 state를 같이 쓴다
            states = dlm.fit_states(self.dl_inputs, self.data.sites, self.cutoff_h, spec.hyperparameters["zca_epsilon"], self.global_q, self.solar,
                                    self.excluded_days)
            store.write("prep", key, {f"{n}__{k}": np.asarray(v) for n, s in states.items() for k, v in s.items()}, {"fit_end_day": self.cutoff_day})
        states = {}
        for name, value in store.load("prep", key, mmap=False)[0].items():
            group, _, k = name.partition("__")
            states.setdefault(group, {})[k] = np.asarray(value)
        prep = dlm.build_prep(self.dl_inputs, self.data.sites, self.solar, self.prep_days, states, self._observed(), self._site())
        if self.loss.aux == "settlement":
            prep.settlement = dlm.settlement_day_weights(self.data.base.smp, self.data.base.k, prep.days, self.cutoff_h, self.not_trained_days)
        self.batcher = dlm.Batcher(prep, self.data.sites, self.device, spec.config["runtime"]["dl_preload_max_gib"])
        active = {p: bool(prep.flags[i]) for i, p in enumerate(PROVIDERS)}
        self.details["dl"] = {"prep_key": key, "layout": prep.layout, "seeds": {}}
        if self.site_aux_ratio > 0:
            self.details["dl"]["site_aux_ratio"] = self.site_aux_ratio
        season = self.season if self.season_applies("dl") else None
        plan = self._season_dl_plan(prep) if season is not None else None
        if plan is not None:
            self.details["dl"]["season"] = plan["record"]
        for seed in self.seeds:
            model_key = self.dl_model_key(seed)
            if not store.exists("models", model_key):
                queries, steps, kwargs = self.train_q, self.steps, {}
                if plan is not None:
                    queries, steps, kwargs = plan["queries"], plan["steps"], dict(plan["kwargs"])
                    if season["strategy"] == "ft":
                        kwargs.update(self._ft_parent(seed))
                print(f"[{time.strftime('%H:%M:%S')}] DL 학습 seed={seed} steps={steps} queries={len(queries)}"
                      + (f" ({season['name']}·{season['strategy']})" if season is not None else ""), flush=True)
                state, details = dlm.train_dl(prep, self.batcher, queries, spec.hyperparameters, spec.quantiles, seed, steps, self.loss,
                                              site_aux_ratio=self.site_aux_ratio, decoder=self.decoder, provider_context=self.provider_context,
                                              day_scale=self.day_scale, plant_routing=self.plant_routing, group_routing=self.group_routing,
                                              **kwargs)
                if plan is not None:
                    details["season"] = {**self._season_key("dl", seed), **plan["record"]}
                store.write("models", model_key, None, {"model": "dl", "seed": seed, "details": details},
                            {"model.pt": lambda path, s=state, sd=seed: torch.save({"state_dict": s, "layout": prep.layout, "seed": sd}, path)})
            net = dlm.build_net(prep, spec.hyperparameters, len(spec.quantiles), active, site_head=self.site_aux_ratio > 0,
                                decoder=self.decoder, provider_context=self.provider_context, day_scale=self.day_scale,
                                plant_routing=self.plant_routing, group_routing=self.group_routing).to(self.device)
            net.load_state_dict(torch.load(store.path("models", model_key) / "model.pt", map_location=self.device)["state_dict"])
            net.eval()
            self.nets[seed] = net
            self.details["dl"]["seeds"][str(seed)] = {"model_key": model_key, **store.load("models", model_key, names=[])[1]["details"]}

    def _season_dl_plan(self, prep) -> dict:
        """계절 DL 학습의 query·step·train_dl 인자와 기록(N_G·N_s·T_s, 계절 표본 가중치 비중)."""
        season = self.season
        name, strategy = season["name"], season["strategy"]
        used = self.global_q.subset(np.flatnonzero(self.global_q.target_mask.any(axis=1)))
        in_q = self.in_season(used.day)
        weight = float(season["weight"]) if strategy == "weighted" else 1.0
        a_q = np.where(in_q, weight, 1.0)
        cells = used.target_mask.sum(axis=1) * used.cap
        train_days = np.unique(used.day)
        in_d = self.in_season(train_days)
        a_d = np.where(in_d, weight, 1.0)

        def share(values, mask, a):
            return {"unweighted": float(values[mask].sum() / values.sum()), "weighted": float((values * a)[mask].sum() / (values * a).sum())}
        record = {"name": name, "strategy": strategy,
                  "sample_weight_share": {"query_cell": share(cells, in_q, a_q), "query": share(used.cap, in_q, a_q),
                                          "unique_day": share(np.ones(len(train_days)), in_d, a_d)}}
        if strategy == "weighted":
            day_weight = np.where(self.in_season(prep.days), weight, 1.0)
            return {"queries": self.global_q, "steps": self.steps, "kwargs": {"day_weight": day_weight}, "record": record}
        budget = self.season_budget()
        record.update(budget)
        if strategy == "only":
            return {"queries": self.season_q, "steps": budget["T_s"], "kwargs": {"warmup_steps": budget["warmup"]}, "record": record}
        return {"queries": self.season_q, "steps": budget["T_s"], "kwargs": {"learning_rate": float(season["ft_lr"])}, "record": record}

    def _ft_parent(self, seed: int) -> dict:
        """Seasonal-FT의 시작 weights와 고정 λ·loss_scales: 같은 seed의 Global DL 모델(cache)과 그 학습 기록."""
        store = self.data.store
        parent_key = self.dl_model_key(seed, season=False)
        if not store.exists("models", parent_key):
            raise ValueError(f"Seasonal-FT의 Global DL 모델(seed {seed})이 cache에 없습니다. Global을 먼저 실행하세요(설정을 바꾼 Global은 --use-cache로).")
        parent = store.load("models", parent_key, names=[])[1]["details"]
        state = torch.load(store.path("models", parent_key) / "model.pt", map_location="cpu")["state_dict"]
        fixed = {"lambda": parent.get("lambda"), "lambda_site": (parent.get("site_aux") or {}).get("lambda"), "loss_scales": parent["loss_scales"]}
        return {"init_state": state, "fixed": fixed}

    # ---- prediction ----
    def enable_fused_cache(self) -> None:
        """대상일별 DL PV 표현을 seed마다 보관한다(같은 대상일을 반복해 예측할 때)."""
        self.fused_cache = {seed: {} for seed in self.seeds}

    def release_caches(self) -> None:
        self.fused_cache = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def fused(self, seed: int, days: np.ndarray, chunk_days: int = 16) -> torch.Tensor:
        """대상일별 DL PV 표현 [D, PV, 24, H](float32, device). 집합과 무관하다."""
        net = self.nets[seed]
        net.eval()
        days = np.asarray(days, dtype=np.int64)
        out = []
        for start in range(0, len(days), chunk_days):
            x, extras = self.batcher.day_inputs(self.batcher.prep.position(days[start:start + chunk_days]))
            out.append(net.fuse(x, extras))
        return torch.cat(out)

    def _fused_days(self, seed: int, days: np.ndarray) -> torch.Tensor:
        cache = None if self.fused_cache is None else self.fused_cache[seed]
        if cache is None:
            return self.fused(seed, days)
        missing = np.array([d for d in days if int(d) not in cache], dtype=np.int64)
        if len(missing):
            for d, value in zip(missing, self.fused(seed, missing)):
                cache[int(d)] = value
        return torch.stack([cache[int(d)] for d in days])

    @torch.no_grad()
    def aggregate(self, days: np.ndarray, members: np.ndarray) -> Aggregates:
        """(대상일, membership)마다 XGB·DL·관측소의 용량 가중합을 계산한다."""
        days, members = np.asarray(days, dtype=np.int64), np.asarray(members, dtype=bool)
        unique = np.unique(days)
        xgb_sums, dl_sums = None, {seed: [None, None] for seed in self.seeds}
        fused = {seed: self._fused_days(seed, unique) for seed in self.seeds}
        capacity = torch.as_tensor(self.data.sites.capacity, dtype=torch.float64, device=self.device)
        for u, d in enumerate(unique):
            rows = np.flatnonzero(days == d)
            part = self.features.sums(self.features.block(int(d)), members[rows])
            if xgb_sums is None:
                xgb_sums = {k: None if v is None else torch.zeros((len(days), *v.shape[1:]), dtype=v.dtype, device=v.device)
                            for k, v in part.items()}
            index = torch.as_tensor(rows, device=self.device)
            for k, v in part.items():
                if v is not None:
                    xgb_sums[k][index] = v
            weights = torch.as_tensor(members[rows], dtype=torch.float64, device=self.device) * capacity[None, :]
            for seed in self.seeds:
                s1, s2 = dlm.pv_sums(fused[seed][u], weights)
                if dl_sums[seed][0] is None:
                    dl_sums[seed] = [torch.zeros((len(days), *s1.shape[1:]), dtype=s1.dtype, device=s1.device) for _ in range(2)]
                dl_sums[seed][0][index], dl_sums[seed][1][index] = s1, s2
        cap_matrix = members * self.data.sites.capacity[None, :]
        station = {p: cap_matrix @ np.eye(len(self.data.sites.stations[p]))[self.data.sites.station_index[p]] for p in PROVIDERS}
        return Aggregates(days, xgb_sums, {seed: tuple(v) for seed, v in dl_sums.items()}, station)

    @torch.no_grad()
    def predict_aggregates(self, agg: Aggregates, hidden: bool = False, per_seed: bool = False,
                           expert_quantiles: bool = False) -> Prediction:
        """용량 가중합에서 결합 분위수·제출값과, hidden=True이면 R 입력용 집합 표현, per_seed=True이면 seed별 결합 제출값."""
        days, sums = agg.days, agg.xgb
        cap = sums["cap"].cpu().numpy()
        n_pv = sums["n_pv"].cpu().numpy()
        hhi = (sums["csq"] / (sums["cap"] * sums["cap"])).cpu().numpy()
        daylight = daylight_of(sums["sunup"]).cpu().numpy()
        rows, curve_x = self.features.from_sums(sums, self.features.day_columns(days))
        rows, pc = with_power_curve(self.features, rows, curve_x, self.curve)
        xgb_z = self._xgb_from_rows(rows, pc, daylight)
        query = self.batcher.query(cap, n_pv, hhi)
        dl_z, h = {}, []
        for seed in self.seeds:
            mean, std = dlm.pooled_from_sums(*agg.dl[seed], sums["cap"])
            q, representation = self.nets[seed].decode(mean.float(), std.float(), query)
            dl_z[seed] = q.float().cpu().numpy()
            if hidden:
                h.append(representation.float().cpu().numpy())
        q_xgb = finalize(np.mean([xgb_z[s] for s in self.seeds], axis=0), cap, daylight)
        q_dl = finalize(np.mean([dl_z[s] for s in self.seeds], axis=0), cap, daylight)
        weight = float(self.model["blend"]["xgb_weight"])
        q = blend(q_xgb, q_dl, weight)
        i = self.median_index
        station_hhi = np.stack([((agg.station[p] / cap[:, None]) ** 2).sum(axis=1) for p in PROVIDERS], axis=1)
        seed_median = None
        if per_seed:          # 같은 seed끼리: expert별 후처리 → 결합 → 정렬 → 0.5 분위수
            seed_median = np.stack([blend(finalize(xgb_z[s], cap, daylight), finalize(dl_z[s], cap, daylight), weight)[..., i]
                                    for s in self.seeds], axis=1)
        extra = {}
        if expert_quantiles:
            for name, values, final in (("xgb", xgb_z, q_xgb), ("dl", dl_z, q_dl)):
                extra[name + "_quantiles"] = final
                extra[name + "_raw"] = np.mean([values[s] for s in self.seeds], axis=0) * cap[:, None, None]
                extra[name + "_seed_raw"] = np.stack([values[s] for s in self.seeds], axis=1) * cap[:, None, None, None]
                extra[name + "_seed_quantiles"] = np.stack([finalize(values[s], cap, daylight) for s in self.seeds], axis=1)
        return Prediction(q, q[..., i], q_xgb[..., i], q_dl[..., i], np.concatenate(h, axis=-1) if hidden else None,
                          cap, n_pv, hhi, daylight, station_hhi, seed_median, **extra)

    def _xgb_from_rows(self, rows: np.ndarray, pc: np.ndarray, daylight: np.ndarray) -> dict:
        """XGB 정규화 분위수 [B, 24, Q](seed별). 야간 칸은 후처리에서 0이 되므로 주간 칸만 예측한다. 행렬 하나를 seed가 같이 쓴다."""
        import xgboost as xgb
        out = {seed: np.zeros((*daylight.shape, len(self.spec.quantiles)), dtype=np.float32) for seed in self.seeds}
        q, s = np.nonzero(daylight)
        if len(q) == 0:
            return out
        matrix = xgb.DMatrix(rows[q, s], feature_names=feature_names(self.features), nthread=self.spec.hyperparameters["xgb_threads"])
        for seed, booster in self.boosters.items():
            out[seed][q, s] = booster.predict(matrix).reshape(len(q), -1) + pc[q, s][:, None]
        return out

    def predict(self, days: np.ndarray, members: np.ndarray, hidden: bool = False, chunk_days: int = 16, per_seed: bool = False,
                expert_quantiles: bool = False) -> Prediction:
        """(대상일, membership)마다 결합 분위수·제출값과, hidden=True이면 R 입력용 집합 표현, per_seed=True이면 seed별 결합 제출값.
        대상일 묶음마다 가중합을 계산한다."""
        days, members = np.asarray(days, dtype=np.int64), np.asarray(members, dtype=bool)
        order = np.argsort(days, kind="stable")
        unique = np.unique(days)
        parts = []
        for start in range(0, len(unique), chunk_days):
            rows = order[np.isin(days[order], unique[start:start + chunk_days])]
            parts.append((rows, self.predict_aggregates(self.aggregate(days[rows], members[rows]), hidden, per_seed,
                                                       expert_quantiles=expert_quantiles)))
        merged = Prediction.concat([part for _, part in parts])
        position = np.empty(len(days), dtype=np.int64)
        position[np.concatenate([rows for rows, _ in parts])] = np.arange(len(days))
        return merged.take(position)


def queries_of(forecaster: Forecaster, days: np.ndarray, settings: dict, stream: str) -> QuerySet:
    """forecaster의 cutoff·cohort·편입 규칙으로 지정한 대상일의 query를 만든다(risk 학습용 held-out 블록 query 등)."""
    return build_queries(forecaster.view, forecaster.data.sites, forecaster.sunup, days, forecaster.cohort, settings, stream,
                         forecaster.share_low)
