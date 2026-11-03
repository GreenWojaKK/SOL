"""실행 config, weather schema, 모델 등록 목록을 읽어 설정 객체를 만든다.

모델 버전(MK)은 계산 그래프의 component 선택을 묶은 정의이다. 등록한 버전의 정의는 바꾸지 않는다.
참조: docs/solarvpp2-design.md#versions.
"""
from __future__ import annotations

import copy
import functools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vppcontract.timeaxis import SEASON_IDS, HolidayCalendar

from .forecast import contexts, decoders, routing
from .store import digest, file_sha, read_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/solarvpp2/runtime.json"
AUX = {"pinball": None, "mae": "mae", "mse": "mse", "ramp": "ramp", "fourier": "fourier", "cosine": "cosine", "day-energy": "day_energy", "settlement-penalty": "settlement"}
MODEL_KEYS = ("sampler", "heldout", "xgb", "dl", "blend", "seeds", "risk", "partition", "recommendation")
SAMPLER_KINDS = ("original", "random_halves", "stratified_halves", "uniform_subsets", "station_subsets")
TIME_WEIGHTS = ("none", "settlement")
# 계절 학습: 정의에 season이 없으면 Global이다. ft는 Global 모델에서 이어 학습, weighted는 계절 대상일 가중, only는 계절 대상일만 학습한다.
SEASON_STRATEGIES = ("ft", "weighted", "only")
SEASON_DEFAULTS = {"weight": 3.0, "ft_lr": 0.0001, "xgb_ft_rounds": 100, "xgb_ft_learning_rate": 0.02}
# 선택 항목: 학습 계절의 앞뒤 확장 개월 수, ft·only의 DL step 예산(scaled: T_s, full: Global 예산), 계절 학습을 적용할 expert.
# 기본값이면 정의에 넣지 않으므로 기존 계절 정의와 key가 그대로다.
SEASON_OPTIONS = {"margin_months": 0, "steps": "scaled", "experts": ["xgb", "dl"]}


@dataclass(frozen=True)
class InputConfig:
    """입력 구성. day_context는 XGB에 하루 단위 예보 집계 열을 더하고, day_state는 DL에 관측소 하루 상태를 둔다."""
    name: str
    input_role: str
    providers: tuple[str, ...]
    disagreement: bool
    revision: bool
    solar_geometry: bool = True
    physics_proxy: bool = False
    time_features: bool = True
    day_context: bool = False
    day_state: bool = False


@dataclass(frozen=True)
class LossConfig:
    """DL loss 구성. time_weight="settlement"는 pinball과 day-energy에 k를 적용한다."""
    name: str
    aux: str | None
    aux_ratio: float
    warmup_steps: int
    calibration_batches: int
    time_weight: str = "none"


def settlement_slot_weight() -> np.ndarray:
    """하루 24개 slot의 정산 가중 k. slot s는 표시 시각 s+1시이다."""
    return np.where((np.arange(24) + 1 >= 11) & (np.arange(24) + 1 <= 15), 3.0, 1.0)


def season_settings(model: dict) -> dict | None:
    """계절 학습 설정(기본값을 채운 사본). 정의에 season이 없으면 None(Global)이다."""
    item = model.get("season")
    return None if item is None else {**SEASON_DEFAULTS, **SEASON_OPTIONS, **item}


def site_aux_ratio(model: dict) -> float:
    """발전소 이용률 보조 head의 세기 비율. 정의에 없으면 0(보조 head 없음)이다. 기본값을 정의에 채워 넣지 않아 기존 정의의 identity가 그대로이다."""
    return float(model["dl"].get("site_aux_ratio", 0.0))


def decoder_settings(model: dict) -> dict | None:
    """pooled 시퀀스 decoder 설정(기본값을 채운 사본). 정의에 dl.decoder가 없거나 BiGRU이면 None(기존 BiGRU 경로)이다."""
    return decoders.resolve(model["dl"].get("decoder"))


def provider_context_settings(model: dict) -> dict | None:
    """provider 표현 context 설정(기본값을 채운 사본). 정의에 dl.provider_context가 없으면 None(context 없음)이다."""
    return contexts.resolve(model["dl"].get("provider_context"))


def plant_routing_settings(model: dict) -> dict | None:
    """집합 pooling 전 provider 표현의 temporal routing 설정(기본값을 채운 사본). 정의에 dl.plant_routing이 없으면 None이다."""
    return routing.resolve_plant(model["dl"].get("plant_routing"))


def group_routing_settings(model: dict) -> dict | None:
    """집합 pooling 뒤 sequence의 temporal routing 설정(기본값을 채운 사본). 정의에 dl.group_routing이 없으면 None이다."""
    return routing.resolve_group(model["dl"].get("group_routing"))


def day_scale_settings(model: dict) -> dict | None:
    """집합의 24시간 decoder 표현으로 하나의 양수 배율을 내는 선택적 head. 없거나 kind=none이면 기존 경로다."""
    item = model["dl"].get("day_scale")
    if item is None:
        return None
    if not isinstance(item, dict):
        raise ValueError("day_scale은 설정 객체입니다.")
    if item.get("kind") == "none":
        return None
    out = {"kind": "bounded-log-gain", "hidden": 32, "max_log_gain": 0.35, **item}
    if set(out) != {"kind", "hidden", "max_log_gain"} or out["kind"] != "bounded-log-gain":
        raise ValueError("day_scale은 kind=bounded-log-gain, hidden, max_log_gain을 담습니다.")
    if type(out["hidden"]) is not int or out["hidden"] < 1:
        raise ValueError("day_scale.hidden은 양의 정수입니다.")
    bound = float(out["max_log_gain"])
    if not np.isfinite(bound) or not 0 < bound <= 1:
        raise ValueError("day_scale.max_log_gain은 0보다 크고 1 이하입니다.")
    out["max_log_gain"] = bound
    return out


def slot_weight(time_weight: str) -> np.ndarray | None:
    """time weight의 24개 slot 가중. "none"이면 None, "settlement"이면 정산 가중 k이다."""
    if time_weight not in TIME_WEIGHTS:
        raise ValueError(f"time weight는 {TIME_WEIGHTS} 중 하나입니다: {time_weight}")
    return None if time_weight == "none" else settlement_slot_weight()


@dataclass
class Spec:
    root: Path
    config: dict
    schema: dict
    registry: dict
    hashes: dict = field(default_factory=dict)

    @property
    def data_dir(self) -> Path:
        return self.root / self.config["paths"]["data_dir"]

    @property
    def artifact_dir(self) -> Path:
        """package 산출물 디렉터리. repo 밖의 상대 경로(`../sol-artifact/solarvpp2`)도 받으므로 기록에는 정규화한 절대 경로가 남는다."""
        return (self.root / self.config["paths"]["artifact_dir"]).resolve()

    @property
    def quantiles(self) -> np.ndarray:
        return np.asarray(self.config["quantiles"], dtype=np.float64)

    @property
    def hyperparameters(self) -> dict:
        return self.config["hyperparameters"]

    @property
    def excluded_inputs(self) -> tuple[str, ...]:
        return tuple(sorted({name for item in self.config["excluded_inputs"] for name in item["inputs"]}))

    @property
    def default_model(self) -> str:
        return self.registry["default"]

    def model(self, name: str | None = None) -> dict:
        name = name or self.default_model
        if name not in self.registry["models"]:
            raise ValueError(f"등록되지 않은 모델 버전입니다: {name} (등록: {', '.join(self.registry['models'])})")
        return dict(self.registry["models"][name])

    def input_config(self, name: str) -> InputConfig:
        if name not in self.config["input_configs"]:
            raise ValueError(f"정의되지 않은 입력 구성입니다: {name}")
        item = self.config["input_configs"][name]
        return InputConfig(name, item["input_role"], tuple(item["providers"]), item["disagreement"], item["revision"],
                       item["solar_geometry"], item["physics_proxy"], item["time_features"], item.get("day_context", False),
                       item.get("day_state", False))

    @functools.cached_property
    def calendar(self) -> HolidayCalendar:
        """휴일 편향 진단에 쓰는 공휴일 목록. solarvpp2는 모델 입력에 쓰지 않는다.

        참조: docs/solarvpp2-design.md#diagnostics.
        """
        return HolidayCalendar(self.root / self.config["paths"]["calendar"])

    def panel(self, name: str) -> dict:
        """validation panel 정의: 평가할 instance, 평가일 묶음([시작, 끝] KST 날짜), 학습에서 함께 빼는 앞뒤 완충 일수.

        참조: docs/solarvpp2-design.md#p60-panel.
        """
        panels = self.config.get("panels") or {}
        if name not in panels:
            raise ValueError(f"정의되지 않은 validation panel입니다: {name} (정의: {', '.join(panels) or '없음'})")
        item = copy.deepcopy(panels[name])
        if (set(item) != {"instance", "buffer_days", "blocks"} or not item["blocks"] or int(item["buffer_days"]) < 0
                or any(len(block) != 2 for block in item["blocks"])):
            raise ValueError(f"validation panel {name}은 instance, buffer_days(0 이상), blocks([시작, 끝] 목록)를 담습니다.")
        return item

    def experiment(self, name: str) -> dict:
        """제출 경로가 없어 MK로 등록하지 않는 실험의 고정 설정(runtime의 experiments).

        참조: docs/transfer-calibration.md.
        """
        experiments = self.config.get("experiments") or {}
        if name not in experiments:
            raise ValueError(f"정의되지 않은 실험입니다: {name} (정의: {', '.join(experiments) or '없음'})")
        item = copy.deepcopy(experiments[name])
        if name == "transfer":
            expected = {"kind", "parent_model", "history_months", "steps", "batch_queries", "target_fraction", "learning_rate", "weight_decay",
                        "gradient_clip", "shrinkage_days"}
            if set(item) != expected or item["kind"] != "fixed-partition-decoder" or item["parent_model"] not in self.registry["models"]:
                raise ValueError("transfer 실험 설정이 잘못되었습니다.")
            if item["history_months"] != 12 or item["batch_queries"] % 2 or item["target_fraction"] != .5:
                raise ValueError("transfer 실험은 최근 12개월·짝수 batch·target/replay 1:1을 사용합니다.")
            if any(not np.isfinite(float(item[k])) or float(item[k]) <= 0
                   for k in ("steps", "batch_queries", "learning_rate", "weight_decay", "gradient_clip", "shrinkage_days")):
                raise ValueError("transfer 실험의 수치 설정은 양수여야 합니다.")
        elif name == "recalibration":
            if (set(item) != {"kind", "shrinkage_days"} or item["kind"] != "seasonal-location-scale-W"
                    or not np.isfinite(float(item["shrinkage_days"])) or item["shrinkage_days"] <= 0):
                raise ValueError("recalibration 실험 설정이 잘못되었습니다.")
        return item

    def loss(self, name: str, aux_ratio: float, smoke: bool = False, time_weight: str = "none") -> LossConfig:
        if name not in AUX:
            raise ValueError(f"알 수 없는 손실 구성입니다: {name}")
        aux = self.config["aux_lambda"]
        warmup = self.config["smoke"]["aux_warmup_steps"] if smoke else aux["warmup_steps"]
        batches = self.config["smoke"]["aux_calibration_batches"] if smoke else aux["calibration_batches"]
        return LossConfig(name, AUX[name], 0.0 if AUX[name] is None else float(aux_ratio), int(warmup), int(batches), time_weight)


def validate_registry(registry: dict, input_configs: dict) -> None:
    """등록 목록 검사: 이름은 MK{번호}, 정의는 MODEL_KEYS를 모두 담고, 값은 구현한 component이다."""
    for name, item in registry["models"].items():
        if not (name.startswith("MK") and name[2:].isdigit()):
            raise ValueError(f"모델 버전 이름은 MK와 번호입니다: {name}")
        if set(item) - {"season"} != set(MODEL_KEYS):
            raise ValueError(f"{name}의 component가 {MODEL_KEYS}(선택: season)와 다릅니다: {sorted(item)}")
        season = season_settings(item)
        if season is not None:
            if season.get("name") not in SEASON_IDS or season.get("strategy") not in SEASON_STRATEGIES:
                raise ValueError(f"{name}의 season은 name({SEASON_IDS})과 strategy({SEASON_STRATEGIES})를 담습니다.")
            if set(season) - {"name", "strategy", *SEASON_DEFAULTS, *SEASON_OPTIONS}:
                raise ValueError(f"{name}의 season에 정의되지 않은 항목이 있습니다: {sorted(season)}")
            experts = season["experts"]
            if (not isinstance(season["margin_months"], int) or not 0 <= season["margin_months"] <= 3 or season["steps"] not in ("scaled", "full")
                    or not experts or not set(experts) <= {"xgb", "dl"} or len(set(experts)) != len(experts)):
                raise ValueError(f"{name}의 season margin_months는 0~3, steps는 scaled·full, experts는 xgb·dl 중 하나 이상입니다.")
            numbers = [float(season[k]) for k in ("weight", "ft_lr", "xgb_ft_learning_rate")]
            if not all(np.isfinite(v) and v > 0 for v in numbers) or int(season["xgb_ft_rounds"]) < 1:
                raise ValueError(f"{name}의 season 가중·학습률은 양수이고 xgb_ft_rounds는 1 이상입니다.")
        for label, sampler in (("sampler", item["sampler"]), ("risk.sampler", item["risk"]["sampler"])):
            if set(sampler) - set(SAMPLER_KINDS) - {"subset_fraction", "seed"}:
                raise ValueError(f"{name}의 {label}에 정의되지 않은 집합 종류가 있습니다.")
        if item["xgb"]["input"] not in input_configs or item["dl"]["input"] not in input_configs or item["dl"]["loss"] not in AUX:
            raise ValueError(f"{name}의 입력·손실이 정의되지 않았습니다.")
        if item["xgb"]["time_weight"] not in TIME_WEIGHTS or item["dl"]["time_weight"] not in TIME_WEIGHTS:
            raise ValueError(f"{name}의 time weight가 정의되지 않았습니다.")
        if not (np.isfinite(site_aux_ratio(item)) and site_aux_ratio(item) >= 0.0):
            raise ValueError(f"{name}의 dl.site_aux_ratio은 유한한 0 이상의 값입니다.")
        for label, settings in (("dl.decoder", decoder_settings), ("dl.provider_context", provider_context_settings),
                                ("dl.day_scale", day_scale_settings), ("dl.plant_routing", plant_routing_settings),
                                ("dl.group_routing", group_routing_settings)):
            try:
                settings(item)
            except ValueError as error:
                raise ValueError(f"{name}의 {label}: {error}") from None
        if item["blend"]["method"] != "weighted-quantile" or not 0.0 <= float(item["blend"]["xgb_weight"]) <= 1.0:
            raise ValueError(f"{name}의 결합은 weighted-quantile이고 xgb_weight는 [0, 1]입니다.")
        held = item["heldout"]
        if held["block_days"] < 1 or held["cycle_blocks"] < 2 or not 0 <= held["offset"] < held["cycle_blocks"]:
            raise ValueError(f"{name}의 held-out 블록 규칙이 잘못되었습니다.")
        risk = item["risk"]
        if risk["objective"] != "tweedie" or not 1.0 < float(risk["power"]) < 2.0:
            raise ValueError(f"{name}의 risk 학습 목적은 1 < p < 2인 Tweedie deviance입니다.")
        part = item["partition"]
        if part["days"] != "heldout" or part["day_weight"] not in ("calendar_month", "uniform") or part["price"] != "month_hour_mean_12m":
            raise ValueError(f"{name}의 배정 목적 component가 정의되지 않았습니다.")
        if item["recommendation"]["baseline"] != "stratified":
            raise ValueError(f"{name}의 기준 배정이 정의되지 않았습니다.")
    if registry["default"] not in registry["models"]:
        raise ValueError(f"default {registry['default']}이 등록 목록에 없습니다.")


OVERRIDES = ("xgb.input", "dl.input", "xgb.time_weight", "dl.time_weight", "dl.loss", "dl.aux_ratio", "dl.site_aux_ratio", "blend.xgb_weight", "seeds",
             "season.name", "season.strategy", "season.weight", "season.ft_lr", "season.xgb_ft_rounds", "season.xgb_ft_learning_rate",
             "season.margin_months", "season.steps", "season.experts",
             "dl.decoder.kind", "dl.decoder.loops", "dl.provider_context.kind",
             "dl.day_scale.kind", "dl.day_scale.hidden", "dl.day_scale.max_log_gain",
             "dl.plant_routing.kind", "dl.plant_routing.hours", "dl.group_routing.kind")
# 모델 정의 안의 component 설정 묶음: 종류(kind)를 바꾸면 그 component의 항목을 새로 채우고, 값은 이 표의 형식으로 바꾼다.
NESTED_OVERRIDES = {"dl.decoder": {"loops": int}, "dl.provider_context": {},
                    "dl.day_scale": {"hidden": int, "max_log_gain": float},
                    "dl.plant_routing": {"hours": int}, "dl.group_routing": {}}
NUMERIC_OVERRIDES = ("dl.aux_ratio", "dl.site_aux_ratio", "blend.xgb_weight", "season.weight", "season.ft_lr", "season.xgb_ft_learning_rate")


def with_overrides(spec: Spec, model_name: str, overrides: dict) -> Spec:
    """등록 정의의 입력 구성, time weight, DL 손실·보조 head 세기, decoder 종류·반복 수, provider context 종류, temporal routing 종류, 결합 비율, seed, 계절 학습
    설정을 바꾼 Spec 사본. 등록 목록 파일은 바꾸지 않고, 실행 기록에 실제 정의와 override를 남긴다. 계절 학습 설정을 처음 주면
    기본값(SEASON_DEFAULTS)을 정의에 채운다. component(decoder, provider context)의 종류를 바꾸면 그 component의 항목만 남긴다."""
    unknown = set(overrides) - set(OVERRIDES)
    if unknown:
        raise ValueError(f"override할 수 있는 항목은 {OVERRIDES}입니다: {sorted(unknown)}")
    registry = copy.deepcopy(spec.registry)
    item = registry["models"][model_name]
    for key, value in overrides.items():
        if key == "seeds":
            seeds = [int(s) for s in value]
            if not seeds or len(set(seeds)) != len(seeds):
                raise ValueError(f"seed override는 겹치지 않는 seed 하나 이상입니다: {value}")
            item["seeds"] = seeds
            continue
        group, _, field_name = key.rpartition(".")
        if group in NESTED_OVERRIDES:
            section, component = group.split(".")
            current = item[section].get(component) or {}
            if field_name == "kind" and current.get("kind") != value:
                current = {}
            convert = NESTED_OVERRIDES[group].get(field_name)
            item[section][component] = {**current, field_name: value if convert is None else convert(value)}
            continue
        section, field_name = key.split(".")
        if section == "season" and "season" not in item:
            item["season"] = dict(SEASON_DEFAULTS)
        if key in NUMERIC_OVERRIDES:
            value = float(value)
        elif key in ("season.xgb_ft_rounds", "season.margin_months"):
            value = int(value)
        elif key == "season.experts":
            value = [str(v) for v in value]
        item[section][field_name] = value
    validate_registry(registry, spec.config["input_configs"])
    out = copy.copy(spec)
    out.registry = registry
    return out


def load_spec(config_path: str | Path = CONFIG, root: Path = ROOT) -> Spec:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = read_json(config_path)
    schema_path = root / config["paths"]["schema"]
    registry = read_json(root / config["paths"]["models"])
    validate_registry(registry, config["input_configs"])
    spec = Spec(root, config, read_json(schema_path), registry)
    spec.hashes = {"config": digest(config), "schema": file_sha(schema_path)}
    if spec.schema["schema_version"] != 1:
        raise ValueError("이 코드는 weather schema version 1을 사용합니다.")
    return spec
