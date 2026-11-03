"""protocol·schema·evidence와 실행 config를 읽어 실험 설정 객체를 만든다."""
from __future__ import annotations

import functools
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

from vppcontract.timeaxis import FOLD_PERIODS, SEASON_IDS, HolidayCalendar
from vppcontract.timeaxis import PERIODS as CONTRACT_PERIODS

from .data import TZ, to_h
from .store import digest, file_sha, read_json

ROOT = Path(__file__).resolve().parents[1]
PERIODS = ("D", "V", "H", "B", "test")
# robust test용 rolling-origin fold: V·H처럼 평가 시작 전날 17시까지 학습하고 다음 12개월을 평가하며, cutoff만 3개월씩 다르다.
# 이름 → 평가 시작일. 기간 정의는 공통 계약층(vppcontract.timeaxis)에 두고, V(2023-06)와 H(2024-06) 사이를 채운다.
FOLDS = {name: CONTRACT_PERIODS[name].start[:10] for name in FOLD_PERIODS if name not in PERIODS}
# fold 배정 탐색의 추정 기간: fold cutoff 직전 12개월을 그보다 1년 앞선 cutoff의 모델로 예측한 OOT 기간이다. fold 판정에는 쓰지 않는다.
PARTITION_ESTIMATION = {"fold-2209": "2022-09-01", "fold-2212": "2022-12-01", "fold-2303": "2023-03-01"}


def fold_period(name: str, start_day: str, role: str = "robust_fold", exposure: str | None = "previously_exposed") -> "Period":
    """평가 시작일 01시부터 12개월 뒤 00시까지를 평가하고, 시작 전날 17시를 학습 cutoff로 두는 기간."""
    start = pd.Timestamp(f"{start_day} 01:00").tz_localize(TZ)
    end = start + pd.DateOffset(years=1) - pd.Timedelta(hours=1)
    return Period(name, start, end, start - pd.Timedelta(hours=8), role, int((end - start) / pd.Timedelta(hours=1)) + 1, exposure)


@dataclass(frozen=True)
class Period:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    fit_cutoff: pd.Timestamp
    role: str
    expected_hours: int
    exposure_status: str | None

    @property
    def start_h(self) -> int:
        return to_h(self.start)

    @property
    def end_h(self) -> int:
        return to_h(self.end)

    @property
    def cutoff_h(self) -> int:
        return to_h(self.fit_cutoff)

    @property
    def hours(self) -> np.ndarray:
        return np.arange(self.start_h, self.end_h + 1)

    @property
    def days(self) -> np.ndarray:
        return np.arange(self.start_h // 24, self.end_h // 24 + 1)

    @property
    def has_labels(self) -> bool:
        return self.name != "test"


_DAYSTATE = "forecast-physics-daystate"
# protocol 기준 구성 위에 항목만 바꾼 입력 구성: 이름 → (기준 구성, 바꾸는 항목)
DERIVED_INPUTS = {
    # temporal routing: time shift, time shift와 그룹 nearby hours(neutral 맥락), nearby hours와 gate
    f"{_DAYSTATE}-time-shift": (_DAYSTATE, {"time_shift": "region+weather", "shift_scales": "0.5,1,2"}),
    f"{_DAYSTATE}-time-shift-group-nearby-hours": (_DAYSTATE, {"time_shift": "region+weather",
                                                                           "shift_scales": "0.5,1,2", "group_nearby_hours": 3,
                                                                           "group_context": "neutral"}),
    f"{_DAYSTATE}-nearby-hours-3h": (_DAYSTATE, {"nearby_hours": 3, "nearby_gate": True}),
    f"{_DAYSTATE}-nearby-hours-1h": (_DAYSTATE, {"nearby_hours": 1, "nearby_gate": True}),
    f"{_DAYSTATE}-nearby-hours-2h": (_DAYSTATE, {"nearby_hours": 2, "nearby_gate": True}),
}
INPUTS = ("forecast-raw", "forecast-solar", "forecast-solar-diff", "forecast-solar-physics", "forecast-physics", "forecast-physics-daycontext", "forecast-physics-daystate") + tuple(DERIVED_INPUTS)
GRID_LOSSES = ("pinball", "mae", "mse", "ramp", "fourier", "cosine")
LOSSES = GRID_LOSSES + ("day-energy",)
AUX = {"pinball": None, "mae": "mae", "mse": "mse", "ramp": "ramp", "fourier": "fourier", "cosine": "cosine", "day-energy": "day_energy"}


@dataclass(frozen=True)
class InputConfig:
    """입력 구성. day_context는 XGB에 하루 단위 예보 집계 열을 더하고, day_state는 DL에 관측소 하루 상태 Z_d를 둔다.
    site_aux_ratio > 0이면 DL의 PV 표현에 발전소 이용률 보조 head를 두고, warm-up 뒤 gradient 비율로 세기를 정한다. 이 값은 입력 이름에
    넣지 않고 모델 정의의 site_aux_ratio에서 읽는다(Spec.dl_input_config).
    time_shift는 group pooling 전 provider 표현의 temporal routing이다. 읽을 시각을 날씨 맥락("weather"), 관측소별 prior("region"),
    또는 둘의 합("region+weather")으로 정한다. shift_scales는 time shift의 시간 범위(참조점 간격, 시간)를 쉼표로 나열한다.
    nearby_hours > 0이면 group pooling 전 provider 표현에 ±nearby_hours시간 창의 residual sliding window attention을 두고,
    group_nearby_hours > 0이면 group pooling 뒤 sequence에 ±group_nearby_hours시간 창의 residual sliding window attention을 둔다.
    nearby_gate이면 window attention에서 날씨 맥락으로 정한 gate(0~1)가 읽어 온 값의 몫을 시각마다 조절한다.
    group_context가 neutral이면 그룹 window attention의 맥락·key를 모든 관측소에 같은 평균 station embedding을 넣고 정적 정보를
    0으로 둔 표현의 그룹 pooling으로 만들고, 읽어 오는 값은 원래 pooled sequence로 둔다.

    참조: docs/solarvpp-reference.md#features.
    """
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
    site_aux_ratio: float = 0.0
    time_shift: str = ""
    shift_scales: str = "1"
    nearby_hours: int = 0
    group_nearby_hours: int = 0
    nearby_gate: bool = False
    group_context: str = "pooled"


EXTENSIONS = ("site_aux_ratio", "time_shift", "shift_scales", "nearby_hours", "group_nearby_hours", "nearby_gate", "group_context")


def input_key(input_config) -> dict:
    """cache key·run identity에 넣는 입력 구성. 기본값인 확장 항목(EXTENSIONS)은 빼서 기존 구성의 key를 유지한다."""
    defaults = {f.name: f.default for f in fields(InputConfig) if f.name in EXTENSIONS}
    return {k: v for k, v in input_config.__dict__.items() if not (k in defaults and v == defaults[k])}


@dataclass(frozen=True)
class LossConfig:
    """DL loss 구성. time_weight="settlement"는 pinball과 day-energy에 k를 적용한다. XGB weight는 experiment에서 별도로 선택한다."""
    name: str
    aux: str | None
    aux_ratio: float
    warmup_steps: int
    calibration_batches: int
    time_weight: str = "none"

    @property
    def cell_suffix(self) -> str:
        base = self.name if self.aux is None else f"{self.name}-aux{self.aux_ratio:g}"
        return base + ("-settlement" if self.time_weight == "settlement" else "")


# 출력제어 후처리 module(forecast.curtailment)의 설정 항목. 기본값은 runtime config의 curtailment이다.
CURTAILMENT_KEYS = ("half_life_days", "clear_cloud_pct", "reference_quantile", "cut_ratio", "shoulder_ratio", "min_plants", "min_cut_kw", "ridge",
                    "forecast_clearness", "min_forecast_share", "prior_strength")


def curtailment_settings(value: dict) -> dict:
    """출력제어 module 설정을 검사해 정해진 항목과 형식으로 돌려준다."""
    if set(value) != set(CURTAILMENT_KEYS):
        raise ValueError(f"출력제어 module 설정 항목은 {CURTAILMENT_KEYS}입니다: {sorted(value)}")
    out = {k: (int(value[k]) if k == "min_plants" else float(value[k])) for k in CURTAILMENT_KEYS}
    if (not all(np.isfinite(v) for v in out.values()) or out["half_life_days"] <= 0 or out["ridge"] < 0 or out["min_plants"] < 1
            or out["prior_strength"] < 0):
        raise ValueError(f"출력제어 module 설정 값이 범위를 벗어났습니다: {out}")
    if not (0 < out["reference_quantile"] <= 1 and 0 < out["cut_ratio"] < 1 and 0 < out["shoulder_ratio"] and 0 < out["clear_cloud_pct"] <= 100
            and 0 <= out["forecast_clearness"] <= 1 and 0 < out["min_forecast_share"] < 1):
        raise ValueError(f"출력제어 module의 비율 설정이 범위를 벗어났습니다: {out}")
    return out


# 계절 학습: Global 학습 query를 만든 뒤 대상일 계절로 이어 학습(ft)·가중(weighted)·선택(only)한다.
# 참조: solarvpp2 산출물 reports/seasonal-training-2026-10-21/plan.md.
SEASON_STRATEGIES = ("ft", "weighted", "only")
SEASON_DEFAULTS = {"weight": 3.0, "ft_lr": 0.0001, "xgb_ft_rounds": 100, "xgb_ft_learning_rate": 0.02}
# 선택 항목: 학습 계절의 앞뒤 확장 개월 수, ft·only의 DL step 예산(scaled: T_s, full: Global 예산), 계절 학습을 적용할 expert.
# 기본값이면 설정에 넣지 않으므로 기존 계절 설정의 run identity와 key가 그대로다.
SEASON_OPTIONS = {"margin_months": 0, "steps": "scaled", "experts": ["xgb", "dl"]}


def season_settings(name: str, strategy: str, **overrides) -> dict:
    """계절 학습 설정을 검사해 기본값(SEASON_DEFAULTS)을 채운 사본으로 돌려준다. 값이 None인 override는 기본값을 쓴다.
    선택 항목(SEASON_OPTIONS)은 기본값과 다를 때만 넣는다."""
    if name not in SEASON_IDS or strategy not in SEASON_STRATEGIES:
        raise ValueError(f"계절 학습은 계절 {SEASON_IDS}과 방식 {SEASON_STRATEGIES}을 정합니다: {name}, {strategy}")
    if set(overrides) - set(SEASON_DEFAULTS) - set(SEASON_OPTIONS):
        raise ValueError(f"정의되지 않은 계절 학습 설정입니다: {sorted(set(overrides) - set(SEASON_DEFAULTS) - set(SEASON_OPTIONS))}")
    options = {k: overrides.pop(k) for k in list(overrides) if k in SEASON_OPTIONS}
    value = {**SEASON_DEFAULTS, **{k: v for k, v in overrides.items() if v is not None}}
    out = {"name": name, "strategy": strategy, "weight": float(value["weight"]), "ft_lr": float(value["ft_lr"]),
           "xgb_ft_rounds": int(value["xgb_ft_rounds"]), "xgb_ft_learning_rate": float(value["xgb_ft_learning_rate"])}
    if not all(np.isfinite(out[k]) and out[k] > 0 for k in ("weight", "ft_lr", "xgb_ft_learning_rate")) or out["xgb_ft_rounds"] < 1:
        raise ValueError(f"계절 가중·학습률은 양수이고 xgb_ft_rounds는 1 이상입니다: {out}")
    for key, given in options.items():
        if given is None:
            continue
        if key == "margin_months":
            given = int(given)
            if not 0 <= given <= 3:
                raise ValueError(f"계절 학습의 margin_months는 0~3입니다: {given}")
        elif key == "steps" and given not in ("scaled", "full"):
            raise ValueError(f"계절 학습의 steps는 scaled·full입니다: {given}")
        elif key == "experts":
            if not given or set(given) - {"xgb", "dl"}:
                raise ValueError(f"계절 학습의 experts는 xgb·dl 중 하나 이상입니다: {given}")
            given = [e for e in ("xgb", "dl") if e in given]
        if given != SEASON_OPTIONS[key]:
            out[key] = given
    return out


def settlement_slot_weight() -> np.ndarray:
    """하루 24개 slot의 정산 가중 k. slot s는 표시 시각 s+1시이고, 표시 시각의 시가 11~15이면 3, 그 밖은 1이다."""
    return np.where((np.arange(24) + 1 >= 11) & (np.arange(24) + 1 <= 15), 3.0, 1.0)


@dataclass
class Spec:
    root: Path
    config: dict
    protocol: dict
    schema: dict
    evidence: dict
    hashes: dict = field(default_factory=dict)
    registry: dict = field(default_factory=dict)
    settings_source: dict = field(default_factory=lambda: {"kind": "files"})

    @property
    def data_dir(self) -> Path:
        return self.root / self.config["paths"]["data_dir"]

    @property
    def artifact_dir(self) -> Path:
        """package 산출물 디렉터리. repo 밖의 상대 경로(`../sol-artifact/solarvpp`)도 받으므로 기록에는 정규화한 절대 경로가 남는다."""
        return (self.root / self.config["paths"]["artifact_dir"]).resolve()

    @property
    def quantiles(self) -> np.ndarray:
        return np.asarray(self.protocol["output"]["quantiles"], dtype=np.float64)

    @property
    def hyperparameters(self) -> dict:
        return self.protocol["hyperparameters"]

    @property
    def excluded_inputs(self) -> tuple[str, ...]:
        return tuple(sorted({name for item in self.config["deviations"] for name in item["inputs"]}))

    @property
    def recorded_decisions(self) -> dict:
        return {"deviations": self.config["deviations"], "interpretations": self.config["interpretations"]}

    def period(self, name: str) -> Period:
        if name in FOLDS:
            return fold_period(name, FOLDS[name])
        if name in PARTITION_ESTIMATION:
            return fold_period(name, PARTITION_ESTIMATION[name], role="partition_estimation")
        if name not in PERIODS:
            raise ValueError(f"알 수 없는 기간입니다: {name}")
        item = self.protocol["periods"][name]
        return Period(name, pd.Timestamp(item["start"]).tz_convert(TZ), pd.Timestamp(item["end"]).tz_convert(TZ),
                      pd.Timestamp(item["fit_cutoff"]).tz_convert(TZ), item["role"], int(item["expected_hours"]),
                      item.get("exposure_status"))

    @functools.cached_property
    def calendar(self) -> HolidayCalendar:
        """공휴일 목록. 출력제어 후처리 module이 대상일의 토요일·일요일·공휴일 여부를 정하는 데 쓴다."""
        return HolidayCalendar(self.root / self.config["paths"]["calendar"])

    def curtailment(self, half_life_days: float | None = None) -> dict:
        """출력제어 module 설정. runtime config의 기본값에 반감기만 바꿀 수 있다."""
        value = dict(self.config["curtailment"])
        if half_life_days is not None:
            value["half_life_days"] = half_life_days
        return curtailment_settings(value)

    @property
    def champion(self) -> str:
        """명시적으로 선택한 기본 MK 버전. 각 실험은 다른 MK·override를 사용할 수 있다."""
        return self.registry["champion"]

    @property
    def blend_weight(self) -> float:
        """등록 모델이 아닌 칸의 weighted-quantile XGB 비율."""
        return float(self.protocol["blending"]["weighted-quantile"]["w"])

    def model(self, name: str) -> dict:
        """등록된 MK 버전의 정의(두 expert의 입력·time weight, DL 손실·aux_ratio, 블렌드)."""
        if name not in self.registry["models"]:
            raise ValueError(f"등록되지 않은 모델 버전입니다: {name} (등록: {', '.join(self.registry['models'])})")
        return dict(self.registry["models"][name])

    def input_config(self, name: str) -> InputConfig:
        """입력 구성을 돌려준다."""
        if name not in INPUTS:
            raise NotImplementedError(f"{name}은 구현하지 않은 입력 구성입니다. 실행 가능한 입력 구성은 {', '.join(INPUTS)}입니다.")
        base, changes = DERIVED_INPUTS.get(name, (name, {}))
        item = {**self.protocol["input_configs"][base], **changes}
        return InputConfig(name, item["input_role"], tuple(item["providers"]), item["disagreement"], item["revision"],
                       item["solar_geometry"], item["physics_proxy"], item["time_features"], item.get("day_context", False),
                       item.get("day_state", False), float(item.get("site_aux_ratio", 0.0)), str(item.get("time_shift", "")),
                       str(item.get("shift_scales", "1")), int(item.get("nearby_hours", 0)), int(item.get("group_nearby_hours", 0)),
                       bool(item.get("nearby_gate", False)), str(item.get("group_context", "pooled")))

    def dl_input_config(self, item: dict) -> InputConfig:
        """모델 정의의 DL 입력 구성. 발전소 보조 head 세기는 정의의 site_aux_ratio(없으면 0)이다."""
        return replace(self.input_config(item["dl_input"]), site_aux_ratio=float(item.get("site_aux_ratio", 0.0)))

    def loss(self, name: str, aux_ratio: float | None = None, smoke: bool = False, time_weight: str = "none") -> LossConfig:
        if name not in AUX:
            raise ValueError(f"알 수 없는 손실 구성입니다: {name}")
        if time_weight not in ("none", "settlement"):
            raise ValueError(f"알 수 없는 시간 가중입니다: {time_weight}")
        lam = self.protocol["aux_lambda"]
        warmup = self.config["smoke"]["aux_warmup_steps"] if smoke else lam["warmup_steps_l0_only"]
        batches = self.config["smoke"]["aux_calibration_batches"] if smoke else lam["calibration_batches"]
        strength = 0.0 if AUX[name] is None else float(lam["aux_ratio_screening"] if aux_ratio is None else aux_ratio)
        return LossConfig(name, AUX[name], strength, int(warmup), int(batches), time_weight)


MODEL_FIELDS = ("xgb_input", "xgb_time_weight", "dl_input", "dl_loss", "aux_ratio", "dl_time_weight", "blend", "xgb_weight")
# 쓰는 모델만 적는 module 항목. 없으면 그 module을 쓰지 않는다.
MODULE_FIELDS = ("curtailment", "site_aux_ratio")


def validate_registry(registry: dict) -> None:
    """모델 등록 목록 검사: 버전 이름은 MK{번호}, 필드는 MODEL_FIELDS(+ 쓰는 MODULE_FIELDS), 값은 구현된 입력·손실·time weight·블렌드이다."""
    for name, item in registry["models"].items():
        if not (name.startswith("MK") and name[2:].isdigit()):
            raise ValueError(f"모델 버전 이름은 MK와 번호입니다: {name}")
        if not set(MODEL_FIELDS) <= set(item) <= set(MODEL_FIELDS) | set(MODULE_FIELDS):
            raise ValueError(f"{name}의 필드가 {MODEL_FIELDS}(+ {MODULE_FIELDS})와 다릅니다: {sorted(item)}")
        if "curtailment" in item:
            curtailment_settings(item["curtailment"])
        if not (np.isfinite(float(item.get("site_aux_ratio", 0.0))) and float(item.get("site_aux_ratio", 0.0)) >= 0.0):
            raise ValueError(f"{name}의 site_aux_ratio는 유한한 0 이상의 값입니다.")
        if item["xgb_input"] not in INPUTS or item["dl_input"] not in INPUTS or item["dl_loss"] not in AUX:
            raise ValueError(f"{name}의 입력·손실이 정의되지 않았습니다.")
        if item["xgb_time_weight"] not in ("none", "settlement") or item["dl_time_weight"] not in ("none", "settlement") or item["blend"] != "weighted-quantile":
            raise ValueError(f"{name}의 time weight·블렌드가 정의되지 않았습니다.")
        if not 0.0 <= float(item["xgb_weight"]) <= 1.0:
            raise ValueError(f"{name}의 xgb_weight는 [0, 1]입니다.")
    if registry["champion"] not in registry["models"]:
        raise ValueError(f"champion {registry['champion']}이 등록 목록에 없습니다.")

def config_hash(config: dict) -> str:
    """출처 기록용 config hash. 경로는 제외하고, run의 계산 identity는 solarvpp.store에서 만든다."""
    return digest({k: v for k, v in config.items() if k != "paths"})
def load_spec(config_path: str | Path = "configs/runtime.json", root: Path = ROOT) -> Spec:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = read_json(config_path)
    paths = {name: root / config["paths"][name] for name in ("protocol", "schema", "evidence")}
    spec = Spec(root, config, *(read_json(paths[name]) for name in ("protocol", "schema", "evidence")))
    spec.hashes = {"config": config_hash(config), **{name: file_sha(path) for name, path in paths.items()}}
    # 모델 등록 목록은 이름표이다. run identity는 모델 내용(입력·손실·비율)으로 정해지므로 hash에 넣지 않는다.
    spec.registry = read_json(root / config["paths"]["models"])
    validate_registry(spec.registry)
    if spec.schema["schema_version"] != 1:
        raise ValueError("이 코드는 weather schema version 1을 사용합니다.")
    return spec
