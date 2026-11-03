"""예측 모델 정의와 후처리.

- XGB: 두 그룹이 공유하는 19분위수 모델. target은 정규화 잔차 z − power_curve이다.
- DL: provider encoder → PV fusion → query pooling → 24시간 BiGRU → 19분위수와 보조손실.
- temporal routing(선택): pooling 전 provider 표현과 pooling 뒤 sequence가 다른 시각을 읽어 원래 값에 더한다. 학습 시작 시 출력은 원래와 같다.
- 하루 상태 확장(state token): 관측소마다 그날 24시간의 예보(두 round·revision·결측 indicator·시간 표현)에서 하루 상태 Z_d를 추론하고,
  발전소로 옮겨 시간별 fusion 입력에 붙인다(Y_1:24 = g(Z_d, 시간별 날씨, 자산 정보)). Z_d의 보조 target은 query 하루 발전량
  (z의 하루 평균)과 관측소 낮 시간 observed 평균(cutoff 이전 학습 구간의 운량·UV·운고·시정, test에는 입력으로 쓰지 않음)이다.
- 후처리: seed 평균 → 모델별 clip·야간·정렬 → 모델 간 결합 → 정렬 → 중앙 분위수.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import torch
import xgboost as xgb
from torch import nn

from .queries import EvalPanel, QuerySet


def xgb_params(hyperparameters: dict, quantiles: np.ndarray, seed: int) -> dict:
    return {"objective": "reg:quantileerror", "quantile_alpha": np.asarray(quantiles), "tree_method": "hist",
            "max_depth": hyperparameters["xgb_max_depth"], "eta": hyperparameters["xgb_learning_rate"], "subsample": hyperparameters["xgb_subsample"],
            "colsample_bytree": hyperparameters["xgb_colsample_bytree"], "min_child_weight": hyperparameters["xgb_min_child_weight"],
            "reg_lambda": hyperparameters["xgb_reg_lambda"], "nthread": hyperparameters["xgb_threads"], "device": hyperparameters["xgb_device"], "seed": seed}


def normalized_weights(capacity: np.ndarray) -> np.ndarray:
    """pinball의 용량 비례를 유지하면서 min_child_weight·lambda의 의미를 보존하도록 평균 1로 맞춘다."""
    capacity = np.asarray(capacity, dtype=np.float64)
    return capacity / capacity.mean()


def fit_xgb(x: np.ndarray, residual: np.ndarray, capacity: np.ndarray, names: list[str], hyperparameters: dict, quantiles: np.ndarray,
            seed: int, rounds: int, init_booster: xgb.Booster | None = None, learning_rate: float | None = None) -> tuple[xgb.Booster, dict]:
    """init_booster를 주면 그 모델에 tree를 rounds만큼 더한다(training continuation). learning_rate는 이번 학습의 학습률이다."""
    started = time.perf_counter()
    weight = normalized_weights(capacity)
    data = xgb.DMatrix(x, label=residual, weight=weight, feature_names=names, nthread=hyperparameters["xgb_threads"])
    params = xgb_params(hyperparameters, quantiles, seed)
    if learning_rate is not None:
        params["eta"] = float(learning_rate)
    booster = xgb.train(params, data, num_boost_round=rounds, xgb_model=init_booster)
    details = {"rows": int(len(x)), "features": int(x.shape[1]), "rounds": rounds, "trees": booster.num_boosted_rounds() * len(quantiles),
               "weight_normalizer_kw": float(np.mean(capacity)), "seconds": time.perf_counter() - started}
    if init_booster is not None or learning_rate is not None:
        details.update({"initial_rounds": 0 if init_booster is None else int(init_booster.num_boosted_rounds()),
                        "total_rounds": int(booster.num_boosted_rounds()), "learning_rate": float(params["eta"])})
    return booster, details


def predict_xgb(booster: xgb.Booster, x: np.ndarray, names: list[str], threads: int) -> np.ndarray:
    out = booster.predict(xgb.DMatrix(x, feature_names=names, nthread=threads))
    return np.asarray(out, dtype=np.float32).reshape(len(x), -1)


def gain_share(booster: xgb.Booster, feature: str) -> float:
    gain = booster.get_score(importance_type="total_gain")
    total = sum(gain.values())
    return float(gain.get(feature, 0.0) / total) if total > 0 else 0.0


def seed_mean(values: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack([np.asarray(v, dtype=np.float64) for v in values])
    if not np.isfinite(stacked).all():
        raise ValueError("예측 분위수에 유한하지 않은 값이 있습니다.")
    return stacked.mean(axis=0)


def crossing_stats(z: np.ndarray, daylight: np.ndarray, median_index: int) -> dict:
    """정렬 전 분위수 교차. 주간 행 기준으로 교차가 있는 행 비율, 인접 α쌍별 비율, 정렬로 중앙값이 바뀐 행 비율."""
    rows = z[daylight]
    if len(rows) == 0:
        return {"rows": 0}
    pairs = rows[:, 1:] < rows[:, :-1]
    changed = np.sort(rows, axis=-1)[:, median_index] != rows[:, median_index]
    return {"rows": int(len(rows)), "any_crossing_share": float(pairs.any(axis=1).mean()),
            "pair_crossing_share": pairs.mean(axis=0).round(6).tolist(), "median_changed_share": float(changed.mean())}


def finalize(z: np.ndarray, capacity: np.ndarray, daylight: np.ndarray) -> np.ndarray:
    """정규화 분위수에 query 용량을 곱하고 [0, C_A]로 자른 뒤 야간을 0으로 두고 정렬한다."""
    cap = np.asarray(capacity, dtype=np.float64)[:, None, None]
    kwh = np.clip(np.asarray(z, dtype=np.float64) * cap, 0.0, cap)
    kwh = np.where(daylight[..., None], kwh, 0.0)
    return np.sort(kwh, axis=-1)


def blend(first: np.ndarray, second: np.ndarray, weight: float) -> np.ndarray:
    """weighted-quantile: 같은 분위수끼리 weight·first + (1-weight)·second를 결합하고 다시 정렬한다."""
    return np.sort(weight * first + (1.0 - weight) * second, axis=-1)


def to_panel(values: np.ndarray, queries: QuerySet, panel: EvalPanel) -> np.ndarray:
    """query별 [Q, 24, ...] 값을 panel 시간축 [시각, 그룹, ...]으로 옮긴다."""
    out = np.full((len(panel.hours), len(panel.cap), *values.shape[2:]), np.nan)
    start_day = panel.hours[0] // 24
    for qi in range(len(queries)):
        rows = slice((queries.day[qi] - start_day) * 24, (queries.day[qi] - start_day + 1) * 24)
        out[rows, queries.group[qi]] = values[qi]
    if np.isnan(out).any():
        raise ValueError("panel 시간축에 예측이 없는 시각이 있습니다.")
    return out


def decision(quantiles: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """절대오차 정산의 제출값은 최종 0.5 분위수이다."""
    return quantiles[..., int(np.flatnonzero(np.isclose(levels, 0.5))[0])]


class ProviderEncoder(nn.Module):
    """station·시각마다 한 번 계산한다. 입력에 station embedding을 붙여 MLP로 변환한다."""

    def __init__(self, n_in: int, n_stations: int, embedding: int, widths: tuple[int, int]):
        super().__init__()
        self.embedding = nn.Embedding(n_stations, embedding)
        self.net = nn.Sequential(nn.Linear(n_in + embedding, widths[0]), nn.ReLU(), nn.Linear(widths[0], widths[1]), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        days, stations, hours, _ = x.shape
        emb = self.embedding.weight[None, :, None, :].expand(days, stations, hours, -1)
        return self.net(torch.cat([x, emb], dim=-1))


class StackedBiGRU(nn.Module):
    """다층 양방향 GRU. 층 사이 dropout은 nn.Dropout으로 둔다.

    cuDNN RNN 내부 dropout은 이 실행 환경(Windows, torch 2.12~2.14, cuDNN 9.2)에서 프로세스 종료 시 비정상 종료를 일으킨다.
    구조는 nn.GRU(num_layers, dropout)와 같다: 마지막 층을 제외한 각 층 출력에 dropout을 적용한다.
    """

    def __init__(self, n_in: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([nn.GRU(n_in if i == 0 else 2 * hidden, hidden, batch_first=True, bidirectional=True) for i in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            if i:
                x = self.dropout(x)
            x, _ = layer(x)
        return x


def interpolation_weights(tau: torch.Tensor, size: int) -> torch.Tensor:
    """시각 index 축의 실수 위치 τ에서 값을 읽는 Catmull-Rom 보간 가중 [..., size].

    정수 위치에서는 원래 값을 읽고, 위치에 대한 기울기는 양옆 값 차분의 절반이다. 위치는 [0, size − 1]로 자르고,
    끝에서 범위를 벗어난 이웃의 몫은 남은 가중의 합을 1로 다시 맞춰 처리한다.
    """
    tau = tau.clamp(0, size - 1)
    grid = torch.arange(size, device=tau.device, dtype=tau.dtype)
    distance = (grid - tau[..., None]).abs()
    near = 1.5 * distance ** 3 - 2.5 * distance ** 2 + 1
    far = -0.5 * distance ** 3 + 2.5 * distance ** 2 - 4 * distance + 2
    weight = torch.where(distance <= 1, near, torch.where(distance < 2, far, torch.zeros_like(distance)))
    return weight / weight.sum(dim=-1, keepdim=True)


class TimeShiftRouting(nn.Module):
    """provider 표현 sequence 하나의 deformable temporal routing.

    시각 t의 참조 시각은 τ_{m,j} = t + σ_m·b_j + 1.5·σ_m·tanh(Δ_{m,j})이다. b_j ∈ {−1, 0, 1}이고 σ_m은 시간 범위(scale)이다.
    Δ와 attention logit은 region prior(관측소별 학습값)와 날씨 맥락에서 만든 동적 항의 합이며, 설정에 따라 한쪽만 쓴다.
    동적 attention의 key는 routing source를 τ에서 보간한 값과 상대 위치 (t − τ)로 만든다. routing(읽는 위치와 attention 가중)은
    날씨 맥락·routing source·region prior로만 정하고, 읽어 오는 값은 언제나 원래 sequence이다. 출력은 그 값의 attention 가중합을
    0에서 시작하는 사영으로 원래 sequence에 더하므로, τ는 읽은 값을 거쳐서만 출력에 영향을 주고 학습 시작 시 출력은 원래와 같다.
    """

    def __init__(self, width: int, attn: int, scales: tuple, n_context: int = 0, hidden: int = 0, n_prior: int = 0,
                 n_source: int | None = None):
        super().__init__()
        if not n_context and not n_prior:
            raise ValueError("temporal routing에는 날씨 맥락이나 region prior가 필요합니다.")
        base = [s * b for s in scales for b in (-1.0, 0.0, 1.0)]
        shift = [1.5 * s for s in scales for _ in range(3)]
        self.register_buffer("base", torch.tensor(base))
        self.register_buffer("shift", torch.tensor(shift))
        self.reach, self.attn = max(abs(b) + s for b, s in zip(base, shift)), attn
        self.value, self.out = nn.Linear(width, width), nn.Linear(width, width)
        self.context = self.offset = self.query = self.key = self.position_key = self.prior_offset = self.prior_logit = None
        if n_context:
            self.context = nn.Sequential(nn.Linear(n_context, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
            self.offset, self.query = nn.Linear(hidden, len(base)), nn.Linear(hidden, attn)
            self.key, self.position_key = nn.Linear(width if n_source is None else n_source, attn), nn.Linear(2, attn)
            nn.init.zeros_(self.offset.weight)
            nn.init.zeros_(self.offset.bias)
        if n_prior:
            self.prior_offset = nn.Parameter(torch.zeros(n_prior, len(base)))
            self.prior_logit = nn.Parameter(torch.zeros(n_prior, len(base)))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def offset_parameters(self) -> list:
        """참조 시각을 정하는 parameter(동적 offset 층, region prior offset)."""
        out = [] if self.offset is None else [self.offset.weight, self.offset.bias]
        return out + ([] if self.prior_offset is None else [self.prior_offset])

    def routing_parameters(self) -> list:
        return self.offset_parameters()

    def reference(self, hidden: torch.Tensor | None, prior: torch.Tensor | None, lead: tuple, hours: int) -> torch.Tensor:
        """참조 시각 τ [*lead, 24, n] (시각 index 단위). hidden은 날씨 맥락 MLP 출력, prior는 [PV, 관측소] one-hot이다."""
        parts = []
        if hidden is not None:
            parts.append(self.offset(hidden))
        if prior is not None:
            parts.append((prior @ self.prior_offset).unsqueeze(-2))
        t = torch.arange(hours, device=self.base.device, dtype=self.base.dtype)[:, None]
        tau = t + self.base + self.shift * torch.tanh(sum(parts))
        return tau.expand(*lead, hours, len(self.base)).clamp(0, hours - 1)

    def weights(self, lead: tuple, hours: int, context: torch.Tensor | None = None, prior: torch.Tensor | None = None,
                source: torch.Tensor | None = None) -> torch.Tensor:
        """읽는 위치의 보간 가중을 attention으로 합친 routing 가중 [*lead, 24, 24]. context [..., 24, n_context](날씨 맥락),
        prior [PV, 관측소] one-hot(PV가 lead의 마지막 축일 때), source [..., 24, n_source](동적 attention의 key를 만드는 sequence)."""
        hidden = None if self.context is None else self.context(context)
        tau = self.reference(hidden, None if self.prior_offset is None else prior, lead, hours)
        weight = interpolation_weights(tau, hours)                                      # [..., 24, n, 24]
        parts = []
        if hidden is not None:
            relative = (torch.arange(hours, device=tau.device, dtype=tau.dtype)[:, None] - tau) / self.reach
            key = torch.einsum("...tnh,...hd->...tnd", weight, self.key(source)) + self.position_key(torch.stack([relative, relative * relative], dim=-1))
            parts.append((self.query(hidden)[..., None, :] * key).sum(dim=-1) / self.attn ** 0.5)
        if self.prior_logit is not None:
            parts.append((prior @ self.prior_logit).unsqueeze(-2))
        attention = torch.softmax(sum(parts), dim=-1).expand(*lead, hours, len(self.base))
        return torch.einsum("...tn,...tnh->...th", attention, weight)

    def read(self, h: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """routing 가중으로 h를 읽어 0에서 시작하는 사영으로 h에 더한다."""
        return h + self.out(torch.einsum("...th,...hd->...td", weights, self.value(h)))

    def forward(self, h: torch.Tensor, context: torch.Tensor | None = None, prior: torch.Tensor | None = None,
                source: torch.Tensor | None = None) -> torch.Tensor:
        """h [..., 24, width]. source를 생략하면 h로 key를 만든다."""
        return self.read(h, self.weights(h.shape[:-2], h.shape[-2], context, prior, h if source is None else source))


class PlantTimeShift(nn.Module):
    """group pooling 전 provider 표현의 temporal routing. provider마다 TimeShiftRouting을 둔다.

    날씨 맥락은 시각 t에 PV가 속한 관측소의 provider 예보 입력(station embedding을 붙이기 전)과 fusion 공통 입력 가운데
    정적 정보·provider 사용 여부를 뺀 채널(disagreement·태양 위치·physics·하루 상태)이고, 동적 attention의 key도 그 provider의
    예보 입력으로 만든다. 그래서 관측소 identity는 region prior로만 routing에 들어가고, station embedding은 읽어 오는 값(provider 표현)과
    fusion에만 남는다. region prior는 PV가 속한 provider 관측소별 학습값이다.
    """

    def __init__(self, providers: tuple, width: int, n_forecast: dict, n_extra: int, static: tuple, hidden: int, scales: tuple,
                 weather: bool, station_to_pv: dict | None):
        super().__init__()
        self.providers, self.weather, self.static, self.region = tuple(providers), weather, tuple(static), station_to_pv is not None
        n_context = sum(n_forecast.values()) + n_extra - (static[1] - static[0]) if weather else 0
        self.routes = nn.ModuleDict({p: TimeShiftRouting(width, width, scales, n_context, hidden,
                                                         station_to_pv[p].shape[1] if self.region else 0, n_forecast[p])
                                     for p in self.providers})
        if self.region:
            for p in self.providers:
                self.register_buffer(f"station_{p}", torch.as_tensor(station_to_pv[p], dtype=torch.float32))

    def offset_parameters(self) -> list:
        return [x for p in self.providers for x in self.routes[p].offset_parameters()]

    def routing_parameters(self) -> list:
        return [x for p in self.providers for x in self.routes[p].routing_parameters()]

    def weights(self, extras: torch.Tensor, forecast: list[torch.Tensor] | None = None) -> list[torch.Tensor]:
        """provider별 routing 가중 [대상일, PV, 24, 24]. forecast는 PV로 옮긴 provider 예보 입력 [대상일, PV, 24, C]이다."""
        context = None
        if self.weather:
            start, stop = self.static
            context = torch.cat(forecast + [extras[..., :start], extras[..., stop:]], dim=-1)
        lead, hours = extras.shape[:-2], extras.shape[-2]
        return [self.routes[p].weights(lead, hours, context, getattr(self, f"station_{p}") if self.region else None,
                                       forecast[i] if self.weather else None)
                for i, p in enumerate(self.providers)]

    def forward(self, pv: list[torch.Tensor], extras: torch.Tensor, forecast: list[torch.Tensor] | None = None) -> list[torch.Tensor]:
        """pv: provider 표현 [대상일, PV, 24, width]. routing 가중은 pv를 보지 않는다."""
        return [self.routes[p].read(h, w) for p, h, w in zip(self.providers, pv, self.weights(extras, forecast))]


def _shift(x: torch.Tensor, offset: int) -> torch.Tensor:
    """시간 축(끝에서 두 번째)을 offset만큼 당긴 sequence. 결과의 시각 t는 x의 t + offset이고, 하루 밖은 0이다."""
    if offset == 0:
        return x
    pad = torch.zeros_like(x[..., :abs(offset), :])
    return torch.cat([x[..., offset:, :], pad], dim=-2) if offset > 0 else torch.cat([pad, x[..., :offset, :]], dim=-2)


class NearbyHoursRouting(nn.Module):
    """sequence 하나의 residual sliding window attention.

    시각 t는 창 [t − W, t + W] 안의 시각 s를 attention으로 읽는다. score는 q_t·k_s/√d − λ_t·(s − t)²이고, q와 λ_t는 날씨 맥락에서,
    k_s는 routing source(관측소 정보가 없는 예보 입력)에서 만들며, 하루 밖 시각은 가린다. gate가 있으면 날씨 맥락에서 정한
    gate(0~1)가 읽어 온 값의 몫을 시각마다 줄인다. 읽어 오는 값은 원래 sequence이고, 출력은 그 가중합을 0에서 시작하는 사영으로
    원래 sequence에 더한다. 학습 시작 시 λ_t는 창 끝 가중이 중심의 e⁻²가 되는 2/W²이다.
    """

    def __init__(self, width: int, attn: int, n_context: int, hidden: int, n_source: int, window: int, gate: bool = False):
        super().__init__()
        self.window, self.attn = window, attn
        self.gate = nn.Linear(hidden, 1) if gate else None
        self.context = nn.Sequential(nn.Linear(n_context, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.query, self.key = nn.Linear(hidden, attn), nn.Linear(n_source, attn)
        self.scale = nn.Linear(hidden, 1)
        nn.init.zeros_(self.scale.weight)
        nn.init.constant_(self.scale.bias, float(np.log(np.expm1(2.0 / window ** 2))))
        self.value, self.out = nn.Linear(width, width), nn.Linear(width, width)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("offsets", torch.arange(-window, window + 1, dtype=torch.float32))

    def routing_parameters(self) -> list:
        """읽을 시각의 가중을 정하는 parameter(query·key·λ, gate)."""
        return [*self.scale.parameters(), *self.query.parameters(), *self.key.parameters()] + ([] if self.gate is None else [*self.gate.parameters()])

    def weights(self, context: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """창 위치별 attention 가중 [..., 24, 2W + 1]."""
        hidden = self.context(context)
        score = -nn.functional.softplus(self.scale(hidden).squeeze(-1))[..., None] * self.offsets ** 2
        q, k = self.query(hidden), self.key(source)
        score = score + torch.stack([(q * _shift(k, d)).sum(dim=-1) for d in range(-self.window, self.window + 1)], dim=-1) / self.attn ** 0.5
        position = torch.arange(score.shape[-2], device=score.device, dtype=self.offsets.dtype)[:, None] + self.offsets
        outside = (position < 0) | (position > score.shape[-2] - 1)
        weights = torch.softmax(score.masked_fill(outside, float("-inf")), dim=-1)
        return weights if self.gate is None else weights * torch.sigmoid(self.gate(hidden))   # 읽어 온 값의 몫(0~1)

    def read(self, h: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """창 가중으로 h를 읽어 0에서 시작하는 사영으로 h에 더한다."""
        value = self.value(h)
        return h + self.out(sum(weights[..., i, None] * _shift(value, d) for i, d in enumerate(range(-self.window, self.window + 1))))


class PlantNearbyHours(nn.Module):
    """group pooling 전 provider 표현의 residual sliding window attention. provider마다 NearbyHoursRouting을 둔다.

    query·λ·gate의 날씨 맥락은 대상 시각에 PV가 속한 관측소의 두 provider 예보 입력(station embedding을 붙이기 전)과 fusion 공통 입력
    가운데 keep 구간(disagreement·태양 위치·하루 상태)이고, key는 그 provider의 예보 입력이다. 관측소 identity는 routing에 들어가지
    않고, station embedding은 읽어 오는 값(provider 표현)과 fusion에만 남는다.
    """

    def __init__(self, providers: tuple, width: int, n_forecast: dict, keep: tuple, hidden: int, window: int, gate: bool = False):
        super().__init__()
        self.providers, self.keep = tuple(providers), tuple(tuple(k) for k in keep)
        n_context = sum(n_forecast.values()) + sum(stop - start for start, stop in self.keep)
        self.routes = nn.ModuleDict({p: NearbyHoursRouting(width, width, n_context, hidden, n_forecast[p], window, gate) for p in self.providers})

    def routing_parameters(self) -> list:
        return [x for p in self.providers for x in self.routes[p].routing_parameters()]

    def weights(self, extras: torch.Tensor, forecast: list[torch.Tensor]) -> list[torch.Tensor]:
        """provider별 창 가중 [대상일, PV, 24, 2W + 1]. forecast는 PV로 옮긴 provider 예보 입력 [대상일, PV, 24, C]이다."""
        context = torch.cat(forecast + [extras[..., start:stop] for start, stop in self.keep], dim=-1)
        return [self.routes[p].weights(context, forecast[i]) for i, p in enumerate(self.providers)]

    def forward(self, pv: list[torch.Tensor], extras: torch.Tensor, forecast: list[torch.Tensor]) -> list[torch.Tensor]:
        """pv: provider 표현 [대상일, PV, 24, width]. 창 가중은 pv를 보지 않는다."""
        return [self.routes[p].read(h, w) for p, h, w in zip(self.providers, pv, self.weights(extras, forecast))]


class GenerationNet(nn.Module):
    def __init__(self, n_in: dict, station_to_pv: dict, n_extra: int, n_query: int, n_quantiles: int, active: dict, settings: dict):
        super().__init__()
        self.providers = tuple(n_in)
        width = settings["dl_provider_width"]
        self.encoders = nn.ModuleDict({p: ProviderEncoder(n_in[p], station_to_pv[p].shape[1], settings["dl_station_embedding_width"],
                                                          tuple(settings["dl_provider_mlp_widths"])) for p in self.providers})
        for p in self.providers:
            self.register_buffer(f"map_{p}", torch.as_tensor(station_to_pv[p], dtype=torch.float32))
            self.register_buffer(f"active_{p}", torch.tensor(float(active[p])))
        fusion = settings["dl_fusion_width"]
        self.fusion = nn.Sequential(nn.Linear(width * len(self.providers) + n_extra, fusion), nn.ReLU(), nn.Linear(fusion, fusion), nn.ReLU())
        hidden = settings["dl_gru_hidden"]
        self.gru = StackedBiGRU(2 * fusion + n_query, hidden, settings["dl_gru_layers"], settings["dl_dropout"])
        self.head = nn.Linear(2 * hidden, n_quantiles)
        self.site_head = None      # build_net이 발전소 보조 head를 마지막에 붙인다(다른 층의 초기값을 바꾸지 않는다)
        self.time_shift = None     # build_net이 pooling 전·후 temporal routing을 그 뒤에 붙인다
        self.nearby_hours = None
        self.group_nearby_hours = None
        self.group_neutral = None  # 그룹 routing을 관측소 정보 없는 표현으로 정할 때 0으로 둘 정적 정보 구간(start, stop)

    def forward(self, x: dict, extras: torch.Tensor, day_onehot: torch.Tensor, weights: torch.Tensor, query: torch.Tensor,
                return_site: bool = False):
        # station 표현을 one-hot 행렬곱으로 PV에 전달한다(gather 역전파의 비결정성을 피한다).
        pv = [torch.einsum("ps,usth->upth", getattr(self, f"map_{p}"), self.encoders[p](x[p])) * getattr(self, f"active_{p}")
              for p in self.providers]
        forecast = None
        if (self.time_shift is not None and self.time_shift.weather) or self.nearby_hours is not None:
            forecast = [torch.einsum("ps,ustc->uptc", getattr(self, f"map_{p}"), x[p]) for p in self.providers]
        if self.time_shift is not None:
            pv = [v * getattr(self, f"active_{p}") for p, v in zip(self.providers, self.time_shift(pv, extras, forecast))]
        if self.nearby_hours is not None:
            pv = [v * getattr(self, f"active_{p}") for p, v in zip(self.providers, self.nearby_hours(pv, extras, forecast))]
        fused = self.fusion(torch.cat(pv + [extras], dim=-1))
        mean = torch.einsum("bu,bp,upth->bth", day_onehot, weights, fused)
        second = torch.einsum("bu,bp,upth->bth", day_onehot, weights, fused * fused)
        std = torch.sqrt(torch.clamp(second - mean * mean, min=0.0) + 1e-4)
        pooled = torch.cat([mean, std, query[:, None, :].expand(-1, mean.shape[1], -1)], dim=-1)
        route = pooled
        if self.group_neutral is not None:                                              # 평균 station embedding·정적 정보 0인 표현의 그룹 pooling
            start, stop = self.group_neutral
            neutral = []
            for p in self.providers:
                encoder = self.encoders[p]
                days, stations, hours, _ = x[p].shape
                emb = encoder.embedding.weight.mean(dim=0)[None, None, None, :].expand(days, stations, hours, -1)
                neutral.append(torch.einsum("ps,usth->upth", getattr(self, f"map_{p}"), encoder.net(torch.cat([x[p], emb], dim=-1)))
                               * getattr(self, f"active_{p}"))
            blank = torch.cat([extras[..., :start], torch.zeros_like(extras[..., start:stop]), extras[..., stop:]], dim=-1)
            plain_fused = self.fusion(torch.cat(neutral + [blank], dim=-1))
            n_mean = torch.einsum("bu,bp,upth->bth", day_onehot, weights, plain_fused)
            n_second = torch.einsum("bu,bp,upth->bth", day_onehot, weights, plain_fused * plain_fused)
            n_std = torch.sqrt(torch.clamp(n_second - n_mean * n_mean, min=0.0) + 1e-4)
            route = torch.cat([n_mean, n_std, query[:, None, :].expand(-1, n_mean.shape[1], -1)], dim=-1)
        if self.group_nearby_hours is not None:
            pooled = self.group_nearby_hours.read(pooled, self.group_nearby_hours.weights(route, route))
        q = self.head(self.gru(pooled))
        if return_site:                                 # 발전소 이용률 [대상일, PV, 24]
            return q, None if self.site_head is None else self.site_head(fused).squeeze(-1)
        return q


def aux_loss(kind: str, q: torch.Tensor, z: torch.Tensor, mask: torch.Tensor, daylight: torch.Tensor, full_day: torch.Tensor,
             capacity: torch.Tensor, median_index: int, slot_weight: torch.Tensor | None = None) -> torch.Tensor | None:
    """중앙분위수 출력 m에 거는 보조손실. 적용할 표본이 없으면 None을 돌려준다.

    참조: docs/solarvpp-reference.md#losses.

    mae·mse는 pinball과 같은 주간 유효 mask·용량 가중을 쓴다. ramp·fourier·cosine는 24시간 정답이 완전한 query-day만 쓰고,
    태양 위치 기준 야간은 예측·정답을 0으로 둔 24시간 profile에서 계산한 뒤 query 용량으로 가중 평균한다.
    day_energy는 complete query-day의 median output에 |Σ_h w_h·D_h·(m_h − z_h)|를 적용한다(slot_weight가 없으면 w_h=1).
    """
    m = q[..., median_index]
    if kind in ("mae", "mse"):
        weight = mask.to(m.dtype) * capacity[:, None]
        if float(weight.sum()) <= 0:
            return None
        error = torch.where(mask, m - z, torch.zeros_like(m))
        value = error.abs() if kind == "mae" else error * error
        return (value * weight).sum() / weight.sum()
    if kind == "day_energy":
        eligible = full_day & (mask | ~daylight).all(dim=1)
        if not bool(eligible.any()):
            return None
        valid = mask[eligible] & daylight[eligible]
        error = torch.where(valid, m[eligible] - z[eligible], torch.zeros_like(m[eligible]))
        if slot_weight is not None:
            error = error * slot_weight[None, :]
        per_day = error.sum(dim=1).abs()
        weight = capacity[eligible]
        return (per_day * weight).sum() / weight.sum()
    day = daylight.to(m.dtype)
    pred, target = m * day, z * day
    if kind == "cosine":
        norm_target = target.norm(dim=1)
        eligible = full_day & (norm_target > 0)
    else:
        eligible = full_day
    if not bool(eligible.any()):
        return None
    if kind == "ramp":
        per_day = ((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs().mean(dim=1)
    elif kind == "fourier":
        spec_pred, spec_target = torch.fft.rfft(pred, dim=1), torch.fft.rfft(target, dim=1)
        amplitude_pred = torch.sqrt(spec_pred.real ** 2 + spec_pred.imag ** 2 + 1e-12)
        amplitude_target = torch.sqrt(spec_target.real ** 2 + spec_target.imag ** 2 + 1e-12)
        per_day = ((amplitude_pred - amplitude_target) ** 2).mean(dim=1)
    elif kind == "cosine":
        per_day = 1 - (pred * target).sum(dim=1) / (pred.norm(dim=1) * norm_target + 1e-6)
    else:
        raise ValueError(f"알 수 없는 보조손실입니다: {kind}")
    weight = eligible.to(m.dtype) * capacity
    return (per_day * weight).sum() / weight.sum()


def l0_loss(q: torch.Tensor, z: torch.Tensor, mask: torch.Tensor, capacity: torch.Tensor, alphas: torch.Tensor,
            slot_weight: torch.Tensor | None = None) -> torch.Tensor:
    """pinball = Σ M·C_A·k_t·ρ_α(z − q) / (|Q|·Σ M·C_A·k_t). slot_weight가 없으면 k_t = 1이다."""
    target = torch.where(mask, z, torch.zeros_like(z))
    error = target[..., None] - q
    rho = torch.maximum(alphas * error, (alphas - 1) * error)
    weight = mask.to(q.dtype) * capacity[:, None]
    if slot_weight is not None:
        weight = weight * slot_weight[None, :]
    return (rho * weight[..., None]).sum() / (alphas.numel() * weight.sum().clamp_min(1e-8))


OBSERVED_CHANNELS = {"wa": ("wa_cloud",), "wb": ("wb_cloud", "wb_uv_idx", "wb_ceiling", "wb_vis")}
OBSERVED_SLOTS = slice(8, 17)   # 표시 시각 9~17시


def observed_day_targets(observed: dict, channel_names: dict, days: np.ndarray, cutoff_h: int) -> dict:
    """관측소-일 낮 시간 observed 평균 [대상일, 관측소, K]과 표준화 상태. cutoff 이후 시각은 쓰지 않는다.

    observed[p]는 [시각, 관측소, 채널], channel_names[p]는 채널 이름이다. 표준화는 cutoff 이전 대상일의 관측소-일 값으로 맞춘다.
    """
    out = {}
    for p, names in OBSERVED_CHANNELS.items():
        index = [channel_names[p].index(n) for n in names]
        values = np.asarray(observed[p], dtype=np.float64)[..., index]
        values[cutoff_h + 1:] = np.nan
        hours = days[:, None] * 24 + np.arange(24)[None, :]
        hours = np.clip(hours, 0, len(values) - 1)
        block = values[hours[:, OBSERVED_SLOTS]]                    # [days, slots, stations, K]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # cutoff 이후 대상일은 값이 없어 NaN이 된다
            daily = np.nanmean(block, axis=1)
        past = days * 24 + 23 <= cutoff_h
        reference = daily[past].reshape(-1, len(names))
        mean = np.nanmean(reference, axis=0)
        std = np.nanstd(reference, axis=0)
        std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
        out[p] = {"target": ((daily - mean) / std).astype(np.float32), "mean": mean, "std": std, "names": list(names)}
    return out


def daily_energy(z: torch.Tensor, mask: torch.Tensor, full_day: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """query 하루 발전량(Σ_h z_h / 24, z와 같은 단위)과 사용 가능 여부. 24시간 정답이 모두 있는 query-day만 쓴다."""
    return torch.where(mask, z, torch.zeros_like(z)).sum(dim=1) / z.shape[1], full_day


class DayEncoder(nn.Module):
    """관측소-일 24시간 예보 [대상일, 관측소, 24, C] → Z [대상일, 관측소, d]."""

    def __init__(self, n_in: int, width: int, out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(24 * n_in, width), nn.ReLU(), nn.Linear(width, out), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        days, stations = x.shape[:2]
        return self.net(x.reshape(days, stations, -1))


class DayStateNet(GenerationNet):
    """GenerationNet의 fusion 입력에 관측소 하루 상태를 붙이고, 일 발전량·observed 보조 head를 둔다."""

    def __init__(self, n_in: dict, station_to_pv: dict, n_extra: int, n_query: int, n_quantiles: int, active: dict, settings: dict,
                 n_observed: dict):
        state = settings["dl_daystate_width"]
        super().__init__(n_in, station_to_pv, n_extra + state * len(n_in), n_query, n_quantiles, active, settings)
        self.day_encoders = nn.ModuleDict({p: DayEncoder(n_in[p], settings["dl_fusion_width"], state) for p in self.providers})
        self.energy_head = nn.Sequential(nn.Linear(state * len(self.providers), state), nn.ReLU(), nn.Linear(state, 1))
        self.observed_heads = nn.ModuleDict({p: nn.Linear(state, n_observed[p]) for p in self.providers})

    def states(self, x: dict) -> tuple[dict, torch.Tensor]:
        station = {p: self.day_encoders[p](x[p]) * getattr(self, f"active_{p}") for p in self.providers}
        pv = torch.cat([torch.einsum("ps,usd->upd", getattr(self, f"map_{p}"), station[p]) for p in self.providers], dim=-1)
        return station, pv

    def forward(self, x: dict, extras: torch.Tensor, day_onehot: torch.Tensor, weights: torch.Tensor, query: torch.Tensor,
                return_aux: bool = False):
        station, pv = self.states(x)
        widened = torch.cat([extras, pv[:, :, None, :].expand(-1, -1, extras.shape[2], -1)], dim=-1)
        if not return_aux:
            return super().forward(x, widened, day_onehot, weights, query)
        q, site = super().forward(x, widened, day_onehot, weights, query, return_site=True)
        pooled = torch.einsum("bu,bp,upd->bd", day_onehot, weights, pv)
        return q, {"energy": self.energy_head(pooled).squeeze(-1), "observed": {p: self.observed_heads[p](station[p]) for p in self.providers},
                   "site": site}


def site_loss(pred: torch.Tensor, target: torch.Tensor, capacity: torch.Tensor, day_weight: torch.Tensor | None = None) -> torch.Tensor | None:
    """발전소 이용률 보조 head의 용량 가중 MAE. pred·target은 [대상일, PV, 24]이고 target이 NaN인 칸(정답 없음·야간·cutoff 이후)은 뺀다.
    day_weight [대상일]을 주면 Σ a_d·C_i·valid로 가중한다."""
    ok = torch.isfinite(target)
    if not bool(ok.any()):
        return None
    weight = ok.to(pred.dtype) * capacity[None, :, None]
    if day_weight is not None:
        weight = weight * day_weight[:, None, None]
    error = torch.where(ok, pred - torch.nan_to_num(target), torch.zeros_like(pred)).abs()
    return (error * weight).sum() / weight.sum()


def loss_cap(b: dict) -> torch.Tensor:
    """손실의 query 용량 가중. 대상일 가중(qw)이 있으면 곱한다(가중이 없으면 C_A 그대로)."""
    return b["cap"] if b.get("qw") is None else b["cap"] * b["qw"]


def observed_huber(predicted: torch.Tensor, target: torch.Tensor, ok: torch.Tensor, day_weight: torch.Tensor | None) -> torch.Tensor:
    """observed 보조항 [unique 대상일, 관측소, 채널]의 Huber 평균. 대상일 가중을 주면 분자와 유효 가중합 분모에 곱한다."""
    if day_weight is None:
        return nn.functional.huber_loss(predicted[ok], target[ok])
    w = day_weight.reshape(-1, *([1] * (target.dim() - 1))).expand_as(target)[ok]
    return (nn.functional.huber_loss(predicted[ok], target[ok], reduction="none") * w).sum() / w.sum()


def daystate_loss(aux: dict, b: dict, energy_weight: float, observed_weight: float) -> tuple[torch.Tensor, dict]:
    """일 발전량 Huber(용량 가중)와 observed Huber(값이 있는 관측소-일)의 가중합. 쓸 표본이 없는 항은 0이다."""
    energy, valid = daily_energy(b["z"], b["mask"], b["full_day"])
    zero = aux["energy"].sum() * 0.0
    if energy_weight > 0 and bool(valid.any()):
        weight = valid.to(energy.dtype) * loss_cap(b)
        loss_energy = (nn.functional.huber_loss(aux["energy"], energy, reduction="none") * weight).sum() / weight.sum()
    else:
        loss_energy = zero
    terms = []
    for p, predicted in aux["observed"].items():
        target = b["observed"][p]
        ok = torch.isfinite(target)
        if bool(ok.any()):
            terms.append(observed_huber(predicted, target, ok, b.get("uw")))
    loss_observed = torch.stack(terms).mean() if terms else zero
    return energy_weight * loss_energy + observed_weight * loss_observed, {"energy": float(loss_energy.detach()),
                                                                           "observed": float(loss_observed.detach())}
