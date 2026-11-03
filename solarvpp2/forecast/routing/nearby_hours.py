"""Residual sliding window attention: PV별 provider 표현이 ±window시간 창을 attention으로 읽고, gate가 읽어 온 값의 몫을 시각마다 조절한다."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

KIND = "nearby-hours"
VERSION = 1
DEFAULTS = {"hours": 3, "gate": True}


def resolve(spec: dict) -> dict:
    if set(spec) - {"kind", *DEFAULTS}:
        raise ValueError(f"{KIND} plant routing에 정의되지 않은 항목이 있습니다: {sorted(set(spec) - {'kind', *DEFAULTS})}")
    out = {"kind": KIND, **{k: spec.get(k, v) for k, v in DEFAULTS.items()}}
    if isinstance(out["hours"], bool) or float(out["hours"]) != int(out["hours"]) or int(out["hours"]) < 1:
        raise ValueError(f"{KIND} plant routing의 hours는 1 이상의 정수입니다: {out['hours']}")
    out["hours"] = int(out["hours"])
    if not isinstance(out["gate"], bool):
        raise ValueError(f"{KIND} plant routing의 gate는 true 또는 false입니다.")
    return out


def build(spec: dict, providers: tuple, width: int, n_forecast: dict, n_common: int, static: tuple, keep: tuple, hidden: int,
          station_to_pv: dict) -> "PlantNearbyHours":
    return PlantNearbyHours(providers, width, n_forecast, keep, hidden, spec["hours"], spec["gate"])


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
