"""Deformable temporal routing: PV별 provider 표현이 시간 범위별 참조 시각을 보간으로 읽어 원래 표현에 더한다.

참조 시각은 τ_{m,j} = t + σ_m·b_j + 1.5·σ_m·tanh(Δ_{m,j}), b_j ∈ {−1, 0, 1}이고, σ_m은 시간 범위(scales)이다. 기본 설정은 범위 3개의 참조점 9개와
관측소별 region prior·날씨 맥락의 합이다.
"""
from __future__ import annotations

import torch
from torch import nn

KIND = "time-shift"
VERSION = 1
DEFAULTS = {"region": True, "weather": True, "scales": [0.5, 1.0, 2.0]}


def resolve(spec: dict) -> dict:
    if set(spec) - {"kind", *DEFAULTS}:
        raise ValueError(f"{KIND} plant routing에 정의되지 않은 항목이 있습니다: {sorted(set(spec) - {'kind', *DEFAULTS})}")
    out = {"kind": KIND, **{k: spec.get(k, v) for k, v in DEFAULTS.items()}}
    if not isinstance(out["region"], bool) or not isinstance(out["weather"], bool) or not (out["region"] or out["weather"]):
        raise ValueError(f"{KIND} plant routing의 region·weather는 true 또는 false이고 적어도 하나는 true입니다.")
    scales = out["scales"]
    if not isinstance(scales, (list, tuple)) or not scales or any(isinstance(s, bool) or float(s) <= 0 for s in scales):
        raise ValueError(f"{KIND} plant routing의 scales는 양수 목록입니다: {scales}")
    out["scales"] = [float(s) for s in scales]
    return out


def build(spec: dict, providers: tuple, width: int, n_forecast: dict, n_common: int, static: tuple, keep: tuple, hidden: int,
          station_to_pv: dict) -> "PlantTimeShift":
    return PlantTimeShift(providers, width, n_forecast, n_common, static, hidden, tuple(spec["scales"]), spec["weather"],
                        station_to_pv if spec["region"] else None)


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
