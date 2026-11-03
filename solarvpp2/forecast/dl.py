"""DL 집합 expert: provider encoder → PV fusion → 집합 pooling → 24시간 decoder(기본 BiGRU) → 19분위수와 보조손실.

참조: docs/solarvpp2-design.md#dl.

PV 표현(fuse)은 집합과 무관하고, 집합 pooling은 구성원 PV 표현의 용량 가중 평균·표준편차이다. 예측과 배정 탐색은 대상일별 PV
표현을 한 번 계산한 뒤 membership만 바꿔 pooling·decode한다. decode는 R 입력으로 쓸 시각별 집합 표현을 함께 돌려준다.
pooled 시퀀스 decoder는 decoders 패키지에 decoder별 모듈로 둔다(BiGRU, shared-memory looped attention). 시간별 provider 표현에
시간별 표현이 다른 시각을 읽는 component(temporal routing)는 routing 패키지에 종류별 모듈로 두고, 설정이 없으면 쓰지 않는다.
관측소 사이 context를 더하는 component(provider context)는 contexts 패키지에 종류별 모듈로 두고, 설정이 있을 때만 PV 표현 계산(fuse)에서 쓴다.
하루 상태 확장(state token)은 관측소마다 그날 예보에서 하루 상태 Z_d를 추론해 PV fusion 입력에 붙이고, 학습 기간 observed 낮 평균을
보조 target으로 쓴다. held-out 블록 대상일의 observed는 보조 target과 표준화에서 뺀다.
발전소 이용률 보조 head(site_aux_ratio > 0)는 PV 표현 위의 선형 head로 발전소별 이용률(발전량 / 설비용량)을 학습한다. 정답은 batch의
unique 대상일마다 발전소별로 한 번씩 쓰고, held-out 블록 대상일은 뺀다. head는 학습에만 쓰며 예측·배정 탐색 경로(fuse → pooling →
decode)는 head와 무관하다.
계절 학습: 대상일 가중(day_weight)을 주면 각 지도 항을 reduction 전에 그 대상일의 가중으로 곱한다(query 항은 query의 대상일, 발전소
이용률·observed 항은 batch의 unique 대상일). 시작 weights와 고정 λ·loss_scales(fixed)를 주면 warm-up과 λ calibration 없이 첫 step부터
같은 손실로 이어 학습한다.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..config import LossConfig, slot_weight
from ..data import PROVIDERS, ROUNDS, Sites
from ..features import (
    CROSS_PROVIDER,
    BranchScaler,
    InputSet,
    Standardizer,
    physics_block,
    time_features,
)
from ..sampler import QuerySet
from . import contexts, decoders, routing

PHYSICS_CHANNELS = ["phys_ghi_clear_sky"] + [f"phys_{k}_{p}_r{r}" for p in PROVIDERS for r in ROUNDS
                                             for k in ("ghi_cloud", "dewpoint_spread", "temp_derate")]
EXTRA_CHANNELS = ([f"dis_{k}_r{r}" for r in ROUNDS for k in CROSS_PROVIDER] + ["solar_elev_90", "solar_cosz_pos", "solar_azim_sin", "solar_azim_cos"]
                  + PHYSICS_CHANNELS)
STATIC_CHANNELS = ("lat", "lon", "capacity_kw")
OBSERVED_CHANNELS = {"wa": ("wa_cloud",), "wb": ("wb_cloud", "wb_uv_idx", "wb_ceiling", "wb_vis")}
OBSERVED_SLOTS = slice(8, 17)   # 표시 시각 9~17시


def configure_determinism(threads: int) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG를 torch import 전에 설정해야 합니다(python -m solarvpp2로 실행).")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


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


def pool_onehot(fused: torch.Tensor, day_onehot: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """학습 batch의 집합 pooling. fused [U, PV, 24, H], day_onehot [B, U], weights [B, PV](합 1) → 평균·표준편차 [B, 24, H]."""
    mean = torch.einsum("bu,bp,upth->bth", day_onehot, weights, fused)
    second = torch.einsum("bu,bp,upth->bth", day_onehot, weights, fused * fused)
    return mean, torch.sqrt(torch.clamp(second - mean * mean, min=0.0) + 1e-4)


def pool_by_day(fused: torch.Tensor, day_index: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """예측·탐색용 집합 pooling. pool_onehot과 같은 값을 대상일마다 행렬곱으로 계산한다(집합 수가 많을 때 메모리를 아낀다)."""
    n_pv, hours, width = fused.shape[1:]
    mean = torch.empty((len(weights), hours, width), dtype=fused.dtype, device=fused.device)
    second = torch.empty_like(mean)
    for u in torch.unique(day_index).tolist():
        rows = torch.nonzero(day_index == u).squeeze(1)
        flat = fused[u].reshape(n_pv, hours * width)
        w = weights.index_select(0, rows)
        mean[rows] = (w @ flat).reshape(-1, hours, width)
        second[rows] = (w @ (flat * flat)).reshape(-1, hours, width)
    return mean, torch.sqrt(torch.clamp(second - mean * mean, min=0.0) + 1e-4)


def pooled_from_sums(s1: torch.Tensor, s2: torch.Tensor, cap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """S⁽¹⁾ = Σ c_i φ_i, S⁽²⁾ = Σ c_i φ_i² [B, 24, H]와 C_A [B]에서 pooled 평균·표준편차. pool_onehot과 같은 식이다."""
    mean = s1 / cap[:, None, None]
    second = s2 / cap[:, None, None]
    return mean, torch.sqrt(torch.clamp(second - mean * mean, min=0.0) + 1e-4)


def pv_sums(fused: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """대상일 하루의 PV 표현 [PV, 24, H]와 용량 membership [G, PV]에서 S⁽¹⁾·S⁽²⁾ [G, 24, H](float64)."""
    n_pv, hours, width = fused.shape
    flat = fused.reshape(n_pv, hours * width).double()
    return ((weights @ flat).reshape(-1, hours, width), (weights @ (flat * flat)).reshape(-1, hours, width))


DAY_SCALE_VERSION = 1


class DailyScaleHead(nn.Module):
    """하루의 decoder 표현 평균·표준편차 → exp(bound·tanh(MLP)) 배율 [B, 1].

    마지막 선형층을 0으로 초기화하므로 초기 배율은 정확히 1이다. 첫 backward는 마지막 층을 학습하고,
    그 층이 움직인 뒤에는 앞 MLP와 decoder로도 이 경로의 gradient가 흐른다. 기존 quantile 경로는 처음부터 유지된다.
    """
    def __init__(self, width: int, hidden: int, max_log_gain: float):
        super().__init__()
        self.max_log_gain = float(max_log_gain)
        self.net = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        summary = torch.cat([hidden.mean(dim=1), torch.sqrt(hidden.var(dim=1, unbiased=False) + 1e-6)], dim=-1)
        return torch.exp(self.max_log_gain * torch.tanh(self.net(summary)))


class GenerationNet(nn.Module):
    """provider encoder → PV fusion → 집합 pooling(용량 가중 평균·표준편차) → pooled 시퀀스의 decoder와 선형 head.
    decoder는 모델 정의의 dl.decoder(decoders.resolve의 결과)이고, 없으면 BiGRU이다."""

    def __init__(self, n_in: dict, station_to_pv: dict, n_extra: int, n_query: int, n_quantiles: int, active: dict, settings: dict,
                 decoder: dict | None = None):
        super().__init__()
        self.providers = tuple(n_in)
        width = settings["dl_provider_width"]
        self.encoders = nn.ModuleDict({p: ProviderEncoder(n_in[p], station_to_pv[p].shape[1], settings["dl_station_embedding_width"],
                                                          tuple(settings["dl_provider_mlp_widths"])) for p in self.providers})
        for p in self.providers:
            self.register_buffer(f"map_{p}", torch.as_tensor(station_to_pv[p], dtype=torch.float32))
            self.register_buffer(f"active_{p}", torch.tensor(float(active[p])))
        fusion = settings["dl_fusion_width"]
        self.fusion_width = fusion
        self.fusion = nn.Sequential(nn.Linear(width * len(self.providers) + n_extra, fusion), nn.ReLU(), nn.Linear(fusion, fusion), nn.ReLU())
        self.decoder_name, sequence = decoders.build(decoder, 2 * fusion + n_query, settings)
        setattr(self, self.decoder_name, sequence)      # BiGRU는 gru(기존 checkpoint 이름), 다른 decoder는 각 모듈의 이름
        self.head = nn.Linear(sequence.out_width, n_quantiles)
        self.site_head = None      # build_net이 발전소 이용률 보조 head를 마지막에 붙인다(다른 층의 초기값을 바꾸지 않는다)
        self.provider_context = None   # build_net이 provider context를 그 다음에 붙인다(설정이 없으면 None)
        self.day_scale = None          # 마지막에 붙이는 집합별 하루 배율 head. 기존 초기값과 dropout RNG를 유지한다.
        self.plant_routing = None      # build_net이 temporal routing을 그 뒤에 붙인다(설정이 없으면 None)
        self.group_routing = None

    def site_output(self, fused: torch.Tensor) -> torch.Tensor | None:
        """PV 표현 [U, PV, 24, H]에서 발전소 이용률 [U, PV, 24]. 보조 head가 없으면 None이다."""
        return None if self.site_head is None else self.site_head(fused[..., :self.fusion_width]).squeeze(-1)

    def fuse(self, x: dict, extras: torch.Tensor) -> torch.Tensor:
        """대상일별 PV 표현 [U, PV, 24, H]. station 표현은 one-hot 행렬곱으로 PV에 전달한다(gather 역전파의 비결정성을 피한다).
        provider context가 있으면 station 표현에 관측소 사이 context를 더한 뒤 PV에 전달한다. plant routing은 PV에 전달한 provider 표현에 둔다.
        group routing의 맥락이 neutral이면 neutral 표현을 뒤에 이어 붙여 [U, PV, 24, 2H]를 돌려주고, 같은 pooling을 거쳐 decode가 나눠 읽는다."""
        station = {p: self.encoders[p](x[p]) for p in self.providers}
        if self.provider_context is not None:
            station = self.provider_context(station, {p: getattr(self, f"active_{p}") for p in self.providers})
        pv = [torch.einsum("ps,usth->upth", getattr(self, f"map_{p}"), station[p]) * getattr(self, f"active_{p}") for p in self.providers]
        if self.plant_routing is not None:
            forecast = [torch.einsum("ps,ustc->uptc", getattr(self, f"map_{p}"), x[p]) for p in self.providers]
            pv = [v * getattr(self, f"active_{p}") for p, v in zip(self.providers, self.plant_routing(pv, extras, forecast))]
        fused = self.fusion(torch.cat(pv + [extras], dim=-1))
        if self.group_routing is not None and self.group_routing.neutral:
            return torch.cat([fused, self._neutral(x, extras)], dim=-1)
        return fused

    def _neutral(self, x: dict, extras: torch.Tensor) -> torch.Tensor:
        """모든 관측소에 같은 평균 station embedding을 넣고 정적 정보를 0으로 둔 PV 표현 [U, PV, 24, H]. group routing의 맥락·key에만 쓴다."""
        start, stop = self.group_routing.blank
        pv = []
        for p in self.providers:
            encoder = self.encoders[p]
            days, stations, hours, _ = x[p].shape
            embedding = encoder.embedding.weight.mean(dim=0)[None, None, None, :].expand(days, stations, hours, -1)
            pv.append(torch.einsum("ps,usth->upth", getattr(self, f"map_{p}"), encoder.net(torch.cat([x[p], embedding], dim=-1)))
                      * getattr(self, f"active_{p}"))
        blank = torch.cat([extras[..., :start], torch.zeros_like(extras[..., start:stop]), extras[..., stop:]], dim=-1)
        return self.fusion(torch.cat(pv + [blank], dim=-1))

    def decode(self, mean: torch.Tensor, std: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """pooled 값에서 정규화 분위수 [B, 24, Q]와 시각별 집합 표현(decoder 출력 [B, 24, out_width])을 계산한다."""
        info = query[:, None, :].expand(-1, mean.shape[1], -1)
        width = self.fusion_width
        pooled = torch.cat([mean[..., :width], std[..., :width], info], dim=-1)
        if self.group_routing is not None:
            route = torch.cat([mean[..., width:], std[..., width:], info], dim=-1) if self.group_routing.neutral else pooled
            pooled = self.group_routing.read(pooled, self.group_routing.weights(route, route))
        hidden = getattr(self, self.decoder_name)(pooled)
        quantiles = self.head(hidden)
        if self.day_scale is not None:
            quantiles = quantiles * self.day_scale(hidden)[:, None, :]
        return quantiles, hidden

    def forward(self, x: dict, extras: torch.Tensor, day_onehot: torch.Tensor, weights: torch.Tensor, query: torch.Tensor,
                return_hidden: bool = False, return_site: bool = False):
        fused = self.fuse(x, extras)
        q, hidden = self.decode(*pool_onehot(fused, day_onehot, weights), query)
        out = [q]
        if return_hidden:
            out.append(hidden)
        if return_site:
            out.append(self.site_output(fused))
        return out[0] if len(out) == 1 else tuple(out)


def submitted_median(q: torch.Tensor, median_index: int) -> torch.Tensor:
    """제출에 쓰는 중앙값: 분위수를 정렬한 뒤의 median_index번째 값. 정렬은 미분 가능하고, 기울기는 그 자리에 온 분위수 head로 간다."""
    return torch.sort(q, dim=-1).values[..., median_index]


def loss_scales(loss: LossConfig, queries: QuerySet, query_weight: np.ndarray | None = None,
                settlement_weight: np.ndarray | None = None) -> dict:
    """용량 가중 손실의 고정 normalizer. 학습 query 전체에서 한 번 계산하므로 표본의 가중이 batch 구성에 따라 바뀌지 않는다.

    cell: 유효 칸의 C_A·시각 가중 평균(pinball), cell_plain: 유효 칸의 C_A 평균(mae·mse), query: query의 C_A 평균(하루 단위 항).
    query_weight(대상일 가중 a)를 주면 C_A 대신 a·C_A의 평균을 쓴다: cell = Σ mask·a·C_A·k / Σ mask, query = mean(a·C_A).
    settlement는 학습 query의 유효 칸에서 a·C_A·SMP·k의 평균이다. 가격은 모델 입력에 들어가지 않는다.
    """
    cap = np.asarray(queries.cap, dtype=np.float64)
    if query_weight is not None:
        cap = cap * np.asarray(query_weight, dtype=np.float64)
    mask = np.asarray(queries.target_mask, dtype=bool)
    slot = slot_weight(loss.time_weight)
    cell = np.broadcast_to(cap[:, None] * (1.0 if slot is None else slot[None, :]), mask.shape)[mask].mean()
    scales = {"cell": float(cell), "cell_plain": float(np.broadcast_to(cap[:, None], mask.shape)[mask].mean()), "query": float(cap.mean())}
    if getattr(loss, "aux", None) == "settlement":
        if settlement_weight is None:
            raise ValueError("settlement-penalty에는 학습 query의 SMP·k 가중이 필요합니다.")
        price = np.asarray(settlement_weight, dtype=np.float64)
        if price.shape != mask.shape or not np.isfinite(price).all() or (price < 0).any():
            raise ValueError("정산 가중은 query·시각과 같은 모양의 유한한 음이 아닌 값입니다.")
        normalizer = float((cap[:, None] * price)[mask].mean())
        if not np.isfinite(normalizer) or normalizer <= 0:
            raise ValueError("settlement-penalty의 학습 가중 평균은 양수여야 합니다.")
        scales["settlement"] = normalizer
    return scales


def aux_loss(kind: str, q: torch.Tensor, z: torch.Tensor, mask: torch.Tensor, daylight: torch.Tensor, full_day: torch.Tensor,
             capacity: torch.Tensor, median_index: int, scales: dict, slot_weight: torch.Tensor | None = None,
             settlement_weight: torch.Tensor | None = None) -> torch.Tensor | None:
    """제출 중앙값 m(정렬 후)에 거는 보조손실. 적용할 표본이 없으면 None을 돌려준다.

    mae·mse는 pinball과 같은 주간 유효 mask·용량 가중을 쓴다. ramp·fourier·cosine는 24시간 정답이 완전한 query-day만 쓰고,
    태양 위치 기준 야간은 예측·정답을 0으로 둔 24시간 profile에서 계산한 뒤 query 용량으로 가중한다.
    day_energy는 complete query-day의 m에 |Σ_h w_h·D_h·(m_h − z_h)|를 적용한다(slot_weight가 없으면 w_h=1).
    settlement는 m을 [0,1]로 clip한 뒤 group 정답과 비교하고 용량·SMP·k를 곱한다. settlement_weight에 k가 있으므로 slot_weight는 다시 곱하지 않는다.
    가중합은 고정 normalizer(scales)와 batch의 유효 칸·query 수로 나눈다.
    """
    m = submitted_median(q, median_index)
    if kind == "settlement":
        if settlement_weight is None:
            raise ValueError("settlement-penalty batch에 SMP·k 가중이 없습니다.")
        weight = mask.to(m.dtype) * capacity[:, None] * settlement_weight
        if not bool((weight > 0).any()):
            return None
        # group truth를 먼저 합한 뒤 정렬·clipping한 DL median과 비교한다. 가격은 loss에만 들어간다.
        error = torch.where(mask, m.clamp(0.0, 1.0) - z, torch.zeros_like(m))
        return (error.abs() * weight).sum() / (scales["settlement"] * mask.sum())
    if kind in ("mae", "mse"):
        if not bool(mask.any()):
            return None
        weight = mask.to(m.dtype) * capacity[:, None]
        error = torch.where(mask, m - z, torch.zeros_like(m))
        value = error.abs() if kind == "mae" else error * error
        return (value * weight).sum() / (scales["cell_plain"] * mask.sum())
    if kind == "day_energy":
        eligible = full_day & (mask | ~daylight).all(dim=1)
        if not bool(eligible.any()):
            return None
        valid = mask[eligible] & daylight[eligible]
        error = torch.where(valid, m[eligible] - z[eligible], torch.zeros_like(m[eligible]))
        if slot_weight is not None:
            error = error * slot_weight[None, :]
        return (error.sum(dim=1).abs() * capacity[eligible]).sum() / (scales["query"] * eligible.sum())
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
    return (per_day * weight).sum() / (scales["query"] * eligible.sum())


def l0_loss(q: torch.Tensor, z: torch.Tensor, mask: torch.Tensor, capacity: torch.Tensor, alphas: torch.Tensor, scale: float,
            slot_weight: torch.Tensor | None = None) -> torch.Tensor:
    """pinball = Σ M·C_A·k_t·ρ_α(z − q) / (|Q|·scale·Σ M). scale은 학습 칸 C_A·k_t의 평균(고정)이고, slot_weight가 없으면 k_t = 1이다."""
    target = torch.where(mask, z, torch.zeros_like(z))
    error = target[..., None] - q
    rho = torch.maximum(alphas * error, (alphas - 1) * error)
    weight = mask.to(q.dtype) * capacity[:, None]
    if slot_weight is not None:
        weight = weight * slot_weight[None, :]
    return (rho * weight[..., None]).sum() / (alphas.numel() * scale * mask.sum().clamp_min(1))


def observed_day_targets(observed: dict, channel_names: dict, days: np.ndarray, cutoff_h: int, excluded_days: np.ndarray | None = None) -> dict:
    """관측소-일 낮 시간 observed 평균 [대상일, 관측소, K]과 표준화 상태. cutoff 이후 시각과 excluded_days는 쓰지 않는다."""
    excluded = np.zeros(len(days), dtype=bool) if excluded_days is None else np.isin(days, excluded_days)
    out = {}
    for p, names in OBSERVED_CHANNELS.items():
        index = [channel_names[p].index(n) for n in names]
        values = np.asarray(observed[p], dtype=np.float64)[..., index]
        values[cutoff_h + 1:] = np.nan
        hours = np.clip(days[:, None] * 24 + np.arange(24)[None, :], 0, len(values) - 1)
        block = values[hours[:, OBSERVED_SLOTS]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # cutoff 이후 대상일은 값이 없어 NaN이 된다
            daily = np.nanmean(block, axis=1)
        daily[excluded] = np.nan
        reference = daily[(days * 24 + 23 <= cutoff_h) & ~excluded].reshape(-1, len(names))
        mean = np.nanmean(reference, axis=0)
        std = np.nanstd(reference, axis=0)
        std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
        out[p] = {"target": ((daily - mean) / std).astype(np.float32), "mean": mean, "std": std, "names": list(names)}
    return out


def daily_energy(z: torch.Tensor, mask: torch.Tensor, full_day: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """query 하루 발전량(Σ_h z_h / 24)과 사용 가능 여부. 24시간 정답이 모두 있는 query-day만 쓴다."""
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
                 n_observed: dict, decoder: dict | None = None):
        state = settings["dl_daystate_width"]
        super().__init__(n_in, station_to_pv, n_extra + state * len(n_in), n_query, n_quantiles, active, settings, decoder)
        self.day_encoders = nn.ModuleDict({p: DayEncoder(n_in[p], settings["dl_fusion_width"], state) for p in self.providers})
        self.energy_head = nn.Sequential(nn.Linear(state * len(self.providers), state), nn.ReLU(), nn.Linear(state, 1))
        self.observed_heads = nn.ModuleDict({p: nn.Linear(state, n_observed[p]) for p in self.providers})

    def states(self, x: dict) -> tuple[dict, torch.Tensor]:
        station = {p: self.day_encoders[p](x[p]) * getattr(self, f"active_{p}") for p in self.providers}
        pv = torch.cat([torch.einsum("ps,usd->upd", getattr(self, f"map_{p}"), station[p]) for p in self.providers], dim=-1)
        return station, pv

    def _fuse_with(self, x: dict, extras: torch.Tensor, pv: torch.Tensor) -> torch.Tensor:
        widened = torch.cat([extras, pv[:, :, None, :].expand(-1, -1, extras.shape[2], -1)], dim=-1)
        return GenerationNet.fuse(self, x, widened)

    def fuse(self, x: dict, extras: torch.Tensor) -> torch.Tensor:
        return self._fuse_with(x, extras, self.states(x)[1])

    def forward(self, x: dict, extras: torch.Tensor, day_onehot: torch.Tensor, weights: torch.Tensor, query: torch.Tensor,
                return_aux: bool = False, return_hidden: bool = False):
        station, pv = self.states(x)
        fused = self._fuse_with(x, extras, pv)
        q, hidden = self.decode(*pool_onehot(fused, day_onehot, weights), query)
        out = [q]
        if return_hidden:
            out.append(hidden)
        if return_aux:
            pooled = torch.einsum("bu,bp,upd->bd", day_onehot, weights, pv)
            out.append({"energy": self.energy_head(pooled).squeeze(-1),
                        "observed": {p: self.observed_heads[p](station[p]) for p in self.providers},
                        "site": self.site_output(fused)})
        return out[0] if len(out) == 1 else tuple(out)


def daystate_parts(aux: dict, b: dict, query_scale: float) -> dict:
    """하루 상태 보조항의 가중 전 값: 일 발전량 Huber(용량 가중, 고정 normalizer)와 observed Huber(값이 있는 관측소-일 평균)."""
    energy, valid = daily_energy(b["z"], b["mask"], b["full_day"])
    zero = aux["energy"].sum() * 0.0
    if bool(valid.any()):
        weight = valid.to(energy.dtype) * loss_cap(b)
        loss_energy = (nn.functional.huber_loss(aux["energy"], energy, reduction="none") * weight).sum() / (query_scale * valid.sum())
    else:
        loss_energy = zero
    terms = []
    for p, predicted in aux["observed"].items():
        target = b["observed"][p]
        ok = torch.isfinite(target)
        if bool(ok.any()):
            terms.append(observed_huber(predicted, target, ok, b.get("uw")))
    return {"energy": loss_energy, "observed": torch.stack(terms).mean() if terms else zero}


def observed_huber(predicted: torch.Tensor, target: torch.Tensor, ok: torch.Tensor, day_weight: torch.Tensor | None) -> torch.Tensor:
    """observed 보조항 [unique 대상일, 관측소, 채널]의 Huber 평균. 대상일 가중을 주면 분자와 유효 가중합 분모에 곱한다."""
    if day_weight is None:
        return nn.functional.huber_loss(predicted[ok], target[ok])
    w = day_weight.reshape(-1, *([1] * (target.dim() - 1))).expand_as(target)[ok]
    return (nn.functional.huber_loss(predicted[ok], target[ok], reduction="none") * w).sum() / w.sum()


def loss_cap(b: dict) -> torch.Tensor:
    """손실의 query 용량 가중. 대상일 가중(qw)이 있으면 곱한다(가중이 없으면 C_A 그대로)."""
    return b["cap"] if b.get("qw") is None else b["cap"] * b["qw"]


def site_loss(pred: torch.Tensor, target: torch.Tensor, capacity: torch.Tensor, day_weight: torch.Tensor | None = None) -> torch.Tensor | None:
    """발전소 이용률 보조 head의 용량 가중 MAE. pred·target은 [대상일, PV, 24]이고, target이 NaN인 칸(정답 없음·야간·cutoff 이후·
    held-out 대상일)은 뺀다. 쓸 칸이 없으면 None을 돌려준다. day_weight [대상일]을 주면 Σ a_d·C_i·valid로 가중한다."""
    ok = torch.isfinite(target)
    if not bool(ok.any()):
        return None
    weight = ok.to(pred.dtype) * capacity[None, :, None]
    if day_weight is not None:
        weight = weight * day_weight[:, None, None]
    error = torch.where(ok, pred - torch.nan_to_num(target), torch.zeros_like(pred)).abs()
    return (error * weight).sum() / weight.sum()


def _indicator_channels(pi) -> list[int]:
    """원본 열마다 결측을 나타낼 canonical channel(풍향은 sin 채널)."""
    names = pi.names
    return [names.index(f"{s}_sin" if s.endswith("_wind_dir") else s) for s in pi.sources]


@dataclass
class Prep:
    days: np.ndarray
    x: dict
    extras: np.ndarray
    static: np.ndarray
    flags: np.ndarray
    station_to_pv: dict
    query_scaler: Standardizer
    layout: dict
    states: dict
    observed: dict | None = None
    site: np.ndarray | None = None          # [대상일, PV, 24] 발전소 이용률 정답(없으면 NaN)
    settlement: np.ndarray | None = None    # [대상일, 24] 학습용 SMP·k. NN 입력과 inference에는 쓰지 않는다.

    def position(self, days: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(self.days, days)
        if (pos >= len(self.days)).any() or (self.days[np.minimum(pos, len(self.days) - 1)] != days).any():
            raise ValueError("DL 입력 tensor에 없는 대상일이 있습니다.")
        return pos


def settlement_day_weights(smp: np.ndarray, k: np.ndarray, days: np.ndarray, cutoff_h: int,
                           excluded_days: np.ndarray | None = None) -> np.ndarray:
    """학습용 SMP·k [대상일, 24]. cutoff 이후와 R0/E는 0이다. Generation label은 읽지 않는다."""
    days = np.asarray(days, dtype=np.int64)
    hours = days[:, None] * 24 + np.arange(24)[None, :]
    keep = (hours >= 0) & (hours <= cutoff_h) & (hours < len(smp)) & (hours < len(k))
    if excluded_days is not None:
        keep &= ~np.isin(days, excluded_days)[:, None]
    out = np.zeros(hours.shape, dtype=np.float32)
    values = np.asarray(smp)[hours[keep]] * np.asarray(k)[hours[keep]]
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("학습용 SMP·k는 유한하고 0 이상이어야 합니다.")
    out[keep] = values
    return out


def query_info(cap: np.ndarray, n_pv: np.ndarray, hhi: np.ndarray) -> np.ndarray:
    return np.column_stack([np.log(cap), np.log1p(np.asarray(n_pv, dtype=np.float64)), hhi])


def fit_hours(cutoff_h: int, excluded_days: np.ndarray | None = None) -> np.ndarray:
    """입력 전처리 fit에 쓰는 시각: cutoff 대상일까지의 모든 시각에서 학습 제외 대상일(validation panel)의 시각을 뺀다."""
    hours = np.arange((cutoff_h // 24 + 1) * 24)
    if excluded_days is None or not len(excluded_days):
        return hours
    return hours[~np.isin(hours // 24, excluded_days)]


def fit_states(inputs: InputSet, sites: Sites, cutoff_h: int, epsilon: float, train_queries: QuerySet, solar: np.ndarray,
               excluded_days: np.ndarray | None = None) -> dict:
    """cutoff 대상일까지의 station·시각(두 round 합침)으로 provider scaler를, PV·시각으로 disagreement·physics scaler를 fit한다.
    예보 입력만 쓰므로 held-out 블록의 정답은 들어가지 않는다. excluded_days(validation panel의 학습 제외 대상일)의 시각은 뺀다."""
    fit_end = (cutoff_h // 24 + 1) * 24
    hours = fit_hours(cutoff_h, excluded_days)
    keep = np.zeros(fit_end, dtype=bool)
    keep[hours] = True
    states = {}
    for p in PROVIDERS:
        pi = inputs.providers[p]
        present = pi.present[:, :fit_end] if keep.all() else pi.present[:, :fit_end] & keep[None, :, None]
        rows = pi.values[:, :fit_end][present]
        states[f"scaler_{p}"] = BranchScaler(epsilon).fit(rows).state()
    dis = _disagreement(inputs, sites, hours)
    states["disagreement"] = Standardizer().fit(dis.reshape(-1, dis.shape[-1])).state()
    if inputs.input_config.physics_proxy:
        physics, _ = physics_block(inputs, sites, solar, hours)
        states["physics"] = Standardizer().fit(physics.reshape(-1, physics.shape[-1])).state()
    states["static"] = Standardizer().fit(_static(sites)).state()
    states["query"] = Standardizer().fit(query_info(train_queries.cap, train_queries.n_pv, train_queries.hhi)).state()
    return states


def _disagreement(inputs: InputSet, sites: Sites, hours: np.ndarray) -> np.ndarray:
    """PV별 짝 station의 WA − WB [시각, PV, 12]."""
    out = []
    wa, wb = inputs.providers["wa"], inputs.providers["wb"]
    for r in range(len(ROUNDS)):
        for (na, nb) in CROSS_PROVIDER.values():
            a = wa.values[r, hours, :, wa.names.index(na)][:, sites.station_index["wa"]]
            b = wb.values[r, hours, :, wb.names.index(nb)][:, sites.station_index["wb"]]
            out.append(a - b)
    return np.stack(out, axis=-1).astype(np.float32)


def _static(sites: Sites) -> np.ndarray:
    return np.column_stack([sites.lat, sites.lon, sites.capacity])


def site_day_targets(energy: np.ndarray, sunup: np.ndarray, capacity: np.ndarray, days: np.ndarray,
                     excluded_days: np.ndarray | None = None) -> np.ndarray:
    """발전소 이용률 보조 정답 [대상일, PV, 24] = 발전량 / 설비용량.

    energy는 cutoff 이후와 무효 status를 NaN으로 둔 발전량 [시각, PV]이다. energy 범위 밖 시각(발전량이 없는 test 기간), 발전소별
    야간(sunup이 거짓), excluded_days(held-out 블록)는 NaN으로 둔다.
    """
    days = np.asarray(days, dtype=np.int64)
    hours = (days[:, None] * 24 + np.arange(24)[None, :]).ravel()
    z = np.full((len(hours), len(capacity)), np.nan, dtype=np.float32)
    known = hours < len(energy)
    z[known] = np.asarray(energy)[hours[known]] / np.asarray(capacity, dtype=np.float64)[None, :]
    z[~np.asarray(sunup)[hours]] = np.nan
    if excluded_days is not None:
        z[np.isin(np.repeat(days, 24), excluded_days)] = np.nan
    return np.ascontiguousarray(z.reshape(len(days), 24, -1).transpose(0, 2, 1))


def build_prep(inputs: InputSet, sites: Sites, solar: np.ndarray, days: np.ndarray, states: dict, observed: dict | None = None,
               site: dict | None = None) -> Prep:
    """observed = {"values", "names", "cutoff_h", "excluded_days"}를 주면 하루 상태 보조 target을, site = {"energy", "sunup",
    "excluded_days"}를 주면 발전소 이용률 보조 target을 함께 만든다."""
    days = np.unique(days)
    hours = (days[:, None] * 24 + np.arange(24)[None, :]).ravel()
    clock = time_features(days).astype(np.float32)
    x, layout = {}, {}
    for p in PROVIDERS:
        pi = inputs.providers[p]
        scaler = BranchScaler.from_state(states[f"scaler_{p}"])
        n_station, n_channel = pi.values.shape[2], pi.values.shape[3]
        raw = [pi.values[r, hours] for r in range(len(ROUNDS))]
        z = [scaler.transform(v.reshape(-1, n_channel)).reshape(v.shape) for v in raw]
        missing = [np.isnan(v) for v in raw]
        revision = np.where(missing[0] | missing[1], 0.0, z[1] - z[0]).astype(np.float32)
        if not inputs.input_config.revision:
            revision[:] = 0.0
        indicator_index = _indicator_channels(pi)
        indicators = [m[..., indicator_index].astype(np.float32) for m in missing]
        tf = np.broadcast_to(clock.reshape(len(days) * 24, 1, 4), (len(hours), n_station, 4))
        stacked = np.concatenate([z[0], z[1], revision, indicators[0], indicators[1], tf], axis=-1)
        if not pi.active:
            stacked = np.zeros_like(stacked)
        x[p] = np.ascontiguousarray(stacked.reshape(len(days), 24, n_station, -1).transpose(0, 2, 1, 3))
        layout[p] = ([f"{n}_r10_zca" for n in pi.names] + [f"{n}_r17_zca" for n in pi.names] + [f"{n}_rev_zca" for n in pi.names]
                     + [f"missing_{s}_r{r}" for r in ROUNDS for s in pi.sources] + ["slot_sin", "slot_cos", "doy_sin", "doy_cos"])
    dis = Standardizer.from_state(states["disagreement"]).transform(_disagreement(inputs, sites, hours).reshape(-1, len(CROSS_PROVIDER) * 2))
    if not inputs.disagreement:
        dis[:] = 0.0
    sol = solar[hours].astype(np.float32)
    sol = np.stack([sol[..., 0] / 90, np.maximum(sol[..., 1], 0), sol[..., 2], sol[..., 3]], axis=-1)
    if not inputs.input_config.solar_geometry:
        sol[:] = 0.0
    if inputs.input_config.physics_proxy:
        physics, _ = physics_block(inputs, sites, solar, hours)
        physics = Standardizer.from_state(states["physics"]).transform(physics.reshape(-1, len(PHYSICS_CHANNELS))).reshape(len(hours), len(sites), -1)
    else:
        physics = np.zeros((len(hours), len(sites), len(PHYSICS_CHANNELS)), dtype=np.float32)
    extras = np.concatenate([dis.reshape(len(hours), len(sites), -1), sol, physics], axis=-1)
    extras = np.ascontiguousarray(extras.reshape(len(days), 24, len(sites), -1).transpose(0, 2, 1, 3))
    static = Standardizer.from_state(states["static"]).transform(_static(sites))
    flags = np.array([float(inputs.providers[p].active) for p in PROVIDERS], dtype=np.float32)
    station_to_pv = {p: np.eye(len(sites.stations[p]), dtype=np.float32)[sites.station_index[p]] for p in PROVIDERS}
    layout["extras"] = EXTRA_CHANNELS + list(STATIC_CHANNELS) + [f"active_{p}" for p in PROVIDERS]
    layout["query"] = ["log_capacity_kw", "log1p_pv_count", "hhi"]
    targets = None
    if observed is not None:
        targets = {p: v["target"] for p, v in observed_day_targets(observed["values"], observed["names"], days, observed["cutoff_h"],
                                                                     observed.get("excluded_days")).items()}
    site_z = None
    if site is not None:
        site_z = site_day_targets(site["energy"], site["sunup"], sites.capacity, days, site.get("excluded_days"))
    return Prep(days, x, extras, static, flags, station_to_pv, Standardizer.from_state(states["query"]), layout, states, targets, site_z)


class Batcher:
    """GPU(또는 pinned CPU)에 올린 대상일 tensor에서 query batch와 대상일 입력을 만든다."""

    def __init__(self, prep: Prep, sites: Sites, device: torch.device, preload_max_gib: float):
        self.prep, self.device = prep, device
        size = sum(a.nbytes for a in prep.x.values()) + prep.extras.nbytes
        self.preloaded = size <= preload_max_gib * 1024 ** 3
        target = device if self.preloaded else torch.device("cpu")
        pin = device.type == "cuda" and not self.preloaded
        convert = (lambda a: torch.as_tensor(a, device=target)) if not pin else (lambda a: torch.as_tensor(a).pin_memory())
        self.x = {p: convert(a) for p, a in prep.x.items()}
        self.extras = convert(prep.extras)
        n_pv = len(sites)
        self.static = torch.as_tensor(np.concatenate([prep.static, np.broadcast_to(prep.flags, (n_pv, len(prep.flags)))], axis=1), device=device)
        self.capacity = torch.tensor(sites.capacity, dtype=torch.float32, device=device)
        self.input_bytes = int(size)
        self.observed = None if prep.observed is None else {p: torch.as_tensor(a, device=device) for p, a in prep.observed.items()}
        self.site = None if prep.site is None else torch.as_tensor(prep.site, device=device)
        self.settlement = None if prep.settlement is None else torch.as_tensor(prep.settlement, device=device)
        self.day_weight = None      # 학습 중 대상일 가중 [prep 대상일](train_dl이 정하고 끝나면 지운다)

    def day_inputs(self, positions: np.ndarray) -> tuple[dict, torch.Tensor]:
        """prep 대상일 위치의 provider 입력과 PV 입력(정적 정보 포함)."""
        pos = torch.as_tensor(np.asarray(positions), device=self.x["wa"].device)
        x = {p: a.index_select(0, pos).to(self.device, non_blocking=True) for p, a in self.x.items()}
        extras = self.extras.index_select(0, pos).to(self.device, non_blocking=True)
        n_day, n_pv = extras.shape[0], extras.shape[1]
        return x, torch.cat([extras, self.static[None, :, None, :].expand(n_day, n_pv, 24, -1)], dim=-1)

    def weights(self, members: np.ndarray) -> torch.Tensor:
        """membership [B, PV]의 용량 가중치(합 1)."""
        w = torch.as_tensor(np.asarray(members), device=self.device, dtype=torch.float32) * self.capacity[None, :]
        return w / w.sum(dim=1, keepdim=True)

    def query(self, cap: np.ndarray, n_pv: np.ndarray, hhi: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(self.prep.query_scaler.transform(query_info(cap, n_pv, hhi)), device=self.device)

    def batch(self, queries: QuerySet, index: np.ndarray) -> dict:
        days = queries.day[index]
        unique, inverse = np.unique(days, return_inverse=True)
        pos = self.prep.position(unique)
        x, extras = self.day_inputs(pos)
        onehot = torch.zeros((len(index), len(unique)), device=self.device)
        onehot[torch.arange(len(index), device=self.device), torch.as_tensor(inverse, device=self.device)] = 1.0
        sub = queries.subset(index)
        day_pos = torch.as_tensor(pos, device=self.device)        # onehot 열과 같은 unique 대상일 순서
        out_observed = None
        if self.observed is not None:
            out_observed = {p: a.index_select(0, day_pos) for p, a in self.observed.items()}
        site = None if self.site is None else self.site.index_select(0, day_pos)
        weighting = {}
        if self.day_weight is not None:        # unique 대상일 가중(uw)과 query의 대상일 가중(qw)
            uw = self.day_weight.index_select(0, day_pos)
            weighting = {"uw": uw, "qw": uw.index_select(0, torch.as_tensor(inverse, device=self.device))}
        if self.settlement is not None:
            weighting["settlement_weight"] = self.settlement.index_select(0, day_pos).index_select(0, torch.as_tensor(inverse, device=self.device))
        return {**weighting, "x": x, "extras": extras, "onehot": onehot, "weights": self.weights(sub.members),
                "query": self.query(sub.cap, sub.n_pv, sub.hhi),
                "z": torch.as_tensor(np.nan_to_num(sub.z), dtype=torch.float32, device=self.device),
                "mask": torch.as_tensor(sub.target_mask, device=self.device),
                "cap": torch.as_tensor(sub.cap, dtype=torch.float32, device=self.device),
                "daylight": torch.as_tensor(sub.daylight, device=self.device),
                "full_day": torch.as_tensor(sub.mask.all(axis=1), device=self.device), "observed": out_observed,
                "site_z": site, "pv_cap": self.capacity}


def batch_schedule(n_queries: int, steps: int, batch: int, rng: np.random.Generator) -> np.ndarray:
    """순회마다 섞은 순열을 이어 붙여 [steps, batch]로 만든다."""
    epochs = math.ceil(steps * batch / n_queries)
    order = np.concatenate([rng.permutation(n_queries) for _ in range(epochs)])[:steps * batch]
    return order.reshape(steps, batch)


def _seeds(seed: int) -> dict:
    children = np.random.SeedSequence(seed).spawn(4)
    return {name: int(c.generate_state(1)[0]) for name, c in zip(("batch", "init", "dropout", "aux"), children)}


def build_net(prep: Prep, settings: dict, n_quantiles: int, active: dict, site_head: bool = False, decoder: dict | None = None,
              provider_context: dict | None = None, day_scale: dict | None = None, plant_routing: dict | None = None,
              group_routing: dict | None = None) -> GenerationNet:
    """site_head=True이면 발전소 이용률 보조 head를 다른 층을 모두 만든 뒤 붙인다. decoder는 decoders.resolve의 결과(None이면 BiGRU)이다.
    provider_context(contexts.resolve의 결과)는 보조 head 다음에 붙이므로 기존 층의 초기값은 설정이 없는 모델과 같다.
    day_scale은 그 다음에 붙이며, 마지막 층을 0으로 초기화해 초기 예측을 보존한다. plant_routing·group_routing(routing.resolve_*의 결과)은
    맨 뒤에 붙이고, 출력 사영을 0으로 초기화해 초기 예측을 보존한다.
    학습·cache 로드·저장 artifact 로드가 같은 구성을 쓴다."""
    n_in = {p: prep.x[p].shape[-1] for p in PROVIDERS}
    n_extra = prep.extras.shape[-1] + prep.static.shape[-1] + len(prep.flags)
    if prep.observed is not None:
        net = DayStateNet(n_in, prep.station_to_pv, n_extra, len(prep.layout["query"]), n_quantiles, active, settings,
                          {p: a.shape[-1] for p, a in prep.observed.items()}, decoder)
    else:
        net = GenerationNet(n_in, prep.station_to_pv, n_extra, len(prep.layout["query"]), n_quantiles, active, settings, decoder)
    if site_head:
        net.site_head = nn.Linear(settings["dl_fusion_width"], 1)
    if provider_context is not None:
        net.provider_context = contexts.build(provider_context, net.providers, {p: prep.station_to_pv[p].shape[1] for p in PROVIDERS},
                                              settings["dl_provider_mlp_widths"][-1], active)
    if day_scale is not None:
        net.day_scale = DailyScaleHead(getattr(net, net.decoder_name).out_width, day_scale["hidden"], day_scale["max_log_gain"])
    width, hidden = settings["dl_provider_width"], settings["dl_fusion_width"]
    start = prep.extras.shape[-1]                               # 이 뒤에 정적 정보·provider 사용 여부가 붙는다(Batcher)
    stop = start + prep.static.shape[-1] + len(prep.flags)
    if plant_routing is not None:
        n_common = net.fusion[0].in_features - width * len(PROVIDERS)
        keep = ((0, EXTRA_CHANNELS.index(PHYSICS_CHANNELS[0])), (stop, n_common))   # disagreement·태양 위치, 정적 정보를 건너뛴 하루 상태
        net.plant_routing = routing.build_plant(plant_routing, PROVIDERS, width, n_in, n_common, (start, stop), keep, hidden, prep.station_to_pv)
    if group_routing is not None:
        net.group_routing = routing.build_group(group_routing, 2 * hidden + len(prep.layout["query"]), width, hidden,
                                                (start, start + prep.static.shape[-1]))
    return net


def slot_weight_tensor(loss: LossConfig, device) -> torch.Tensor | None:
    """정산 시간 가중 k를 쓰는 손실이면 24개 slot의 k를 돌려준다."""
    weight = slot_weight(loss.time_weight)
    return None if weight is None else torch.as_tensor(weight, dtype=torch.float32, device=device)


def _aux(loss: LossConfig, q: torch.Tensor, b: dict, median_index: int, scales: dict):
    return aux_loss(loss.aux, q, b["z"], b["mask"], b["daylight"], b["full_day"], loss_cap(b), median_index, scales,
                    slot_weight_tensor(loss, q.device), b.get("settlement_weight"))


def _grad_norm(value: torch.Tensor, params: list, retain: bool) -> float:
    grads = torch.autograd.grad(value, params, retain_graph=retain, allow_unused=True)
    return float(torch.sqrt(sum((g.detach() ** 2).sum() for g in grads if g is not None)))


def daystate_aux_weights(loss: LossConfig, settings: dict) -> dict:
    return {"energy": 0.0 if loss.aux == "day_energy" else float(settings["dl_daystate_energy_weight"]),
            "observed": float(settings["dl_daystate_observed_weight"])}


TOTAL_TERMS = ("main", "aux_weighted", "daystate_energy", "daystate_observed", "site_weighted")


def batch_terms(q: torch.Tensor, b: dict, loss: LossConfig, alphas: torch.Tensor, median_index: int, lam: float, aux_enabled: bool,
                state_aux: dict | None, settings: dict, scales: dict, site_pred: torch.Tensor | None = None, lam_site: float = 0.0) -> dict:
    """손실 항별 기여. main은 주손실, aux는 보조손실의 가중 전 값, aux_weighted는 λ(또는 λ_Q)를 곱한 값,
    daystate_energy·daystate_observed는 하루 상태 보조항에 가중을 곱한 값, site는 발전소 이용률 손실의 가중 전 값, site_weighted는
    λ_site를 곱한 값이다. 보조손실과 발전소 이용률 항은 warm-up 뒤(aux_enabled)에만 둔다. 총손실은 TOTAL_TERMS의 합이다.
    """
    terms = {}
    slot = slot_weight_tensor(loss, q.device)
    terms["main"] = l0_loss(q, b["z"], b["mask"], loss_cap(b), alphas, scales["cell"], slot)
    aux = _aux(loss, q, b, median_index, scales) if loss.aux is not None and aux_enabled else None
    if aux is not None:
        terms["aux"], terms["aux_weighted"] = aux, lam * aux
    if state_aux is not None:
        weights = daystate_aux_weights(loss, settings)
        parts = daystate_parts(state_aux, b, scales["query"])
        terms["daystate_energy"] = weights["energy"] * parts["energy"]
        terms["daystate_observed"] = weights["observed"] * parts["observed"]
    if site_pred is not None and aux_enabled:
        value = site_loss(site_pred, b["site_z"], b["pv_cap"], b.get("uw"))
        if value is not None:
            terms["site"], terms["site_weighted"] = value, lam_site * value
    return terms


def total_of(terms: dict) -> torch.Tensor:
    """학습 총손실: 가중을 곱한 기여(TOTAL_TERMS)만 한 번씩 더한다. 가중 전 값(aux, site)은 기록용이다."""
    return sum(terms[k] for k in TOTAL_TERMS if k in terms)


def _forward_train(net: GenerationNet, b: dict, site: bool = False):
    """학습 forward: 분위수, 하루 상태 보조 출력(DayStateNet이 아니면 None), 발전소 이용률 출력(site=False이거나 head가 없으면 None)."""
    if isinstance(net, DayStateNet):
        q, state_aux = net(b["x"], b["extras"], b["onehot"], b["weights"], b["query"], return_aux=True)
        site_pred = state_aux.pop("site")
        return q, state_aux, site_pred if site else None
    if site:
        q, site_pred = net(b["x"], b["extras"], b["onehot"], b["weights"], b["query"], return_site=True)
        return q, None, site_pred
    return _forward(net, b), None, None


def term_gradient_norms(net: GenerationNet, batcher: Batcher, queries: QuerySet, batches: np.ndarray, loss: LossConfig, alphas: torch.Tensor,
                        median_index: int, lam: float, aux_enabled: bool, settings: dict, scales: dict, site: bool = False,
                        lam_site: float = 0.0) -> dict:
    """총손실에 들어가는 항별 기여(가중 후, TOTAL_TERMS)의 기울기 크기 평균. 학습 RNG 순서는 바꾸지 않는다."""
    params = [p for p in net.parameters() if p.requires_grad]
    norms: dict = {}
    devices = [batcher.device.index if batcher.device.index is not None else torch.cuda.current_device()] if batcher.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        net.train()
        for index in batches:
            b = batcher.batch(queries, index)
            q, state_aux, site_pred = _forward_train(net, b, site)
            terms = {k: v for k, v in batch_terms(q, b, loss, alphas, median_index, lam, aux_enabled, state_aux, settings, scales, site_pred,
                                                  lam_site).items()
                     if k in TOTAL_TERMS and v.requires_grad}
            names = list(terms)
            for j, name in enumerate(names):
                norms.setdefault(name, []).append(_grad_norm(terms[name], params, retain=j < len(names) - 1))
    return {name: float(np.mean(v)) for name, v in norms.items()}


def _forward(net: GenerationNet, b: dict) -> torch.Tensor:
    return net(b["x"], b["extras"], b["onehot"], b["weights"], b["query"])


def calibrate_lambda(net: GenerationNet, batcher: Batcher, queries: QuerySet, batches: np.ndarray, loss: LossConfig, alphas: torch.Tensor,
                     median_index: int, scales: dict) -> tuple[float, dict]:
    """λ_k = r·E‖∇L_pinball‖/E‖∇L_k‖. Calibration forward 뒤에도 후속 batch의 dropout RNG 순서를 유지한다."""
    params = [p for p in net.parameters() if p.requires_grad]
    norms0, normsk = [], []
    devices = [batcher.device.index if batcher.device.index is not None else torch.cuda.current_device()] if batcher.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        net.train()
        for index in batches:
            b = batcher.batch(queries, index)
            q = _forward(net, b)
            aux = _aux(loss, q, b, median_index, scales)
            if aux is None:
                continue
            norms0.append(_grad_norm(l0_loss(q, b["z"], b["mask"], loss_cap(b), alphas, scales["cell"], slot_weight_tensor(loss, q.device)),
                                     params, retain=True))
            normsk.append(_grad_norm(aux, params, retain=False))
    if not norms0 or np.mean(normsk) <= 0:
        raise RuntimeError(f"{loss.name}의 λ를 정할 batch가 없습니다.")
    lam = loss.aux_ratio * float(np.mean(norms0)) / max(float(np.mean(normsk)), 1e-12)
    return lam, {"lambda": lam, "ratio": loss.aux_ratio, "grad_l0_mean": float(np.mean(norms0)), "grad_aux_mean": float(np.mean(normsk)),
                 "batches_used": len(norms0), "batches": len(batches)}


def calibrate_site(net: GenerationNet, batcher: Batcher, queries: QuerySet, batches: np.ndarray, loss: LossConfig, alphas: torch.Tensor,
                   scales: dict, ratio: float) -> tuple[float, dict]:
    """λ_site = r·E‖∇L_main‖/E‖∇L_site‖. L_main은 학습과 같은 pinball(고정 normalizer·mask·용량·slot 가중)이고 다른 보조항을 더하지 않는다.

    기울기 크기는 학습 가능한 parameter 전체의 L2 norm이고, 한 손실에 연결되지 않은 parameter의 기울기는 0으로 센다. 따라서 r은
    calibration batch에서 전체 parameter 기울기 크기의 평균 비율이다. calibrate_lambda처럼 RNG를 되돌려 이후 batch의 dropout 순서를
    유지하고, optimizer·scheduler는 진행하지 않는다.
    """
    params = [p for p in net.parameters() if p.requires_grad]
    norms0, norms = [], []
    devices = [batcher.device.index if batcher.device.index is not None else torch.cuda.current_device()] if batcher.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        net.train()
        for index in batches:
            b = batcher.batch(queries, index)
            q, _, site_pred = _forward_train(net, b, site=True)
            value = site_loss(site_pred, b["site_z"], b["pv_cap"], b.get("uw"))
            if value is None:
                continue
            norms0.append(_grad_norm(l0_loss(q, b["z"], b["mask"], loss_cap(b), alphas, scales["cell"], slot_weight_tensor(loss, q.device)),
                                     params, retain=True))
            norms.append(_grad_norm(value, params, retain=False))
    if not norms0 or np.mean(norms) <= 0:
        raise RuntimeError("발전소 이용률 보조 head의 λ를 정할 batch가 없습니다.")
    lam = ratio * float(np.mean(norms0)) / max(float(np.mean(norms)), 1e-12)
    return lam, {"lambda": lam, "ratio": ratio, "parameter_scope": "all_trainable", "grad_main_mean": float(np.mean(norms0)),
                 "grad_site_mean": float(np.mean(norms)), "batches_used": len(norms0), "batches": len(batches)}


def _state_sha(net: GenerationNet) -> str:
    h = hashlib.sha256()
    for key, value in net.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train_dl(prep: Prep, batcher: Batcher, queries: QuerySet, settings: dict, quantiles: np.ndarray, seed: int, steps: int,
             loss: LossConfig, site_aux_ratio: float = 0.0, day_weight: np.ndarray | None = None, init_state: dict | None = None,
             fixed: dict | None = None, learning_rate: float | None = None, warmup_steps: int | None = None,
             decoder: dict | None = None, provider_context: dict | None = None, day_scale: dict | None = None,
             plant_routing: dict | None = None, group_routing: dict | None = None) -> tuple[dict, dict]:
    """pinball warm-up 뒤 median auxiliary의 λ를 정하고 고정한다. site_aux_ratio > 0이면 발전소 이용률 보조 head를 두고 warm-up 직후
    λ_site를 정해 그 뒤부터 더한다. history에는 50 step마다 항별 기여(main, aux, aux_weighted, daystate_energy, daystate_observed,
    site, site_weighted, total)의 평균을, term_gradient_norms에는 warm-up 시점과 학습 끝의 항별 기울기 크기를 남긴다.

    계절 학습용 선택 인자:
    - day_weight [prep 대상일]: 각 지도 항과 고정 normalizer(loss_scales)에 대상일 가중을 넣는다. calibration도 같은 가중 손실을 쓴다.
    - init_state: 이 weights에서 시작한다(초기화 대신).
    - fixed {"lambda", "lambda_site", "loss_scales"}: warm-up과 λ calibration을 하지 않고 첫 step부터 이 λ와 normalizer를 쓴다.
    - learning_rate: 최대 학습률(없으면 settings), warmup_steps: warm-up step 수(없으면 손실 설정).
    decoder는 pooled 시퀀스 decoder 설정(decoders.resolve의 결과, None이면 BiGRU)이고, provider_context는 provider 표현 context 설정
    (contexts.resolve의 결과, None이면 없음)이다.
    day_scale이 있으면 집합별 하루 배율 head를 기존 손실로 함께 학습한다. 예측·배정 탐색에도 같은 decode 경로를 쓴다.
    plant_routing·group_routing은 temporal routing 설정(routing.resolve_*의 결과, None이면 없음)이다.
    """
    device = batcher.device
    median_index = int(np.flatnonzero(np.isclose(quantiles, 0.5))[0])
    seeds = _seeds(seed)
    used = queries.subset(np.flatnonzero(queries.target_mask.any(axis=1)))
    use_site = site_aux_ratio > 0
    if use_site and prep.site is None:
        raise ValueError("site_aux_ratio > 0에는 발전소 이용률 정답이 있는 Prep이 필요합니다.")
    torch.manual_seed(seeds["init"])
    active = {p: bool(prep.flags[i]) for i, p in enumerate(PROVIDERS)}
    net = build_net(prep, settings, len(quantiles), active, site_head=use_site, decoder=decoder,
                    provider_context=provider_context, day_scale=day_scale, plant_routing=plant_routing, group_routing=group_routing).to(device)
    if init_state is not None:
        net.load_state_dict(init_state)
    schedule = batch_schedule(len(used), steps, settings["dl_batch_queries"], np.random.default_rng(seeds["batch"]))
    torch.manual_seed(seeds["dropout"])
    lr = float(settings["dl_learning_rate"] if learning_rate is None else learning_rate)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=settings["dl_weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    alphas = torch.as_tensor(quantiles, dtype=torch.float32, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    if fixed is not None:
        warmup = 0
    else:
        warmup = min(loss.warmup_steps if warmup_steps is None else int(warmup_steps), steps)
    if use_site and warmup >= steps:
        raise ValueError("발전소 이용률 보조 head는 warm-up 뒤 학습 step이 있어야 합니다.")
    lam, calibration, warmup_sha = 0.0, None, None
    lam_site, site_calibration = 0.0, None
    query_weight = None if day_weight is None else np.asarray(day_weight, dtype=np.float64)[prep.position(used.day)]
    if fixed is not None:
        lam, lam_site, scales = float(fixed.get("lambda") or 0.0), float(fixed.get("lambda_site") or 0.0), dict(fixed["loss_scales"])
    else:
        price = None if prep.settlement is None else prep.settlement[prep.position(used.day)]
        scales = loss_scales(loss, used, query_weight, settlement_weight=price)
    probe = settings.get("dl_term_probe_batches", 5)
    batcher.day_weight = None if day_weight is None else torch.as_tensor(np.asarray(day_weight, dtype=np.float32), device=device)
    try:
        history, gradient_norms, warmup_sha, lam, calibration, lam_site, site_calibration = _train_loop(
            net, batcher, used, schedule, optimizer, scheduler, loss, alphas, median_index, settings, scales, use_site, site_aux_ratio, steps, warmup,
            probe, fixed is None, lam, lam_site)
    finally:
        batcher.day_weight = None
    details = _train_details(seed, seeds, steps, settings, queries, used, schedule, batcher, net, started, history, loss, warmup, warmup_sha,
                             lam, calibration, scales, gradient_norms)
    if any(v is not None for v in (day_weight, init_state, fixed, learning_rate, warmup_steps)):     # 계절 학습 경로의 실제 설정
        details["training_mode"] = {
            "learning_rate": lr, "init_from_state": init_state is not None, "calibrated": fixed is None,
            "fixed": None if fixed is None else {"lambda": lam, "lambda_site": lam_site, "loss_scales": scales},
            "day_weight": None if query_weight is None else {"min": float(query_weight.min()), "max": float(query_weight.max()),
                                                              "mean_used_query": float(query_weight.mean())}}
    if use_site:
        details["site_aux"] = {"ratio": site_aux_ratio, "lambda": lam_site, "calibration": site_calibration}
    if day_scale is not None:
        details["day_scale"] = {**day_scale, "version": DAY_SCALE_VERSION}
    if plant_routing is not None:
        details["plant_routing"] = routing.plant_identity(plant_routing)
    if group_routing is not None:
        details["group_routing"] = routing.group_identity(group_routing)
    if device.type == "cuda":
        details["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    return {k: v.detach().cpu() for k, v in net.state_dict().items()}, details


def _routing_grad_norm(net: GenerationNet) -> float | None:
    """읽을 시각과 가중을 정하는 parameter가 받은 gradient의 L2 norm(clip 전). routing이 없으면 None이다."""
    modules = [m for m in (net.plant_routing, net.group_routing) if m is not None]
    if not modules:
        return None
    grads = [p.grad for m in modules for p in m.routing_parameters() if p.grad is not None]
    return float(torch.sqrt(sum((g ** 2).sum() for g in grads))) if grads else 0.0


def _train_loop(net, batcher, used, schedule, optimizer, scheduler, loss, alphas, median_index, settings, scales, use_site, site_aux_ratio, steps, warmup,
                probe, calibrate, lam, lam_site):
    """학습 step을 돌리고 (history, 항별 기울기 크기, warm-up state sha, λ, λ calibration, λ_site, λ_site calibration)을 돌려준다.
    calibrate가 거짓이면(고정 λ) calibration 분기를 건너뛴다."""
    history, window, window_steps = [], {}, 0
    gradient_norms, warmup_sha, calibration, site_calibration = {}, None, None, None
    for step in range(steps):
        if step == warmup:
            warmup_sha = _state_sha(net)
            if calibrate and loss.aux is not None:
                lam, calibration = calibrate_lambda(net, batcher, used, schedule[step:step + loss.calibration_batches], loss, alphas, median_index,
                                                    scales)
            if calibrate and use_site:
                lam_site, site_calibration = calibrate_site(net, batcher, used, schedule[step:step + loss.calibration_batches], loss, alphas,
                                                            scales, site_aux_ratio)
            gradient_norms["warmup"] = term_gradient_norms(net, batcher, used, schedule[step:step + probe], loss, alphas, median_index, lam,
                                                           True, settings, scales, use_site, lam_site)
        net.train()
        b = batcher.batch(used, schedule[step])
        optimizer.zero_grad(set_to_none=True)
        q, state_aux, site_pred = _forward_train(net, b, use_site)
        terms = batch_terms(q, b, loss, alphas, median_index, lam, step >= warmup, state_aux, settings, scales, site_pred, lam_site)
        total = total_of(terms)
        if not torch.isfinite(total):
            raise RuntimeError(f"DL 손실이 유한하지 않습니다(step {step}).")
        total.backward()
        routing_grad = _routing_grad_norm(net)
        torch.nn.utils.clip_grad_norm_(net.parameters(), settings["dl_gradient_clip"])
        optimizer.step()
        scheduler.step()
        window_steps += 1
        if routing_grad is not None:
            window["routing_offset_grad"] = window.get("routing_offset_grad", 0.0) + routing_grad
        for name, value in {**terms, "total": total}.items():
            window[name] = window.get(name, 0.0) + float(value.detach())
        if (step + 1) % 50 == 0:
            # 항이 없는 step(warm-up 전 보조손실)은 0으로 세어 모든 항을 같은 step 수로 평균한다. 그래야 total이 항별 평균의 합이다.
            history.append({"step": step + 1, **{name: v / window_steps for name, v in window.items()}})
            window, window_steps = {}, 0
    gradient_norms["final"] = term_gradient_norms(net, batcher, used, schedule[-probe:], loss, alphas, median_index, lam, True, settings, scales,
                                                  use_site, lam_site)
    return history, gradient_norms, warmup_sha, lam, calibration, lam_site, site_calibration


def _train_details(seed, seeds, steps, settings, queries, used, schedule, batcher, net, started, history, loss, warmup, warmup_sha, lam,
                   calibration, scales, gradient_norms) -> dict:
    return {"seed": seed, "seed_streams": seeds, "steps": steps, "batch_queries": settings["dl_batch_queries"],
               "queries_total": len(queries), "queries_used": len(used), "labelled_hours": int(used.target_mask.sum()),
               "effective_epochs": float(steps * schedule.shape[1] / len(used)),
               "schedule_sha": hashlib.sha256(schedule.tobytes()).hexdigest(), "preloaded": batcher.preloaded,
               "input_bytes": batcher.input_bytes, "parameters": int(sum(p.numel() for p in net.parameters())),
               "seconds": time.perf_counter() - started, "history": history, "loss": loss.__dict__, "day_state": isinstance(net, DayStateNet),
               "warmup_steps": warmup, "warmup_state_sha": warmup_sha, "lambda": lam, "calibration": calibration,
               "loss_scales": scales, "term_gradient_norms": gradient_norms}
