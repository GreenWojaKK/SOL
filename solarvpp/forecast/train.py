"""DL 입력 tensor 구성과 학습·예측.

provider 입력은 학습 구간에서 fit한 대체·표준화·ZCA로 변환하고, round 10·17·revision·결측 indicator·시간 표현을 채널로 둔다.
PV 단위 공통 입력(x_common)은 disagreement·태양 위치·PV physics proxy·정적 정보이며, 입력 구성에서 끈 묶음은 0으로 둔다.
보조손실 칸은 pinball만으로 warm-up한 뒤 gradient 비율로 λ를 한 번 정하고 고정한다.
참조: docs/solarvpp-reference.md#aux-calibration.
계절 학습: 대상일 가중(day_weight)을 주면 각 지도 항을 reduction 전에 그 대상일의 가중으로 곱한다(query 항은 query의 대상일, 발전소
이용률·observed 항은 batch의 unique 대상일). 시작 weights와 고정 λ(fixed)를 주면 warm-up과 λ calibration 없이 첫 step부터 이어 학습한다.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from ..config import LossConfig, settlement_slot_weight
from ..data import PROVIDERS, ROUNDS, Sites
from .features import CROSS_PROVIDER, BranchScaler, InputSet, Standardizer, physics_block, time_features
from .models import (DayStateNet, GenerationNet, PlantTimeShift, PlantNearbyHours, NearbyHoursRouting, aux_loss, daystate_loss, l0_loss, loss_cap,
                     observed_day_targets, site_loss)
from .queries import QuerySet

PHYSICS_CHANNELS = ["phys_ghi_clear_sky"] + [f"phys_{k}_{p}_r{r}" for p in PROVIDERS for r in ROUNDS
                                             for k in ("ghi_cloud", "dewpoint_spread", "temp_derate")]


EXTRA_CHANNELS = ([f"dis_{k}_r{r}" for r in ROUNDS for k in CROSS_PROVIDER] + ["solar_elev_90", "solar_cosz_pos", "solar_azim_sin", "solar_azim_cos"]
                  + PHYSICS_CHANNELS)


STATIC_CHANNELS = ("lat", "lon", "capacity_kw")


def configure_determinism(threads: int) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG를 torch import 전에 설정해야 합니다(python -m solarvpp으로 실행).")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


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
    temporal: dict | None = None            # temporal routing 설정(time_shift·scales·nearby_hours·gate·group_nearby_hours·group_context)

    def position(self, days: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(self.days, days)
        if (pos >= len(self.days)).any() or (self.days[np.minimum(pos, len(self.days) - 1)] != days).any():
            raise ValueError("DL 입력 tensor에 없는 대상일이 있습니다.")
        return pos


def fit_states(inputs: InputSet, sites: Sites, cutoff_h: int, epsilon: float, train_queries: QuerySet, solar: np.ndarray) -> dict:
    """cutoff 대상일까지의 station·시각(두 round 합침)으로 provider scaler를, PV·시각으로 disagreement·physics scaler를 fit한다."""
    fit_end = (cutoff_h // 24 + 1) * 24
    states = {}
    for p in PROVIDERS:
        pi = inputs.providers[p]
        rows = pi.values[:, :fit_end][pi.present[:, :fit_end]]
        states[f"scaler_{p}"] = BranchScaler(epsilon).fit(rows).state()
    dis = _disagreement(inputs, sites, np.arange(fit_end))
    states["disagreement"] = Standardizer().fit(dis.reshape(-1, dis.shape[-1])).state()
    if inputs.input_config.physics_proxy:
        physics, _ = physics_block(inputs, sites, solar, np.arange(fit_end))
        states["physics"] = Standardizer().fit(physics.reshape(-1, physics.shape[-1])).state()
    states["static"] = Standardizer().fit(_static(sites)).state()
    states["query"] = Standardizer().fit(_query_info(train_queries)).state()
    return states


def _disagreement(inputs: InputSet, sites: Sites, hours: np.ndarray) -> np.ndarray:
    """PV별 짝 station의 WA − WB [시각, PV, 12]. 두 provider가 모두 있을 때만 정의한다."""
    out = []
    wa, wb = inputs.providers["wa"], inputs.providers["wb"]
    for r in range(len(ROUNDS)):
        for key, (na, nb) in CROSS_PROVIDER.items():
            a = wa.values[r, hours, :, wa.names.index(na)][:, sites.station_index["wa"]]
            b = wb.values[r, hours, :, wb.names.index(nb)][:, sites.station_index["wb"]]
            out.append(a - b)
    return np.stack(out, axis=-1).astype(np.float32)


def _static(sites: Sites) -> np.ndarray:
    return np.column_stack([sites.lat, sites.lon, sites.capacity])


def _query_info(queries: QuerySet) -> np.ndarray:
    return np.column_stack([np.log(queries.cap), np.log1p(queries.n_pv.astype(np.float64)), queries.hhi])


def build_prep(inputs: InputSet, sites: Sites, solar: np.ndarray, days: np.ndarray, states: dict, observed: dict | None = None,
               site: dict | None = None) -> Prep:
    """observed = {"values", "names", "cutoff_h"}를 주면 하루 상태 보조 target(관측소-일 observed 평균)을 함께 만든다.
    site = {"energy": cutoff 이후·무효 시각을 NaN으로 둔 발전량 [시각, PV], "sunup": [시각, PV]}를 주면 발전소 이용률 보조 target을 만든다."""
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
        targets = {p: v["target"] for p, v in observed_day_targets(observed["values"], observed["names"], days, observed["cutoff_h"]).items()}
    site_z = None
    if site is not None:
        z = np.full((len(hours), len(sites)), np.nan, dtype=np.float32)
        known = hours < len(site["energy"])
        z[known] = site["energy"][hours[known]] / sites.capacity[None, :]
        z[~np.asarray(site["sunup"])[hours]] = np.nan
        site_z = np.ascontiguousarray(z.reshape(len(days), 24, -1).transpose(0, 2, 1))
    return Prep(days, x, extras, static, flags, station_to_pv, Standardizer.from_state(states["query"]), layout, states, targets, site_z,
                temporal_config(inputs.input_config))


class Batcher:
    """GPU(또는 pinned CPU)에 올린 대상일 tensor에서 query batch를 만든다."""

    def __init__(self, prep: Prep, sites: Sites, device: torch.device, preload_max_gib: float):
        self.prep, self.device = prep, device
        size = sum(a.nbytes for a in prep.x.values()) + prep.extras.nbytes
        self.preloaded = size <= preload_max_gib * 1024 ** 3
        target = device if self.preloaded else torch.device("cpu")
        convert = (lambda a: torch.as_tensor(a, device=target)) if self.preloaded else (lambda a: torch.as_tensor(a).pin_memory())
        self.x = {p: convert(a) for p, a in prep.x.items()}
        self.extras = convert(prep.extras)
        n_pv = len(sites)
        static = torch.as_tensor(np.concatenate([prep.static, np.broadcast_to(prep.flags, (n_pv, len(prep.flags)))], axis=1), device=device)
        self.static = static
        self.capacity = torch.tensor(sites.capacity, dtype=torch.float32, device=device)
        self.input_bytes = int(size)
        self.observed = None if prep.observed is None else {p: torch.as_tensor(a, device=device) for p, a in prep.observed.items()}
        self.site = None if prep.site is None else torch.as_tensor(prep.site, device=device)
        self.day_weight = None      # 학습 중 대상일 가중 [prep 대상일](train_dl이 정하고 끝나면 지운다)

    def batch(self, queries: QuerySet, index: np.ndarray) -> dict:
        days = queries.day[index]
        unique, inverse = np.unique(days, return_inverse=True)
        pos = torch.as_tensor(self.prep.position(unique), device=self.x["wa"].device)
        x = {p: a.index_select(0, pos).to(self.device, non_blocking=True) for p, a in self.x.items()}
        extras = self.extras.index_select(0, pos).to(self.device, non_blocking=True)
        n_day, n_pv = extras.shape[0], extras.shape[1]
        extras = torch.cat([extras, self.static[None, :, None, :].expand(n_day, n_pv, 24, -1)], dim=-1)
        onehot = torch.zeros((len(index), len(unique)), device=self.device)
        onehot[torch.arange(len(index), device=self.device), torch.as_tensor(inverse, device=self.device)] = 1.0
        members = torch.as_tensor(queries.members[index], device=self.device, dtype=torch.float32)
        weights = members * self.capacity[None, :]
        weights = weights / weights.sum(dim=1, keepdim=True)
        query = torch.as_tensor(self.prep.query_scaler.transform(_query_info(queries.subset(index))), device=self.device)
        z = torch.as_tensor(np.nan_to_num(queries.z[index]), dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(queries.target_mask[index], device=self.device)
        cap = torch.as_tensor(queries.cap[index], dtype=torch.float32, device=self.device)
        daylight = torch.as_tensor(queries.daylight[index], device=self.device)
        full_day = torch.as_tensor(queries.mask[index].all(axis=1), device=self.device)
        out_observed = None
        if self.observed is not None:
            observed_pos = pos.to(self.device)
            out_observed = {p: a.index_select(0, observed_pos) for p, a in self.observed.items()}
        site = None if self.site is None else self.site.index_select(0, pos.to(self.device))
        weighting = {}
        if self.day_weight is not None:        # unique 대상일 가중(uw)과 query의 대상일 가중(qw)
            uw = self.day_weight.index_select(0, pos.to(self.device))
            weighting = {"uw": uw, "qw": uw.index_select(0, torch.as_tensor(inverse, device=self.device))}
        return {**weighting, "x": x, "extras": extras, "onehot": onehot, "weights": weights, "query": query, "z": z, "mask": mask, "cap": cap,
                "observed": out_observed, "site_z": site,
                "daylight": daylight, "full_day": full_day}


def batch_schedule(n_queries: int, steps: int, batch: int, rng: np.random.Generator) -> np.ndarray:
    """순회마다 섞은 순열을 이어 붙여 [steps, batch]로 만든다."""
    epochs = math.ceil(steps * batch / n_queries)
    order = np.concatenate([rng.permutation(n_queries) for _ in range(epochs)])[:steps * batch]
    return order.reshape(steps, batch)


def _seeds(seed: int) -> dict:
    children = np.random.SeedSequence(seed).spawn(4)
    return {name: int(c.generate_state(1)[0]) for name, c in zip(("batch", "init", "dropout", "aux"), children)}


def temporal_config(input_config) -> dict | None:
    """입력 구성의 temporal routing 설정. 기본값인 항목은 넣지 않는다(routing이 없으면 None)."""
    if not (input_config.time_shift or input_config.nearby_hours or input_config.group_nearby_hours):
        return None
    if input_config.nearby_gate and not (input_config.nearby_hours or input_config.group_nearby_hours):
        raise ValueError("nearby_gate는 window attention(nearby_hours·group_nearby_hours)에만 둔다.")
    if input_config.group_context not in ("pooled", "neutral"):
        raise ValueError(f"알 수 없는 group_context입니다: {input_config.group_context}")
    temporal = {"time_shift": input_config.time_shift}
    if input_config.time_shift:
        temporal["scales"] = [float(s) for s in input_config.shift_scales.split(",")]
    if input_config.nearby_hours:
        temporal["nearby_hours"] = input_config.nearby_hours
    if input_config.nearby_gate:
        temporal["gate"] = True
    if input_config.group_context != "pooled":
        temporal["group_context"] = input_config.group_context
    if input_config.group_nearby_hours:
        temporal["group_nearby_hours"] = input_config.group_nearby_hours
    return temporal


def build_net(prep: Prep, settings: dict, n_quantiles: int, active: dict) -> GenerationNet:
    n_in = {p: prep.x[p].shape[-1] for p in PROVIDERS}
    n_extra = prep.extras.shape[-1] + prep.static.shape[-1] + len(prep.flags)
    if prep.observed is not None:
        net = DayStateNet(n_in, prep.station_to_pv, n_extra, len(prep.layout["query"]), n_quantiles, active, settings,
                          {p: a.shape[-1] for p, a in prep.observed.items()})
    else:
        net = GenerationNet(n_in, prep.station_to_pv, n_extra, len(prep.layout["query"]), n_quantiles, active, settings)
    if prep.site is not None:
        net.site_head = torch.nn.Linear(settings["dl_fusion_width"], 1)
    if prep.temporal is not None:
        cfg, width, hidden = prep.temporal, settings["dl_provider_width"], settings["dl_fusion_width"]
        start = prep.extras.shape[-1]                           # 이 뒤에 정적 정보·provider 사용 여부가 붙는다(Batcher)
        stop = start + prep.static.shape[-1] + len(prep.flags)
        n_forecast = {p: prep.x[p].shape[-1] for p in PROVIDERS}
        n_common = net.fusion[0].in_features - width * len(PROVIDERS)                  # fusion 공통 입력 채널 수
        if cfg["time_shift"]:
            net.time_shift = PlantTimeShift(PROVIDERS, width, n_forecast, n_common, (start, stop), hidden, tuple(cfg["scales"]),
                                          "weather" in cfg["time_shift"], prep.station_to_pv if "region" in cfg["time_shift"] else None)
        if cfg.get("group_context") == "neutral":
            net.group_neutral = (start, start + prep.static.shape[-1])                    # 정적 정보(위경도·용량)만 0으로 둔다
        if cfg.get("nearby_hours"):                                   # 날씨 맥락: disagreement·태양 위치, 정적 정보를 건너뛴 하루 상태
            keep = ((0, EXTRA_CHANNELS.index(PHYSICS_CHANNELS[0])), (stop, n_common))
            net.nearby_hours = PlantNearbyHours(PROVIDERS, width, n_forecast, keep, hidden, cfg["nearby_hours"], cfg.get("gate", False))
        if cfg.get("group_nearby_hours"):
            n_group = net.gru.layers[0].input_size
            net.group_nearby_hours = NearbyHoursRouting(n_group, width, n_group, hidden, n_group, cfg["group_nearby_hours"], cfg.get("gate", False))
    return net


def _forward(net: GenerationNet, b: dict) -> torch.Tensor:
    return net(b["x"], b["extras"], b["onehot"], b["weights"], b["query"])


def _forward_site(net: GenerationNet, b: dict) -> tuple[torch.Tensor, torch.Tensor | None, dict | None]:
    """분위수 출력, 발전소 이용률 보조 출력(head가 없으면 None), DayState 보조 출력(DayStateNet이 아니면 None)."""
    if isinstance(net, DayStateNet):
        q, state_aux = net(b["x"], b["extras"], b["onehot"], b["weights"], b["query"], return_aux=True)
        return q, state_aux.pop("site"), state_aux
    q, site = net(b["x"], b["extras"], b["onehot"], b["weights"], b["query"], return_site=True)
    return q, site, None


def _aux(loss: LossConfig, q: torch.Tensor, b: dict, median_index: int):
    return aux_loss(loss.aux, q, b["z"], b["mask"], b["daylight"], b["full_day"], loss_cap(b), median_index,
                   slot_weight_tensor(loss, q.device))


def _grad_norm(value: torch.Tensor, params: list, retain: bool) -> float:
    grads = torch.autograd.grad(value, params, retain_graph=retain, allow_unused=True)
    return float(torch.sqrt(sum((g.detach() ** 2).sum() for g in grads if g is not None)))


def slot_weight_tensor(loss: LossConfig, device) -> torch.Tensor | None:
    """정산 시간 가중 k를 쓰는 손실이면 24개 slot의 k를 돌려준다."""
    return torch.as_tensor(settlement_slot_weight(), dtype=torch.float32, device=device) if loss.time_weight == "settlement" else None


def daystate_aux_weights(loss: LossConfig, settings: dict) -> dict:
    """day-energy는 median trajectory를 직접 supervise하고, DayStateNet에는 observed supervision을 유지한다."""
    return {"energy": 0.0 if loss.aux == "day_energy" else float(settings["dl_daystate_energy_weight"]),
            "observed": float(settings["dl_daystate_observed_weight"])}


def batch_objective(q: torch.Tensor, b: dict, loss: LossConfig, alphas: torch.Tensor, median_index: int,
                    lam: float, aux_enabled: bool, state_aux: dict | None, settings: dict):
    """Weighted pinball, DayState supervision, median auxiliary를 합산하고 각 항을 반환한다."""
    base = l0_loss(q, b["z"], b["mask"], loss_cap(b), alphas, slot_weight_tensor(loss, q.device))
    total, state_parts = base, None
    if state_aux is not None:
        weights = daystate_aux_weights(loss, settings)
        state_loss, state_parts = daystate_loss(state_aux, b, weights["energy"], weights["observed"])
        total = total + state_loss
    aux = _aux(loss, q, b, median_index) if loss.aux is not None and aux_enabled else None
    if aux is not None:
        total = total + lam * aux
    return total, base, aux, state_parts


def calibrate_lambda(net: GenerationNet, batcher: Batcher, queries: QuerySet, batches: np.ndarray, loss: LossConfig, alphas: torch.Tensor,
                     median_index: int) -> tuple[float, dict]:
    """λ_k = r·E‖∇L_pinball‖/E‖∇L_k‖. Calibration forward 뒤에도 후속 batch의 dropout RNG 순서를 유지한다."""
    params = [p for p in net.parameters() if p.requires_grad]
    norms0, normsk = [], []
    device_index = batcher.device.index if batcher.device.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[device_index]):
        net.train()
        for index in batches:
            b = batcher.batch(queries, index)
            q = _forward(net, b)
            aux = _aux(loss, q, b, median_index)
            if aux is None:
                continue
            norms0.append(_grad_norm(l0_loss(q, b["z"], b["mask"], loss_cap(b), alphas, slot_weight_tensor(loss, q.device)), params, retain=True))
            normsk.append(_grad_norm(aux, params, retain=False))
    if not norms0 or np.mean(normsk) <= 0:
        raise RuntimeError(f"{loss.name}의 λ를 정할 batch가 없습니다.")
    lam = loss.aux_ratio * float(np.mean(norms0)) / max(float(np.mean(normsk)), 1e-12)
    return lam, {"lambda": lam, "ratio": loss.aux_ratio, "grad_l0_mean": float(np.mean(norms0)), "grad_aux_mean": float(np.mean(normsk)),
                 "batches_used": len(norms0), "batches": int(len(batches))}


def calibrate_site(net: GenerationNet, batcher: Batcher, queries: QuerySet, batches: np.ndarray, loss: LossConfig, alphas: torch.Tensor,
                   ratio: float) -> tuple[float, dict]:
    """λ_site = r·E‖∇L_pinball‖/E‖∇L_site‖. calibrate_lambda처럼 RNG를 되돌려 후속 batch의 dropout 순서를 유지한다."""
    params = [p for p in net.parameters() if p.requires_grad]
    norms0, norms = [], []
    device_index = batcher.device.index if batcher.device.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[device_index]):
        net.train()
        for index in batches:
            b = batcher.batch(queries, index)
            q, site, _ = _forward_site(net, b)
            value = site_loss(site, b["site_z"], batcher.capacity, b.get("uw"))
            if value is None:
                continue
            norms0.append(_grad_norm(l0_loss(q, b["z"], b["mask"], loss_cap(b), alphas, slot_weight_tensor(loss, q.device)), params, retain=True))
            norms.append(_grad_norm(value, params, retain=False))
    if not norms0 or np.mean(norms) <= 0:
        raise RuntimeError("발전소 보조 head의 λ를 정할 batch가 없습니다.")
    lam = ratio * float(np.mean(norms0)) / max(float(np.mean(norms)), 1e-12)
    return lam, {"lambda": lam, "ratio": ratio, "grad_l0_mean": float(np.mean(norms0)), "grad_site_mean": float(np.mean(norms)),
                 "batches_used": len(norms0), "batches": int(len(batches))}


def _routing(net: GenerationNet) -> list:
    return [m for m in (net.time_shift, net.nearby_hours, net.group_nearby_hours) if m is not None]


def _offset_grad_norm(net: GenerationNet) -> float:
    """읽을 시각과 가중을 정하는 parameter가 받은 gradient의 L2 norm(clip 전). routing은 읽어 온 값을 거쳐서만 출력에 닿는다."""
    grads = [p.grad for m in _routing(net) for p in m.routing_parameters() if p.grad is not None]
    return float(torch.sqrt(sum((g ** 2).sum() for g in grads))) if grads else 0.0


def _state_sha(net: GenerationNet) -> str:
    h = hashlib.sha256()
    for key, value in net.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train_dl(prep: Prep, batcher: Batcher, queries: QuerySet, settings: dict, quantiles: np.ndarray, seed: int, steps: int,
             loss: LossConfig, monitor=None, monitor_every: int = 250, site_aux_ratio: float = 0.0, day_weight: np.ndarray | None = None,
             init_state: dict | None = None, fixed: dict | None = None, learning_rate: float | None = None,
             warmup_steps: int | None = None) -> tuple[dict, dict]:
    """같은 input·DayState auxiliary 설정에서 warm-up을 공유하고, median auxiliary의 λ를 warm-up 직후 정한다.
    site_aux_ratio > 0이면 발전소 이용률 보조 head도 warm-up 직후 λ_site를 정해 그 뒤부터 더한다.

    계절 학습용 선택 인자:
    - day_weight [prep 대상일]: 각 지도 항에 대상일 가중을 넣는다(기존 batch 정규화 유지). calibration도 같은 가중 손실을 쓴다.
    - init_state: 이 weights에서 시작한다(초기화 대신).
    - fixed {"lambda", "lambda_site"}: warm-up과 λ calibration을 하지 않고 첫 step부터 이 λ를 쓴다.
    - learning_rate: 최대 학습률(없으면 settings), warmup_steps: warm-up step 수(없으면 손실 설정).
    """
    device = batcher.device
    median_index = int(np.flatnonzero(np.isclose(quantiles, 0.5))[0])
    seeds = _seeds(seed)
    usable = np.flatnonzero(queries.target_mask.any(axis=1))
    used = queries.subset(usable)
    torch.manual_seed(seeds["init"])
    active = {p: bool(prep.flags[i]) for i, p in enumerate(PROVIDERS)}
    net = build_net(prep, settings, len(quantiles), active).to(device)
    if init_state is not None:
        net.load_state_dict(init_state)
    schedule = batch_schedule(len(used), steps, settings["dl_batch_queries"], np.random.default_rng(seeds["batch"]))
    torch.manual_seed(seeds["dropout"])
    lr = float(settings["dl_learning_rate"] if learning_rate is None else learning_rate)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=settings["dl_weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    alphas = torch.as_tensor(quantiles, dtype=torch.float32, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    warmup = 0 if fixed is not None else min(loss.warmup_steps if warmup_steps is None else int(warmup_steps), steps)
    lam, lam_site = 0.0, 0.0
    use_site = site_aux_ratio > 0 and net.site_head is not None
    if site_aux_ratio > 0 and not use_site:
        raise ValueError("site_aux_ratio > 0에는 발전소 이용률 target이 있는 Prep이 필요합니다.")
    if use_site and warmup >= steps:
        raise ValueError("발전소 이용률 보조 head는 warm-up 뒤 학습 step이 있어야 합니다.")
    if fixed is not None:
        lam, lam_site = float(fixed.get("lambda") or 0.0), float(fixed.get("lambda_site") or 0.0)
    batcher.day_weight = None if day_weight is None else torch.as_tensor(np.asarray(day_weight, dtype=np.float32), device=device)
    try:
        history, curve, unique_days, warmup_sha, lam, calibration, lam_site, site_calibration = _train_loop(
            net, batcher, used, schedule, optimizer, scheduler, loss, alphas, median_index, settings, use_site, site_aux_ratio, steps, warmup,
            fixed is None, lam, lam_site, monitor, monitor_every)
    finally:
        batcher.day_weight = None
    torch.cuda.synchronize(device)
    details = _train_details(seed, seeds, steps, settings, queries, used, schedule, unique_days, batcher, net, started, device, history,
                             curve, loss, warmup, warmup_sha, lam, calibration)
    if use_site:
        details["site_aux"] = {"ratio": site_aux_ratio, "lambda": lam_site, "calibration": site_calibration}
    if prep.temporal is not None:
        details["temporal"] = prep.temporal
    if any(v is not None for v in (day_weight, init_state, fixed, learning_rate, warmup_steps)):     # 계절 학습 경로의 실제 설정
        query_weight = None if day_weight is None else np.asarray(day_weight, dtype=np.float64)[prep.position(used.day)]
        details["training_mode"] = {
            "learning_rate": lr, "init_from_state": init_state is not None, "calibrated": fixed is None,
            "fixed": None if fixed is None else {"lambda": lam, "lambda_site": lam_site},
            "day_weight": None if query_weight is None else {"min": float(query_weight.min()), "max": float(query_weight.max()),
                                                              "mean_used_query": float(query_weight.mean())}}
    return {k: v.detach().cpu() for k, v in net.state_dict().items()}, details


def _train_loop(net, batcher, used, schedule, optimizer, scheduler, loss, alphas, median_index, settings, use_site, site_aux_ratio, steps, warmup,
                calibrate, lam, lam_site, monitor, monitor_every):
    """학습 step을 돌리고 (history, 평가 구간 손실 곡선, batch별 unique 대상일 수, warm-up state sha, λ, λ calibration, λ_site,
    λ_site calibration)을 돌려준다. calibrate가 거짓이면(고정 λ) calibration 분기를 건너뛴다."""
    history, window, aux_window, curve, state_window, site_window, offset_window = [], [], [], [], [], [], []
    unique_days, warmup_sha, calibration, site_calibration = [], None, None, None
    for step in range(steps):
        if step == warmup:
            warmup_sha = _state_sha(net)
            if calibrate and loss.aux is not None:
                lam, calibration = calibrate_lambda(net, batcher, used, schedule[step:step + loss.calibration_batches], loss, alphas, median_index)
            if calibrate and use_site:
                lam_site, site_calibration = calibrate_site(net, batcher, used, schedule[step:step + loss.calibration_batches], loss, alphas,
                                                            site_aux_ratio)
        net.train()
        b = batcher.batch(used, schedule[step])
        unique_days.append(b["onehot"].shape[1])
        optimizer.zero_grad(set_to_none=True)
        state_aux, site = None, None
        if isinstance(net, DayStateNet) or use_site:
            q, site, state_aux = _forward_site(net, b)
        else:
            q = _forward(net, b)
        total, base, aux, state_parts = batch_objective(q, b, loss, alphas, median_index, lam, step >= warmup, state_aux, settings)
        if use_site and step >= warmup:
            value = site_loss(site, b["site_z"], batcher.capacity, b.get("uw"))
            if value is not None:
                total = total + lam_site * value
                site_window.append(float(value.detach()))
        if aux is not None:
            aux_window.append(float(aux.detach()))
        if not torch.isfinite(total):
            raise RuntimeError(f"DL 손실이 유한하지 않습니다(step {step}).")
        total.backward()
        if _routing(net):
            offset_window.append(_offset_grad_norm(net))
        torch.nn.utils.clip_grad_norm_(net.parameters(), settings["dl_gradient_clip"])
        optimizer.step()
        scheduler.step()
        window.append(float(base.detach()))
        if state_parts is not None:
            state_window.append(state_parts)
        if (step + 1) % 50 == 0:
            row = {"step": step + 1, "train_l0": float(np.mean(window)), "train_aux": float(np.mean(aux_window)) if aux_window else None}
            if loss.aux == "day_energy":
                row["train_median_day_energy"] = row["train_aux"]
            if state_window:
                row["train_day_energy"] = float(np.mean([v["energy"] for v in state_window]))
                row["train_day_observed"] = float(np.mean([v["observed"] for v in state_window]))
            if site_window:
                row["train_site"] = float(np.mean(site_window))
            if offset_window:
                row["train_routing_offset_grad"] = float(np.mean(offset_window))
            history.append(row)
            window, aux_window, state_window, site_window, offset_window = [], [], [], [], []
        if monitor is not None and ((step + 1) % monitor_every == 0 or step + 1 == steps):
            curve.append({"step": step + 1, "oot_l0": monitor(net)})
    return history, curve, unique_days, warmup_sha, lam, calibration, lam_site, site_calibration


def _train_details(seed, seeds, steps, settings, queries, used, schedule, unique_days, batcher, net, started, device, history, curve, loss,
                   warmup, warmup_sha, lam, calibration) -> dict:
    return {"seed": seed, "seed_streams": seeds, "steps": steps, "batch_queries": settings["dl_batch_queries"],
               "queries_total": len(queries), "queries_used": len(used), "queries_without_daylight_label": int(len(queries) - len(used)),
               "labelled_hours": int(used.target_mask.sum()), "samples_seen": int(steps * schedule.shape[1]),
               "effective_epochs": float(steps * schedule.shape[1] / len(used)), "mean_unique_days_per_batch": float(np.mean(unique_days)),
               "schedule_sha": __import__("hashlib").sha256(schedule.tobytes()).hexdigest(), "preloaded": batcher.preloaded,
               "input_bytes": batcher.input_bytes, "parameters": int(sum(p.numel() for p in net.parameters())),
               "seconds": time.perf_counter() - started, "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
               "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024 ** 2, "history": history, "oot_curve": curve,
               "loss": loss.__dict__, "day_state": isinstance(net, DayStateNet),
               "daystate_aux_weights": daystate_aux_weights(loss, settings) if isinstance(net, DayStateNet) else None,
               "warmup_steps": warmup, "warmup_state_sha": warmup_sha, "lambda": lam, "calibration": calibration}


@torch.no_grad()
def predict_dl(net: GenerationNet, batcher: Batcher, queries: QuerySet, chunk: int = 64) -> np.ndarray:
    net.eval()
    out = []
    for start in range(0, len(queries), chunk):
        b = batcher.batch(queries, np.arange(start, min(start + chunk, len(queries))))
        out.append(net(b["x"], b["extras"], b["onehot"], b["weights"], b["query"]).float().cpu().numpy())
    return np.concatenate(out)


def oot_monitor(batcher: Batcher, queries: QuerySet, quantiles: np.ndarray):
    """진단용 평가 구간 pinball. 학습 종료 시점 결정에는 쓰지 않는다."""
    alphas = torch.as_tensor(quantiles, dtype=torch.float32, device=batcher.device)

    @torch.no_grad()
    def measure(net: GenerationNet) -> float:
        was_training = net.training
        net.eval()
        total, weight = 0.0, 0.0
        for start in range(0, len(queries), 64):
            index = np.arange(start, min(start + 64, len(queries)))
            b = batcher.batch(queries, index)
            w = float((b["mask"].float() * b["cap"][:, None]).sum())
            if w > 0:
                total += float(l0_loss(net(b["x"], b["extras"], b["onehot"], b["weights"], b["query"]), b["z"], b["mask"], b["cap"], alphas)) * w
                weight += w
        net.train(was_training)
        return total / weight if weight else float("nan")

    return measure
