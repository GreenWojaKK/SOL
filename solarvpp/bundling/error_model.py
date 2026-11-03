"""공동 오차 모델: 발전소 예측 오차의 공동분포를 예보 상태에 따라 학습하고 오차 시나리오를 만든다.

- 입력: OOT 잔차 cutoff, 이력 지연, train-only 표준화
- 모델: 조건부 잔차의 평균·저랭크 공동요인·개별 분산과 관측 부분집합 likelihood. CUDA에서 학습·저장·추론하며,
  평가기간 정답은 학습 함수에 전달하지 않는다.
- 시나리오: 시각별 공통 난수의 잔차 시나리오, 결측을 공동 완성하는 조건부 analog, pairwise 2차 moment 비교 모델
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..data import ROUNDS
from ..experiment import cuda_lock, log
from ..forecast.features import resolve_inputs, time_features
from ..forecast.train import configure_determinism
from ..store import array_hash, digest, environment, package_versions, versions

BLOCKS = {"k3": 3, "k1": 1}


def pairwise_moments(u: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """w 가중 2차 moment S_ij(둘 다 값이 있는 시각)와 겹친 시간 수 n_ij."""
    finite = np.isfinite(u)
    u0 = np.where(finite, u, 0.0)
    f = finite.astype(np.float64)
    num = (u0 * w[:, None]).T @ u0
    den = (f * w[:, None]).T @ f
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0), f.T @ f


def nearest_psd(matrix: np.ndarray) -> tuple[np.ndarray, dict]:
    sym = (matrix + matrix.T) / 2
    values, vectors = np.linalg.eigh(sym)
    negative = values < 0
    info = {"clipped_eigenvalues": int(negative.sum()),
            "clipped_mass": float(-values[negative].sum() / max(values[~negative].sum(), 1e-300))}
    return (vectors * np.clip(values, 0, None)) @ vectors.T, info


def history_features(error_per_kw: np.ndarray, hours: np.ndarray, prior_hours: float = 200.0, issue_slot: int = 16):
    """하루 전 예측 시점에 알려진 이력과 전체 입력 마지막의 고정 이력.

    대상일 d의 모든 시각은 전일 slot `issue_slot`(17시)까지의 잔차만 쓴다. 같은 날 앞 시각의 잔차는 예측 시점에 알 수 없으므로 넣지 않는다.
    hours는 오름차순 전역 시각이고 error_per_kw의 행 순서와 같다.
    """
    hours = np.asarray(hours, dtype=np.int64)
    if len(hours) != len(error_per_kw) or (np.diff(hours) <= 0).any():
        raise ValueError("이력 계산의 시각은 잔차 행과 같은 길이의 오름차순이어야 합니다.")
    ok = np.isfinite(error_per_kw)
    e = np.where(ok, error_per_kw, 0.0).astype(np.float64)
    zero = np.zeros((1,) + e.shape[1:])
    count = np.concatenate([zero, np.cumsum(ok, axis=0)])        # [j] = 앞 j개 행의 누적
    total = np.concatenate([zero, np.cumsum(e, axis=0)])
    square = np.concatenate([zero, np.cumsum(e * e, axis=0)])
    known = np.searchsorted(hours, (hours // 24 - 1) * 24 + issue_slot, side="right")

    def make(n, s, ss):
        mean = s / (n + prior_hours)
        rms = np.sqrt((ss + prior_hours * 0.1 ** 2) / (n + prior_hours))
        return np.stack([mean, rms, np.log1p(n)], axis=-1).astype(np.float32)

    return make(count[known], total[known], square[known]), make(count[-1], total[-1], square[-1])


STATIC_FIELDS = {
    "log_capacity": lambda sites, index: np.log(sites.capacity[index]),
    "wa_dist": lambda sites, index: sites.distance["wa"][index] / 1000,
    "wb_dist": lambda sites, index: sites.distance["wb"][index] / 1000,
}


@dataclass
class RiskScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, mask: np.ndarray, maximum_rows: int, seed: int):
        rows = np.flatnonzero(mask.ravel())
        if len(rows) == 0:
            raise ValueError("공동 오차 모델 표준화를 fit할 유효 잔차가 없습니다.")
        if len(rows) > maximum_rows:
            rows = np.random.default_rng(seed).choice(rows, maximum_rows, replace=False)
        v = x.reshape(-1, x.shape[-1])[rows].astype(np.float64)
        finite = np.isfinite(v)
        count = finite.sum(axis=0)
        clean = np.where(finite, v, 0.0)
        mean = clean.sum(axis=0) / np.maximum(count, 1)
        variance = np.where(finite, (v - mean) ** 2, 0.0).sum(axis=0) / np.maximum(count, 1)
        scale = np.sqrt(variance)
        return cls(mean.astype(np.float32), np.where(scale > 1e-5, scale, 1.0).astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        missing = ~np.isfinite(x)
        z = np.clip((np.where(missing, self.mean, x) - self.mean) / self.scale, -8.0, 8.0)
        return np.concatenate([z, missing.astype(np.float32)], axis=-1).astype(np.float32)


class RiskFeatures:
    """같은 feature 함수를 모든 PV에 적용한다. 예측 구간의 history는 cutoff에서 고정한다."""

    def __init__(self, spec, data, site_index: np.ndarray, cfg: dict):
        self.data, self.index, self.cfg = data, np.asarray(site_index), cfg
        self.inputs = resolve_inputs(data.weather, spec.input_config(cfg["forecast_input"]), spec.excluded_inputs, spec.schema)
        self.fields = cfg["risk"]["forecast_fields"]
        self.names = []
        for p, fields in self.fields.items():
            known = self.inputs.providers[p].names
            missing = set(fields) - set(known)
            if missing:
                raise ValueError(f"위험 입력 schema에 없는 channel: {p} {sorted(missing)}")
            for suffix in ("r17", "revision"):
                self.names += [f"{f}_{suffix}" for f in fields]
        self.weather_count = len(self.names)
        self.names += ["solar_elev_90", "solar_cosz_pos", "solar_azim_sin", "solar_azim_cos"]
        self.names += ["slot_sin", "slot_cos", "doy_sin", "doy_cos"]
        self.static_fields = list(cfg["risk"]["static_fields"])
        unknown = set(self.static_fields) - set(STATIC_FIELDS)
        if unknown:
            raise ValueError(f"공동 오차 모델 static 입력은 {sorted(STATIC_FIELDS)} 중에서 고릅니다(좌표는 쓰지 않음): {sorted(unknown)}")
        self.names += self.static_fields + cfg["risk"]["history_fields"]

    def raw(self, hours: np.ndarray, history: np.ndarray) -> np.ndarray:
        hours = np.asarray(hours, dtype=np.int64)
        n = len(self.index)
        out = np.empty((len(hours), n, len(self.names)), dtype=np.float32)
        sites = self.data.sites
        static = np.column_stack([STATIC_FIELDS[f](sites, self.index) for f in self.static_fields])
        for lo in range(0, len(hours), 512):
            h = hours[lo:lo + 512]
            parts = []
            for p, fields in self.fields.items():
                pi = self.inputs.providers[p]
                channels = [pi.names.index(f) for f in fields]
                station = sites.station_index[p][self.index]
                v = [pi.values[r, h][:, station][..., channels] for r in (0, 1)]
                parts.extend([v[1], v[1] - v[0]])
            sol = np.array(self.data.base.solar[h][:, self.index], dtype=np.float32)
            sol[..., 0] /= 90
            sol[..., 1] = np.maximum(sol[..., 1], 0)
            days, inverse = np.unique(h // 24, return_inverse=True)
            clock = time_features(days)[inverse, h % 24]
            hist = np.broadcast_to(history, (len(h), n, 3)) if history.ndim == 2 else history[lo:lo + len(h)]
            parts += [sol, np.broadcast_to(clock[:, None, :], (len(h), n, 4)),
                      np.broadcast_to(static, (len(h), n, static.shape[1])), hist]
            out[lo:lo + len(h)] = np.concatenate(parts, axis=-1)
        return out


def prepare_training(spec, data, instance, cfg: dict):
    """outer cutoff를 먼저 검사하고 주간 OOT 잔차와 입력을 만든다."""
    if len(instance.est_hours) == 0 or instance.est_hours.max() > instance.cutoff_h:
        raise ValueError("공동 오차 모델 학습에는 cutoff 이전 OOT 잔차만 사용합니다.")
    error = instance.est_error / instance.capacity[None, :]
    daylight = np.asarray(data.base.sunup)[instance.est_hours][:, instance.sites]
    error = np.where(daylight, error, np.nan).astype(np.float32)
    history, frozen = history_features(error, instance.est_hours, cfg["risk"]["history_prior_hours"], cfg["risk"]["history_issue_slot"])
    features = RiskFeatures(spec, data, instance.sites, cfg)
    raw = features.raw(instance.est_hours, history)
    mask = np.isfinite(error)
    scaler = RiskScaler.fit(raw, mask, cfg["risk"]["scaler_max_rows"], cfg["risk"]["seed"])
    usable = mask.any(axis=1)
    x = scaler.transform(raw[usable])
    metadata = {"cutoff_h": int(instance.cutoff_h), "first_target_h": int(instance.est_hours.min()),
                "last_target_h": int(instance.est_hours.max()), "joint_training_hours": int(usable.sum()),
                "valid_site_hours": int(mask.sum()), "sites_with_targets": int(mask.any(axis=0).sum()),
                "sites_without_targets": int((~mask.any(axis=0)).sum()), "feature_names": features.names,
                "history_policy": "training rows of day d use residuals up to day d-1 slot history_issue_slot; inference history frozen at cutoff",
                "label_policy": "observed subset marginal likelihood; no generation imputation"}
    return {"x": x, "target": error[usable], "mask": mask[usable], "hours": instance.est_hours[usable],
            "features": features, "scaler": scaler, "history": frozen, "metadata": metadata}


class ConditionalRisk(nn.Module):
    """공유 PV 함수가 μ_i, L_i, σ_i를 출력한다. Σ = LLᵀ + diag(σ²)."""

    def __init__(self, features: int, width: int = 64, rank: int = 6, minimum_sd: float = 0.01):
        super().__init__()
        self.rank, self.minimum_sd = rank, minimum_sd
        self.body = nn.Sequential(nn.Linear(features, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.head = nn.Linear(width, rank + 2)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[-1] = math.log(math.expm1(0.08 - minimum_sd))

    def forward(self, x: torch.Tensor):
        raw = self.head(self.body(x))
        mean = raw[..., 0]
        loading = raw[..., 1:1 + self.rank]
        sd = torch.nn.functional.softplus(raw[..., -1]) + self.minimum_sd
        return mean, loading, sd


def masked_nll(mean: torch.Tensor, loading: torch.Tensor, sd: torch.Tensor,
               target: torch.Tensor, mask: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """시각별 관측 PV 부분공분산의 정확한 Gaussian NLL / 관측 PV 수. Woodbury로 계산한다."""
    count = mask.sum(-1)
    inv = mask.to(sd.dtype) / sd.square()
    residual = torch.where(mask, torch.nan_to_num(target) - mean, torch.zeros_like(mean))
    weighted = loading * inv[..., None]
    eye = torch.eye(loading.shape[-1], dtype=loading.dtype, device=loading.device)
    small = eye + loading.transpose(-2, -1) @ weighted
    chol = torch.linalg.cholesky(small)
    b = (weighted * residual[..., None]).sum(-2)[..., None]
    solved = torch.cholesky_solve(b, chol)
    quadratic = (residual.square() * inv).sum(-1) - (b * solved).sum((-2, -1))
    logdet = (2 * sd.log() * mask).sum(-1) + 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    values = 0.5 * (count * math.log(2 * math.pi) + logdet + quadratic) / count.clamp_min(1)
    if reduction == "none":
        return torch.where(count > 0, values, torch.zeros_like(values))
    if reduction != "mean":
        raise ValueError(reduction)
    return values[count > 0].mean() if bool((count > 0).any()) else mean.sum() * 0.0


def partition_sum_nll(mean: torch.Tensor, loading: torch.Tensor, sd: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                      capacity: torch.Tensor, assign: torch.Tensor) -> torch.Tensor:
    """무작위 분할 두 그룹의 kWh 합(관측 PV만)에 대한 Gaussian NLL. 용량 정규화를 되돌린 집계 분산을 직접 맞춘다.

    mean·sd·target·mask [B, N]은 kW당 값, loading [B, N, R], capacity [N], assign [B, P, N]은 G1이면 True.
    그룹 합 S = Σ c_i u_i의 예측분포는 N(Σ c_i μ_i, ‖Σ c_i L_i‖² + Σ c_i² σ_i²)이고, 그룹 용량으로 나눈 kW당 척도에서 NLL을 평균한다.
    """
    observed = mask.to(mean.dtype)
    values = []
    for group in (assign, ~assign):
        w = group.to(mean.dtype) * observed[:, None, :] * capacity[None, None, :]
        total = w.sum(-1)
        valid = total > 0
        scale = total.clamp_min(1e-9)
        realized = (w * torch.nan_to_num(target)[:, None, :]).sum(-1) / scale
        center = (w * mean[:, None, :]).sum(-1) / scale
        factor = torch.einsum("bpn,bnr->bpr", w, loading) / scale[..., None]
        variance = factor.square().sum(-1) + (w.square() * sd.square()[:, None, :]).sum(-1) / scale.square()
        # 관측 구성원이 없는 그룹은 분산이 0이다. 걸러낸 항의 backward가 0·∞ = NaN이 되지 않도록 비선형 연산 전에 안전한 값으로 바꾼다.
        variance = torch.where(valid, variance, torch.ones_like(variance))
        difference = torch.where(valid, realized - center, torch.zeros_like(realized))
        nll = 0.5 * (torch.log(2 * math.pi * variance) + difference.square() / variance)
        values.append(nll[valid])
    values = torch.cat(values)
    return values.mean() if values.numel() else mean.sum() * 0.0


def conditional_complete(mean: np.ndarray, loading: np.ndarray, sd: np.ndarray,
                         observed: np.ndarray, factor_noise: np.ndarray, individual_noise: np.ndarray) -> np.ndarray:
    """관측 잔차를 보존하고 공통 저랭크 요인의 조건부분포에서 결측 성분을 생성한다.

    mean [N], loading [N,R], sd [N], observed [N], noise [S,R]・[S,N]。
    관측이 없는 시각은 사전 공동분포에서 생성한다.
    """
    mean, loading, sd = (np.asarray(x, dtype=np.float64) for x in (mean, loading, sd))
    observed = np.asarray(observed, dtype=np.float64)
    ok = np.isfinite(observed)
    rank = loading.shape[-1]
    inv = 1.0 / sd[ok] ** 2
    precision = np.eye(rank) + (loading[ok].T * inv) @ loading[ok]
    chol = np.linalg.cholesky(precision)
    b = loading[ok].T @ (inv * (observed[ok] - mean[ok]))
    factor_mean = np.linalg.solve(chol.T, np.linalg.solve(chol, b))
    factor = factor_mean + np.linalg.solve(chol.T, np.asarray(factor_noise).T).T
    sample = mean + factor @ loading.T + np.asarray(individual_noise) * sd
    sample[:, ok] = observed[ok]
    return sample


def forecast_similarity(data, instance, fields: dict, round_value: int = 17) -> np.ndarray:
    """발전소 쌍의 유사도 = 대응 관측소 예보 시계열 상관의 제공사 평균. 좌표를 쓰지 않는다.

    추정기간(instance cutoff 이전) 중 instance 발전소 하나라도 해가 떠 있는 시각만 쓴다. 같은 관측소를 쓰는 쌍은 1이다.
    """
    hours = instance.est_hours
    hours = hours[np.asarray(data.base.sunup)[hours][:, instance.sites].any(axis=1)]
    r = ROUNDS.index(int(round_value))
    out = []
    for p, name in fields.items():
        channels = [c["name"] for c in data.weather.channels[f"forecast_{p}"]]
        series = np.asarray(data.weather.values[f"forecast_{p}"])[r, hours, :, channels.index(name)].astype(np.float64)
        ok = np.isfinite(series)
        centered = np.where(ok, series - np.nanmean(np.where(ok, series, np.nan), axis=0), 0.0)
        cov = centered.T @ centered / np.maximum(ok.T.astype(np.float64) @ ok, 1.0)
        sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        station = data.sites.station_index[p][instance.sites]
        out.append((cov / np.outer(sd, sd))[np.ix_(station, station)])
    return np.clip(np.mean(out, axis=0), -1.0, 1.0)


def similarity_target(S: np.ndarray, n: np.ndarray, similarity: np.ndarray, cfg: dict, neighbor_similarity: float):
    """T_ij = s_i·s_j·ρ_ij, ρ_ij = a + b·similarity_ij. s_i는 예보 유사도가 neighbor_similarity 이상인 발전소의 중앙값 쪽으로 shrink한다."""
    kappa, min_overlap = float(cfg["pseudo_count_hours"]), float(cfg["min_overlap_hours"])
    own = np.sqrt(np.clip(np.diag(S), 0, None))
    count = np.diag(n)
    good = count >= min_overlap
    if good.sum() < 2:
        good = count >= max(count.max() * 0.5, 1.0)
    overall = float(np.median(own[good]))
    near = (similarity >= neighbor_similarity) & ~np.eye(len(own), dtype=bool) & good[None, :]
    prior = np.array([np.median(own[row]) if row.any() else overall for row in near])
    scale = (count * own + kappa * prior) / (count + kappa)
    iu = np.triu_indices(len(own), 1)
    ok = (n[iu] >= min_overlap) & good[iu[0]] & good[iu[1]] & (own[iu[0]] > 0) & (own[iu[1]] > 0)
    r = S[iu][ok] / (own[iu[0]][ok] * own[iu[1]][ok])
    info = {"pairs": int(ok.sum()), "scale_median": overall, "sites_below_min_overlap": int((~good).sum())}
    if ok.sum() >= 10:
        design = np.column_stack([np.ones(int(ok.sum())), similarity[iu][ok]])
        (a, b), *_ = np.linalg.lstsq(design, r, rcond=None)
        rho = np.clip(a + b * similarity, -1.0, 1.0)
        info.update(a=float(a), b=float(b), rho_at_similarity_1=float(np.clip(a + b, -1, 1)),
                    r2=float(1 - ((design @ np.array([a, b]) - r) ** 2).sum() / max(((r - r.mean()) ** 2).sum(), 1e-300)))
    else:
        rho = np.full_like(similarity, float(np.median(r)) if len(r) else 0.9)
        info.update(fallback=True)
    np.fill_diagonal(rho, 1.0)
    return np.outer(scale, scale) * rho, info


def coordinate_free_moments(instance, cfg: dict, similarity: np.ndarray, neighbor_similarity: float):
    """블록(k=3, k=1)별 kW당 오차의 pairwise 2차 moment를 예보 유사도 target으로 shrink하고 PSD로 만든다."""
    u = instance.est_error / instance.capacity[None, :]
    kappa = float(cfg["pseudo_count_hours"])
    out, info = [], {}
    for name, kv in BLOCKS.items():
        rows = instance.est_k == kv
        S, n = pairwise_moments(u[rows], instance.est_w[rows])
        target, target_info = similarity_target(S, n, similarity, cfg, neighbor_similarity)
        mixed, psd_info = nearest_psd((n * S + kappa * target) / (n + kappa))
        out.append(mixed)
        info[name] = {"hours": int(rows.sum()), "site_hours_median": float(np.median(np.diag(n))), "target": target_info, "psd": psd_info}
    return out, info


def joint_draw(mean, loading, sd, samples, seed, hour):
    """호출 순서·chunk 크기와 무관한 시각별 공통 요인과 PV별 혁신."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(hour)]))
    factors = rng.standard_normal((samples, loading.shape[-1]))
    individual = rng.standard_normal((samples, len(mean)))
    return mean + factors @ loading.T + individual * sd


class AnalogDays:
    """전체 fleet의 예보 일중 profile로 이웃 일자를 찾고 같은 일자를 모든 PV에 적용한다."""

    def __init__(self, instance, predictor, target_hours, cfg):
        self.instance, self.predictor, self.cfg = instance, predictor, cfg
        self.days = np.array([d for d in np.unique(instance.est_hours // 24)
                              if (d + 1) * 24 - 1 <= instance.cutoff_h], dtype=np.int64)
        if len(self.days) == 0:
            raise ValueError("analog 이웃으로 쓸 cutoff 이전 예보 일자가 없습니다.")
        lookup = {int(h): i for i, h in enumerate(instance.est_hours)}
        self.days = np.array([d for d in self.days if all(int(d * 24 + s) in lookup for s in range(24))])
        rows = np.array([[lookup[int(d * 24 + s)] for s in range(24)] for d in self.days])
        self.errors = instance.est_error[rows] / instance.capacity[None, None, :]
        historical_daylight = np.asarray(predictor.features.data.base.sunup)[instance.est_hours[rows]][..., instance.sites]
        self.target_days = np.unique(target_hours // 24)
        historical = self._descriptor(self.days)
        target = self._descriptor(self.target_days)
        distance = np.maximum((target * target).sum(1)[:, None] + (historical * historical).sum(1)[None, :]
                              - 2 * target @ historical.T, 0.0) / historical.shape[1]
        k = min(int(cfg["analog"]["neighbors"]), len(self.days))
        self.neighbors = np.argsort(distance, axis=1, kind="stable")[:, :k]
        selected = np.take_along_axis(distance, self.neighbors, axis=1)
        temperature = np.maximum(np.median(selected - selected[:, :1], axis=1, keepdims=True), 1e-6)
        weights = np.exp(-(selected - selected[:, :1]) / temperature)
        self.probability = weights / weights.sum(1, keepdims=True)
        self.target_day_index = {int(d): i for i, d in enumerate(self.target_days)}
        self._choice_cache = {}
        self.metadata = {"past_days": len(self.days), "neighbors": k, "last_analog_hour": int(self.days.max() * 24 + 23),
                         "cutoff_h": int(instance.cutoff_h), "observed_residual_fraction": float(np.isfinite(self.errors).mean()),
                         "joint_complete_daylight_days": int((np.isfinite(self.errors) | ~historical_daylight).all(axis=(1, 2)).sum()),
                         "completion": "observed analog PVs retained; missing PVs drawn conditional on them under target low-rank risk",
                         "descriptor": "capacity-weighted mean and std of train-standardized weather + solar 24h profiles",
                         "scenario_probability": "weighted neighbors sampled once per scenario/day; equal probability Monte Carlo draws"}

    def _descriptor(self, days):
        result = []
        features, scaler = self.predictor.features, self.predictor.scaler
        width = features.weather_count + 4
        c = self.instance.ctil.astype(np.float32)
        for lo in range(0, len(days), 16):
            hours = (days[lo:lo + 16, None] * 24 + np.arange(24)).ravel()
            raw = features.raw(hours, self.predictor.history)[..., :width]
            x = np.clip((np.where(np.isfinite(raw), raw, scaler.mean[:width]) - scaler.mean[:width]) / scaler.scale[:width], -8, 8)
            mean = np.einsum('hnf,n->hf', x, c)
            square = np.einsum('hnf,n->hf', x * x, c)
            std = np.sqrt(np.maximum(square - mean * mean, 0))
            result.append(np.concatenate([mean, std], axis=-1).reshape(-1, 24 * width * 2))
        return np.concatenate(result)

    def choices(self, day, samples, seed):
        key = (int(day), int(samples), int(seed))
        if key not in self._choice_cache:
            i = self.target_day_index[int(day)]
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(day), 901]))
            self._choice_cache[key] = rng.choice(self.neighbors[i], samples, p=self.probability[i])
        return self._choice_cache[key]

    def draw(self, hour, mean, loading, sd, samples, seed):
        chosen = self.choices(int(hour) // 24, samples, seed)
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(hour)]))
        factor = rng.standard_normal((samples, loading.shape[-1]))
        individual = rng.standard_normal((samples, len(mean)))
        out = np.empty((samples, len(mean)))
        for donor in np.unique(chosen):
            pick = np.flatnonzero(chosen == donor)
            observed = self.errors[int(donor), int(hour) % 24]
            out[pick] = conditional_complete(mean, loading, sd, observed, factor[pick], individual[pick])
        return out


class ScenarioSource:
    """전체 평가시간의 μ·L·σ를 고정하고, 시나리오는 필요할 때 재생성한다."""

    def __init__(self, spec, data, instance, predictor, cfg):
        self.instance, self.predictor, self.cfg = instance, predictor, cfg
        self.hours = spec.period(instance.name).hours
        self.daylight = np.asarray(data.base.sunup)[self.hours][:, instance.sites]
        self.mean, self.loading, self.sd = predictor.predict(self.hours)
        if not all(np.isfinite(a).all() for a in (self.mean, self.loading, self.sd)):
            raise ValueError("공동 오차 모델 출력에 비유한 값이 있습니다.")
        unc = cfg["unconditional"]
        similarity = forecast_similarity(data, instance, unc["similarity_fields"], unc["similarity_round"])
        covariance, self.unconditional_metadata = coordinate_free_moments(instance, spec.protocol["bundling_search"]["covariance"],
                                                                          similarity, unc["neighbor_similarity"])
        self.unconditional_loading = []
        for cov in covariance:
            values, vectors = np.linalg.eigh(cov)
            self.unconditional_loading.append((vectors * np.sqrt(np.maximum(values, 0))).astype(np.float32))
        self.analog = AnalogDays(instance, predictor, self.hours, cfg)
        self.metadata = {"risk_key": predictor.key, "first_hour": int(self.hours[0]), "last_hour": int(self.hours[-1]),
                         "hours": len(self.hours), "sites": len(instance.sites), "analog": self.analog.metadata,
                         "unconditional": self.unconditional_metadata,
                         "night_rule": "zero residual at site-specific night", "point_forecast": "site XGB median frozen per outer cutoff"}

    def components(self, method):
        """혼합 방법은 설정의 구성 방법 시나리오를 같은 비중으로 합친다. 다른 방법은 자기 자신이다."""
        return list(self.cfg["mixture"]["components"]) if method == "mixture" else [method]

    def count(self, method, samples):
        """시각당 시나리오 수. 혼합은 구성 방법마다 samples개씩 뽑는다."""
        return samples * len(self.components(method))

    def batch(self, method, positions, samples, seed):
        if method == "mixture":
            # 구성 방법마다 seed를 달리해 lowrank와 analog가 같은 정규 난수를 공유하지 않게 한다.
            return np.concatenate([self.batch(c, positions, samples, seed + 1000 * (k + 1))
                                   for k, c in enumerate(self.components(method))], axis=1)
        positions = np.asarray(positions, dtype=int)
        n = len(self.instance.sites)
        out = np.empty((len(positions), samples, n), dtype=np.float32)
        for row, i in enumerate(positions):
            h = int(self.hours[i])
            if not self.daylight[i].any():
                out[row] = 0
                continue
            if method == "unconditional":
                block = 0 if self.instance.target_k[i] == 3 else 1
                values = joint_draw(np.zeros(n), self.unconditional_loading[block], np.zeros(n), samples, seed, h)
            elif method == "lowrank":
                values = joint_draw(self.mean[i], self.loading[i], self.sd[i], samples, seed, h)
            elif method == "analog":
                values = self.analog.draw(h, self.mean[i], self.loading[i], self.sd[i], samples, seed)
            else:
                raise ValueError(method)
            out[row] = np.where(self.daylight[i], values * self.instance.capacity[None, :], 0.0)
        if not np.isfinite(out).all():
            raise ValueError(f"시나리오에 비유한 값이 있습니다: {method}")
        return out

    def materialize(self, method, samples, seed):
        # 야간은 오차가 0이지만 목적 정규화의 Σw_h에 포함하므로 전체 시간축을 보존한다.
        k = self.count(method, samples)
        out = np.empty((len(self.hours) * k, len(self.instance.sites)), dtype=np.float32)
        batch = self.cfg["scenarios"]["batch_hours"]
        for lo in range(0, len(self.hours), batch):
            rows = np.arange(lo, min(lo + batch, len(self.hours)))
            out[lo * k:(lo + len(rows)) * k] = self.batch(method, rows, samples, seed).reshape(-1, out.shape[-1])
        weight = np.repeat(self.instance.target_w / k, k)
        return out, weight

    def score_partitions(self, method, partitions, samples, seed, center="none"):
        """독립 평가 seed로 모든 시각의 예상 벌점을 계산한다. 모든 후보가 같은 난수를 공유한다.

        center="median"이면 시각별 그룹 합 분포의 중앙값을 제출값으로 보고 E|R − median(R)|을 쓴다.
        """
        x = np.asarray(partitions, dtype=np.float32)
        result = np.zeros(len(x), dtype=np.float64)
        batch = self.cfg["scenarios"]["batch_hours"]
        for lo in range(0, len(self.hours), batch):
            positions = np.arange(lo, min(lo + batch, len(self.hours)))
            e = self.batch(method, positions, samples, seed).astype(np.float64)
            r1 = (e.reshape(-1, e.shape[-1]) @ x.T).reshape(len(positions), e.shape[1], len(x))
            r2 = e.sum(-1)[..., None] - r1
            if center == "median":
                r1, r2 = r1 - np.median(r1, axis=1, keepdims=True), r2 - np.median(r2, axis=1, keepdims=True)
            losses = (np.abs(r1) + np.abs(r2)).mean(1)
            result += (losses * self.instance.target_w[positions, None]).sum(0)
        return result


def risk_key(data, instance, cfg: dict) -> str:
    return digest({"base": data.base.key, "weather": data.weather.key, "cutoff": instance.cutoff_h,
                   "sites": instance.sites.tolist(), "training": array_hash(instance.est_hours, instance.est_error),
                   "risk": cfg["risk"], "input": cfg["forecast_input"],
                   "packages": package_versions(("numpy", "torch")),
                   "versions": versions("features", "error_model")})


def fit_risk(spec, data, instance, cfg: dict):
    key = risk_key(data, instance, cfg)
    store = data.store
    if not store.exists("error_model", key):
        prep = prepare_training(spec, data, instance, cfg)
        settings = cfg["risk"]
        configure_determinism(spec.config["runtime"]["torch_threads"])
        if not torch.cuda.is_available():
            raise RuntimeError("조건부 공동 오차 모델 학습에는 FAI의 CUDA GPU가 필요합니다.")
        started = time.perf_counter()
        with cuda_lock(spec.artifact_dir / ".cuda.lock"):
            device = torch.device("cuda")
            torch.manual_seed(settings["seed"])
            net = ConditionalRisk(prep["x"].shape[-1], settings["width"], settings["rank"], settings["minimum_sd"]).to(device)
            x = torch.as_tensor(prep["x"], device=device)
            y = torch.as_tensor(prep["target"], device=device)
            mask = torch.as_tensor(prep["mask"], device=device)
            capacity = torch.as_tensor(instance.capacity, dtype=torch.float32, device=device)
            parts, aggregate_weight = int(settings["aggregate_partitions"]), float(settings["aggregate_weight"])
            optimizer = torch.optim.AdamW(net.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["steps"])
            rng = np.random.default_rng(settings["seed"])
            history, losses, aggregates = [], [], []
            torch.cuda.reset_peak_memory_stats()
            for step in range(settings["steps"]):
                index = torch.as_tensor(rng.integers(0, len(x), settings["batch_hours"]), device=device)
                optimizer.zero_grad(set_to_none=True)
                mean, loading, sd = net(x[index])
                loss = masked_nll(mean, loading, sd, y[index], mask[index])
                if aggregate_weight > 0:
                    # 정산은 kWh 그룹 합에서 일어나므로, 무작위 분할의 용량 가중 합 분포를 함께 맞춘다.
                    assign = torch.as_tensor(rng.random((len(index), parts, capacity.numel())) < 0.5, device=device)
                    aggregate = partition_sum_nll(mean, loading, sd, y[index], mask[index], capacity, assign)
                    aggregates.append(float(aggregate.detach()))
                    loss = loss + aggregate_weight * aggregate
                if not torch.isfinite(loss):
                    raise RuntimeError(f"조건부 위험 NLL이 유한하지 않습니다: {instance.name} step={step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), settings["gradient_clip"])
                optimizer.step()
                schedule.step()
                losses.append(float(loss.detach()))
                if (step + 1) % 250 == 0 or step == settings["steps"] - 1:
                    row = {"step": step + 1, "loss": float(np.mean(losses)),
                           "partition_sum_nll": float(np.mean(aggregates)) if aggregates else None}
                    history.append(row)
                    log(f"risk {instance.name} step={step + 1}/{settings['steps']} loss={row['loss']:.5f} sum_nll={row['partition_sum_nll']}")
                    losses, aggregates = [], []
            net.eval()
            state = {name: value.detach().cpu() for name, value in net.state_dict().items()}
            memory = torch.cuda.max_memory_allocated() / 1024 ** 2
            del net, optimizer, schedule, x, y, mask
            torch.cuda.empty_cache()
        meta = {"instance": instance.name, "settings": settings, "features": int(prep["x"].shape[-1]),
                "lineage": prep["metadata"], "history": history, "steps": settings["steps"],
                "seconds": time.perf_counter() - started, "peak_allocated_mib": memory, "environment": environment(),
                "checkpoint_policy": "last fixed optimizer step; outer evaluation labels unused"}
        store.write("error_model", key, {"mean": prep["scaler"].mean, "scale": prep["scaler"].scale,
                                               "frozen_history": prep["history"]}, meta,
                    {"model.pt": lambda path: torch.save(state, path)})
    arrays, meta = store.load("error_model", key, mmap=False)
    settings = meta["settings"]
    net = ConditionalRisk(meta["features"], settings["width"], settings["rank"], settings["minimum_sd"])
    state = torch.load(store.path("error_model", key) / "model.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    return RiskPredictor(key, net, RiskFeatures(spec, data, instance.sites, cfg), RiskScaler(arrays["mean"], arrays["scale"]),
                         arrays["frozen_history"], meta)


class RiskPredictor:
    def __init__(self, key, model, features, scaler, history, metadata):
        self.key, self.model, self.features = key, model.eval(), features
        self.scaler, self.history, self.metadata = scaler, history, metadata

    def predict(self, hours: np.ndarray, device="cuda", batch_hours: int = 128):
        self.model.to(device)
        result = [[], [], []]
        with torch.inference_mode():
            for start in range(0, len(hours), batch_hours):
                raw = self.features.raw(hours[start:start + batch_hours], self.history)
                x = torch.as_tensor(self.scaler.transform(raw), device=device)
                for values, out in zip(result, self.model(x)):
                    values.append(out.cpu().numpy())
        self.model.cpu()
        return tuple(np.concatenate(v) for v in result)
