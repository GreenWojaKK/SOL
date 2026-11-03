"""Risk model: 제출 forecaster의 실제 제출값 절대오차의 조건부 평균 R_h(I_d, A).

참조: docs/solarvpp2-design.md#risk.

- target은 |Y_{A,h} − a_θ,h| / C_A이고, 학습 행은 forecaster가 학습하지 않은 held-out 블록 대상일의 정답 mask·주간 시각이다.
- 학습 목적은 조건부 평균을 내는 Tweedie deviance(log link)이다. 정산 가중 w_h는 R에 넣지 않는다.
- R은 forecaster를 고정한 뒤 별도 head로 학습한다. 입력은 forecaster의 시각별 집합 표현과 출력, 집합 정보, 시간 표현이다.
- held-out 블록마다 그 블록을 빼고 학습한 head(cross-fitting)와 전체 블록으로 학습한 head를 둔다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .data import Sites
from .features import time_features
from .forecast.forecaster import Forecaster, Prediction, queries_of
from .sampler import heldout_block
from .store import digest, versions

SCALAR_FEATURES = ("median_over_cap", "spread_over_cap", "xgb_median_over_cap", "dl_median_over_cap", "log_capacity_kw",
                   "log1p_pv_count", "hhi", "wa_station_hhi", "wb_station_hhi", "slot_sin", "slot_cos", "doy_sin", "doy_cos")
FULL = -1


def station_concentration(members: np.ndarray, sites: Sites) -> np.ndarray:
    """집합 안 관측소별 용량 비율의 제곱합 [B, 2](WA, WB)."""
    cap = np.asarray(members, dtype=np.float64) * sites.capacity[None, :]
    total = cap.sum(axis=1, keepdims=True)
    out = []
    for p in ("wa", "wb"):
        index = sites.station_index[p]
        by_station = np.zeros((len(cap), int(index.max()) + 1))
        for s in np.unique(index):
            by_station[:, s] = cap[:, index == s].sum(axis=1)
        out.append(((by_station / total) ** 2).sum(axis=1))
    return np.stack(out, axis=1)


def risk_inputs(pred: Prediction, days: np.ndarray, quantile_low: int = 0, quantile_high: int = -1) -> np.ndarray:
    """R 입력 [B, 24, F]. 시각별 집합 표현 뒤에 SCALAR_FEATURES 순서의 값을 붙인다."""
    if pred.hidden is None or pred.station_hhi is None:
        raise ValueError("R 입력에는 forecaster의 집합 표현(hidden)과 관측소 집중도가 필요합니다.")
    cap = pred.cap[:, None]
    spread = pred.quantiles[..., quantile_high] - pred.quantiles[..., quantile_low]
    conc = pred.station_hhi
    clock = time_features(np.asarray(days))
    scalars = np.stack(np.broadcast_arrays(
        pred.median / cap, spread / cap, pred.xgb_median / cap, pred.dl_median / cap,
        np.log(pred.cap)[:, None], np.log1p(pred.n_pv.astype(np.float64))[:, None], pred.hhi[:, None],
        conc[:, :1], conc[:, 1:], clock[..., 0], clock[..., 1], clock[..., 2], clock[..., 3]), axis=-1)
    return np.concatenate([pred.hidden, scalars.astype(np.float32)], axis=-1).astype(np.float32)


def tweedie_loss(eta: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, power: float) -> torch.Tensor:
    """Tweedie deviance의 μ 의존 항(log link, μ = exp(η)). 가중 평균을 최소화하는 μ는 가중 조건부 평균이다."""
    loss = -y * torch.exp((1.0 - power) * eta) / (1.0 - power) + torch.exp((2.0 - power) * eta) / (2.0 - power)
    return (loss * weight).sum() / weight.sum()


class RiskHead(nn.Module):
    def __init__(self, n_in: int, widths: list[int]):
        super().__init__()
        layers, prev = [], n_in
        for width in widths:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class RiskData:
    x: np.ndarray       # [N, F] 학습 행 입력
    y: np.ndarray       # [N] |Y − a| / C_A
    weight: np.ndarray  # [N] 집합 용량
    day: np.ndarray     # [N]
    block: np.ndarray   # [N] held-out 블록 번호


def build_risk_data(forecaster: Forecaster, settings: dict, chunk_queries: int = 4096) -> tuple[RiskData, dict]:
    """held-out 블록 대상일마다 risk sampler 집합의 실현 오차를 만든다."""
    held = forecaster.model["heldout"]
    queries = queries_of(forecaster, forecaster.heldout_days, settings["sampler"], "risk")
    xs, ys, ws, ds = [], [], [], []
    for start in range(0, len(queries), chunk_queries):
        sub = queries.subset(np.arange(start, min(start + chunk_queries, len(queries))))
        pred = forecaster.predict(sub.day, sub.members, hidden=True)
        x = risk_inputs(pred, sub.day)
        select = sub.target_mask
        q, s = np.nonzero(select)
        xs.append(x[q, s])
        ys.append(np.abs(sub.y_kwh[q, s] - pred.median[q, s]) / sub.cap[q])
        ws.append(sub.cap[q])
        ds.append(sub.day[q])
    day = np.concatenate(ds)
    data = RiskData(np.concatenate(xs), np.concatenate(ys).astype(np.float64), np.concatenate(ws), day, heldout_block(day, held["block_days"]))
    meta = {"queries": len(queries), "rows": len(data.y), "days": len(np.unique(day)), "blocks": len(np.unique(data.block)),
            "queries_hash": queries.hash, "kinds": np.bincount(queries.kind, minlength=5).tolist()}
    return data, meta


def fit_head(x: np.ndarray, y: np.ndarray, weight: np.ndarray, settings: dict, seed: int, device: torch.device) -> dict:
    """입력 표준화와 Tweedie head 학습. 반환은 state·표준화·학습 기록이다."""
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    torch.manual_seed(seed)
    head = RiskHead(x.shape[1], list(settings["hidden"])).to(device)
    init = float(np.log(max(np.average(y, weights=weight), 1e-6)))
    with torch.no_grad():
        head.net[-1].bias.fill_(init)
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    xt = torch.as_tensor((x - mean) / scale, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    wt = torch.as_tensor(weight / weight.mean(), dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    batch = int(settings["batch_rows"])
    history = []
    for epoch in range(int(settings["epochs"])):
        order = rng.permutation(len(y))
        total = 0.0
        for start in range(0, len(order), batch):
            index = torch.as_tensor(order[start:start + batch], device=device)
            loss = tweedie_loss(head(xt.index_select(0, index)), yt.index_select(0, index), wt.index_select(0, index), float(settings["power"]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        history.append({"epoch": epoch + 1, "loss": total / len(order)})
    return {"state": {k: v.detach().cpu() for k, v in head.state_dict().items()}, "mean": mean, "scale": scale, "history": history}


class RiskModel:
    """결정 cutoff forecaster 하나에 묶인 R. heads[b]는 held-out 블록 b를 빼고 학습했고 heads[FULL]은 전체 블록으로 학습했다."""

    def __init__(self, forecaster: Forecaster, smoke: bool = False):
        self.forecaster, self.smoke = forecaster, smoke
        self.settings = dict(forecaster.model["risk"])
        if smoke:
            self.settings["epochs"] = forecaster.spec.config["smoke"]["risk_epochs"]
        self.device = forecaster.device
        self.heads: dict = {}

    def key(self) -> str:
        if hasattr(self, "cached_key"):
            return self.cached_key
        return digest({"forecaster": self.forecaster.identity(), "risk": self.settings, "versions": versions("sampler", "risk")})

    def load_cached(self, record: dict) -> dict:
        """저장된 risk head를 읽으며 학습은 호출하지 않는다."""
        self.settings = dict(record["settings"])
        self.cached_key = record["key"]
        store = self.forecaster.data.store
        self.heads = {int(b): h for b, h in torch.load(store.path("risk", self.cached_key) / "heads.pt",
                     map_location="cpu", weights_only=False).items()}
        return store.load("risk", self.cached_key, names=[])[1]

    def fit(self) -> dict:
        store = self.forecaster.data.store
        key = self.key()
        if not store.exists("risk", key):
            started = time.perf_counter()
            print(f"[{time.strftime('%H:%M:%S')}] risk 학습 자료 생성", flush=True)
            data, meta = build_risk_data(self.forecaster, self.settings)
            blocks = np.unique(heldout_block(self.forecaster.heldout_days, self.forecaster.model["heldout"]["block_days"]))
            heads, crossfit = {}, np.full(len(data.y), np.nan)
            for b in [FULL, *blocks.tolist()]:
                train = data.block != b
                if not train.any():
                    raise ValueError(f"held-out 블록 {b}을 빼면 risk 학습 행이 없습니다.")
                heads[b] = fit_head(data.x[train], data.y[train], data.weight[train], self.settings, int(self.settings["seed"]), self.device)
                if b != FULL and (~train).any():
                    crossfit[~train] = self._apply(heads[b], data.x[~train])
            calibration = calibration_table(crossfit, data.y, data.weight, int(self.forecaster.spec.config["evaluation"]["calibration_bins"]))
            payload = {str(b): h for b, h in heads.items()}
            store.write("risk", key, {"y": data.y, "weight": data.weight, "day": data.day, "block": data.block, "crossfit": crossfit},
                        {**meta, "heads": [int(b) for b in heads], "settings": self.settings, "crossfit_calibration": calibration,
                         "seconds": time.perf_counter() - started},
                        {"heads.pt": lambda path: torch.save(payload, path)})
        self.heads = {int(b): h for b, h in torch.load(store.path("risk", key) / "heads.pt", map_location="cpu", weights_only=False).items()}
        return store.load("risk", key, names=[])[1]

    def _apply(self, head: dict, x: np.ndarray, chunk: int = 65536) -> np.ndarray:
        settings = self.settings
        module = RiskHead(x.shape[1], list(settings["hidden"])).to(self.device)
        module.load_state_dict(head["state"])
        module.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(x), chunk):
                xt = torch.as_tensor((x[start:start + chunk] - head["mean"]) / head["scale"], dtype=torch.float32, device=self.device)
                out.append(torch.exp(module(xt)).cpu().numpy())
        return np.concatenate(out).astype(np.float64) if out else np.zeros(0)

    def predict(self, pred: Prediction, days: np.ndarray, block: np.ndarray | None = None) -> np.ndarray:
        """집합·대상일마다 시각별 R [B, 24](kWh). block을 주면 그 held-out 블록을 빼고 학습한 head를 쓰고, 없으면 전체 head를 쓴다.
        집합의 모든 구성원이 야간인 시각은 0이다."""
        x = risk_inputs(pred, days)
        out = np.zeros(pred.median.shape)
        block = np.full(len(days), FULL) if block is None else np.asarray(block)
        for b in np.unique(block):
            rows = np.flatnonzero(block == b)
            if int(b) not in self.heads:
                raise ValueError(f"held-out 블록 {b}의 head가 없습니다.")
            r = self._apply(self.heads[int(b)], x[rows].reshape(-1, x.shape[-1])).reshape(len(rows), 24)
            out[rows] = r * pred.cap[rows, None]
        return np.where(pred.daylight, out, 0.0)


def calibration_table(predicted: np.ndarray, realized: np.ndarray, weight: np.ndarray, bins: int) -> list[dict]:
    """예측 R의 분위 구간별 가중 평균 예측과 실현 절대오차. 가장 높은 구간의 비교가 high-predicted-risk mean calibration이다."""
    ok = np.isfinite(predicted)
    if not ok.any():
        return []
    p, r, w = predicted[ok], realized[ok], weight[ok]
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    index = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    table = []
    for i in range(bins):
        sel = index == i
        if sel.any():
            table.append({"bin": i + 1, "rows": int(sel.sum()), "predicted_mean": float(np.average(p[sel], weights=w[sel])),
                          "realized_mean": float(np.average(r[sel], weights=w[sel]))})
    return table
