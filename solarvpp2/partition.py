"""배정 결정.

참조: docs/solarvpp2-design.md#partition.

목적은 결정 cutoff 이전 정보로 추정한 다가오는 평가 기간의 기대 정산손실이다.

    Ĵ(x) = Σ_{d∈𝒟} ω_d Σ_h π_{d,h} [R_h(I_d, A_1(x)) + R_h(I_d, A_2(x))]

R은 실제 집합에서만 정의되므로 x는 이산 변수이고 연속 완화는 쓰지 않는다. Σ π R은 I_d가 주어졌을 때 SMP와 절대오차를 독립으로
두는 근사이고, π를 월·시각 평균 SMP로 두면 일자별 SMP 변동을 평균으로 바꾸는 근사가 더해진다.
배정은 instance 발전소(결정 cutoff까지 유효 발전량이 있는 발전소)에 대한 boolean이고 True가 G1이다.

Objective는 임의의 후보 배정을 전체 계산으로 평가한다. 국소 탐색은 SearchState가 현재 배정의 그룹별 용량 가중합을 들고,
후보 이동마다 이동한 PV의 기여만 더하고 뺀다. 두 경로는 가중합 뒤의 예측(Forecaster.predict_aggregates)과 R을 같이 쓴다.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from vppcontract.partition import capacity_ok, repair
from vppcontract.timeaxis import calendar_month

from .data import PROVIDERS
from .forecast.forecaster import Aggregates, Forecaster
from .risk import RiskModel
from .sampler import heldout_block, stratified_split
from .store import digest, versions

PENALTY_RATE = 0.10


def day_weights(days: np.ndarray, mode: str) -> np.ndarray:
    """ω_d. calendar_month는 달력 월마다 가중치 합이 1/12이고 같은 월 안에서 균등하다. uniform은 모두 같다."""
    days = np.asarray(days)
    if mode == "uniform":
        return np.full(len(days), 1.0 / len(days))
    if mode != "calendar_month":
        raise ValueError(f"알 수 없는 일자 가중치입니다: {mode}")
    month = calendar_month(days)
    present = np.unique(month)
    count = {m: int((month == m).sum()) for m in present}
    return np.array([1.0 / (len(present) * count[m]) for m in month])


def price_profile(smp: np.ndarray, cutoff_h: int, months: int = 12) -> np.ndarray:
    """결정 cutoff 직전 months개월(대상일 기준)의 달력 월·slot별 평균 SMP [12, 24]. cutoff 이후 시각은 쓰지 않는다."""
    last_day = cutoff_h // 24
    first_day = max(last_day - round(months * 365.25 / 12) + 1, 0)
    hours = np.arange(first_day * 24, cutoff_h + 1)
    if hours.max() > cutoff_h:
        raise RuntimeError("SMP 추정에 cutoff 이후 시각이 들어갔습니다.")
    month = calendar_month(hours // 24)
    slot = hours % 24
    table = np.full((12, 24), np.nan)
    for m in range(1, 13):
        for s in range(24):
            sel = (month == m) & (slot == s)
            if sel.any():
                table[m - 1, s] = float(np.mean(smp[hours[sel]]))
    overall = np.array([np.nanmean(table[:, s]) for s in range(24)])
    return np.where(np.isfinite(table), table, overall[None, :])


def objective_price(days: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """π_{d,h} = 0.10·k_h·SMP̄(월, slot) [D, 24]."""
    k = np.where((np.arange(24) + 1 >= 11) & (np.arange(24) + 1 <= 15), 3.0, 1.0)
    return PENALTY_RATE * k[None, :] * profile[calendar_month(days) - 1]


class Objective:
    """Ĵ(x)를 후보 배정 묶음에 대해 전체 계산으로 구한다. 𝒟 대상일의 R 값은 그 held-out 블록을 빼고 학습한 head로 계산한다."""

    def __init__(self, forecaster: Forecaster, risk: RiskModel, instance_sites: np.ndarray, days: np.ndarray, settings: dict):
        self.forecaster, self.risk, self.sites_index = forecaster, risk, np.asarray(instance_sites)
        self.days = np.asarray(days, dtype=np.int64)
        if self.days.max() > forecaster.cutoff_day:
            raise RuntimeError("목적 대상일이 결정 cutoff 이후에 있습니다.")
        held = forecaster.model["heldout"]
        self.block = heldout_block(self.days, held["block_days"])
        self.omega = day_weights(self.days, settings["day_weight"])
        self.pi = objective_price(self.days, price_profile(np.asarray(forecaster.data.base.smp), forecaster.cutoff_h))
        self.chunk_days = int(settings.get("chunk_days", 16))
        self.evaluations = 0

    def accumulate(self, value: np.ndarray, start: int, r: np.ndarray) -> None:
        """대상일 묶음의 R [Dc, G, 24]을 ω·π로 가중해 그룹별 값 value [G]에 더한다."""
        sl = slice(start, start + len(r))
        value += np.einsum("d,dh,dgh->g", self.omega[sl], self.pi[sl], r)

    def __call__(self, candidates: np.ndarray) -> np.ndarray:
        """candidates [C, n_instance] → Ĵ [C](원/일 단위의 기대 벌점)."""
        candidates = np.asarray(candidates, dtype=bool)
        n_pv = len(self.forecaster.data.sites)
        groups = np.zeros((2 * len(candidates), n_pv), dtype=bool)
        groups[0::2, self.sites_index] = candidates
        groups[1::2, self.sites_index] = ~candidates
        if (groups.sum(axis=1) == 0).any():
            raise ValueError("구성원이 없는 그룹이 있습니다.")
        g = len(groups)
        value = np.zeros(g)
        for start in range(0, len(self.days), self.chunk_days):
            chunk = self.days[start:start + self.chunk_days]
            days = np.repeat(chunk, g)
            pred = self.forecaster.predict(days, np.tile(groups, (len(chunk), 1)), hidden=True, chunk_days=self.chunk_days)
            r = self.risk.predict(pred, days, np.repeat(self.block[start:start + len(chunk)], g)).reshape(len(chunk), g, 24)
            self.accumulate(value, start, r)
        self.evaluations += len(candidates)
        return value[0::2] + value[1::2]


class SearchState:
    """국소 탐색의 증분 상태.

    𝒟 대상일마다 instance PV별 기여(XGB PV 입력·유효 여부·태양·결측, DL PV 표현)를 한 번 만들어 두고, 현재 배정의 그룹별 용량
    가중합(XGB 집합 feature의 합·제곱합·유효 용량·태양·결측, DL의 S⁽¹⁾·S⁽²⁾, 용량·PV 수·용량 제곱합, 관측소별 용량)을 들고 있다.
    후보 이동은 이동한 PV의 기여만 더하고 빼서 후보 가중합을 만든다. 받아들인 이동은 가중합에 반영하고, refresh번마다 전체를 다시 합한다.
    """

    def __init__(self, objective: Objective, max_gib: float, refresh: int = 25):
        f = objective.forecaster
        self.objective, self.f, self.risk, self.refresh = objective, f, objective.risk, refresh
        self.index, self.days, self.chunk = objective.sites_index, objective.days, objective.chunk_days
        self.device = f.device
        self.capacity = torch.as_tensor(f.data.sites.capacity[self.index], dtype=torch.float64, device=self.device)
        self.onehot = {p: torch.as_tensor(np.eye(len(f.data.sites.stations[p]))[f.data.sites.station_index[p][self.index]],
                                          dtype=torch.float64, device=self.device) for p in PROVIDERS}
        n, d = len(self.index), len(self.days)
        width = f.pv_width
        n_feature = len(f.features.pv_names)
        size = n * d * 24 * (n_feature * 9 + 10) + len(f.seeds) * n * d * 24 * width * 4
        self.store_device = self.device if size <= max_gib * 1024 ** 3 else torch.device("cpu")
        self.stored_bytes = int(size)
        self._build()
        self.x, self.sums, self.accepted, self.evaluations = None, None, 0, 0

    def _empty(self, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        pin = self.store_device.type == "cpu" and self.device.type == "cuda"
        return torch.empty(shape, dtype=dtype, device=self.store_device, pin_memory=pin)

    @torch.no_grad()
    def _build(self) -> None:
        """PV를 첫 축으로 둔 기여 저장소: x·finite [n, D, 24, F], cosz·sunup [n, D, 24], present [n, D, K, 24], DL [n, D, 24, H]."""
        index = torch.as_tensor(self.index, device=self.device)
        n, d, dev = len(self.index), len(self.days), self.store_device
        first = self.f.features.block(int(self.days[0]))
        n_feature = first["x"].shape[-1]
        n_present = None if first["present"] is None else first["present"].shape[0]
        store = {"x": self._empty((n, d, 24, n_feature), torch.float64), "finite": self._empty((n, d, 24, n_feature), torch.bool),
                 "cosz": self._empty((n, d, 24), torch.float64), "sunup": self._empty((n, d, 24), torch.bool),
                 "present": None if n_present is None else self._empty((n, d, n_present, 24), torch.bool)}
        for j, day in enumerate(self.days):
            block = first if j == 0 else self.f.features.block(int(day))
            store["x"][:, j] = block["x"].index_select(1, index).permute(1, 0, 2).to(dev)
            store["finite"][:, j] = (block["finite"].index_select(1, index) > 0).permute(1, 0, 2).to(dev)
            store["cosz"][:, j] = block["cosz"].index_select(1, index).permute(1, 0).to(dev)
            store["sunup"][:, j] = (block["sunup"].index_select(1, index) > 0).permute(1, 0).to(dev)
            if n_present is not None:
                store["present"][:, j] = (block["present"].index_select(2, index) > 0).permute(2, 0, 1).to(dev)
        self.store = store
        self.fused = {}
        for seed in self.f.seeds:
            out = self._empty((n, d, 24, self.f.pv_width), torch.float32)
            for start in range(0, d, self.chunk):
                part = self.f.fused(seed, self.days[start:start + self.chunk]).index_select(1, index)
                out[:, start:start + len(part)] = part.permute(1, 0, 2, 3).to(dev)
            self.fused[seed] = out

    def _gather(self, name: str, sel: torch.Tensor, days: slice, seed: int | None = None) -> torch.Tensor:
        """선택한 PV의 대상일 묶음 기여. 날짜 묶음을 먼저 잘라(복사 없는 view) PV를 고르므로 묶음 크기만 복사한다."""
        source = self.fused[seed] if seed is not None else self.store[name]
        return source[:, days].index_select(0, sel.to(source.device)).to(self.device, dtype=torch.float64, non_blocking=True)

    def _weighted(self, coef: torch.Tensor, sel: np.ndarray, days: slice) -> dict:
        """coef [m, |sel|](float64)와 선택한 instance PV의 기여 → 대상일 묶음 [m, Dc, ...]의 가중합."""
        m = len(coef)
        dc = len(range(*days.indices(len(self.days))))
        t = torch.as_tensor(sel, dtype=torch.long, device=self.device)
        x = self._gather("x", t, days)
        out = {"num": torch.einsum("mu,udtf->mdtf", coef, x), "sq": torch.einsum("mu,udtf->mdtf", coef, x * x),
               "den": torch.einsum("mu,udtf->mdtf", coef, self._gather("finite", t, days)),
               "cosz": torch.einsum("mu,udt->mdt", coef, self._gather("cosz", t, days)),
               "sunup": torch.einsum("mu,udt->mdt", coef, self._gather("sunup", t, days)),
               "present": None if self.store["present"] is None
               else torch.einsum("mu,udkt->mdtk", coef, self._gather("present", t, days))}
        for seed in self.f.seeds:
            phi = self._gather("", t, days, seed)
            out[("s1", seed)] = torch.einsum("mu,udth->mdth", coef, phi)
            out[("s2", seed)] = torch.einsum("mu,udth->mdth", coef, phi * phi)
        assert all(v is None or v.shape[:2] == (m, dc) for v in out.values())
        return out

    def _scalars(self, coef: torch.Tensor, sel: np.ndarray) -> dict:
        """용량·PV 수·용량 제곱합·관측소별 용량의 가중합 [m]. coef는 (부호 × 용량)이다."""
        t = torch.as_tensor(sel, dtype=torch.long, device=self.device)
        c = self.capacity.index_select(0, t)
        return {"cap": coef.sum(dim=1), "n_pv": (coef / c).sum(dim=1), "csq": (coef * c).sum(dim=1),
                **{("station", p): coef @ self.onehot[p].index_select(0, t) for p in PROVIDERS}}

    @torch.no_grad()
    def _group_sums(self, x: np.ndarray, pv_chunk: int = 32) -> dict:
        """배정 x의 G1·G2 가중합을 처음부터 계산한다. 키마다 [2, D, ...] 또는 [2]이다."""
        member = torch.as_tensor(np.stack([x, ~x]), dtype=torch.float64, device=self.device) * self.capacity[None, :]
        all_days = slice(0, len(self.days))
        total = None
        for start in range(0, len(self.index), pv_chunk):
            sel = np.arange(start, min(start + pv_chunk, len(self.index)))
            part = self._weighted(member[:, sel], sel, all_days)
            total = part if total is None else {k: None if v is None else total[k] + v for k, v in part.items()}
        return {**total, **self._scalars(member, np.arange(len(self.index)))}

    def reset(self, x: np.ndarray) -> None:
        self.x = np.asarray(x, dtype=bool).copy()
        self.sums = self._group_sums(self.x)
        self.accepted = 0

    def _coefficients(self, moves: list) -> tuple[np.ndarray, torch.Tensor]:
        sel = np.array(sorted({i for move in moves for i, _ in move}), dtype=np.int64)
        position = {int(i): k for k, i in enumerate(sel)}
        coef = np.zeros((len(moves), len(sel)))
        capacity = self.capacity.cpu().numpy()
        for k, move in enumerate(moves):
            for i, sign in move:
                coef[k, position[int(i)]] += sign * capacity[i]
        return sel, torch.as_tensor(coef, dtype=torch.float64, device=self.device)

    @torch.no_grad()
    def evaluate(self, moves: list) -> np.ndarray:
        """moves[k] = [(instance PV 위치, +1이면 G2→G1, −1이면 G1→G2), ...]. 각 이동을 적용한 배정의 Ĵ [K]. 빈 이동은 현재 배정이다."""
        k = len(moves)
        sel, coef = self._coefficients(moves)
        if len(sel) == 0:
            sel, coef = np.zeros(1, dtype=np.int64), torch.zeros((k, 1), dtype=torch.float64, device=self.device)
        scalars = self._scalars(coef, sel)
        current = self.sums
        cap = torch.stack([current["cap"][0] + scalars["cap"], current["cap"][1] - scalars["cap"]], dim=1)
        if bool((cap <= 0).any()):
            raise ValueError("구성원이 없는 그룹이 생기는 이동이 있습니다.")
        group_scalar = {name: torch.stack([current[name][0] + scalars[name], current[name][1] - scalars[name]], dim=1)
                        for name in ("cap", "n_pv", "csq")}
        station = {p: torch.stack([current[("station", p)][0] + scalars[("station", p)],
                                   current[("station", p)][1] - scalars[("station", p)]], dim=1) for p in PROVIDERS}
        value = np.zeros(2 * k)
        for start in range(0, len(self.days), self.chunk):
            days = slice(start, min(start + self.chunk, len(self.days)))
            dc = days.stop - days.start
            delta = self._weighted(coef, sel, days)

            def groups(name, delta=delta, days=days, dc=dc):
                base = current[name]
                g1 = base[0][days][None] + delta[name]
                g2 = base[1][days][None] - delta[name]
                return torch.stack([g1, g2], dim=2).transpose(0, 1).reshape(dc * 2 * k, *g1.shape[2:])

            def per_query(value_kg, dc=dc):
                return value_kg[None].expand(dc, *value_kg.shape).reshape(dc * 2 * k, *value_kg.shape[2:])

            xgb = {name: groups(name) for name in ("num", "sq", "den", "cosz", "sunup")}
            xgb["present"] = None if current["present"] is None else groups("present")
            xgb.update({name: per_query(group_scalar[name]) for name in ("cap", "n_pv", "csq")})
            dl = {seed: (groups(("s1", seed)), groups(("s2", seed))) for seed in self.f.seeds}
            stations = {p: per_query(station[p]).cpu().numpy() for p in PROVIDERS}
            query_days = np.repeat(self.days[days], 2 * k)
            pred = self.f.predict_aggregates(Aggregates(query_days, xgb, dl, stations), hidden=True)
            r = self.risk.predict(pred, query_days, np.repeat(self.objective.block[days], 2 * k)).reshape(dc, 2 * k, 24)
            self.objective.accumulate(value, start, r)
        self.evaluations += k
        return value[0::2] + value[1::2]

    @torch.no_grad()
    def apply(self, move: list) -> None:
        """받아들인 이동을 배정과 가중합에 반영한다. refresh번마다 가중합을 처음부터 다시 계산해 누적 오차를 막는다."""
        for i, sign in move:
            self.x[i] = sign > 0
        self.accepted += 1
        if self.accepted % self.refresh == 0:
            self.sums = self._group_sums(self.x)
            return
        sel, coef = self._coefficients([move])
        delta = self._weighted(coef, sel, slice(0, len(self.days)))
        for name, v in delta.items():
            if v is not None:
                self.sums[name][0] += v[0]
                self.sums[name][1] -= v[0]
        for name, v in self._scalars(coef, sel).items():
            self.sums[name][0] += v[0]
            self.sums[name][1] -= v[0]

    def describe(self) -> dict:
        return {"store_device": self.store_device.type, "stored_bytes": self.stored_bytes, "evaluations": self.evaluations,
                "refresh": self.refresh}


def sample_moves(x: np.ndarray, capacity: np.ndarray, share: float, count: int, rng: np.random.Generator) -> list:
    """용량 제약을 지키는 flip·swap 이동을 count개까지 뽑는다. 이동은 [(PV 위치, +1 G2→G1 / −1 G1→G2), ...]이다."""
    moves = []
    for _ in range(count * 4):
        if len(moves) >= count:
            break
        if rng.random() < 0.5:
            i, j = int(rng.choice(np.flatnonzero(x))), int(rng.choice(np.flatnonzero(~x)))
            move = [(i, -1), (j, 1)]
        else:
            i = int(rng.integers(len(x)))
            move = [(i, -1 if x[i] else 1)]
        y = x.copy()
        for pv, sign in move:
            y[pv] = sign > 0
        if y.any() and (~y).any() and capacity_ok(y, capacity, share):
            moves.append(move)
    return moves


def local_search(state: SearchState, x0: np.ndarray, capacity: np.ndarray, share: float, settings: dict,
                 rng: np.random.Generator) -> tuple[np.ndarray, float, list[dict]]:
    """flip·swap 국소 탐색. 매 반복 용량 제약을 지키는 이동을 moves_per_iteration개 뽑아 한 번에 평가하고, 가장 좋은 이동이
    tolerance보다 크게 줄이면 받아들인다. patience번 연속으로 받아들이지 못하거나 max_iterations에 닿으면 멈춘다.
    목적값이 유한하지 않은 후보는 고르지 않고 개수를 기록한다. 시작 배정이나 한 반복의 모든 후보가 유한하지 않으면 멈춘다."""
    state.reset(repair(x0, capacity, share))
    value = float(state.evaluate([[]])[0])
    if not np.isfinite(value):
        raise RuntimeError("시작 배정의 목적값이 유한하지 않습니다.")
    trace = [{"iteration": 0, "objective": value}]
    stale, nonfinite = 0, 0
    for it in range(1, int(settings["max_iterations"]) + 1):
        moves = sample_moves(state.x, capacity, share, int(settings["moves_per_iteration"]), rng)
        if not moves:
            break
        values = np.asarray(state.evaluate(moves), dtype=np.float64)
        bad = ~np.isfinite(values)
        if bad.all():
            raise RuntimeError(f"반복 {it}의 후보 목적값이 모두 유한하지 않습니다.")
        nonfinite += int(bad.sum())
        values = np.where(bad, np.inf, values)
        best = int(np.argmin(values))
        if values[best] < value - float(settings["tolerance"]) * max(abs(value), 1.0):
            state.apply(moves[best])
            value, stale = float(values[best]), 0
            trace.append({"iteration": it, "objective": value})
        else:
            stale += 1
            if stale >= int(settings["patience"]):
                break
        if it % 10 == 0:
            print(f"[{time.strftime('%H:%M:%S')}]   반복 {it}: 목적 {value:.1f}, 연속 미개선 {stale}", flush=True)
    if nonfinite:
        trace.append({"nonfinite_candidates": nonfinite})
    return state.x.copy(), value, trace


def search(state: SearchState, starts: dict, capacity: np.ndarray, share: float, settings: dict) -> dict:
    """시작점마다 국소 탐색하고 Ĵ가 가장 낮은 해를 고른다."""
    rng = np.random.default_rng(int(settings["seed"]))
    records, best = {}, None
    for name, x0 in starts.items():
        started = time.perf_counter()
        x, value, trace = local_search(state, x0, capacity, share, settings, rng)
        steps = [t for t in trace if "iteration" in t]
        records[name] = {"objective": value, "accepted": len(steps) - 1, "trace": trace, "seconds": time.perf_counter() - started,
                         "start_objective": steps[0]["objective"]}
        print(f"[{time.strftime('%H:%M:%S')}] 배정 탐색 {name}: {steps[0]['objective']:.1f} → {value:.1f}", flush=True)
        if best is None or value < best[1]:
            best = (name, value, x)
    return {"x": best[2], "objective": best[1], "best_start": best[0], "starts": records}


def starting_points(capacity: np.ndarray, index: np.ndarray, sites, share: float, settings: dict) -> dict:
    """original(instance 발전소의 sites.csv 배정), 관측소 층화 이분할, 무작위 이분할."""
    rng = np.random.default_rng(int(settings["seed"]) + 1)
    starts = {}
    if settings["starts"].get("original", True):
        starts["original"] = sites.group[index] == 0
    if settings["starts"].get("stratified", True):
        starts["stratified"] = stratified_split(index, sites)
    for i in range(int(settings["starts"].get("random_halves", 0))):
        starts[f"random{i}"] = rng.random(len(index)) < 0.5
    return {name: repair(x, capacity, share) for name, x in starts.items()}


def partition_key(risk: RiskModel, instance_sites: np.ndarray, days: np.ndarray, settings: dict) -> str:
    return digest({"risk": risk.key(), "sites": np.asarray(instance_sites).tolist(), "days": np.asarray(days).tolist(),
                   "settings": settings, "versions": versions("partition")})
