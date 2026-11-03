"""배정 최적화: 공통 시나리오의 정확한 가중 MAE와 soft balance 목적(CUDA에서 flip·swap 값 계산),
QUBO zᵀQz(z = 2x − 1)의 Gurobi 이진 2차계획·tabu 해법, flip·swap local search.
"""
from __future__ import annotations

import time

import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB


def local_search(obj, x0: np.ndarray, band: float, max_iter: int = 20_000, tol: float = 1e-12) -> tuple[np.ndarray, float, int]:
    """flip·swap 중 가장 좋은 개선 move를 반복한다. band를 벗어나는 move는 하지 않는다."""
    x = x0.copy()
    st = obj.state(x)
    J = obj.current(st)
    for it in range(max_iter):
        flip_values, flip_bal = obj.flips(x, st)
        allowed = np.where(np.abs(flip_bal) <= band + 1e-12, flip_values, np.inf)
        i = int(np.argmin(allowed))
        best_value, best_moves = float(allowed[i]), [i]
        i1, i2, swap_values, swap_bal = obj.swaps(x, st, flip_values)
        if swap_values.size:
            swap_values = np.where(np.abs(swap_bal) <= band + 1e-12, swap_values, np.inf)
            a, b = np.unravel_index(int(np.argmin(swap_values)), swap_values.shape)
            if swap_values[a, b] < best_value:
                best_value, best_moves = float(swap_values[a, b]), [int(i1[a]), int(i2[b])]
        if not best_value < J - tol:
            return x, J, it
        obj.apply(x, st, best_moves)
        J = obj.current(st)
    return x, J, max_iter


STATUS = {GRB.OPTIMAL: "optimal", GRB.WORK_LIMIT: "work_limit", GRB.TIME_LIMIT: "time_limit",
          GRB.SUBOPTIMAL: "suboptimal", GRB.INTERRUPTED: "interrupted"}


def gurobi_version() -> str:
    return ".".join(map(str, gp.gurobi.version()))


def qubo_coefficients(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """zᵀQz를 x의 2차식으로 쓴 계수 (4Q, −4Q1, 1ᵀQ1)."""
    return 4 * Q, -4 * Q.sum(axis=1), float(Q.sum())


def qubo_energy(Q: np.ndarray, x: np.ndarray) -> float:
    z = 2 * np.asarray(x, dtype=np.float64) - 1
    return float(z @ Q @ z)


def orient(x: np.ndarray, anchor: int) -> np.ndarray:
    x = np.asarray(x, dtype=bool)
    return x.copy() if x[anchor] else ~x


def set_starts(model: gp.Model, assignments: list[list[tuple]]) -> None:
    """여러 MIP start. assignment는 (MVar, 값) 쌍의 목록이다."""
    model.NumStart = len(assignments)
    model.update()
    for i, assignment in enumerate(assignments):
        model.Params.StartNumber = i
        for var, value in assignment:
            var.Start = value


def summary(model: gp.Model, scale: float, build_seconds: float) -> dict:
    if model.SolCount == 0:
        raise RuntimeError(f"Gurobi가 가능해를 찾지 못했습니다: status={model.Status}")
    return {"status": STATUS.get(model.Status, str(model.Status)), "objective": model.ObjVal / scale, "bound": model.ObjBound / scale,
            "gap": model.MIPGap, "seconds": model.Runtime, "work": model.Work, "nodes": model.NodeCount,
            "solutions": model.SolCount, "build_seconds": build_seconds}


def environment(log_path) -> gp.Env:
    base = {"LogToConsole": 0}
    if log_path is None:
        base["OutputFlag"] = 0
    else:
        base["LogFile"] = str(log_path)
    return gp.Env(params=base)


def tabu_qubo(Q: np.ndarray, capacity: np.ndarray, starts: list[np.ndarray], restarts: int, iterations: int, seed: int) -> tuple[np.ndarray, dict]:
    """min zᵀQz의 1-flip tabu search. Gurobi 없이 같은 QUBO를 푸는 대체 해법이다.

    주어진 배정과 무작위 배정 restarts개에서 각각 iterations번 이동한다. 이동은 tabu가 아닌 flip 중 에너지 증가가 가장 작은 것이고,
    그 시작점의 최선값을 낮추는 flip은 tabu여도 허용한다. 용량 최대 발전소를 G1에 고정하고 두 그룹을 비우지 않는다.
    """
    n = len(Q)
    anchor = int(np.argmax(capacity))
    diag = np.diag(Q).copy()
    rng = np.random.default_rng(seed)
    tenure = max(7, n // 10)
    initial = [orient(s, anchor) for s in starts]
    for _ in range(restarts):
        x = rng.random(n) < 0.5
        x[anchor] = True
        if x.all():
            x[int(np.argmin(capacity))] = False
        initial.append(x)
    best_z, best_e = None, np.inf
    for x0 in initial:
        z = np.where(x0, 1.0, -1.0)
        Qz = Q @ z
        e = float(z @ Qz)
        tabu = np.zeros(n, dtype=np.int64)
        local_e, local_z = e, z.copy()
        for it in range(iterations):
            delta = 4.0 * (diag - z * Qz)
            allowed = np.ones(n, dtype=bool)
            allowed[anchor] = False
            members = z > 0
            if (~members).sum() == 1:
                allowed[~members] = False
            candidate = allowed & ((tabu <= it) | (e + delta < local_e))
            if not candidate.any():
                break
            i = int(np.argmin(np.where(candidate, delta, np.inf)))
            Qz -= 2.0 * z[i] * Q[i]  # Q는 대칭이므로 연속된 행을 쓴다
            e += float(delta[i])
            z[i] = -z[i]
            tabu[i] = it + tenure + int(rng.integers(0, tenure // 2 + 1))
            if e < local_e:
                local_e, local_z = e, z.copy()
        energy = qubo_energy(Q, local_z > 0)
        if energy < best_e:
            best_e, best_z = energy, local_z.copy()
    return best_z > 0, {"energy": best_e, "starts": len(initial), "iterations": iterations, "tenure": tenure, "seed": seed}


def solve_qubo(Q: np.ndarray, capacity: np.ndarray, starts: list[np.ndarray], params: dict, log_path=None) -> tuple[np.ndarray, dict]:
    """min zᵀQz, z = 2x − 1. 두 그룹을 맞바꿔도 값이 같으므로 용량이 가장 큰 발전소를 G1에 고정하고 두 그룹을 비우지 않는다.

    계수는 최대 절댓값이 1이 되도록 나누고 보고값은 원래 척도로 되돌린다.
    """
    n = len(Q)
    scale = 1.0 / float(np.abs(Q).max())
    quadratic, linear, constant = qubo_coefficients(Q * scale)
    anchor = int(np.argmax(capacity))
    started = time.perf_counter()
    with environment(log_path) as env, gp.Model("qubo", env=env) as model:
        x = model.addMVar(n, vtype=GRB.BINARY, name="x")
        model.addConstr(x[anchor] == 1, name="anchor")
        model.addConstr(x.sum() <= n - 1, name="nonempty")
        model.setMObjective(quadratic, linear, constant, xQ_L=x, xQ_R=x, xc=x, sense=GRB.MINIMIZE)
        set_starts(model, [[(x, orient(s, anchor).astype(float))] for s in starts])
        for key, value in params.items():
            model.setParam(key, value)
        build = time.perf_counter() - started
        model.optimize()
        record = summary(model, scale, build)
        solution = x.X > 0.5
    record.update(energy=qubo_energy(Q, solution), gurobi=gurobi_version())
    return solution, record


def group_deviation(r: torch.Tensor, total: torch.Tensor, center: str) -> torch.Tensor:
    """두 그룹의 시나리오별 절대오차. r은 G1 합, total은 전체 합이며 시나리오 축은 1이다.

    center="median"이면 시각별 그룹 합 분포의 중앙값을 제출값으로 보아 |R − median(R)|을 쓴다(그룹 모델은 그룹 중앙값을 낸다).
    center="none"이면 고정 점예측의 합을 제출값으로 보아 |R|을 쓴다.
    """
    r2 = total - r
    if center == "median":
        return (r - lower_median(r)).abs() + (r2 - lower_median(r2)).abs()
    return r.abs() + r2.abs()


def balance_coefficients(relative: float, merge: float, moment: np.ndarray, band: float) -> tuple[float, float]:
    """λ̃를 MAE 목적과 QUBO의 균형 계수로 바꾼다. |b| = β에서 각 penalty가 λ̃ × 그 척도의 merge 값이 된다.

    MAE의 merge 값은 전체 합 분포의 중앙값 기준 가중 절대오차, QUBO의 merge 값은 전체 합의 MSE ½·𝟙ᵀM𝟙이다.
    """
    return relative * merge / band ** 2, relative * 0.5 * float(np.sum(moment)) / band ** 2


def first_within_band(imbalances, band: float) -> int:
    """오름차순 λ̃ grid의 해 가운데 |b| ≤ β인 첫 해의 위치. 없으면 마지막(가장 큰 λ̃) 위치이다."""
    for i, b in enumerate(imbalances):
        if abs(b) <= band + 1e-12:
            return i
    return len(imbalances) - 1


def lower_median(x: torch.Tensor) -> torch.Tensor:
    """시나리오 축(1)의 하위 중앙값. CUDA의 torch.median은 결정성 모드에서 쓸 수 없어 정렬로 구한다."""
    return x.sort(dim=1).values.narrow(1, (x.shape[1] - 1) // 2, 1)


class ScenarioObjective:
    """시나리오 잔차의 가중 기대 절대오차와 soft balance. 정규화 척도는 C·Σw_h이다.

    center="none": error [시각×시나리오, PV], weight = w_h·p_hs.
    center="median": 같은 행 순서를 [시각, 시나리오, PV]로 보고(시각마다 samples개, 같은 확률), 그룹 중앙값을 제출값으로 쓴다.
    """

    def __init__(self, error, weight, capacity, lam=1.0, swap_candidates=8, device="cuda", samples=1, center="none"):
        error, weight, capacity = (np.asarray(x) for x in (error, weight, capacity))
        if error.ndim != 2 or error.shape != (len(weight), len(capacity)):
            raise ValueError("시나리오·가중치·용량의 축이 일치해야 합니다.")
        if not np.isfinite(error).all() or not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
            raise ValueError("시나리오와 가중치는 유한해야 하고 가중치 합은 양수여야 합니다.")
        if not np.isfinite(capacity).all() or (capacity <= 0).any():
            raise ValueError("설비용량은 유한한 양수여야 합니다.")
        if center not in ("none", "median"):
            raise ValueError(center)
        self.device, self.center = torch.device(device), center
        dtype = torch.float32 if self.device.type == "cuda" else torch.float64
        if center == "median":
            samples = int(samples)
            if len(weight) % samples:
                raise ValueError("시나리오 행 수는 시각 수 × samples여야 합니다.")
            per_hour = weight.reshape(-1, samples)
            if not np.allclose(per_hour, per_hour[:, :1]):
                raise ValueError("중앙값 기준 목적은 시각 안에서 시나리오 확률이 같아야 합니다.")
            hour_weight = per_hour.sum(axis=1)
            active = hour_weight > 0
            values = error.reshape(-1, samples, error.shape[-1])[active]
            self.w = torch.as_tensor(hour_weight[active] / hour_weight.sum(), dtype=dtype, device=self.device)
            self.row_weight = torch.as_tensor(np.repeat(hour_weight[active] / hour_weight.sum() / samples, samples), dtype=dtype, device=self.device)
        else:
            active = weight > 0
            values = error[active]
            self.w = torch.as_tensor(weight[active] / weight.sum(), dtype=dtype, device=self.device)
            self.row_weight = self.w
        self.E = torch.as_tensor(values / capacity.sum(), dtype=dtype, device=self.device)
        self.Et = self.E.sum(dim=-1)
        self.c = capacity.astype(np.float64) / capacity.sum()
        self.lam, self.K = float(lam), int(swap_candidates)

    def state(self, x):
        select = torch.as_tensor(x, dtype=torch.bool, device=self.device)
        return {"R1": self.E[..., select].sum(-1), "b": float(self.c @ (2 * np.asarray(x, float) - 1))}

    def _loss(self, r):
        extra = r.ndim - self.Et.ndim
        total = self.Et.reshape(self.Et.shape + (1,) * extra)
        deviation = group_deviation(r, total, self.center)
        if self.center == "median":
            deviation = deviation.mean(dim=1)
        return (deviation * self.w.reshape((-1,) + (1,) * extra)).sum(dim=0, dtype=torch.float64)

    def current(self, state):
        return float(self._loss(state["R1"])) + self.lam * state["b"] ** 2

    def value(self, x):
        return self.current(self.state(x))

    def flips(self, x, state):
        sign = np.where(x, -1.0, 1.0)
        bal = state["b"] + 2 * sign * self.c
        shifts = self.E * torch.as_tensor(sign, dtype=self.E.dtype, device=self.device)
        values = self._loss(state["R1"][..., None] + shifts).cpu().numpy() + self.lam * bal ** 2
        # 연구 단계는 soft balance를 쓰되 두 그룹에 적어도 한 발전소를 둔다.
        if x.sum() == 1:
            values[x] = np.inf
        if (~x).sum() == 1:
            values[~x] = np.inf
        return values, bal

    def swaps(self, x, state, flip_values):
        a, b = np.flatnonzero(x), np.flatnonzero(~x)
        a, b = a[np.argsort(flip_values[a])[:self.K]], b[np.argsort(flip_values[b])[:self.K]]
        bal = state["b"] - 2 * self.c[a, None] + 2 * self.c[b][None, :]
        values = np.empty((len(a), len(b)))
        # 행렬 하나의 메모리가 전체 시나리오 수×후보 수에 비례하도록 한쪽 축을 순회한다.
        for i, site in enumerate(a):
            r = state["R1"][..., None] - self.E[..., int(site), None] + self.E[..., b]
            values[i] = self._loss(r).cpu().numpy() + self.lam * bal[i] ** 2
        return a, b, values, bal

    def apply(self, x, state, moves):
        for i in moves:
            x[i] = not x[i]
        # FP32 incremental 누적 대신 선택된 배정의 합을 다시 계산한다.
        state.update(self.state(x))

    def merge_value(self):
        """x = ½ 연속완화의 값(균형항 제외): 전체 합 시나리오의 중앙값 기준 가중 절대오차. center="none"이면 |전체 합|이다."""
        if self.center == "median":
            deviation = (self.Et - lower_median(self.Et)).abs().mean(dim=1)
        else:
            deviation = self.Et.abs()
        return float((deviation * self.w).sum(dtype=torch.float64))

    def second_moment(self):
        flat = self.E.reshape(-1, self.E.shape[-1])
        return (flat.T @ (flat * self.row_weight[:, None])).double().cpu().numpy()

    def close(self):
        del self.E, self.Et, self.w, self.row_weight
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def optimize_scenarios(obj, capacity, initial, random_starts, seed, max_iterations, tolerance, qubo_params,
                       qubo_lambda=None, moment=None, tabu=None):
    """QUBO MIQP 해·주어진 배정·무작위 초기해를 soft balance 영역에서 개선하고 목적이 가장 낮은 해를 고른다.

    QUBO는 같은 시나리오의 2차 moment 요약 J_MSE = zᵀ(½M + λ_Q c̃c̃ᵀ)z이고, 주어진 배정을 MIP start로 Gurobi로 푼다.
    λ_Q를 주지 않으면 local search의 λ와 같다. tabu 설정을 주면 같은 QUBO를 tabu search로도 풀어 기록만 한다.
    """

    c = np.asarray(capacity) / np.sum(capacity)
    moment = obj.second_moment() if moment is None else moment
    Q = 0.5 * moment + (obj.lam if qubo_lambda is None else qubo_lambda) * np.outer(c, c)
    x_qubo, qubo_record = solve_qubo(Q, np.asarray(capacity), list(initial.values()), qubo_params)
    if tabu is not None:
        x_tabu, shadow = tabu_qubo(Q, np.asarray(capacity), list(initial.values()), **tabu)
        same = np.array_equal(x_tabu, x_qubo) or np.array_equal(x_tabu, ~x_qubo)
        qubo_record["shadow_tabu"] = {**shadow, "relative_gap_to_gurobi": (shadow["energy"] - qubo_record["energy"]) / qubo_record["energy"],
                                      "same_partition": bool(same)}
    starts = {"qubo": x_qubo, **initial}
    rng = np.random.default_rng(seed)
    for i in range(random_starts):
        x = rng.random(len(c)) < 0.5
        if x.all() or not x.any():
            x[:len(x) // 2] = True
            x[len(x) // 2:] = False
        starts[f"random{i}"] = x
    best, records = None, {}
    for name, value in starts.items():
        x0 = value.copy()
        if x0.all() or not x0.any():
            x0[int(np.argmin(c))] = not x0[0]
        x, score, iterations = local_search(obj, x0, 1.0, max_iterations, tolerance)
        records[name] = {"objective": score, "iterations": iterations, "imbalance": float(c @ (2 * x - 1.0))}
        if best is None or score < best[1]:
            best = (x.copy(), score, name)
    return best[0], {"objective": best[1], "best_start": best[2], "starts": records, "qubo_solver": qubo_record,
                     "constraint": "nonempty groups; soft balance"}


def restore_feasibility(obj, x, capacity, minimum_kw):
    """최종 적용 시 부족한 그룹으로 목적 증가가 가장 작은 발전소를 옮기고 다시 채점한다."""
    cap = np.asarray(capacity, dtype=float)
    if cap.sum() < 2 * minimum_kw:
        raise ValueError("최종 두 그룹의 최소 용량을 충족할 총용량이 없습니다.")
    x = x.copy()
    initial = obj.value(x)
    moves = []
    for _ in range(len(x)):
        c1, c2 = cap[x].sum(), cap[~x].sum()
        if min(c1, c2) >= minimum_kw - 1e-8:
            return x, {"moves": moves, "before_objective": initial, "after_objective": obj.value(x),
                       "capacity_kw": [float(c1), float(c2)]}
        donor = x if c1 > c2 else ~x
        donor_capacity = max(c1, c2)
        allowable = donor & (donor_capacity - cap >= minimum_kw)
        if not allowable.any():
            raise ValueError("단일 이동으로 최소 용량을 복구할 수 없습니다.")
        state = obj.state(x)
        values, _ = obj.flips(x, state)
        i = int(np.argmin(np.where(allowable, values, np.inf)))
        x[i] = not x[i]
        moves.append(i)
    raise RuntimeError("최소 용량 복구가 종료되지 않았습니다.")
