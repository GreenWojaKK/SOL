"""배정 표현과 instance: 구성 파일 읽기·검증과 Sites의 그룹 배정 override, 상대 용량 band, instance 구성, 무작위 배정,
253개소 배정 완성.

분할 x는 instance 발전소마다 G1이면 True인 bool 배열이고, z = 2x − 1, c̃ = c / C_τ이다. 발전소 좌표는 쓰지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..data import GROUPS, Sites
from ..forecast.queries import actual_view, freeze_cohort
from ..scoring import PENALTY_RATE
from ..store import digest
from ..submission import MINIMUM_GROUP_CAPACITY_KW

if TYPE_CHECKING:
    from ..config import Spec
    from ..experiment import Data


def settings(spec: Spec) -> dict:
    return spec.protocol["bundling_search"]


@dataclass
class Instance:
    name: str
    cutoff_h: int
    sites: np.ndarray                    # 전역 발전소 index
    capacity: np.ndarray                 # kW
    original: np.ndarray                       # original에서 G1이면 True
    est_hours: np.ndarray
    est_error: np.ndarray                # [H, n] kWh. 정답 없음·야간은 NaN
    est_w: np.ndarray
    est_k: np.ndarray
    target_w: np.ndarray                 # 목적의 시각 가중치를 정하는 기간(평가 또는 test)의 w_h
    target_k: np.ndarray
    target_daylight: np.ndarray
    eval_hours: np.ndarray | None = None
    eval_error: np.ndarray | None = None  # [H, n] kWh. 모든 발전소에 정답이 있는 시각만
    eval_w: np.ndarray | None = None

    @property
    def total(self) -> float:
        return float(self.capacity.sum())

    @property
    def ctil(self) -> np.ndarray:
        return self.capacity / self.total


def estimation_errors(forecasts: dict, names: list[str], cutoff_h: int, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cutoff 이전 시각의 발전소 오차를 모은다. 나중 기간의 예측이 겹치는 시각을 덮어쓴다."""
    parts = [forecasts[n] for n in names]
    first = min(int(f["hours"][0]) for f in parts)
    if first > cutoff_h:
        raise ValueError("cutoff 이전 추정 오차가 없습니다.")
    error = np.full((cutoff_h - first + 1, len(index)), np.nan)
    covered = np.zeros(cutoff_h - first + 1, dtype=bool)
    for f in parts:
        keep = f["hours"] <= cutoff_h
        rows = f["hours"][keep] - first
        error[rows] = f["error"][keep][:, index]
        covered[rows] = True
    hours = np.arange(first, cutoff_h + 1)[covered]
    if hours.max() > cutoff_h:
        raise ValueError("M 추정에 cutoff 이후 시각이 들어갔습니다.")
    return hours, error[covered]


def build_instance(spec: Spec, data: Data, name: str, forecasts: dict, item: dict | None = None) -> Instance:
    """item({"cutoff", "estimation", "evaluation"})을 주지 않으면 protocol의 bundling instance 정의를 쓴다."""
    item = item or settings(spec)["instances"][name]
    period = spec.period(item["cutoff"])
    sites, base = data.sites, data.base
    energy, status, sunup, smp, k = (np.asarray(a) for a in (base.energy, base.status, base.sunup, base.smp, base.k))
    members = freeze_cohort(actual_view(energy, status, period.cutoff_h), sites, period).members.any(axis=0)
    index = np.flatnonzero(members)
    hours, error = estimation_errors(forecasts, item["estimation"], period.cutoff_h, index)
    error[~sunup[hours][:, index]] = np.nan
    weight = lambda h: PENALTY_RATE * smp[h] * k[h]  # noqa: E731
    target = forecasts[item["evaluation"]]["hours"] if item["evaluation"] else period.hours
    inst = Instance(name, period.cutoff_h, index, sites.capacity[index], sites.group[index] == 0, hours, error, weight(hours), k[hours],
                    weight(target), k[target], sunup[target][:, index].any(axis=1))
    if item["evaluation"]:
        f = forecasts[item["evaluation"]]
        e = f["error"][:, index].astype(np.float64)
        complete = np.isfinite(e).all(axis=1)
        inst.eval_hours, inst.eval_error, inst.eval_w = f["hours"][complete], e[complete], weight(f["hours"][complete])
    return inst


def random_partition(ctil: np.ndarray, band: float, rng: np.random.Generator) -> np.ndarray:
    """무작위 배정 뒤, 큰 그룹에서 무작위로 고른 발전소를 옮겨 band 안에 넣는다."""
    x = rng.random(len(ctil)) < 0.5
    b = float(ctil @ (2 * x - 1.0))
    while abs(b) > band:
        side = b > 0
        candidates = np.flatnonzero(x == side)
        after = np.abs(b - (2 if side else -2) * ctil[candidates])
        ok = candidates[after < abs(b)]
        if len(ok) == 0:
            break
        landing = ok[np.abs(b - (2 if side else -2) * ctil[ok]) <= band]
        i = int(rng.choice(landing if len(landing) else ok))
        x[i] = not x[i]
        b = float(ctil @ (2 * x - 1.0))
    return x


def full_assignment(n_sites: int, capacity: np.ndarray, index: np.ndarray, x: np.ndarray) -> np.ndarray:
    """instance 밖 발전소는 용량이 작은 그룹에 큰 순서로 배정해 253개소 전체 구성을 만든다."""
    group = np.full(n_sites, -1, dtype=np.int8)
    group[index] = np.where(x, 0, 1)
    load = [float(capacity[group == g].sum()) for g in (0, 1)]
    for i in np.flatnonzero(group < 0)[np.argsort(-capacity[group < 0], kind="stable")]:
        g = 0 if load[0] <= load[1] else 1
        group[i] = g
        load[g] += capacity[i]
    return group


def partition_hash(ids, group) -> str:
    """`contract.load_sites`의 original hash와 같은 정의이므로 original 배정이면 sites.original_hash와 같다."""
    return digest(list(zip([str(s) for s in ids], [GROUPS[int(g)] for g in group])))


def read_partition(path: str | Path, sites: Sites) -> np.ndarray:
    """composition.csv(site_id, group)를 sites 순서의 그룹 index 배열로 읽는다. 253개소 전부가 한 번씩 있어야 한다."""
    frame = pd.read_csv(path, dtype={"site_id": str, "group": str})
    if list(frame.columns) != ["site_id", "group"]:
        raise ValueError("partition 파일의 열은 site_id, group이어야 합니다.")
    if not frame.site_id.is_unique or sorted(frame.site_id) != sorted(sites.ids.astype(str)):
        raise ValueError("partition 파일은 sites.csv의 발전소를 한 번씩 모두 담아야 합니다.")
    if not frame.group.isin(GROUPS).all():
        raise ValueError(f"partition의 group은 {GROUPS} 중 하나여야 합니다.")
    order = pd.Index(frame.site_id).get_indexer(sites.ids.astype(str))
    return np.array([GROUPS.index(g) for g in frame.group.to_numpy()[order]], dtype=np.int8)


def apply_partition(sites: Sites, path: str | Path) -> tuple[Sites, dict]:
    """그룹 배정과 hash만 바꾼 Sites와 구성 기록을 돌려준다. base·weather cache key는 그룹과 무관하다."""
    group = read_partition(path, sites)
    new_hash = partition_hash(sites.ids, group)
    capacity = [float(sites.capacity[group == g].sum()) for g in range(len(GROUPS))]
    record = {"path": str(Path(path).resolve()), "hash": new_hash, "is_original": new_hash == sites.original_hash,
              "capacity_kw": dict(zip(GROUPS, capacity)), "sites": dict(zip(GROUPS, [int((group == g).sum()) for g in range(len(GROUPS))]))}
    return replace(sites, group=group, original_hash=new_hash), record


def write_partition(path: str | Path, sites: Sites, group: np.ndarray) -> Path:
    """제출 형식(site_id, group)으로 쓴다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"site_id": sites.ids.astype(str), "group": [GROUPS[int(g)] for g in group]}).to_csv(path, index=False, lineterminator="\n")
    return path


def relative_band(total_capacity_kw: float, minimum_kw: float = MINIMUM_GROUP_CAPACITY_KW) -> float:
    """그룹당 최소 용량 제약을 상대 불균형 |C_1 − C_2| / C의 상한으로 바꾼다. 253개소 전체에서 0.0623이다."""
    return 1.0 - 2.0 * minimum_kw / total_capacity_kw
