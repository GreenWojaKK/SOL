"""배정 계약: 그룹 최소 용량 조건과 기준 층화 배정.

그룹마다 용량이 20,000 kW 이상이어야 한다. 배정할 발전소 집합에서는 이 조건을 253개소 전체 용량에 대한 비율로 바꿔,
각 그룹 용량이 집합 용량의 그 비율 이상인지 본다. 기준 층화 배정은 같은 층(예: WA·WB 관측소 쌍)의 발전소를 두 그룹에
용량이 비슷하게 나누고, 비율 조건을 만족할 때까지 큰 그룹에서 발전소를 옮겨 고친다.
"""
from __future__ import annotations

import numpy as np

from .timeaxis import MINIMUM_GROUP_CAPACITY_KW


def minimum_share(full_capacity_kw: float) -> float:
    """그룹당 최소 용량 비율. 253개소 전체 용량 42,655.37 kW에서 20,000 kW에 해당하는 0.4689이다."""
    share = MINIMUM_GROUP_CAPACITY_KW / float(full_capacity_kw)
    if share >= 0.5:
        raise ValueError(f"전체 용량 {full_capacity_kw:.1f} kW로는 두 그룹이 각각 20,000 kW 이상일 수 없습니다.")
    return share


def capacity_ok(x: np.ndarray, capacity: np.ndarray, share: float) -> bool:
    """두 그룹(x가 True인 첫 그룹과 나머지)이 각각 전체 용량의 share 이상인지."""
    total = capacity.sum()
    first = capacity[x].sum()
    return bool(first >= share * total - 1e-9 and total - first >= share * total - 1e-9)


def repair(x: np.ndarray, capacity: np.ndarray, share: float) -> np.ndarray:
    """용량 비율 제약을 만족할 때까지 큰 그룹에서 제약 위반을 가장 많이 줄이는 발전소를 옮긴다."""
    x = np.asarray(x, dtype=bool).copy()
    total = capacity.sum()
    for _ in range(len(x)):
        if capacity_ok(x, capacity, share):
            return x
        first = capacity[x].sum()
        big = first > total - first
        candidates = np.flatnonzero(x == big)
        after = np.abs((first - capacity[candidates] if big else first + capacity[candidates]) - total / 2)
        x[candidates[int(np.argmin(after))]] = not big
    if not capacity_ok(x, capacity, share):
        raise RuntimeError("용량 제약을 만족하는 배정으로 고칠 수 없습니다.")
    return x


def stratified_split(capacity: np.ndarray, keys: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """같은 층(keys의 같은 행)의 발전소를 용량이 비슷하게 두 그룹에 나누고, 첫 그룹이면 True인 배정을 돌려준다.

    rng가 없으면 층 순서는 key 순서, 층 안은 용량 큰 순서이고, 두 그룹의 층 안 용량이 같으면 전체 용량이 작은 그룹을 고른다
    (결정적 기준 배정). rng가 있으면 층 순서와 층 안 순서를 섞는다.
    """
    capacity = np.asarray(capacity, dtype=np.float64)
    keys = np.asarray(keys).reshape(len(capacity), -1)
    strata = np.unique(keys, axis=0)
    if rng is not None:
        strata = strata[rng.permutation(len(strata))]
    first = np.zeros(len(capacity), dtype=bool)
    total = np.zeros(2)
    for key in strata:
        local_index = np.flatnonzero((keys == key).all(axis=1))
        order = rng.permutation(local_index) if rng is not None else local_index[np.argsort(-capacity[local_index], kind="stable")]
        local = np.zeros(2)
        for i in order:
            g = int(np.argmin(local)) if local[0] != local[1] else int(np.argmin(total))
            first[i] = g == 0
            local[g] += capacity[i]
            total[g] += capacity[i]
    return first
