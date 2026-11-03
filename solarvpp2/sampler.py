"""학습·risk query sampler, held-out 블록, validation panel 날짜, 편입 규칙.

참조: docs/solarvpp2-design.md#query-sampler, docs/solarvpp2-design.md#training-days.

Query의 집합 A는 sampler가 뽑은 그대로 쓴다. 구성원 중 하나라도 시각 h의 정답이 없으면 그 query의 시각 h 정답 mask를 0으로 두고,
결측 때문에 구성원을 빼거나 query를 나누지 않는다. A는 대상일에 편입된 발전소, 즉 첫 유효 발전량이 그날까지 나타난 cohort
발전소로만 구성한다. 학습 fit은 실적을 ActualView로만 받는다(cutoff 이후 generation을 먼저 지운다).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vppcontract.partition import stratified_split as split_strata
from vppcontract.timeaxis import to_h

from .data import Sites
from .store import array_hash

KINDS = ("original", "random_half", "stratified_half", "uniform_subset", "station_subset")
SETTING_KINDS = {"original": "original", "random_halves": "random_half", "stratified_halves": "stratified_half",
                 "uniform_subsets": "uniform_subset", "station_subsets": "station_subset"}
NEVER = np.iinfo(np.int32).max


@dataclass
class ActualView:
    cutoff_h: int
    energy: np.ndarray
    valid: np.ndarray

    def observed(self, values: np.ndarray) -> np.ndarray:
        """observed 배열 [h, ...]에서 cutoff 이후를 결측으로 둔 사본."""
        out = np.array(values, dtype=np.float32, copy=True)
        out[self.cutoff_h + 1:] = np.nan
        return out


def actual_view(energy: np.ndarray, status: np.ndarray, cutoff_h: int) -> ActualView:
    valid = np.asarray(status) == 1
    valid[cutoff_h + 1:] = False
    return ActualView(cutoff_h, np.where(valid, energy, np.nan), valid)


@dataclass
class QuerySet:
    day: np.ndarray
    kind: np.ndarray
    members: np.ndarray
    mask: np.ndarray
    daylight: np.ndarray
    y_kwh: np.ndarray
    cap: np.ndarray
    n_pv: np.ndarray
    hhi: np.ndarray
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.day)

    @property
    def z(self) -> np.ndarray:
        return self.y_kwh / self.cap[:, None]

    @property
    def target_mask(self) -> np.ndarray:
        return self.mask & self.daylight

    def subset(self, index) -> QuerySet:
        index = np.asarray(index)
        return QuerySet(self.day[index], self.kind[index], self.members[index], self.mask[index], self.daylight[index],
                        self.y_kwh[index], self.cap[index], self.n_pv[index], self.hhi[index], dict(self.meta))

    @property
    def hash(self) -> str:
        return array_hash(self.day, self.kind, self.members, self.mask, self.daylight, np.nan_to_num(self.y_kwh, nan=-1))


def describe(members: np.ndarray, capacity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """집합 용량, PV 수, 용량 집중도(HHI)."""
    cap_matrix = np.asarray(members, dtype=np.float64) * capacity[None, :]
    cap = cap_matrix.sum(axis=1)
    share = cap_matrix / cap[:, None]
    return cap, np.asarray(members).sum(axis=1).astype(np.int16), (share ** 2).sum(axis=1)


def daylight_of(members: np.ndarray, days: np.ndarray, sunup: np.ndarray) -> np.ndarray:
    """구성 PV 중 하나라도 구간 시작·중앙·끝에 태양이 떠 있으면 주간이다."""
    out = np.zeros((len(days), 24), dtype=bool)
    for d in np.unique(days):
        rows = np.flatnonzero(days == d)
        out[rows] = (members[rows][:, None, :] & sunup[d * 24:(d + 1) * 24][None]).any(axis=2)
    return out


def labels_of(members: np.ndarray, days: np.ndarray, view: ActualView) -> tuple[np.ndarray, np.ndarray]:
    """정답 mask [Q, 24]는 구성원 모두의 정답이 있는 시각이고, 정답은 그 시각의 구성원 발전량 합이다."""
    mask = np.zeros((len(days), 24), dtype=bool)
    y = np.full((len(days), 24), np.nan)
    n = len(view.valid)
    for d in np.unique(days):
        if d * 24 >= n:
            continue
        rows = np.flatnonzero(days == d)
        valid = view.valid[d * 24:(d + 1) * 24]
        energy = np.nan_to_num(view.energy[d * 24:(d + 1) * 24])
        m = ~((members[rows][:, None, :] & ~valid[None]).any(axis=2))
        mask[rows] = m
        y[rows] = np.where(m, members[rows].astype(np.float64) @ energy.T, np.nan)
    return mask, y


def heldout_mask(days, block_days: int, cycle_blocks: int, offset: int) -> np.ndarray:
    """대상일을 block_days일 블록으로 나누고 cycle_blocks개마다 offset번째 블록을 held-out으로 둔다."""
    return (np.asarray(days) // block_days) % cycle_blocks == offset


def panel_days(blocks) -> np.ndarray:
    """validation panel의 평가일. blocks는 [시작, 끝] 대상일(KST 날짜, 양끝 포함) 목록이고 묶음끼리 겹치지 않는다."""
    parts = []
    for start, end in blocks:
        first, last = (to_h(f"{d} 01:00:00+09:00") // 24 for d in (start, end))
        if last < first:
            raise ValueError(f"panel 묶음의 끝이 시작보다 앞섭니다: {start}~{end}")
        parts.append(np.arange(first, last + 1))
    days = np.sort(np.concatenate(parts))
    if len(np.unique(days)) != len(days):
        raise ValueError("panel 묶음이 겹칩니다.")
    return days


def buffered_days(days, buffer_days: int) -> np.ndarray:
    """평가일과 앞뒤 buffer_days일. validation panel에서 학습에서 빼는 날짜이다."""
    days = np.asarray(days, dtype=np.int64)
    return np.unique((days[:, None] + np.arange(-int(buffer_days), int(buffer_days) + 1)[None, :]).ravel())


def heldout_block(days, block_days: int) -> np.ndarray:
    """held-out 블록 번호. cross-fitting 단위로 쓴다."""
    return np.asarray(days) // block_days


def cohort_mask(view: ActualView) -> np.ndarray:
    """cutoff까지 유효 발전량이 한 번 이상 있는 발전소."""
    return view.valid[:view.cutoff_h + 1].any(axis=0)


def joined_days(view: ActualView) -> np.ndarray:
    """발전소별 첫 유효 발전량의 대상일. cutoff까지 없으면 NEVER이다."""
    valid = view.valid[:view.cutoff_h + 1]
    first = np.argmax(valid, axis=0)
    return np.where(valid.any(axis=0), first // 24, NEVER).astype(np.int64)


def balanced_split(capacity: np.ndarray, rng: np.random.Generator, share_low: float, share_high: float = 0.5) -> np.ndarray:
    """무작위 순서로 발전소를 쌓아 첫 그룹의 용량 비율이 [share_low, share_high]의 무작위 목표에 닿으면 멈춘다."""
    target = rng.uniform(share_low, share_high) * capacity.sum()
    order = rng.permutation(len(capacity))
    cumulative = np.cumsum(capacity[order])
    k = int(np.searchsorted(cumulative, target)) + 1
    first = np.zeros(len(capacity), dtype=bool)
    first[order[:min(k, len(capacity) - 1)]] = True
    return first


def stratified_split(index: np.ndarray, sites: Sites, rng: np.random.Generator | None = None) -> np.ndarray:
    """같은 (WA station, WB station) 쌍의 발전소를 용량이 비슷하게 두 그룹에 나눈다(vppcontract.partition의 기준 층화 배정).

    rng가 없으면 결정적 기준 배정이고, rng가 있으면 층 순서와 층 안 순서를 섞는다(sampler).
    """
    index = np.asarray(index)
    keys = np.stack([sites.station_index["wa"][index], sites.station_index["wb"][index]], axis=1)
    return split_strata(sites.capacity[index], keys, rng)


def uniform_subset(n: int, rng: np.random.Generator, fraction: tuple[float, float]) -> np.ndarray:
    size = max(1, round(rng.uniform(*fraction) * n))
    out = np.zeros(n, dtype=bool)
    out[rng.choice(n, size=min(size, n), replace=False)] = True
    return out


def station_subset(index: np.ndarray, sites: Sites, rng: np.random.Generator, fraction: tuple[float, float]) -> np.ndarray:
    """WB station을 무작위 순서로 하나씩 더해 대상 수에 닿을 때까지 그 station의 발전소를 모은다(관측소 집중 집합)."""
    station = sites.station_index["wb"][index]
    target = max(1, round(rng.uniform(*fraction) * len(index)))
    out = np.zeros(len(index), dtype=bool)
    for s in rng.permutation(np.unique(station)):
        out |= station == s
        if out.sum() >= target:
            break
    return out


def sample_day_sets(eligible: np.ndarray, sites: Sites, settings: dict, rng: np.random.Generator,
                    share_low: float) -> list[tuple[int, np.ndarray]]:
    """하루의 편입 발전소 eligible에서 설정한 종류·개수의 집합을 뽑는다. 반환은 (종류 번호, 전체 PV membership)이다."""
    n_pv = len(sites)
    out = []

    def add(kind: str, local: np.ndarray) -> None:
        if local.any():
            members = np.zeros(n_pv, dtype=bool)
            members[eligible[local]] = True
            out.append((KINDS.index(kind), members))

    fraction = tuple(settings.get("subset_fraction", (0.05, 0.9)))
    if settings.get("original", False):
        for g in (0, 1):
            add("original", sites.group[eligible] == g)
    if len(eligible) >= 2:
        for _ in range(int(settings.get("random_halves", 0))):
            first = balanced_split(sites.capacity[eligible], rng, share_low)
            add("random_half", first)
            add("random_half", ~first)
        for _ in range(int(settings.get("stratified_halves", 0))):
            first = stratified_split(eligible, sites, rng)
            add("stratified_half", first)
            add("stratified_half", ~first)
    for _ in range(int(settings.get("uniform_subsets", 0))):
        add("uniform_subset", uniform_subset(len(eligible), rng, fraction))
    for _ in range(int(settings.get("station_subsets", 0))):
        add("station_subset", station_subset(eligible, sites, rng, fraction))
    return out


def build_queries(view: ActualView, sites: Sites, sunup: np.ndarray, days: np.ndarray, cohort: np.ndarray, settings: dict,
                  stream: str, share_low: float) -> QuerySet:
    """대상일마다 그날 편입된 cohort 발전소에서 집합을 뽑아 query를 만든다. 대상일별 난수는 (seed, stream, 대상일)로 정한다."""
    joined = joined_days(view)
    seed = int(settings.get("seed", 0))
    stream_id = int.from_bytes(stream.encode()[:8].ljust(8, b"\0"), "little")
    day_list, kind_list, member_list = [], [], []
    for d in np.asarray(days, dtype=np.int64):
        eligible = np.flatnonzero(cohort & (joined <= d))
        if len(eligible) == 0:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, stream_id, int(d)]))
        for kind, members in sample_day_sets(eligible, sites, settings, rng, share_low):
            day_list.append(d)
            kind_list.append(kind)
            member_list.append(members)
    if not day_list:
        raise ValueError("query를 만들 대상일이 없습니다.")
    day = np.array(day_list, dtype=np.int32)
    members = np.stack(member_list)
    mask, y = labels_of(members, day, view)
    cap, n_pv, hhi = describe(members, sites.capacity)
    return QuerySet(day, np.array(kind_list, dtype=np.int8), members, mask, daylight_of(members, day, sunup), y, cap, n_pv, hhi,
                    {"stream": stream, "cutoff_h": view.cutoff_h, "settings": settings})
