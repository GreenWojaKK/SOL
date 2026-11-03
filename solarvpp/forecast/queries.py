"""portfolio query, 평가 cohort, evaluation panel.

학습 fit은 실적을 `ActualView`로만 받는다. cutoff 이후 generation·observed를 먼저 지운 뒤 query와 24시간 시퀀스를 만든다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Period
from ..data import GROUPS, N_ACTUAL, Sites, delivery_month, k_weight
from ..store import array_hash, digest

QUERY_POLICY = "partition each original group/day by identical 24-bit valid-generation mask; skip all-zero masks"
COHORT_POLICY = "valid generation at or before fit cutoff; membership frozen for the period"


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
    valid = status == 1
    valid[cutoff_h + 1:] = False
    masked = np.where(valid, energy, np.nan)
    return ActualView(cutoff_h, masked, valid)


@dataclass
class QuerySet:
    day: np.ndarray
    group: np.ndarray
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

    def weights(self, capacity: np.ndarray) -> np.ndarray:
        """구성 PV의 용량 비중 [Q, PV]. 합은 1이다."""
        w = self.members * capacity[None, :]
        return w / w.sum(axis=1, keepdims=True)

    def subset(self, index) -> "QuerySet":
        index = np.asarray(index)
        return QuerySet(self.day[index], self.group[index], self.members[index], self.mask[index], self.daylight[index],
                        self.y_kwh[index], self.cap[index], self.n_pv[index], self.hhi[index], dict(self.meta))

    @property
    def hash(self) -> str:
        return array_hash(self.day, self.group, self.members, self.mask, self.daylight, np.nan_to_num(self.y_kwh, nan=-1))


def _describe(members: np.ndarray, sites: Sites) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cap_matrix = members * sites.capacity[None, :]
    cap = cap_matrix.sum(axis=1)
    share = cap_matrix / cap[:, None]
    return cap, members.sum(axis=1).astype(np.int16), (share ** 2).sum(axis=1)


def _daylight(members: np.ndarray, days: np.ndarray, sunup: np.ndarray) -> np.ndarray:
    """query 구성 PV 중 하나라도 구간 시작·중앙·끝에 태양이 떠 있으면 주간이다."""
    out = np.zeros((len(days), 24), dtype=bool)
    for i, (d, m) in enumerate(zip(days, members)):
        out[i] = sunup[d * 24:(d + 1) * 24][:, m].any(axis=1)
    return out


def training_queries(view: ActualView, sites: Sites, sunup: np.ndarray, first_day: int = 0) -> QuerySet:
    last_day = view.cutoff_h // 24
    bits = (1 << np.arange(24, dtype=np.int64))
    rows = []
    for d in range(first_day, last_day + 1):
        valid = view.valid[d * 24:(d + 1) * 24]
        if not valid.any():
            continue
        masks = bits @ valid.astype(np.int64)
        for g in range(len(GROUPS)):
            in_group = (sites.group == g) & (masks > 0)
            for value in np.unique(masks[in_group]):
                members = in_group & (masks == value)
                rows.append((d, g, members, ((value >> np.arange(24)) & 1).astype(bool)))
    if not rows:
        raise ValueError("cutoff까지 학습 query가 없습니다.")
    day = np.array([r[0] for r in rows], dtype=np.int32)
    group = np.array([r[1] for r in rows], dtype=np.int8)
    members = np.stack([r[2] for r in rows])
    mask = np.stack([r[3] for r in rows])
    y = np.full((len(rows), 24), np.nan)
    for i, (d, m) in enumerate(zip(day, members)):
        y[i] = np.where(mask[i], view.energy[d * 24:(d + 1) * 24][:, m].sum(axis=1), np.nan)
    cap, n_pv, hhi = _describe(members, sites)
    return QuerySet(day, group, members, mask, _daylight(members, day, sunup), y, cap, n_pv, hhi,
                    {"policy": QUERY_POLICY, "cutoff_h": view.cutoff_h})


@dataclass
class Cohort:
    period: str
    cutoff_h: int
    members: np.ndarray

    def site_ids(self, sites: Sites, group: int) -> list[str]:
        return [str(s) for s in sites.ids[self.members[group]]]

    @property
    def hash(self) -> str:
        return array_hash(self.members)


def freeze_cohort(view: ActualView, sites: Sites, period: Period) -> Cohort:
    if period.name == "test":
        members = np.stack([sites.group == g for g in range(len(GROUPS))])
    else:
        ever = view.valid[:view.cutoff_h + 1].any(axis=0)
        members = np.stack([(sites.group == g) & ever for g in range(len(GROUPS))])
    return Cohort(period.name, view.cutoff_h, members)


def period_queries(cohort: Cohort, sites: Sites, days: np.ndarray, sunup: np.ndarray) -> QuerySet:
    """기간의 매 대상일·그룹에 cohort 전체를 query로 둔다. 정답은 넣지 않는다."""
    day = np.repeat(days, len(GROUPS)).astype(np.int32)
    group = np.tile(np.arange(len(GROUPS), dtype=np.int8), len(days))
    members = cohort.members[group]
    cap, n_pv, hhi = _describe(members, sites)
    empty = np.zeros((len(day), 24), dtype=bool)
    return QuerySet(day, group, members, empty, _daylight(members, day, sunup), np.full((len(day), 24), np.nan), cap, n_pv, hhi,
                    {"cohort": cohort.hash, "period": cohort.period})


@dataclass
class EvalPanel:
    period: str
    hours: np.ndarray
    cohort: Cohort
    cap: np.ndarray
    n_pv: np.ndarray
    y: np.ndarray
    m: np.ndarray
    daylight: np.ndarray
    smp: np.ndarray
    k: np.ndarray
    month: np.ndarray
    smp_bin: np.ndarray
    has_labels: bool
    panel_hash: str = ""

    @property
    def cohort_id(self) -> str:
        return f"{self.period}-{self.cohort.hash[:8]}"


def eval_panel(period: Period, cohort: Cohort, sites: Sites, energy: np.ndarray, status: np.ndarray, smp: np.ndarray,
               sunup: np.ndarray, queries: QuerySet, days: np.ndarray | None = None) -> EvalPanel:
    """days를 주면 그 대상일만 담은 축소 panel(실행 확인용)을 만든다."""
    if days is None:
        hours = period.hours
        if len(hours) != period.expected_hours:
            raise ValueError(f"{period.name}의 시간 수가 protocol과 다릅니다: {len(hours)}")
    else:
        hours = (np.asarray(days)[:, None] * 24 + np.arange(24)[None, :]).ravel()
    n = len(GROUPS)
    cap = np.array([sites.capacity[cohort.members[g]].sum() for g in range(n)])
    n_pv = cohort.members.sum(axis=1)
    y = np.full((len(hours), n), np.nan)
    m = np.zeros((len(hours), n), dtype=bool)
    if period.has_labels:
        if hours[-1] >= N_ACTUAL:
            raise ValueError("정답 구간이 generation 범위를 벗어납니다.")
        for g in range(n):
            valid = status[hours][:, cohort.members[g]] == 1
            m[:, g] = valid.all(axis=1)
            y[:, g] = np.where(m[:, g], energy[hours][:, cohort.members[g]].sum(axis=1), np.nan)
    daylight = np.zeros((len(hours), n), dtype=bool)
    start_day = hours[0] // 24
    for qi in range(len(queries)):
        rows = slice((queries.day[qi] - start_day) * 24, (queries.day[qi] - start_day + 1) * 24)
        daylight[rows, queries.group[qi]] = queries.daylight[qi]
    price = smp[hours]
    edges = np.quantile(price, [0.2, 0.4, 0.6, 0.8])
    panel = EvalPanel(period.name, hours, cohort, cap, n_pv, y, m, daylight, price, k_weight(hours), delivery_month(hours),
                      np.searchsorted(edges, price, side="right").astype(np.int8), period.has_labels)
    panel.panel_hash = array_hash(hours, cohort.members, cap, np.nan_to_num(y, nan=-1), m, daylight, price, panel.k)
    return panel


def check_evidence(panel: EvalPanel, sites: Sites, evidence: dict) -> dict:
    """cohort evidence와 PV 목록·용량·완전 시간·대상일 월별 시간이 모두 같아야 한다."""
    item = evidence["periods"][panel.period]
    if int(item["hours"]) != len(panel.hours):
        raise ValueError(f"{panel.period} 시간 수가 evidence와 다릅니다.")
    for g, name in enumerate(GROUPS):
        ref = item["groups"][name]
        ids = panel.cohort.site_ids(sites, g)
        checks = {
            "site_ids": ids == ref["site_ids"],
            "sites": len(ids) == ref["sites"],
            "capacity_kw": abs(panel.cap[g] - ref["capacity_kw"]) < 1e-6,
            "complete_hours": int(panel.m[:, g].sum()) == ref["complete_hours"],
            "missing_label_hours": int((~panel.m[:, g]).sum()) == ref["missing_label_hours"],
            "monthly_complete_hours": pd.Series(panel.m[:, g]).groupby(panel.month).sum().astype(int).to_dict() == ref["monthly_complete_hours"],
        }
        failed = [k for k, ok in checks.items() if not ok]
        if failed:
            raise ValueError(f"{panel.period} {name} cohort가 evidence와 다릅니다: {failed}")
    return {"period": panel.period, "status": "matched", "evidence_hash": digest(item)}
