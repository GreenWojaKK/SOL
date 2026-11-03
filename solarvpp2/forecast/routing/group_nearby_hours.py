"""집합 pooling 뒤 sequence의 residual sliding window attention.

읽어 오는 값은 언제나 원래 pooled sequence이다. context가 neutral이면 맥락·key를 모든 관측소에 같은 평균 station embedding을 넣고 정적 정보를
0으로 둔 PV 표현의 pooling으로 만들어, 읽을 시각을 정할 때 관측소·발전소 identity를 보지 않는다. pooled이면 pooled sequence를 그대로 쓴다.
neutral 표현은 PV 표현과 함께 pooling해야 하므로, GenerationNet.fuse가 PV 표현 뒤에 이어 붙이고 decode가 나눠 읽는다.
"""
from __future__ import annotations

from .nearby_hours import NearbyHoursRouting

KIND = "nearby-hours"
VERSION = 1
DEFAULTS = {"hours": 3, "gate": False, "context": "neutral"}
CONTEXTS = ("neutral", "pooled")


def resolve(spec: dict) -> dict:
    if set(spec) - {"kind", *DEFAULTS}:
        raise ValueError(f"{KIND} group routing에 정의되지 않은 항목이 있습니다: {sorted(set(spec) - {'kind', *DEFAULTS})}")
    out = {"kind": KIND, **{k: spec.get(k, v) for k, v in DEFAULTS.items()}}
    if isinstance(out["hours"], bool) or float(out["hours"]) != int(out["hours"]) or int(out["hours"]) < 1:
        raise ValueError(f"{KIND} group routing의 hours는 1 이상의 정수입니다: {out['hours']}")
    out["hours"] = int(out["hours"])
    if not isinstance(out["gate"], bool):
        raise ValueError(f"{KIND} group routing의 gate는 true 또는 false입니다.")
    if out["context"] not in CONTEXTS:
        raise ValueError(f"{KIND} group routing의 context는 {CONTEXTS} 중 하나입니다: {out['context']}")
    return out


def build(spec: dict, n_group: int, attn: int, hidden: int, blank: tuple) -> "GroupNearbyHours":
    return GroupNearbyHours(n_group, attn, hidden, spec["hours"], spec["gate"], spec["context"], blank)


class GroupNearbyHours(NearbyHoursRouting):
    def __init__(self, n_group: int, attn: int, hidden: int, window: int, gate: bool, context: str, blank: tuple):
        super().__init__(n_group, attn, n_group, hidden, n_group, window, gate)
        self.neutral, self.blank = context == "neutral", tuple(blank)
