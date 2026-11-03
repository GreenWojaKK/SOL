"""Provider 표현 context 구성요소: 시간별 provider 표현 e_p [U, S_p, 24, R]에 관측소 사이 context를 더한다.

종류마다 모듈을 하나 두고, 각 모듈은 KIND(모델 정의의 dl.provider_context.kind), VERSION(구현 version), DEFAULTS, resolve(설정 검사·기본값),
build(module 생성)를 정의한다. 모델 정의에 dl.provider_context가 없으면 context를 더하지 않는다(기존 경로). 설정과 VERSION은 DL model key에
들어간다.
"""
from __future__ import annotations

from . import inverted_attention

MODULES = {m.KIND: m for m in (inverted_attention,)}


def resolve(spec: dict | None) -> dict | None:
    """모델 정의의 dl.provider_context를 검사해 기본값을 채운 사본으로 돌려준다. 없으면 None(context 없음)이다."""
    if spec is None:
        return None
    kind = spec.get("kind")
    if kind not in MODULES:
        raise ValueError(f"provider context kind는 {sorted(MODULES)} 중 하나입니다: {kind}")
    return MODULES[kind].resolve(spec)


def identity(spec: dict | None) -> dict | None:
    """DL model key에 넣을 provider context 설정과 구현 version. 없으면 None이다."""
    return None if spec is None else {**spec, "version": MODULES[spec["kind"]].VERSION}


def build(spec: dict, providers: tuple, n_stations: dict, rep_width: int, active: dict):
    """spec은 resolve의 결과, n_stations는 제공사별 관측소 수, rep_width는 provider 표현의 시각별 폭, active는 제공사별 사용 여부이다."""
    return MODULES[spec["kind"]].build(spec, providers, n_stations, rep_width, active)
