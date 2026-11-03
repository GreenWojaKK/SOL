"""Temporal routing 구성요소: 시각 t의 표현이 같은 하루의 다른 시각 표현을 읽어 원래 값에 더한다.

자리는 둘이다. dl.plant_routing은 집합 pooling 전 PV별 provider 표현에 두고, 집합과 무관한 PV 표현 계산 안에 있다.
dl.group_routing은 집합 pooling 뒤 sequence에 두고 decode 안에서 계산한다. 종류마다 모듈을 하나 두고, 각 모듈은 KIND(모델 정의의 kind),
VERSION(구현 version), DEFAULTS, resolve(설정 검사·기본값), build(module 생성)를 정의한다. 모델 정의에 항목이 없으면 그 자리의 routing은
없다(기존 경로). 설정과 VERSION은 DL model key에 들어간다. 출력 사영을 0으로 초기화하므로 학습 시작 시 출력은 routing이 없는 모델과 같다.
"""
from __future__ import annotations

from . import group_nearby_hours, nearby_hours, time_shift

PLANT_MODULES = {m.KIND: m for m in (time_shift, nearby_hours)}
GROUP_MODULES = {m.KIND: m for m in (group_nearby_hours,)}


def _resolve(spec: dict | None, modules: dict, label: str) -> dict | None:
    if spec is None:
        return None
    kind = spec.get("kind")
    if kind not in modules:
        raise ValueError(f"{label} kind는 {sorted(modules)} 중 하나입니다: {kind}")
    return modules[kind].resolve(spec)


def resolve_plant(spec: dict | None) -> dict | None:
    """모델 정의의 dl.plant_routing을 검사해 기본값을 채운 사본으로 돌려준다. 없으면 None이다."""
    return _resolve(spec, PLANT_MODULES, "plant routing")


def resolve_group(spec: dict | None) -> dict | None:
    """모델 정의의 dl.group_routing을 검사해 기본값을 채운 사본으로 돌려준다. 없으면 None이다."""
    return _resolve(spec, GROUP_MODULES, "group routing")


def plant_identity(spec: dict) -> dict:
    """DL model key에 넣을 plant routing 설정과 구현 version."""
    return {**spec, "version": PLANT_MODULES[spec["kind"]].VERSION}


def group_identity(spec: dict) -> dict:
    """DL model key에 넣을 group routing 설정과 구현 version."""
    return {**spec, "version": GROUP_MODULES[spec["kind"]].VERSION}


def build_plant(spec: dict, providers: tuple, width: int, n_forecast: dict, n_common: int, static: tuple, keep: tuple, hidden: int,
                station_to_pv: dict):
    """spec은 resolve_plant의 결과, width는 provider 표현 폭, n_forecast는 제공사별 예보 입력 채널 수, n_common은 fusion 공통 입력 채널 수,
    static은 공통 입력 안의 정적 정보·provider 사용 여부 구간, keep은 window 날씨 맥락으로 쓰는 공통 입력 구간, hidden은 날씨 맥락 폭이다."""
    return PLANT_MODULES[spec["kind"]].build(spec, providers, width, n_forecast, n_common, static, keep, hidden, station_to_pv)


def build_group(spec: dict, n_group: int, attn: int, hidden: int, blank: tuple):
    """spec은 resolve_group의 결과, n_group은 pooled sequence 폭, attn은 attention 폭, blank는 neutral 표현에서 0으로 둘 정적 정보 구간이다."""
    return GROUP_MODULES[spec["kind"]].build(spec, n_group, attn, hidden, blank)
