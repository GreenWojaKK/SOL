"""집합 pooling 뒤의 시퀀스 decoder 구성요소. pooled 시퀀스 [B, 24, n_in]를 시각별 집합 표현 [B, 24, out_width]로 바꾼다.

decoder마다 모듈을 하나 두고, 각 모듈은 다음을 정의한다.
- KIND: 모델 정의의 dl.decoder.kind 이름
- ATTRIBUTE: GenerationNet에서 decoder module의 속성 이름(checkpoint의 parameter 이름 앞부분)
- VERSION: 구현 version(BiGRU가 아닌 decoder의 DL model key에 들어간다)
- DEFAULTS, resolve(설정 검사·기본값), build(module 생성, module.out_width는 출력 폭)

모델 정의에 decoder가 없으면 BiGRU이다. BiGRU는 runtime hyperparameters의 dl_gru_*를 쓰고 DL model key에 들어가지 않으므로 MK3·MK5의 key와
checkpoint가 그대로이다. 다른 decoder는 resolve한 설정과 VERSION이 DL model key에 들어간다.
"""
from __future__ import annotations

from . import bigru, looped

MODULES = {m.KIND: m for m in (bigru, looped)}


def resolve(spec: dict | None) -> dict | None:
    """모델 정의의 dl.decoder를 검사해 기본값을 채운 사본으로 돌려준다. 없거나 BiGRU이면 None(기존 BiGRU 경로)이다."""
    if spec is None:
        return None
    kind = spec.get("kind")
    if kind not in MODULES:
        raise ValueError(f"decoder kind는 {sorted(MODULES)} 중 하나입니다: {kind}")
    out = MODULES[kind].resolve(spec)
    return None if kind == bigru.KIND else out


def identity(spec: dict | None) -> dict | None:
    """DL model key에 넣을 decoder 설정과 구현 version. BiGRU(None)이면 None이다."""
    return None if spec is None else {**spec, "version": MODULES[spec["kind"]].VERSION}


def build(spec: dict | None, n_in: int, settings: dict):
    """(GenerationNet 속성 이름, decoder module). spec은 resolve의 결과이다."""
    module = bigru if spec is None else MODULES[spec["kind"]]
    return module.ATTRIBUTE, module.build(spec, n_in, settings)
