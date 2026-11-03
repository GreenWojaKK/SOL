"""package를 넘는 champion 기록. package와 그 package의 모델 버전을 함께 가리킨다."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .outputs import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs" / "champion.json"
PACKAGES = ("solarvpp", "solarvpp2")


def read_champion(path: Path = PATH) -> dict:
    record = read_json(path)
    if record.get("package") not in PACKAGES or not str(record.get("model", "")).startswith("MK"):
        raise ValueError(f"champion 기록은 package({', '.join(PACKAGES)})와 MK 버전을 담아야 합니다: {path}")
    return record


def set_champion(package: str, model: str, evidence: str, path: Path = PATH) -> dict:
    """champion을 바꾼다. 근거(비교 보고서 경로 등)를 함께 기록한다."""
    if package not in PACKAGES or not model.startswith("MK"):
        raise ValueError(f"champion은 {PACKAGES}의 package와 MK 버전입니다: {package}:{model}")
    record = {"package": package, "model": model, "evidence": evidence, "updated": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    write_json(path, record)
    return record
