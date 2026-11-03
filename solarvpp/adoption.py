"""명시적인 모델 채택. 계산·검증 보고서와 champion pointer 변경을 분리한다."""
from copy import deepcopy
from pathlib import Path

import pandas as pd

from .config import validate_registry
from .store import digest, read_json, write_json


def adopt_model(spec, name: str, run_dir: Path | None = None) -> dict:
    """등록 모델을 선택하거나, pipeline이 바뀐 run의 설정을 새 MK로 등록해 champion으로 선택한다.

    parameter만 다른 run은 새 MK로 등록하지 않는다. 그 run은 manifest의 model_version·parameters로 기준 MK와 차이를 기록한다.
    """
    registry = deepcopy(spec.registry)
    source = None
    if run_dir is None:
        definition = spec.model(name)
    else:
        manifest = read_json(Path(run_dir) / "manifest.json")
        if manifest.get("execution_status") != "completed" or "model" not in manifest:
            raise ValueError("완료한 run의 model definition이 필요합니다.")
        definition = manifest["model"]
        source = {"run": str(Path(run_dir).resolve()), "run_key": manifest["run_key"],
                  "experts": manifest["experts"], "evaluation_status": manifest.get("evaluation_status")}
        if name in registry["models"] and registry["models"][name] != definition:
            raise ValueError(f"{name}의 정의가 이미 다릅니다. 새 MK 번호를 사용합니다.")
        registry["models"][name] = definition
    record = {"previous_champion": registry["champion"], "champion": name, "model": definition,
              "model_hash": digest(definition), "source": source, "selection": "explicit",
              "selected_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    registry["champion"] = name
    registry["adoption"] = record
    validate_registry(registry)
    write_json(spec.root / spec.config["paths"]["models"], registry)
    spec.registry = registry
    return record
