"""실행 진입점. torch를 import하기 전에 cuBLAS 결정성 조건(CUBLAS_WORKSPACE_CONFIG)을 고정한다."""
import json
import os
import sys
from pathlib import Path

_runtime = json.loads((Path(__file__).resolve().parents[1] / "configs" / "runtime.json").read_text(encoding="utf-8"))["runtime"]
os.environ["CUBLAS_WORKSPACE_CONFIG"] = _runtime["cublas_workspace"]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from .cli import main  # noqa: E402

main()
