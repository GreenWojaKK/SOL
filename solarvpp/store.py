"""content key 기반 cache store, JSON·hash 도구, 구성요소 버전, 실행 환경 기록(실제 사용한 패키지 버전과 GPU)."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from vppcontract.identity import ArtifactAdapter, ArtifactIndex, digest, file_sha

PACKAGE_DIR = Path(__file__).resolve().parent




# 구성요소 버전. cache key와 run identity에 들어간다. 해당 구성요소의 출력이 바뀌는 변경(계산식·정책·기본값)을 하면 그 번호를 올린다.
# 출력이 같은 리팩터링은 번호를 바꾸지 않으며, 코드 출처는 key 밖의 source_sha로 manifest에 기록한다.
VERSIONS = {
    "base": 1,              # data: 발전량·status·SMP·k·태양 위치 base cache
    "weather": 1,           # data: 예보·observed weather cache와 schema 변환
    "queries": 1,           # forecast.queries: 학습 query·cohort·evaluation panel 정책
    "features": 1,          # forecast.features: XGB 행렬, physics proxy·시간 표현, 대체·표준화·ZCA, power curve(XGB·DL·공동 오차 모델 입력)
    "xgb": 1,               # forecast.models: XGB 학습·예측
    "dl_prep": 1,           # forecast.train: DL 입력 tensor 구성·전처리 fit
    "dl": 1,                # forecast.models·forecast.train: DL 구조·손실·학습
    "postprocess": 1,       # forecast.models: 분위수 clip·야간·정렬·블렌드·중앙값 결정
    "scoring": 1,           # scoring: 정산 S·W 계산
    "partition": 1,         # bundling.partition: instance 구성·배정 표현
    "residuals": 1,         # bundling.residuals: 발전소 단위 primitive와 잔차
    "error_model": 1,       # bundling.error_model: 공동 오차 모델 학습·시나리오
    "optimize": 1,          # bundling.optimize: 시나리오 목적·QUBO·탐색
    "pipeline": 2,          # scoped 실행·독립 transfer·명시적 recommendation·snapshot audit
    "test_input_check": 1,  # submission: test 입력 분포 점검
    "curtailment": 3,       # forecast.curtailment: 출력제어 표시·전역 확률·발전소별 손실·OOT 잔차의 추가 보정(module을 쓰는 run에만 들어간다)
}


def versions(*names: str) -> dict:
    """구성요소 버전 묶음. 이름을 주지 않으면 전체이다."""
    return dict(VERSIONS) if not names else {name: VERSIONS[name] for name in names}


def source_sha() -> str:
    """패키지 전체 소스의 hash. 출처 기록용이며 cache key에는 넣지 않는다."""
    return digest({p.relative_to(PACKAGE_DIR).as_posix(): file_sha(p) for p in sorted(PACKAGE_DIR.rglob("*.py"))})


def array_hash(*arrays) -> str:
    h = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        h.update(str(array.dtype).encode() + str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def _plain(value):
    """numpy 값을 JSON 기본형으로 바꾸고, 유한하지 않은 실수는 null로 둔다."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_json(path: Path, value) -> None:
    Path(path).write_text(json.dumps(_plain(value), ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# 설정을 바꾼 실행이 기본으로 임시 폴더에 쓰는 학습 cache 종류. 입력 원자료 cache(base, weather)는 항상 본래 위치에 쓴다.
TEMPORARY_KINDS = ("models", "features", "prep")


def experiment_cache_enabled() -> bool:
    """이 PC에서 설정을 바꾼 실행의 학습 cache도 저장하는지(환경 변수 VPP_EXPERIMENT_CACHE=1)."""
    return os.environ.get("VPP_EXPERIMENT_CACHE") == "1"


def open_temporary_cache(artifact_dir: Path) -> None:
    """이후 새 학습 cache를 실행 동안만 쓰는 임시 폴더에 쓴다. 이미 있는 cache는 그대로 읽는다."""
    Store.temporary = Path(artifact_dir) / "cache" / "temporary" / uuid.uuid4().hex[:12]


def close_temporary_cache() -> None:
    """임시 폴더를 지우고 본래 cache 위치로 돌아간다."""
    folder, Store.temporary = Store.temporary, None
    if folder is None:
        return
    shutil.rmtree(folder, ignore_errors=True)
    if folder.exists():
        print(f"임시 cache를 지우지 못했습니다. 실행이 끝난 뒤 지워도 됩니다: {folder}", flush=True)
    elif folder.parent.exists() and not any(folder.parent.iterdir()):
        folder.parent.rmdir()


class Store:
    """내용 기반 key로 이름을 정하는 불변 cache 디렉터리. temporary가 있으면 TEMPORARY_KINDS의 새 항목을 그 폴더에 쓰고,
    이미 있는 항목은 본래 위치에서 읽는다."""

    temporary: Path | None = None

    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, kind: str, key: str) -> Path:
        final = self.root / kind / key[:16]
        if Store.temporary is None or kind not in TEMPORARY_KINDS or (final / "meta.json").exists():
            return final
        return Store.temporary / kind / key[:16]

    def exists(self, kind: str, key: str) -> bool:
        return (self.path(kind, key) / "meta.json").exists()

    def write(self, kind: str, key: str, arrays: dict | None = None, meta: dict | None = None, files: dict | None = None) -> Path:
        final = self.path(kind, key)
        if (final / "meta.json").exists():
            return final
        temporary = final.parent / f"{final.name}.tmp-{uuid.uuid4().hex[:8]}"
        temporary.mkdir(parents=True)
        try:
            for name, array in (arrays or {}).items():
                np.save(temporary / f"{name}.npy", np.asarray(array), allow_pickle=False)
            for name, writer in (files or {}).items():
                writer(temporary / name)
            write_json(temporary / "meta.json", {"key": key, **(meta or {})})
            if final.exists():
                shutil.rmtree(final)
            os.replace(temporary, final)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return final

    def load(self, kind: str, key: str, names=None, mmap=True) -> tuple[dict, dict]:
        directory = self.path(kind, key)
        meta = read_json(directory / "meta.json")
        if meta.get("key") != key:
            raise ValueError(f"cache key가 일치하지 않습니다: {directory}")
        names = names if names is not None else [p.stem for p in directory.glob("*.npy")]
        arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r" if mmap else None, allow_pickle=False) for name in names}
        return arrays, meta


PACKAGES = ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "xgboost", "torch")
ENV_FLAGS = ("MKL_THREADING_LAYER", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED")


def package_versions(names=PACKAGES) -> dict:
    out = {"python": platform.python_version()}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _driver_version() -> str | None:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return result.stdout.strip().splitlines()[0] if result.returncode == 0 and result.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def environment(include_torch: bool = True) -> dict:
    info = {"packages": package_versions(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "env": {name: os.environ.get(name) for name in ENV_FLAGS}}
    try:
        import xgboost
        build = xgboost.build_info()
        info["xgboost_build"] = {k: build.get(k) for k in ("USE_CUDA", "USE_OPENMP", "GCC_VERSION", "MSVC_VERSION") if k in build}
    except Exception as error:  # 기록용 정보이므로 실패해도 실행은 계속한다.
        info["xgboost_build"] = {"error": str(error)}
    if include_torch:
        import torch
        info["torch"] = {"cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available(),
                         "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                         "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
                         "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                         "num_threads": torch.get_num_threads()}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = {"name": props.name, "memory_mib": props.total_memory // 1024 ** 2, "capability": f"{props.major}.{props.minor}",
                           "driver": _driver_version()}
    return info


IDENTITY_VERSION = 1


def calculation_identity(record: dict) -> dict:
    """실제 expert cache, cutoff·배정·평가 panel과 후처리 조건으로 run을 식별한다."""
    seeds = [int(s) for s in record["seeds"]]
    experts = {}
    for expert in ("xgb", "dl"):
        if expert not in record["experts"]:
            continue
        details = record["experts_detail"][expert]["details"]
        experts[expert] = [{"seed": s, "model_key": details[str(s)]["model_key"]} for s in seeds]
    if not experts or any(not x["model_key"] for values in experts.values() for x in values):
        raise ValueError("실행 expert의 model cache key가 필요합니다.")
    runtime = {}
    if "dl" in experts:
        runtime = record.get("effective_runtime")
        if runtime is None:
            env = record["environment"]
            runtime = {"torch_threads": int(env["torch"]["num_threads"]),
                       "cublas_workspace": env["env"]["CUBLAS_WORKSPACE_CONFIG"]}
    identity = {
        "identity_version": IDENTITY_VERSION,
        "experts": experts,
        "runtime": runtime,
        "blend": {"method": "weighted-quantile", "xgb_weight": float(record["xgb_weight"])} if len(experts) == 2 else None,
        "cutoff_h": int(record["lineage"]["outer_cutoff_h"]),
        "partition": record["original_hash"],
        "panel": record["panel_hash"],
        "period": record["period"],
        "execution_check": record["level"] == "smoke",
        "diagnostics": record["level"] == "diagnostic",
        "postprocess_version": int(record["versions"]["postprocess"]),
        "scoring_version": int(record["versions"]["scoring"]),
    }
    if record.get("curtailment"):                           # module을 쓰지 않는 run의 key는 그대로 둔다
        identity["curtailment"] = record["curtailment"]
    if record.get("season"):                                # 계절 학습 run만 넣는다(Global run의 key는 그대로다)
        identity["season"] = record["season"]
    return identity


def identity_key(record: dict) -> str:
    return digest(calculation_identity(record))


def completed(record: dict, directory: Path) -> bool:
    if record.get("execution_status") != "completed":
        return False
    outputs = ["quantiles.npz", "report.json",
               "forecast.csv" if record["period"] == "test" else "validation-forecast.csv"]
    if record["period"] != "test":
        outputs.append("daily_penalty.npz")
    return all((directory / name).is_file() for name in outputs)


ADAPTER = ArtifactAdapter(
    record_name="manifest.json",
    record_glob="runs/*/*/manifest.json",
    calculation=lambda record, directory: calculation_identity(record),
    completed=completed,
)


def find_run(root: Path, key: str) -> Path | None:
    return ArtifactIndex(root, ADAPTER).find(key)


def index_runs(root: Path, directories=None, *, write: bool = False) -> dict:
    return ArtifactIndex(root, ADAPTER).index(directories, write=write)
