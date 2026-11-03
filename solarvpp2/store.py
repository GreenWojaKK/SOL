"""solarvpp2의 content key 기반 cache store, JSON·hash 도구, 구성요소 버전, 실행 환경 기록(실제 사용한 패키지 버전과 GPU)."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
import uuid
from importlib.metadata import PackageNotFoundError, version
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from vppcontract.identity import digest, file_sha

PACKAGE_DIR = Path(__file__).resolve().parent




# 구성요소 버전. cache key와 실행 identity에 들어간다. 해당 구성요소의 출력이 바뀌는 변경(계산식·정책·기본값)을 하면 그 번호를 올린다.
# 출력이 같은 리팩터링은 번호를 바꾸지 않으며, 코드 출처는 key 밖의 source_sha로 manifest에 기록한다.
VERSIONS = {
    "base": 1,          # data: 발전량·status·SMP·k·태양 위치 base cache
    "weather": 1,       # data: 예보·observed weather cache와 schema 변환
    "sampler": 1,       # sampler: 편입 규칙, held-out 블록, 학습·risk query와 정답 mask
    "features": 1,      # features: PV 단위 feature, 집합 집계, XGB 행렬, power curve, 대체·표준화·ZCA
    "xgb": 1,           # forecast.xgb: XGB 집합 expert 학습·예측
    "dl_prep": 1,       # forecast.dl: DL 입력 tensor 구성·전처리 fit
    "dl": 2,            # forecast.dl: DL 집합 expert 구조·손실(제출 중앙값에 거는 손실, 고정 normalizer)·학습
    "postprocess": 1,   # forecast.forecaster: seed 평균·clip·야간·결합·중앙값
    "risk": 1,          # risk: R 입력·학습 목적·cross-fitting
    "partition": 1,     # partition: 목적 일자·가중치·SMP 추정·탐색
    "instance": 3,      # instance: 실행 단계·배정·예측 산출물·forecast 단계의 seed별 채점. 재평가 version은 evaluation.EVALUATION_VERSION을 쓴다.
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


def transfer_payload(parent, partition_sha, queries, settings, version):
    """고정 배정 적응의 parent·학습/보정 query·고정 설정 identity."""
    return {"parent": parent, "partition_sha": partition_sha, "queries": queries,
            "settings": settings, "transfer_version": version}


def recalibration_payload(transfer_key, source_hashes, price_sha, settings, version):
    """저장된 transfer·R0 예측과 정산 weight로 재현하는 보정 identity."""
    return {"transfer_key": transfer_key, "source_hashes": source_hashes, "price_sha": price_sha,
            "settings": settings, "calibration_version": version}


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
TEMPORARY_KINDS = ("models", "features", "prep", "risk")


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


@contextmanager
def temporary_cache(artifact_dir: Path, enabled: bool = True) -> Iterator[None]:
    """enabled이면 이 구간의 새 학습 cache를 임시 폴더에 쓰고, 구간이 끝나면 지운다."""
    if enabled:
        open_temporary_cache(artifact_dir)
    try:
        yield
    finally:
        close_temporary_cache()


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
            for attempt in range(20):
                try:
                    os.replace(temporary, final)
                    break
                except PermissionError:   # Windows에서 방금 쓴 파일을 다른 프로세스가 잠시 잡고 있으면 이름 바꾸기가 거부된다
                    if attempt == 19:
                        raise
                    time.sleep(0.25)
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
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=20, check=False)
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
    except Exception as error:  # noqa: BLE001  기록용 정보이므로 실패해도 실행은 계속한다.
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


def forecaster_payload(model_name: str, model: dict, cutoff_h: int, xgb_keys: list, dl_keys: list,
                       postprocess: dict, smoke: bool) -> dict:
    keys = ("sampler", "heldout", "xgb", "dl", "blend", "seeds") + (("season",) if model.get("season") else ())
    return {"model": model_name, "definition": {k: model[k] for k in keys},
            "cutoff_h": cutoff_h, "xgb": xgb_keys, "dl": dl_keys, "postprocess": postprocess, "smoke": smoke}


def forecast_payload(instance: str, forecaster: dict, eval_days: list, evaluation: dict, versions: dict) -> dict:
    return {"instance": instance, "stage": "forecast", "forecaster": forecaster,
            "eval_days": eval_days, "evaluation": evaluation, "versions": versions}


def panel_payload(instance: str, forecaster: dict, panel: dict, evaluation: dict, versions: dict) -> dict:
    return {"instance": instance, "stage": "panel", "forecaster": forecaster, "panel": panel, "evaluation": evaluation, "versions": versions}


def full_payload(instance: str, partition: str, eval_days: list, evaluation: dict, versions: dict) -> dict:
    return {"instance": instance, "partition": partition, "eval_days": eval_days,
            "evaluation": evaluation, "versions": versions}


def evaluation_payload(evaluation: dict, *, calendar=None, candidate_pool: int | None = None) -> dict:
    """평가에 쓰는 calendar 규칙과 실제 후보 수를 기록한다. 설명·출처 문구는 포함하지 않는다."""
    payload = dict(evaluation)
    if calendar is not None:
        payload["calendar"] = calendar.rules
    if candidate_pool is not None:
        payload["candidate_pool_size"] = candidate_pool
    return payload


def prediction_payload(calculation: dict) -> dict:
    """평가 설정을 제외한 배정·예측 생성의 dependency."""
    payload = {k: v for k, v in calculation.items() if k != "evaluation"}
    payload.setdefault("stage", "full")
    return payload

