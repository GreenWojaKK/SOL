"""Package별 계산 payload를 hash하고 기존 산출물 디렉터리에 연결한다."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def digest(value) -> str:
    """기존 package cache와 같은 JSON 직렬화로 SHA-256을 계산한다."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ArtifactAdapter:
    """Package가 제공하는 기록 위치, 계산 payload, 완료 조건."""
    record_name: str
    record_glob: str
    calculation: Callable[[dict, Path], dict | None]
    completed: Callable[[dict, Path], bool]
    stored_key: str = "run_key"
    index_name: str = "run-index"


class ArtifactIndex:
    """Hash·기록 무결성·index I/O를 담당한다. 계산 정의는 adapter가 제공한다."""

    def __init__(self, root: Path, adapter: ArtifactAdapter):
        self.root = Path(os.path.abspath(root)).resolve()
        self.adapter = adapter

    def _record(self, directory: Path) -> dict | None:
        path = directory / self.adapter.record_name
        if not path.is_file():
            return None
        record = _read(path)
        return record if self.adapter.completed(record, directory) else None

    def _identity(self, record: dict, directory: Path) -> tuple[str, dict | None]:
        payload = self.adapter.calculation(record, directory)
        # Payload를 저장하지 않은 이전 기록은 그 기록에 남은 계산 key를 보존한다.
        key = digest(payload) if payload is not None else record[self.adapter.stored_key]
        if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("산출물 계산 key는 SHA-256이어야 합니다.")
        return key, payload

    def find(self, key: str) -> Path | None:
        index_dir = (self.root / self.adapter.index_name).resolve()
        path = (index_dir / f"{key}.json").resolve()
        if not path.is_relative_to(index_dir) or not path.is_file():
            return None
        for reference in _read(path)["runs"]:
            directory = (self.root / reference["directory"]).resolve()
            if not directory.is_relative_to(self.root):
                continue
            record = self._record(directory)
            if record is None or file_sha(directory / self.adapter.record_name) != reference["manifest_sha"]:
                continue
            if self._identity(record, directory)[0] == key:
                return directory
        return None

    def index(self, directories=None, *, write: bool = False) -> dict:
        """기존 기록은 읽고, write=True일 때 index 파일만 생성·갱신한다."""
        paths = sorted(self.root.glob(self.adapter.record_glob)) if directories is None else [
            Path(p) / self.adapter.record_name for p in directories]
        entries, skipped = {}, []
        for path in paths:
            directory = path.parent.resolve()
            if not directory.is_relative_to(self.root):
                raise ValueError(f"산출물 root 밖의 기록입니다: {directory}")
            try:
                record = self._record(directory)
                if record is None:
                    skipped.append({"directory": str(directory), "reason": "완료된 산출물이 없음"})
                    continue
                key, payload = self._identity(record, directory)
            except (KeyError, TypeError, ValueError) as exc:
                skipped.append({"directory": str(directory), "reason": str(exc)})
                continue
            entry = entries.setdefault(key, {"identity_version": 1, "calculation": payload, "runs": []})
            entry["runs"].append({"directory": directory.relative_to(self.root).as_posix(),
                                  "run_key": record[self.adapter.stored_key], "manifest_sha": file_sha(path)})
        if write:
            index_dir = self.root / self.adapter.index_name
            index_dir.mkdir(parents=True, exist_ok=True)
            for key, entry in entries.items():
                path = index_dir / f"{key}.json"
                if path.is_file():
                    prior = _read(path)
                    by_path = {x["directory"]: x for x in prior["runs"]}
                    by_path.update({x["directory"]: x for x in entry["runs"]})
                    entry["runs"] = [by_path[p] for p in sorted(by_path)]
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=index_dir,
                                                     prefix=f"{key}.", suffix=".tmp", delete=False) as stream:
                        temporary = Path(stream.name)
                        json.dump(entry, stream, ensure_ascii=False, indent=2)
                        stream.write("\n")
                    os.replace(temporary, path)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
        return {"indexed_runs": sum(len(e["runs"]) for e in entries.values()), "calculation_keys": len(entries),
                "written": write, "skipped": skipped}
