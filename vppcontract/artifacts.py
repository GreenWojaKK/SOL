"""산출물 수명 관리: 실행 기록의 상태 판정, 참조 수집, archive 이동.

package 산출물 디렉터리(`sol-artifact/<package>/`) 아래 실행 기록(run, instance, 조건부 실험)은 현재 코드와 등록 목록에 대해 다음 상태 가운데 하나다.

- current: 등록 MK의 정의 위에 현재 코드가 받아들이는 parameter만 바꿨고, component version이 현재와 같다.
- superseded: current 조건 가운데 component version만 다르다. 등록 MK의 기록이지만 현재 코드로는 다시 만들 수 없다.
- replaced: superseded이면서 같은 package·kind·stage·MK·parameter의 current 기록이 있다.
- dead: 등록되지 않은 모델이거나, 등록 정의에서 벗어난 정의이거나, 현재 코드에 없는 parameter를 쓴다.
- aborted: 실패했거나 기록이 없거나, 시작 뒤 STALE_HOURS가 지나도 끝나지 않았다.
- running: 시작 뒤 STALE_HOURS 안이고 아직 끝나지 않았다.
- unreadable: 디렉터리를 읽을 수 없다. 사람이 권한을 확인한다.

champion·adoption·제출·latest·추천 기록이 가리키는 디렉터리는 상태와 무관하게 archive하지 않는다.
archive는 replaced·dead·aborted 기록을 같은 package 디렉터리의 `archive/<원래 상대 경로>`로 옮기고 `archive/index.json`에
무엇이었는지(MK, parameter, 상태, 결과 요약)를 적는다. 각 package는 자기 기록을 Entry로 만들고, 이 module은 판정·참조·이동만 한다.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from .outputs import read_json, write_json

STALE_HOURS = 24
ARCHIVED = ("replaced", "dead", "aborted")
STATES = ("current", "superseded", "replaced", "dead", "aborted", "running", "unreadable")
INDEX = "index.json"


@dataclass
class Entry:
    """실행 기록 하나. package가 manifest를 읽어 채우고, state·reason은 classify가 채운다."""
    package: str
    kind: str                                   # run | instance | experiment
    directory: Path
    container: Path | None = None               # 기록을 나열한 디렉터리. 이동 뒤 비어 있으면 그 사이의 디렉터리를 지운다.
    model: str | None = None                    # 기록된 MK 또는 recipe 이름
    stage: str | None = None                    # 기간·level, instance·stage처럼 같은 종류의 기록을 묶는 이름
    status: str | None = None                   # completed | running | failed | missing | unreadable
    started: str | None = None
    finished: str | None = None
    registered: bool = True
    definition_ok: bool = True
    parameter_error: str | None = None          # 현재 코드가 받아들이지 않는 parameter의 설명
    versions_ok: bool = True
    parameters: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    referenced_by: list[str] = field(default_factory=list)
    state: str | None = None
    reason: str | None = None

    @property
    def archivable(self) -> bool:
        return self.state in ARCHIVED and not self.referenced_by

    def key(self) -> tuple:
        """replaced 판정에 쓰는 동일 기록의 열쇠."""
        return (self.package, self.kind, self.stage, self.model, _frozen(self.parameters))


def _frozen(value):
    if isinstance(value, dict):
        return tuple(sorted((str(k), _frozen(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(v) for v in value)
    return value


def classify(entry: Entry, now: pd.Timestamp, stale_hours: float = STALE_HOURS) -> tuple[str, str]:
    """기록 하나의 상태와 이유."""
    if entry.status == "unreadable":
        return "unreadable", "디렉터리를 읽을 수 없다"
    if entry.status in ("failed", "missing", None):
        return "aborted", {"failed": "실행이 실패했다", "missing": "manifest가 없다"}.get(entry.status, "실행 상태 기록이 없다")
    if entry.status != "completed":
        started = pd.Timestamp(entry.started) if entry.started else None
        if started is None or now - started > pd.Timedelta(hours=stale_hours):
            return "aborted", f"{entry.status} 상태로 {stale_hours:g}시간 넘게 끝나지 않았다"
        return "running", f"{entry.status} 상태이고 시작 뒤 {stale_hours:g}시간이 지나지 않았다"
    if not entry.registered:
        return "dead", f"등록되지 않은 모델 {entry.model}"
    if entry.parameter_error:
        return "dead", entry.parameter_error
    if not entry.definition_ok:
        return "dead", "기록된 정의가 등록 정의(parameter 적용 뒤)와 다르다"
    if not entry.versions_ok:
        return "superseded", "component version이 현재와 다르다"
    return "current", "현재 등록 정의·component version과 같다"


def classify_all(entries: list[Entry], references: dict[str, list[str]], now: pd.Timestamp, base: Path | None = None,
                 stale_hours: float = STALE_HOURS) -> list[Entry]:
    """모든 기록의 상태를 정한다. superseded 가운데 같은 열쇠의 current 기록이 있으면 replaced로 바꾸고, 참조된 기록을 표시한다.

    참조는 기록의 package 디렉터리(base) 기준 상대 경로가 pointer 경로 안에 경로 단위로 들어 있는지로 찾는다. 상대 경로에는
    run key·실험 hash가 들어 있어 유일하고, 산출물 디렉터리를 옮기기 전의 절대 경로를 적은 pointer도 같은 기록을 가리킨다.
    """
    for entry in entries:
        entry.state, entry.reason = classify(entry, now, stale_hours)
    current = {entry.key() for entry in entries if entry.state == "current"}
    for entry in entries:
        if entry.state == "superseded" and entry.key() in current:
            entry.state, entry.reason = "replaced", "같은 stage·MK·parameter의 current 기록이 있다"
        needle = _needle(entry.directory, base)
        entry.referenced_by = sorted({source for path, sources in references.items() if _mentions(path, needle) for source in sources})
    return entries


def _posix(text: str) -> str:
    return text.replace("\\", "/").rstrip("/").lower()


def _needle(directory: Path, base: Path | None) -> str:
    """참조를 찾을 경로 조각. base 아래 기록은 base 기준 상대 경로, 그 밖은 절대 경로이다."""
    directory = Path(directory).resolve()
    if base is not None:
        try:
            return "/" + _posix(directory.relative_to(Path(base).resolve()).as_posix())
        except ValueError:
            pass
    return "/" + _posix(str(directory)).lstrip("/")


def _mentions(path: str, needle: str) -> bool:
    """path가 needle을 경로 단위로 담는가. needle 다음 글자는 끝이거나 구분자여야 한다."""
    start = path.find(needle)
    while start >= 0:
        end = start + len(needle)
        if end == len(path) or path[end] == "/":
            return True
        start = path.find(needle, start + 1)
    return False


def collect_references(files: Iterable[Path]) -> dict[str, list[str]]:
    """JSON 기록 안의 경로 문자열을 모아 {'/'로 시작하는 소문자·'/' 구분 경로: [기록 파일]}로 돌려준다.
    구분자가 없는 문자열은 경로로 보지 않는다."""
    out: dict[str, list[str]] = {}
    for file in files:
        file = Path(file)
        if not file.exists():
            continue
        for text in _strings(read_json(file)):
            if "\n" in text or len(text) > 1000 or not any(sep in text for sep in ("\\", "/")):
                continue
            out.setdefault("/" + _posix(text).lstrip("/"), []).append(str(file))
    return out


def _strings(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)
    elif isinstance(value, str):
        yield value


def describe(entry: Entry, base: Path) -> dict:
    """보고서와 archive index에 적는 기록 요약."""
    try:
        location = Path(entry.directory).resolve().relative_to(Path(base).resolve()).as_posix()
    except ValueError:
        location = str(entry.directory)
    return {"path": location, "package": entry.package, "kind": entry.kind, "state": entry.state, "reason": entry.reason,
            "model": entry.model, "stage": entry.stage, "parameters": entry.parameters, "status": entry.status,
            "started": entry.started, "finished": entry.finished, "referenced_by": entry.referenced_by, "summary": entry.summary}


def report(entries: list[Entry], base: Path) -> dict:
    """상태별 수와 기록 목록. archivable은 archive가 옮길 기록, kept_by_reference는 상태로는 옮길 대상이지만 참조되어 남기는 기록이다."""
    counts = {state: sum(entry.state == state for entry in entries) for state in STATES}
    return {"counts": {k: v for k, v in counts.items() if v},
            "archivable": [describe(e, base)["path"] for e in entries if e.archivable],
            "kept_by_reference": [describe(e, base)["path"] for e in entries if e.state in ARCHIVED and e.referenced_by],
            "entries": [describe(e, base) for e in entries]}


def archive(entries: list[Entry], base: Path, archive_root: Path, now: pd.Timestamp) -> dict:
    """archivable 기록을 archive_root/<base 기준 상대 경로>로 옮기고 index에 적는다. 비어 버린 중간 디렉터리는 지운다.

    옮기지 못한 기록(권한, 같은 경로의 archive 존재)은 failed에 이유와 함께 남기고 나머지는 계속 옮긴다.
    """
    base, archive_root = Path(base).resolve(), Path(archive_root)
    index_path = archive_root / INDEX
    index = read_json(index_path) if index_path.exists() else []
    moved, failed = [], []
    for entry in entries:
        if not entry.archivable:
            continue
        relative = Path(entry.directory).resolve().relative_to(base)
        target = archive_root / relative
        try:
            if target.exists():
                raise FileExistsError(f"archive에 같은 경로가 이미 있습니다: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(entry.directory), str(target))
        except OSError as exc:
            failed.append({"path": relative.as_posix(), "error": f"{type(exc).__name__}: {exc}"})
            continue
        _prune(Path(entry.directory).parent, entry.container)
        record = {"archived": now.isoformat(), **describe(entry, base), "to": target.resolve().relative_to(base).as_posix()}
        index.append(record)
        moved.append(record)
    if moved:
        archive_root.mkdir(parents=True, exist_ok=True)
        write_json(index_path, index)
    return {"moved": moved, "failed": failed}


def _prune(directory: Path, container: Path | None) -> None:
    """옮긴 기록과 container 사이의 빈 디렉터리를 지운다."""
    if container is None:
        return
    container = Path(container).resolve()
    directory = Path(directory).resolve()
    while directory != container and container in directory.parents and directory.exists() and not any(directory.iterdir()):
        directory.rmdir()
        directory = directory.parent
