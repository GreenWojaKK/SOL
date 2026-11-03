"""solarvpp2 instance 기록의 상태 판정과 archive.

instance.json의 model·definition·parameters·versions·status를 읽어 vppcontract.artifacts의 Entry로 만든다. 판정 규칙과 이동은
계약층이 맡는다. 추천 기록(`recommendation/<MK>.json`)이 가리키는 instance는 옮기지 않는다.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from vppcontract.artifacts import Entry, archive, classify_all, collect_references, report

from .config import Spec, with_overrides
from .store import read_json, versions


def instance_entries(spec: Spec) -> list[Entry]:
    """instances/<instance>/<run>/instance.json마다 Entry 하나."""
    root = spec.artifact_dir / "instances"
    current = versions()
    out = []
    for container in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        for directory in sorted(p for p in container.iterdir() if p.is_dir()):
            try:
                path = directory / "instance.json"
                record = read_json(path) if path.exists() else None
            except OSError:
                out.append(Entry("solarvpp2", "instance", directory, container, stage=container.name, status="unreadable"))
                continue
            if record is None:
                out.append(Entry("solarvpp2", "instance", directory, container, stage=container.name, status="missing"))
                continue
            out.append(instance_entry(spec, directory, container, record, current))
    return out


def instance_entry(spec: Spec, directory: Path, container: Path, record: dict, current: dict) -> Entry:
    model = record.get("model")
    parameters = record.get("parameters") or record.get("overrides") or {}
    registered = model in spec.registry["models"]
    definition_ok, error = True, None
    if registered:
        try:
            definition_ok = with_overrides(spec, model, parameters).model(model) == record.get("definition")
        except (ValueError, KeyError, TypeError) as exc:
            error = f"현재 코드에 없는 parameter: {exc}"
    panel = (record.get("panel") or {}).get("name")
    stage = f"{container.name}/{record.get('stage', 'full')}" + (f":{panel}" if panel else "") + ("/smoke" if record.get("smoke") else "")
    evaluation = record.get("evaluation") or {}
    summary = {"W_krw": evaluation.get("W_krw"), "source_sha": record.get("source_sha"), "versions": record.get("versions")}
    return Entry("solarvpp2", "instance", directory, container, model=model, stage=stage, status=record.get("status"),
                 started=record.get("started"), finished=record.get("finished"), registered=registered, definition_ok=definition_ok,
                 parameter_error=error, versions_ok=record.get("versions") == current, parameters=parameters, summary=summary)


def reference_files(spec: Spec) -> list[Path]:
    """instance를 가리킬 수 있는 기록: 추천 기록과 package를 넘는 champion 기록."""
    recommendation = spec.artifact_dir / "recommendation"
    files = sorted(recommendation.glob("*.json")) if recommendation.exists() else []
    return [*files, spec.root / "configs" / "champion.json"]


def audit(spec: Spec, move: bool = False, now: pd.Timestamp | None = None) -> dict:
    """instance 기록의 상태를 보고하고, move이면 replaced·dead·aborted 기록을 package 산출물 디렉터리의 `archive/`로 옮긴다."""
    now = now or pd.Timestamp.now(tz="Asia/Seoul")
    base = spec.artifact_dir
    entries = classify_all(instance_entries(spec), collect_references(reference_files(spec)), now, base)
    result = {"base": str(base), **report(entries, base)}
    if move:
        result["archived"] = archive(entries, base, base / "archive", now)
    return result
