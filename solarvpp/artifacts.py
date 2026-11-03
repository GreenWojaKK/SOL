"""solarvpp run·조건부 실험 기록의 상태 판정과 archive.

run manifest의 기준 MK·parameter·versions·execution_status와 조건부 실험의 primitive·versions·execution 상태를 읽어
vppcontract.artifacts의 Entry로 만든다. 판정 규칙과 이동은 계약층이 맡는다. adoption 기록, 조건부 champion·latest, 제출 manifest,
비교 보고서, 남기는 실험의 decision·transfer가 가리키는 디렉터리는 옮기지 않는다.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from vppcontract.artifacts import ARCHIVED, Entry, archive, classify_all, collect_references, report

from .config import AUX, INPUTS, Spec
from .experiment import parameter_label, run_parameters
from .store import read_json, versions

TIME_WEIGHTS = ("none", "settlement")
COMPLETED = ("completed", "completed_command")


def run_entries(spec: Spec) -> list[Entry]:
    """runs/<period>/<run>/manifest.json마다 Entry 하나."""
    root = spec.artifact_dir / "runs"
    current = versions()
    out = []
    for container in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        for directory in sorted(p for p in container.iterdir() if p.is_dir()):
            try:
                path = directory / "manifest.json"
                manifest = read_json(path) if path.exists() else None
            except OSError:
                out.append(Entry("solarvpp", "run", directory, container, stage=container.name, status="unreadable"))
                continue
            if manifest is None:
                out.append(Entry("solarvpp", "run", directory, container, stage=container.name, status="missing"))
                continue
            out.append(run_entry(spec, directory, container, manifest, current))
    return out


def model_record(manifest: dict) -> dict:
    """run의 모델 구성. `model`이 없는 이전 manifest는 expert별 입력·손실·time weight·블렌드 비율 기록으로 같은 형태를 만든다."""
    if manifest.get("model"):
        return dict(manifest["model"])
    inputs, loss, weights = manifest.get("model_inputs") or {}, manifest.get("loss_config") or {}, manifest.get("model_time_weights") or {}
    if not (inputs.get("xgb") and inputs.get("dl") and loss.get("name")):
        return {}
    return {"xgb_input": inputs["xgb"]["name"], "xgb_time_weight": weights.get("xgb", "none"), "dl_input": inputs["dl"]["name"],
            "dl_loss": loss["name"], "aux_ratio": float(loss.get("aux_ratio", 0.0)), "dl_time_weight": weights.get("dl", loss.get("time_weight", "none")),
            "blend": "weighted-quantile", "xgb_weight": float(manifest.get("xgb_weight", 0.5))}


def run_entry(spec: Spec, directory: Path, container: Path, manifest: dict, current: dict) -> Entry:
    manifest = {**manifest, "model": model_record(manifest)}
    recipe = manifest.get("recipe")
    base = None if recipe else (manifest.get("model_version") or manifest.get("base_model"))
    registered = (recipe in spec.protocol.get("forecast_recipes", {})) if recipe else (base is None or base in spec.registry["models"])
    error = parameter_error(spec, manifest)
    parameters, label = {}, base or (f"recipe:{recipe}" if recipe else None)
    if registered and error is None:
        version, parameters = run_parameters(spec, manifest)
        label = parameter_label(version, parameters) if version or parameters else label
    experts = manifest.get("experts") or []
    stage = f"{manifest.get('period', container.name)}/{manifest.get('level')}" + (f"/{'+'.join(experts)}" if set(experts) != {"xgb", "dl"} else "")
    recorded = manifest.get("versions") or {}
    summary = {"label": label, "evaluation_status": manifest.get("evaluation_status"), "cell": manifest.get("cell"),
               "W_krw": _run_scores(directory, manifest), "source_sha": manifest.get("source_sha")}
    return Entry("solarvpp", "run", directory, container, model=label if recipe else base, stage=stage, status=manifest.get("execution_status"),
                 started=manifest.get("started"), finished=manifest.get("finished"), registered=registered, parameter_error=error,
                 versions_ok=all(current.get(k) == v for k, v in recorded.items()), parameters=parameters, summary=summary)


def parameter_error(spec: Spec, manifest: dict) -> str | None:
    """run의 모델 구성(두 expert의 입력, 손실, time weight, 블렌드)과 배정 파일이 현재 코드에 있는지 확인한다."""
    model = manifest.get("model") or {}
    if not model:
        return "모델 구성 기록이 없다"
    problems = [f"{key}={model.get(key)}" for key in ("xgb_input", "dl_input") if model.get(key) not in INPUTS]
    if model.get("dl_loss") not in AUX:
        problems.append(f"dl_loss={model.get('dl_loss')}")
    problems += [f"{key}={model.get(key)}" for key in ("xgb_time_weight", "dl_time_weight") if model.get(key, "none") not in TIME_WEIGHTS]
    if model.get("blend", "weighted-quantile") != "weighted-quantile":
        problems.append(f"blend={model.get('blend')}")
    partition = manifest.get("partition") or {}
    if partition.get("path") and not partition.get("is_original", False) and not (spec.root / partition["path"]).exists():
        problems.append(f"partition 파일 없음: {partition['path']}")
    return "현재 코드에 없는 parameter: " + ", ".join(problems) if problems else None


def _run_scores(directory: Path, manifest: dict) -> dict | None:
    path = directory / "report.json"
    if not path.exists() or not manifest.get("cell"):
        return None
    components = read_json(path).get("components", {})
    return {name: item.get("W_cohort_krw") for name, item in components.items()}


def experiment_entries(spec: Spec) -> list[Entry]:
    """conditional/<hash>/experiment.json마다 Entry 하나. experiment.json이 없는 디렉터리는 기록이 아니므로 aborted로 본다."""
    root = spec.artifact_dir / "conditional"
    current = versions()
    out = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        try:
            path = directory / "experiment.json"
            experiment = read_json(path) if path.exists() else None
            execution = read_json(directory / "execution.json") if (directory / "execution.json").exists() else None
        except OSError:
            out.append(Entry("solarvpp", "experiment", directory, root, stage="conditional", status="unreadable"))
            continue
        if experiment is None:
            out.append(Entry("solarvpp", "experiment", directory, root, stage="conditional", status="missing"))
            continue
        out.append(experiment_entry(spec, directory, root, experiment, execution, current))
    return out


def experiment_entry(spec: Spec, directory: Path, container: Path, experiment: dict, execution: dict | None, current: dict) -> Entry:
    config = experiment.get("config") or {}
    primitive = config.get("primitive") or {}
    model = primitive.get("model")
    registered = model is not None and model in spec.registry["models"]
    error = None if model is not None else "primitive가 MK 버전이 아닌 옛 형식으로 기록되어 현재 코드가 읽지 않는다"
    definition_ok = (not registered) or experiment.get("primitive_model") == spec.model(model)
    recorded = {**(experiment.get("upstream_versions") or {}), **(experiment.get("versions") or {})}
    versions_ok = bool(recorded) and all(current.get(k) == v for k, v in recorded.items())
    status, started, finished = _experiment_status(directory, experiment, execution)
    parameters = {k: config.get(k) for k in ("forecast_input", "models") if k in config}
    decision = read_json(directory / "decision.json") if (directory / "decision.json").exists() else {}
    summary = {"config_version": config.get("version"), "instances": config.get("instances"),
               "decision": {k: decision.get(k) for k in ("method", "deploy", "group_delta_W_total_krw")} if decision else None}
    return Entry("solarvpp", "experiment", directory, container, model=model, stage="conditional", status=status, started=started,
                 finished=finished, registered=registered, definition_ok=definition_ok, parameter_error=error, versions_ok=versions_ok,
                 parameters=parameters, summary=summary)


def _experiment_status(directory: Path, experiment: dict, execution: dict | None) -> tuple[str, str | None, str | None]:
    """execution.json이 있으면 그 상태, 없으면 instance 보고서가 모두 완료인지로 상태를 정한다."""
    if execution is not None:
        status = execution.get("status")
        status = "completed" if status in COMPLETED else ("failed" if status == "failed" else (status or "running"))
        return status, execution.get("started") or experiment.get("started"), execution.get("finished")
    reports = [read_json(directory / f"{name}.json") for name in ("V", "H", "B", "test") if (directory / f"{name}.json").exists()]
    status = "completed" if reports and all(r.get("status") == "completed" for r in reports) else "running"
    return status, experiment.get("started"), None


def reference_files(spec: Spec, experiments: list[Entry]) -> list[Path]:
    """run·실험을 가리킬 수 있는 기록. 남기는 실험의 decision·transfer·selection도 run을 가리킨다."""
    files = [spec.root / spec.config["paths"]["models"], spec.root / "configs" / "champion.json",
             spec.artifact_dir / "conditional" / "champion.json", spec.artifact_dir / "conditional" / "latest.json"]
    for folder, pattern in (("submissions", "*/submission-manifest.json"), ("reports", "*.json")):
        files += sorted((spec.artifact_dir / folder).glob(pattern)) if (spec.artifact_dir / folder).exists() else []
    for entry in experiments:
        if entry.state not in ARCHIVED or entry.referenced_by:
            files += [p for name in ("decision.json", "selection.json") for p in [entry.directory / name] if p.exists()]
            files += sorted(entry.directory.glob("transfer-*.json"))
    return files


def audit(spec: Spec, move: bool = False, now: pd.Timestamp | None = None) -> dict:
    """run·조건부 실험 기록의 상태를 보고하고, move이면 replaced·dead·aborted 기록을 package 산출물 디렉터리의 `archive/`로 옮긴다.

    실험을 먼저 판정하고, 남는 실험이 가리키는 run을 참조로 더한 뒤 run을 판정한다.
    """
    now = now or pd.Timestamp.now(tz="Asia/Seoul")
    base = spec.artifact_dir
    experiments = classify_all(experiment_entries(spec), collect_references(reference_files(spec, [])), now, base)
    references = collect_references(reference_files(spec, experiments))
    entries = classify_all([*experiments, *run_entries(spec)], references, now, base)
    result = {"base": str(base), **report(entries, base)}
    if move:
        result["archived"] = archive(entries, base, base / "archive", now)
    return result
