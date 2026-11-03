"""선택한 run의 configuration 검증. Experiment 실행은 snapshot 검증을 선택적으로 사용한다."""
from pathlib import Path

from .store import digest, read_json


def selection_context(period, definition: dict, settings: str, selection: str | None, level: str, experts=None) -> dict:
    """--selection을 준 경우에만 frozen model·settings를 검사한다."""
    role = "inference" if not period.has_labels else ("execution_check" if level == "smoke" else "exploratory")
    context = {"exposure_status": period.exposure_status, "reporting_role": role, "selection_checked": False}
    if selection is None:
        return context
    record = read_json(Path(selection))
    body = {k: v for k, v in record.items() if k != "sha"}
    if record.get("sha") != digest(body) or not record.get("frozen"):
        raise ValueError("선택 기록이 고정되지 않았거나 내용이 바뀌었습니다.")
    frozen = [record[role].get("model") for role in ("finalist", "baseline")]
    if any(model is None for model in frozen):
        raise ValueError("선택 기록에 모델 구성이 없습니다.")
    matching = [record[role] for role in ("finalist", "baseline") if record[role]["model"] == definition]
    if not matching:
        raise ValueError("모델 구성(입력·손실·블렌드 비율)이 선택 기록과 다릅니다.")
    if experts is not None:
        matching = [item for item in matching if set(item.get("experts", ["xgb", "dl"])) == set(experts)]
        if not matching:
            raise ValueError("실행 expert가 선택 기록과 다릅니다.")
    if not any(item.get("settings_hash", record.get("settings_hash")) == settings for item in matching):
        raise ValueError("학습 설정이 선택 기록과 다릅니다.")
    if period.has_labels and level != "smoke":
        context["reporting_role"] = "retrospective_confirmation" if period.name == "H" else "frozen_comparison"
    context.update(selection=str(Path(selection).resolve()), selection_sha=record["sha"], selection_checked=True)
    return context


def verify_run_selection(spec, run_dir: Path, selection: str) -> dict:
    """저장된 run이 명시한 snapshot과 일치하는지 학습 없이 확인한다."""
    manifest = read_json(Path(run_dir) / "manifest.json")
    context = selection_context(spec.period(manifest["period"]), manifest["model"], manifest["settings_hash"],
                                selection, manifest["level"], manifest.get("experts"))
    return {"run_key": manifest["run_key"], "status": "passed", **context}
