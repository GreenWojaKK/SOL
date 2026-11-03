"""저장된 배정·예측의 재평가와 진단 후보 점수 cache. 학습·배정 탐색은 호출하지 않는다."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vppcontract.evaluate import Actuals, group_truth, load_actuals
from vppcontract.identity import digest, file_sha
from vppcontract.outputs import read_output
from vppcontract.settlement import daily, hourly_penalty, paired
from vppcontract.timeaxis import period
from .store import evaluation_payload
from .sampler import heldout_mask, balanced_split
from .partition import repair
from .store import Store, read_json, write_json

EVALUATION_VERSION = 2


def output_directories(directory: Path, report: dict) -> dict:
    return {name: Path(directory) / "outputs" / name for name in report["outputs"]}


def objective_days(report: dict) -> np.ndarray:
    cutoff = int(report["cutoff_h"]) // 24
    first = cutoff + 1 - int(report["train_days"]) - int(report["heldout_days"])
    days = np.arange(max(first, 0), cutoff + 1)
    held = report["definition"]["heldout"]
    days = days[heldout_mask(days, held["block_days"], held["cycle_blocks"], held["offset"])]
    return days[-int(report["objective_days"]):]


def output_hashes(outputs):
    return {name: {f: file_sha(o.directory / f) for f in ("composition.csv", "forecast.csv")}
            for name, o in outputs.items()}


def candidate_partitions(source, output, capacity, count, seed):
    rng = np.random.default_rng(seed)
    share = float(source["minimum_share"])
    return [repair(balanced_split(capacity, rng, share), capacity, share) for _ in range(count)]


def save_artifacts(spec, directory, forecaster):
    """모델 설정과 최초 평가의 dependency를 run의 artifact 기록에 보존한다."""
    directory = Path(directory)
    source = read_json(directory / "instance.json")
    outputs = {k: read_output(p) for k, p in output_directories(directory, source).items()}
    settings = source["evaluation_settings"]
    base = outputs["stratified"]
    index = pd.Index(forecaster.data.sites.ids.astype(str)).get_indexer(base.site_ids.astype(str))
    candidates = {}
    pool = source.get("evaluation", {}).get("candidate_pool")
    if pool:
        partitions = candidate_partitions(source, base, forecaster.data.sites.capacity[index],
                                          pool["random_halves"], settings["candidate_pool_seed"])
        by_label = {f"random{i}": x for i, x in enumerate(partitions)}
        by_label.update({name: o.group[pd.Index(o.site_ids.astype(str)).get_indexer(base.site_ids.astype(str))] == 0
                         for name, o in outputs.items()})
        for label, predicted, w in zip(pool["labels"], pool["predicted_objective_per_day_krw"], pool["realized_W_krw"]):
            candidates[digest(by_label[label].astype(int).tolist())] = {"objective": predicted, "W_krw": w}
    record = {"inference": forecaster.inference_spec(),
              "evaluation": {"actuals": forecaster.data.base.key, "outputs": output_hashes(outputs),
                             "settings": settings, "candidates": candidates}}
    write_json(directory / "artifacts.json", record)
    return record


def inference_spec(spec, source, saved):
    """원 run의 resolved 설정으로 추론용 Spec을 복원한다."""
    config, registry = deepcopy(spec.config), deepcopy(spec.registry)
    config.update(deepcopy(saved["config"]))
    config["paths"]["artifact_dir"] = saved["artifact_root"]
    registry["models"][source["model"]] = deepcopy(source["definition"])
    return replace(spec, config=config, registry=registry, schema=deepcopy(saved["schema"]),
                   hashes={**spec.hashes, "schema": saved["schema_sha"]})


def actual_identity(spec) -> str:
    from .data import base_key
    return base_key(spec)


def finish_prediction(spec, directory: Path, data=None):
    """지정한 예측의 평가 결과를 별도 디렉터리에 기록한다."""
    source = read_json(Path(directory) / "instance.json")
    if not period(source["instance"]).has_labels:
        return None
    if data is None:
        return evaluate_saved(spec, directory)
    base = data.base
    actuals = Actuals(data.sites.ids, data.sites.capacity, np.asarray(base.energy), np.asarray(base.status) == 1,
                      np.asarray(base.smp), np.asarray(base.k))
    return evaluate_saved(spec, directory, actuals=actuals, actual_key=base.key)


def _score(actuals, output):
    truth, complete = group_truth(actuals, output, output.hours)
    loss = hourly_penalty(truth, output.forecast, actuals.smp[output.hours], actuals.k[output.hours],
                          np.broadcast_to(complete[:, None], truth.shape))
    return {"W_krw": float(loss.sum()), "scored_hours": int(complete.sum()), "daily": daily(output.hours, loss)}


def _calendar_rows(spec, actuals, outputs):
    from datetime import timedelta
    from vppcontract.timeaxis import delivery_dates
    result = {}
    for name, output in outputs.items():
        truth, complete = group_truth(actuals, output, output.hours)
        k3 = (actuals.k[output.hours] == 3) & complete
        errors = ((output.forecast - truth).sum(1) * k3).reshape(-1, 24).sum(1)
        generation = (truth.sum(1) * k3).reshape(-1, 24).sum(1)
        w = _score(actuals, output)["daily"].sum(1)
        dates = [d.date() for d in delivery_dates(np.unique(output.hours // 24))]
        off = np.array([spec.calendar.is_off(d, d - timedelta(days=1)) for d in dates])
        seasons = np.array(["spring" if d.month in (3, 4, 5) else "autumn" if d.month in (9, 10, 11) else "other" for d in dates])
        rows = []
        for season in ("spring", "autumn", "other"):
            for label, flag in (("workday", False), ("off", True)):
                m = (seasons == season) & (off == flag)
                if m.any():
                    rows.append({"season": season, "day": label, "days": int(m.sum()), "W_krw": float(w[m].sum()),
                                 "k3_bias": float(errors[m].sum() / max(generation[m].sum(), 1e-9))})
        result[name] = rows
    return result


class CachedCandidateInference:
    """선택된 source의 저장 모델만 로딩해 새 진단 후보를 추론한다."""
    def __init__(self, spec, directory, report, output, saved):
        self.spec, self.directory, self.report, self.output = spec, directory, report, output
        self.saved = saved
        self.context = None

    def __call__(self, partitions, actuals):
        from .data import load_data
        from .forecast.forecaster import Forecaster
        from .risk import RiskModel
        from .partition import Objective
        from .instance import predict_partitions
        if self.context is None:
            if not self.saved:
                raise ValueError("추가 후보 추론에 필요한 artifacts.json의 inference 설정이 없습니다.")
            spec = inference_spec(self.spec, self.report, self.saved)
            data = load_data(spec)
            if (data.base.key, data.weather.key) != (self.saved["base_key"], self.saved["weather_key"]):
                raise ValueError("추론 입력 데이터가 원 run의 base·weather와 다릅니다.")
            days = np.unique(self.output.hours // 24)
            f = Forecaster(data, self.report["model"], self.report["cutoff_h"], days_needed=days, smoke=self.report.get("smoke", False))
            f.load_cached(self.report["forecaster"], self.output.manifest["forecaster"], self.saved["power_curve"])
            risk = RiskModel(f, self.report.get("smoke", False))
            risk.load_cached(self.report["risk"])
            index = pd.Index(data.sites.ids.astype(str)).get_indexer(self.output.site_ids.astype(str))
            objective = Objective(f, risk, index, objective_days(self.report), self.report["definition"]["partition"])
            self.context = f, risk, index, objective, days
        f, risk, index, objective, days = self.context
        predicted = objective(np.asarray(partitions))
        forecasts = predict_partitions(f, index, np.asarray(partitions), days)
        records = []
        for x, forecast, predicted_value in zip(partitions, forecasts, predicted):
            output = replace(self.output, group=np.where(x, 0, 1), forecast=forecast)
            records.append({"objective": float(predicted_value), "W_krw": _score(actuals, output)["W_krw"]})
        return records


def _candidate_scores(spec, source, saved, outputs, actuals, context, infer):
    cfg = spec.config["evaluation"]
    smoke = source.get("smoke", False)
    count = int(spec.config["smoke"]["candidate_pool"] if smoke else cfg["candidate_pool_size"])
    seed = int(cfg["candidate_pool_seed"])
    output = outputs["stratified"]
    positions = pd.Index(actuals.site_ids.astype(str)).get_indexer(output.site_ids.astype(str))
    capacity = actuals.capacity[positions]
    partitions = candidate_partitions(source, output, capacity, count, seed)
    labels = [f"random{i}" for i in range(count)] + list(outputs)
    partitions += [o.group[pd.Index(o.site_ids.astype(str)).get_indexer(output.site_ids.astype(str))] == 0 for o in outputs.values()]
    store = Store(spec.artifact_dir / "cache")
    keys = [digest({"context": context, "partition": x.astype(int).tolist()}) for x in partitions]
    records = {}
    prior = saved.get("evaluation", {})
    if prior.get("actuals") == context["actuals"] and prior.get("outputs") == context["outputs"]:
        for key, x in zip(keys, partitions):
            value = prior.get("candidates", {}).get(digest(x.astype(int).tolist()))
            if value is not None:
                records[key] = value
    for key in keys:
        if store.exists("evaluation_candidates", key):
            records[key] = store.load("evaluation_candidates", key, names=[])[1]["score"]
    missing = list(dict.fromkeys(key for key in keys if key not in records))
    if missing:
        by_key = dict(zip(keys, partitions))
        results = infer([by_key[key] for key in missing], actuals)
        if len(results) != len(missing):
            raise ValueError("추가 후보의 추론 개수가 일치하지 않습니다.")
        records.update(zip(missing, results))
    for key, record in records.items():
        if not store.exists("evaluation_candidates", key):
            store.write("evaluation_candidates", key, meta={"context": context, "score": record})
    values = [records[key] for key in keys]
    pred = np.array([r["objective"] for r in values])
    real = np.array([r["W_krw"] for r in values])
    result = {"random_halves": count, "labels": labels, "spearman": float(spearmanr(pred, real).statistic),
              "predicted_objective_per_day_krw": pred.tolist(), "realized_W_krw": real.tolist(),
              "searched_share_of_random_with_lower_W": float((real[:count] < _score(actuals, outputs["searched"])["W_krw"]).mean()) if count else None}
    return result, len(missing)


def evaluate_saved(spec, directory: Path, *, actuals=None, actual_key=None, infer=None) -> Path:
    """저장된 partition·forecast로 평가하고, 점수가 없는 추가 후보만 추론한다."""
    directory = Path(directory).resolve()
    source = read_json(directory / "instance.json")
    if not period(source["instance"]).has_labels:
        raise ValueError("정답이 있는 instance만 평가합니다.")
    outputs = {name: read_output(path) for name, path in output_directories(directory, source).items()}
    prediction_key = source["run_key"]
    artifact_path = directory / "artifacts.json"
    saved = read_json(artifact_path) if artifact_path.is_file() else {}
    actual_key = actual_key or actual_identity(spec)
    evaluation = evaluation_payload(spec.config["evaluation"], calendar=spec.calendar,
                                    candidate_pool=spec.config["smoke"]["candidate_pool"] if source.get("smoke") and source.get("stage", "full") == "full" else None)
    content = output_hashes(outputs)
    evaluation.pop("calibration_bins", None)
    payload = {"prediction": prediction_key, "outputs": content, "actuals": actual_key,
               "inference": digest(saved.get("inference")), "evaluation": evaluation, "version": EVALUATION_VERSION}
    key = digest(payload)
    output = spec.artifact_dir / "evaluations" / key
    record_path = output / "evaluation.json"
    if record_path.is_file():
        record = read_json(record_path)
        if record.get("status") == "completed" and record.get("calculation") == payload:
            return output
    actuals = actuals or load_actuals(spec.data_dir)
    scores = {label: _score(actuals, output) for label, output in outputs.items()}
    cfg = spec.config["evaluation"]
    pair = lambda a, b: paired(scores[a]["daily"], scores[b]["daily"], tuple(cfg["bootstrap_blocks"]),
                               int(cfg["bootstrap_replicates"]), int(cfg["bootstrap_seed"]))
    result = {}
    result.update(W_krw={k: v["W_krw"] for k, v in scores.items()}, scored_hours=scores["stratified"]["scored_hours"],
                  regime_bias=_calendar_rows(spec, actuals, outputs))
    if "original" in scores:
        result["stratified_vs_original"] = pair("stratified", "original")
    if "searched" in scores:
        result["searched_vs_stratified"] = pair("searched", "stratified")
    inferred = 0
    if source.get("stage", "full") == "full":
        context = {"prediction": prediction_key, "actuals": actual_key, "outputs": content,
                   "inference": payload["inference"], "version": EVALUATION_VERSION}
        infer = infer or CachedCandidateInference(spec, directory, source, outputs["stratified"], saved.get("inference"))
        result["candidate_pool"], inferred = _candidate_scores(spec, source, saved, outputs, actuals, context, infer)
    output = spec.artifact_dir / "evaluations" / key
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "evaluation.json", {"package": "solarvpp2", "status": "completed", "run_key": key,
               "calculation": payload, "prediction_key": prediction_key, "source_instance": str(directory),
               "evaluation": result, "additional_candidates_inferred": inferred})
    return output
