"""조건부 bundling 실행, group transfer, 결과 audit, 최종 추천을 별도 명령으로 제공한다.

audit는 저장된 실험을 원본 발전량·SMP·고정 발전소 예측으로 독립 재계산하며, 정산 계산에 scoring 모듈을 쓰지 않는다.
"""
from __future__ import annotations

import os
import time
import traceback
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from ..experiment import cuda_lock, load_data, log, model_signature, run
from ..forecast.train import configure_determinism
from ..scoring import block_bootstrap, score_partition
from ..store import array_hash, digest, environment, file_sha, read_json, versions, write_json
from .error_model import ScenarioSource, fit_risk, masked_nll
from .optimize import ScenarioObjective, balance_coefficients, first_within_band, gurobi_version, optimize_scenarios, restore_feasibility
from .partition import build_instance, full_assignment, random_partition, relative_band, write_partition
from .residuals import primitive_descriptor, site_primitive


def frozen_pool(instance, cfg):
    """미래 잔차를 참조하지 않는 공통 후보: original과 상대 불균형 10% 이내 무작위 배정. 좌표 기반 배정은 넣지 않는다."""
    rng = np.random.default_rng(cfg["evaluation"]["candidate_pool_seed"])
    parts = [instance.original.copy()]
    names = ["original"]
    for i in range(cfg["evaluation"]["candidate_pool_size"]):
        parts.append(random_partition(instance.ctil, 0.1, rng))
        names.append(f"random{i}")
    return names, np.stack(parts)


def actual_comparison(instance, selected, random_partitions, cfg):
    """선택 이후의 전체 평가기간에서 original과 paired 금액 차이를 계산한다."""
    actual, daily = score_partition(instance, selected)
    baseline, base_daily = score_partition(instance, instance.original)
    random_w = np.array([score_partition(instance, p)[0]["W_krw"] for p in random_partitions])
    delta = actual["W_krw"] - baseline["W_krw"]
    return {"selected": actual, "original": baseline, "delta_W_krw": delta, "delta_S_krw": -delta,
            "random_median_W_krw": float(np.median(random_w)), "random_percentile": float(np.mean(random_w <= actual["W_krw"])),
            "evaluation_hours": len(instance.eval_hours), "evaluation_first_h": int(instance.eval_hours.min()),
            "evaluation_last_h": int(instance.eval_hours.max()),
            "bootstrap": {f"{b}d": block_bootstrap(daily - base_daily, b, cfg["evaluation"]["bootstrap_replicates"], 42)
                          for b in cfg["evaluation"]["bootstrap_days"]}}


def evaluate_scenarios(source, method, pool, selected_index, instance, cfg):
    """같은 시나리오로 후보 예상 W와 선택 배정의 주간 그룹 오차 calibration을 집계한다."""
    samples = cfg["scenarios"]["evaluation_samples_per_hour"]
    seed = cfg["scenarios"]["evaluation_seed"]
    batch = cfg["scenarios"]["batch_hours"]
    center = cfg["optimization"]["center"]
    predicted = np.zeros(len(pool))
    medians = np.zeros((2, len(source.hours), len(pool)), dtype=np.float32)
    coverage, width, pinball, count = np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2, dtype=int)
    target = np.full((len(source.hours), 2), np.nan)
    if instance.eval_error is not None:
        x = pool[selected_index]
        pos = instance.eval_hours - source.hours[0]
        target[pos, 0] = instance.eval_error[:, x].sum(1)
        target[pos, 1] = instance.eval_error[:, ~x].sum(1)
    levels = np.array([0.05, 0.5, 0.95])
    for lo in range(0, len(source.hours), batch):
        index = np.arange(lo, min(lo + batch, len(source.hours)))
        scenarios = source.batch(method, index, samples, seed).astype(np.float64)
        r1 = (scenarios.reshape(-1, scenarios.shape[-1]) @ pool.astype(float).T).reshape(len(index), scenarios.shape[1], len(pool))
        total = scenarios.sum(-1)[..., None]
        r2 = total - r1
        if center == "median":
            medians[0, index], medians[1, index] = np.median(r1, axis=1), np.median(r2, axis=1)
        losses = (np.abs(r1 - medians[0, index][:, None, :]) + np.abs(r2 - medians[1, index][:, None, :])).mean(1)
        predicted += (losses * instance.target_w[index, None]).sum(0)
        group_samples = (r1[..., selected_index], total[..., 0] - r1[..., selected_index])
        for g, values in enumerate(group_samples):
            valid = np.isfinite(target[index, g]) & source.daylight[index].any(1)
            if not valid.any():
                continue
            q = np.quantile(values[valid], levels, axis=1).T
            y = target[index, g][valid]
            err = y[:, None] - q
            coverage[g] += np.sum((y >= q[:, 0]) & (y <= q[:, 2]))
            width[g] += np.sum(q[:, 2] - q[:, 0])
            pinball[g] += np.maximum(levels * err, (levels - 1) * err).mean(1).sum()
            count[g] += int(valid.sum())
    calibration = {f"G{i + 1}": {"daylight_labelled_hours": int(count[i]),
                                  "coverage_90": float(coverage[i] / count[i]) if count[i] else None,
                                  "mean_width_kwh": float(width[i] / count[i]) if count[i] else None,
                                  "mean_pinball_kwh": float(pinball[i] / count[i]) if count[i] else None} for i in range(2)}
    return predicted, calibration, medians


def realized_losses(instance, pool, medians, hours):
    """평가기간 실현 시각별 벌점 [시각, 후보]. 제출값은 발전소 점예측의 합에 사전 시나리오 그룹 중앙값을 더한 값이다."""
    positions = instance.eval_hours - hours[0]
    r1 = instance.eval_error @ pool.astype(np.float64).T
    r2 = instance.eval_error.sum(axis=1)[:, None] - r1
    return instance.eval_w[:, None] * (np.abs(r1 - medians[0, positions]) + np.abs(r2 - medians[1, positions]))


def compare_realized(instance, hourly, selected_index, random_index, cfg):
    """선택 배정과 original(후보 0)의 실현 벌점 차이, 날짜 bootstrap, 무작위 후보 분포 안 위치."""
    day = instance.eval_hours // 24 - instance.eval_hours.min() // 24

    def daily(j):
        return np.bincount(day, weights=hourly[:, j], minlength=int(day.max()) + 1)

    total = hourly.sum(axis=0)
    random_w = total[random_index]
    delta = float(total[selected_index] - total[0])
    return {"selected": {"W_krw": float(total[selected_index])}, "original": {"W_krw": float(total[0])}, "delta_W_krw": delta,
            "delta_S_krw": -delta, "random_median_W_krw": float(np.median(random_w)),
            "random_percentile": float(np.mean(random_w <= total[selected_index])), "evaluation_hours": len(instance.eval_hours),
            "bootstrap": {f"{b}d": block_bootstrap(daily(selected_index) - daily(0), b, cfg["evaluation"]["bootstrap_replicates"], 42)
                          for b in cfg["evaluation"]["bootstrap_days"]}}


def validate_gaussian(source, instance):
    """관측 가능한 평가 PV의 주간 marginal likelihood. checkpoint 선택에는 사용하지 않는다."""
    if instance.eval_error is None:
        return None
    positions = instance.eval_hours - source.hours[0]
    target = instance.eval_error / instance.capacity[None, :]
    mask = np.isfinite(target) & source.daylight[positions]
    usable = mask.any(axis=1)
    values = []
    with torch.inference_mode():
        for lo in range(0, len(positions), 128):
            p = positions[lo:lo + 128]
            tensors = [torch.as_tensor(a, dtype=torch.float32, device="cuda") for a in
                       (source.mean[p], source.loading[p], source.sd[p], target[lo:lo + len(p)])]
            m = torch.as_tensor(mask[lo:lo + len(p)], dtype=torch.bool, device="cuda")
            values.append(masked_nll(*tensors, m, reduction="none").cpu().numpy())
    all_values = np.concatenate(values)
    return {"mean_nll_per_observed_pv": float(all_values[usable].mean()),
            "median_nll_per_observed_pv": float(np.median(all_values[usable])),
            "daylight_hours": int(usable.sum()), "observed_site_hours": int(mask.sum()),
            "role": "outer evaluation only; training uses fixed last-step checkpoint"}


def experiment_dir(spec, data, forecasts, cfg):
    catalog = {name: primitive_descriptor(spec, data, name, cfg["primitive"]) for name in ("D", "V", "H", "B")}
    for name, forecast in forecasts.items():
        if catalog[name] != {"kind": forecast["kind"], "key": forecast["key"]}:
            raise ValueError(f"{name}의 residual identity가 실험 설정과 다릅니다.")
    identity = {"config": cfg, "base": data.base.key, "weather": data.weather.key,
                "site_point_forecasts": catalog,
                "primitive_model": spec.model(cfg["primitive"]["model"]), "periods": spec.protocol["periods"],
                "forecast_budgets": spec.hyperparameters, "forecast_output": spec.protocol["output"],
                "forecast_settings": {key: spec.config[key] for key in ("runtime", "smoke", "deviations", "interpretations")},
                "forecast_blending": spec.protocol["blending"], "forecast_aux_lambda": spec.protocol["aux_lambda"],
                "upstream_bundling": spec.protocol["bundling_search"], "versions": versions("error_model", "optimize", "pipeline"), "gurobi": gurobi_version(),
                "upstream_versions": versions("queries", "residuals", "partition", "scoring")}
    key = digest(identity)
    directory = spec.artifact_dir / "conditional" / key[:16]
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "experiment.json").exists():
        write_json(directory / "experiment.json", {"key": key, **identity, "environment": environment(),
                                                    "started": pd.Timestamp.now(tz="Asia/Seoul").isoformat()})
    write_json(directory.parent / "latest.json", {"directory": str(directory), "key": key})
    return directory


def search_partitions(obj, instance, opt, band):
    """균형 설정에 따라 주 배정, frontier, test 후보를 만든다.

    fixed_lambda: 고정 λ로 탐색하고 lambda_frontier의 다른 λ는 주 배정에서 시작해 frontier_qubo 예산으로 다시 푼다.
    balance_grid: 오름차순 λ̃마다 λ_MAE = λ̃·J_ref,MAE/β², λ_QUBO = λ̃·J_ref,MSE/β²로 QUBO·original·직전 λ̃ 해·무작위 초기해에서
    탐색하고 모든 λ̃ 해를 남긴다. 주 배정은 V·H·B 전체에서 고른 λ̃*로 나중에 정한다(choose_balance, finalize_instance).
    """
    def entry(candidate, record, index, **extra):
        return {"objective": record["objective"], "imbalance": float(instance.ctil @ (2 * candidate - 1.0)),
                "capacity_kw": [float(instance.capacity[candidate].sum()), float(instance.capacity[~candidate].sum())],
                "partition_index": index, **extra}

    frontier, parts = {}, []
    if "balance_grid" in opt:
        grid = sorted(float(v) for v in opt["balance_grid"])
        moment, merge = obj.second_moment(), obj.merge_value()
        previous = None
        for relative in grid:
            obj.lam, qubo_lambda = balance_coefficients(relative, merge, moment, band)
            starts = {"original": instance.original} if previous is None else {"original": instance.original, "previous": previous}
            candidate, record = optimize_scenarios(obj, instance.capacity, starts, opt["random_starts"], 42, opt["max_iterations"],
                                                   opt["tolerance"], opt["qubo"], qubo_lambda=qubo_lambda, moment=moment, tabu=opt.get("tabu"))
            frontier[str(relative)] = entry(candidate, record, len(parts), relative=relative, qubo_lambda=qubo_lambda,
                                            search=record, **{"lambda": obj.lam})
            parts.append(candidate.copy())
            previous = candidate
        return None, None, frontier, parts, list(parts), {"mode": "relative", "band": band, "merge_objective": merge}, grid
    obj.lam = float(opt["fixed_lambda"])
    primary, trace = optimize_scenarios(obj, instance.capacity, {"original": instance.original}, opt["random_starts"], 42, opt["max_iterations"],
                                        opt["tolerance"], opt["qubo"], tabu=opt.get("tabu"))
    candidates = [primary.copy()]
    for lam in opt["lambda_frontier"]:
        obj.lam = float(lam)
        if lam == opt["fixed_lambda"]:
            candidate, record = primary.copy(), trace
        else:
            candidate, record = optimize_scenarios(obj, instance.capacity, {"primary": primary}, 0, 42,
                                                   opt["max_iterations"], opt["tolerance"], opt["frontier_qubo"])
        frontier[str(lam)] = entry(candidate, record, len(parts))
        parts.append(candidate.copy())
        candidates.append(candidate.copy())
    obj.lam = float(opt["fixed_lambda"])
    return primary, trace, frontier, parts, candidates, {"mode": "fixed", "lambda": obj.lam}, list(opt["lambda_frontier"])


def run_instance(spec, data, forecasts, name, cfg, directory):
    path = directory / f"{name}.json"
    if path.exists() and read_json(path).get("status") == "completed":
        log(f"조건부 bundling 재사용 {name}")
        return read_json(path)
    started = time.perf_counter()
    instance = build_instance(spec, data, name, forecasts)
    pool_names, initial_pool = frozen_pool(instance, cfg)
    initial_hash = array_hash(initial_pool)
    write_json(path, {"instance": name, "status": "training", "cutoff_h": instance.cutoff_h,
                      "candidate_pool_hash": initial_hash, "settings": cfg})
    risk = fit_risk(spec, data, instance, cfg)
    with cuda_lock(spec.artifact_dir / ".cuda.lock"):
        source = ScenarioSource(spec, data, instance, risk, cfg)
    pool, names = list(initial_pool), list(pool_names)
    random_pool = initial_pool[1:]
    selected, reports = {}, {}
    opt = cfg["optimization"]
    band = relative_band(float(data.sites.capacity.sum()), opt["minimum_final_group_kw"])
    for method in cfg["models"]:
        log(f"조건부 탐색 {name} {method}: 전체 {len(source.hours)}시간 시나리오 생성")
        e, weight = source.materialize(method, cfg["scenarios"]["search_samples_per_hour"], cfg["scenarios"]["search_seed"])
        with cuda_lock(spec.artifact_dir / ".cuda.lock"):
            obj = ScenarioObjective(e, weight, instance.capacity, 0.0, opt["swap_candidates"], opt["device"],
                                    source.count(method, cfg["scenarios"]["search_samples_per_hour"]), opt["center"])
            del e, weight
            primary, trace, frontier, frontier_parts, candidates, balance, lambdas = search_partitions(obj, instance, opt, band)
            feasibility, repairs = None, {}
            if balance["mode"] == "relative" and name == "test":
                # 각 λ̃ 해의 20 MW 판정과 복구를 그 λ̃의 목적으로 미리 계산한다. 적용할 λ̃*는 V·H·B를 본 뒤 정한다.
                for key, entry in frontier.items():
                    obj.lam = entry["lambda"]
                    repairs[key], entry["repair"] = restore_feasibility(obj, frontier_parts[entry["partition_index"]], instance.capacity,
                                                                        opt["minimum_final_group_kw"])
            elif name == "test":
                feasible = []
                for candidate in candidates:
                    repaired, repair_info = restore_feasibility(obj, candidate, instance.capacity, opt["minimum_final_group_kw"])
                    feasible.append((repaired, repair_info))
                values = source.score_partitions(method, np.stack([x for x, _ in feasible]),
                                                  cfg["scenarios"]["evaluation_samples_per_hour"], cfg["scenarios"]["evaluation_seed"],
                                                  opt["center"])
                best = int(np.argmin(values))
                primary, feasibility = feasible[best]
                feasibility["expected_W_after_rescore_krw"] = float(values[best])
                feasibility["selection"] = "minimum predicted W among feasible frontier/repair candidates"
            obj.close()
        frontier_path = directory / f"frontier-{name}-{method}.npz"
        np.savez_compressed(frontier_path, partitions=np.stack(frontier_parts), site_index=instance.sites, lambdas=np.asarray(lambdas))
        if balance["mode"] == "relative":
            for key, entry in frontier.items():
                entry["pool_index"] = len(pool)
                pool.append(frontier_parts[entry["partition_index"]])
                names.append(f"{method}@{key}")
                if name == "test":
                    entry["repaired_pool_index"] = len(pool)
                    pool.append(repairs[key])
                    names.append(f"{method}@{key}-repaired")
            reports[method] = {"lambda_frontier": frontier, "balance": balance, "frontier_partitions": str(frontier_path)}
        else:
            selected[method] = len(pool)
            pool.append(primary)
            names.append(f"selected_{method}")
            group = full_assignment(len(data.sites), data.sites.capacity, instance.sites, primary)
            partition = directory / f"partition-{name}-{method}.csv"
            write_partition(partition, data.sites, group)
            reports[method] = {"search": trace, "lambda_frontier": frontier, "balance": balance, "feasibility": feasibility,
                               "frontier_partitions": str(frontier_path),
                               "partition": str(partition), "selected_pool_index": selected[method]}
        log(f"조건부 탐색 {name} {method}: 후보 배정 생성")
    pool = np.stack(pool)
    np.save(directory / f"partitions-{name}.npy", pool, allow_pickle=False)
    # 후보 집합과 선택을 고정한 다음 평가기간 실제 잔차를 채점한다.
    plain_w = None if instance.eval_error is None else np.array([score_partition(instance, x)[0]["W_krw"] for x in pool])
    random_index = np.arange(1, len(initial_pool))
    for method in cfg["models"]:
        log(f"조건부 평가 {name} {method}: 독립 시나리오, 전체 기간")
        report = reports[method]
        if report["balance"]["mode"] == "relative":
            # λ̃*가 아직 없으므로 calibration은 original로 계산하고, 모든 λ̃ 해의 예상·실현 값을 남긴다.
            predicted, calibration, medians = evaluate_scenarios(source, method, pool, 0, instance, cfg)
            report.update(predicted_W_krw=predicted.tolist(), calibration=calibration, calibration_partition="original")
            entries = list(report["lambda_frontier"].values())
            index = [0] + [e["pool_index"] for e in entries] + [e["repaired_pool_index"] for e in entries if "repaired_pool_index" in e]
            grid_medians = directory / f"medians-grid-{name}-{method}.npz"
            np.savez_compressed(grid_medians, hours=source.hours, pool_index=np.asarray(index), medians=medians[:, :, index])
            report["grid_medians"] = str(grid_medians)
            hourly = None if instance.eval_error is None else realized_losses(instance, pool, medians, source.hours)
            for entry in entries:
                entry["predicted_W_krw"] = float(predicted[entry["pool_index"]])
                if "repaired_pool_index" in entry:
                    entry["repaired_predicted_W_krw"] = float(predicted[entry["repaired_pool_index"]])
                if hourly is not None:
                    entry["actual"] = compare_realized(instance, hourly, entry["pool_index"], random_index, cfg)
            if hourly is not None:
                realized = hourly.sum(axis=0)
                report["ranking_spearman_fixed_pool"] = float(spearmanr(predicted[:len(initial_pool)], realized[:len(initial_pool)]).statistic)
            continue
        predicted, calibration, medians = evaluate_scenarios(source, method, pool, selected[method], instance, cfg)
        report["predicted_W_krw"] = predicted.tolist()
        report["calibration"] = calibration
        i = selected[method]
        median_path = directory / f"medians-{name}-{method}.npz"
        np.savez_compressed(median_path, hours=source.hours, selected=medians[:, :, i], original=medians[:, :, 0])
        report["group_medians"] = str(median_path)
        if instance.eval_error is not None:
            hourly = realized_losses(instance, pool, medians, source.hours)
            realized = hourly.sum(axis=0)
            report["actual"] = compare_realized(instance, hourly, i, random_index, cfg)
            report["actual_plain_bottom_up"] = actual_comparison(instance, pool[i], random_pool, cfg)
            report["selection_gap_to_best_pool_krw"] = float(realized[i] - realized.min())
            report["ranking_spearman_fixed_pool"] = float(spearmanr(predicted[:len(initial_pool)], realized[:len(initial_pool)]).statistic)
            report["predicted_selected_W_krw"] = float(predicted[i])
    with cuda_lock(spec.artifact_dir / ".cuda.lock"):
        gaussian_validation = validate_gaussian(source, instance)
    result = {"instance": name, "status": "completed", "cutoff_h": int(instance.cutoff_h),
              "risk_key": risk.key, "risk_training": risk.metadata, "scenario": source.metadata,
              "gaussian_validation": gaussian_validation,
              "sites": len(instance.sites), "capacity_kw": instance.total, "forecast_hours": len(source.hours),
              "candidate_pool_hash": initial_hash, "comparison_pool_hash": array_hash(pool), "candidate_names": names,
              "plain_bottom_up_W_krw": plain_w, "models": reports, "seconds": time.perf_counter() - started,
              "scenario_samples": cfg["scenarios"], "balance_band": band,
              "finished": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    write_json(path, result)
    return result


LABELLED = ("V", "H", "B")


def select_balance(reports: dict) -> dict:
    """현재 선택 scope의 모든 방법에서 band를 만족하는 가장 작은 λ̃. 없으면 grid의 마지막 값을 쓴다."""
    names = [name for name in LABELLED if name in reports]
    if not names:
        raise ValueError("balance 선택에는 완료한 labelled instance가 필요합니다.")
    first = reports[names[0]]
    band = first["balance_band"]
    grid = sorted(float(k) for k in next(iter(first["models"].values()))["lambda_frontier"])
    table = {str(r): {f"{name}-{method}": abs(model["lambda_frontier"][str(r)]["imbalance"])
                      for name in names for method, model in reports[name]["models"].items()} for r in grid}
    worst = [max(table[str(r)].values()) for r in grid]
    chosen = grid[first_within_band(worst, band)]
    return {"relative": chosen, "band": band, "grid": grid, "instances": names,
            "band_satisfied": max(table[str(chosen)].values()) <= band + 1e-12,
            "max_abs_imbalance": dict(zip(map(str, grid), worst)), "abs_imbalance": table,
            "rule": "smallest relative balance weight satisfying the band for every method in the recorded instance scope"}


def choose_balance(reports: dict, directory) -> dict:
    record = select_balance(reports)
    record["applied_instances"] = list(reports)
    write_json(directory / "balance_selection.json", record)
    return record


def finalize_instance(report: dict, chosen: float, data, directory) -> dict:
    """λ̃* 해로 방법별 선택 배정·구성·실현 결과·test 복구를 확정한다."""
    name = report["instance"]
    key = str(chosen)
    pool = np.load(directory / f"partitions-{name}.npy")
    for method, model in report["models"].items():
        entry = model["lambda_frontier"][key]
        index = entry.get("repaired_pool_index", entry["pool_index"])
        x = pool[index]
        with np.load(model["frontier_partitions"]) as frontier:
            site_index = frontier["site_index"]
        partition = directory / f"partition-{name}-{method}.csv"
        write_partition(partition, data.sites, full_assignment(len(data.sites), data.sites.capacity, site_index, x))
        with np.load(model["grid_medians"]) as grid:
            position = {int(i): k for k, i in enumerate(grid["pool_index"])}
            median_path = directory / f"medians-{name}-{method}.npz"
            np.savez_compressed(median_path, hours=grid["hours"], selected=grid["medians"][:, :, position[index]],
                                original=grid["medians"][:, :, position[0]])
        model.update(selected_pool_index=int(index), search=entry["search"], partition=str(partition), group_medians=str(median_path),
                     balance={**model["balance"], "relative": chosen, "lambda": entry["lambda"], "qubo_lambda": entry["qubo_lambda"],
                              "band_satisfied": abs(entry["imbalance"]) <= model["balance"]["band"] + 1e-12},
                     predicted_selected_W_krw=model["predicted_W_krw"][index])
        if "actual" in entry:
            model["actual"] = entry["actual"]
        if "repair" in entry:
            model["feasibility"] = {**entry["repair"], "expected_W_after_rescore_krw": entry["repaired_predicted_W_krw"],
                                    "selection": "λ̃* solution; 20 MW repair when a group is below the minimum"}
    report["finalized_balance"] = chosen
    write_json(directory / f"{name}.json", report)
    return report


def choose_method(reports: dict) -> tuple[str, dict]:
    """정답이 있는 instance 전체에서 방법별 선택 배정의 실현 결과로 고른다.

    점수는 instance 용량 가중 평균 상대 ΔW = Σ C_τ·ΔW_τ / W_original,τ / Σ C_τ이다. 중간 단계에서 유의성 관문은 두지 않는다.
    """
    table = {}
    names = [name for name in LABELLED if name in reports]
    if not names:
        raise ValueError("method ranking에는 완료한 labelled instance가 필요합니다.")
    for method in reports[names[0]]["models"]:
        rows = {name: {"delta_W_krw": reports[name]["models"][method]["actual"]["delta_W_krw"],
                       "relative": reports[name]["models"][method]["actual"]["delta_W_krw"] / reports[name]["models"][method]["actual"]["original"]["W_krw"],
                       "bootstrap_7d": reports[name]["models"][method]["actual"]["bootstrap"]["7d"],
                       "capacity_kw": reports[name]["capacity_kw"]} for name in names}
        weight = sum(r["capacity_kw"] for r in rows.values())
        table[method] = {"instances": rows, "score": sum(r["capacity_kw"] * r["relative"] for r in rows.values()) / weight}
    return min(sorted(table), key=lambda m: table[m]["score"]), table


def freeze_method(reports: dict, directory, method: str | None = None) -> dict:
    recommended, table = choose_method(reports)
    if method is not None and method not in table:
        raise ValueError(f"선택한 method의 결과가 없습니다: {method}")
    names = [name for name in LABELLED if name in reports]
    record = {"method": method or recommended, "recommended_method": recommended, "choice": "explicit" if method else "ranked",
              "scores": table, "instances": names,
              "criterion": "capacity-weighted mean relative realized ΔW over the recorded instance scope",
              "reports_hash": {name: digest(reports[name]) for name in names}}
    write_json(directory / "selection.json", record)
    return record


def transfer_method(spec, report, selection, directory, cfg):
    period, method = report["instance"], selection["method"]
    path = directory / f"transfer-{period}-{method}.json"
    partition = report["models"][method]["partition"]
    model = cfg["primitive"]["model"]
    seeds = cfg["primitive"]["seeds"]
    runs = {"original": run(spec, period, "bundling", model=model, seeds=seeds),
            "candidate": run(spec, period, "bundling", model=model, seeds=seeds, partition=partition)}
    loaded, days = {}, {}
    for label, run_path in runs.items():
        manifest = read_json(run_path / "manifest.json")
        metrics = read_json(run_path / "report.json")["components"][manifest["cell"]]
        if manifest["execution_status"] != "completed" or manifest["evaluation_status"] != "evaluated":
            raise ValueError("그룹 재학습과 전체 기간 평가가 완료되어야 합니다.")
        loaded[label] = {"run": str(run_path), "W_krw": metrics["W_cohort_krw"], "revenue_krw": metrics["revenue_krw"],
                         "group_hours": metrics["group_hours"], "seeds": manifest["seeds"], "cohort_sites": manifest["cohort"]["site_ids"]}
        days[label] = np.load(run_path / "daily_penalty.npz")[manifest["cell"]].sum(1)
    if not np.isclose(loaded["original"]["revenue_krw"], loaded["candidate"]["revenue_krw"], rtol=0, atol=1e-3):
        raise ValueError("그룹 재학습 비교에서 실제 총 발전량 수익이 달라졌습니다.")
    result = {"period": period, "method": method, "partition": partition, "partition_sha": file_sha(Path(partition)),
              "primitive_model": model, "runs": loaded,
              "delta_W_krw": loaded["candidate"]["W_krw"] - loaded["original"]["W_krw"],
              "bootstrap": {f"{b}d": block_bootstrap(days["candidate"] - days["original"], b, cfg["evaluation"]["bootstrap_replicates"], 42)
                            for b in cfg["evaluation"]["bootstrap_days"]}}
    write_json(path, result)
    return result


def deployment_rule(group_delta_total_krw: float, test_capacity_kw, minimum_kw: float, primitive_model: str, champion: str) -> dict:
    """평가한 model·partition pair의 추천. Champion 일치는 context이며 추천 조건과 분리한다."""
    feasible = bool(min(test_capacity_kw) >= minimum_kw - 1e-8)
    is_champion = primitive_model == champion
    return {"deploy": bool(group_delta_total_krw < 0 and feasible), "test_capacity_valid": feasible,
            "primitive_is_champion": is_champion}


def verify_decision(decision: dict, group_delta_total_krw: float, test_capacity_kw, minimum_kw: float, primitive_model: str,
                    champion: str) -> None:
    """기록 당시의 context로 추천을 재계산한다. 현재 champion 변경은 과거 계산의 유효성과 별개이다."""
    if decision["primitive_model"] != primitive_model:
        raise AssertionError(f"판정 기록의 primitive {decision['primitive_model']}이 실험 설정의 {primitive_model}과 다릅니다.")
    if abs(group_delta_total_krw - decision["group_delta_W_total_krw"]) >= 0.02:
        raise AssertionError("그룹 단위 합산 ΔW가 기록과 다릅니다.")
    expected = deployment_rule(group_delta_total_krw, test_capacity_kw, minimum_kw, primitive_model, decision["champion"])
    if decision.get("rule_version", 1) == 1:
        expected["deploy"] = expected["deploy"] and expected["primitive_is_champion"]
    recorded = {key: decision[key] for key in expected}
    if recorded != expected:
        raise AssertionError(f"배포 판정이 규칙과 다릅니다: 기록 {recorded}, 재계산 {expected}")


def decide(spec, reports: dict, transfers: dict, selection: dict, directory, cfg: dict) -> dict:
    """명시적으로 요청한 model·partition pair의 추천을 기록한다. Registry와 제출 파일은 변경하지 않는다."""
    method, primitive_model = selection["method"], cfg["primitive"]["model"]
    group_total = sum(transfers[name]["delta_W_krw"] for name in ("H", "B"))
    test = reports["test"]["models"][method]
    capacity = test["feasibility"]["capacity_kw"]
    rule = deployment_rule(group_total, capacity, cfg["optimization"]["minimum_final_group_kw"], primitive_model, spec.champion)
    deploy = rule["deploy"]
    evidence = {name: {"bottom_up_delta_W_krw": reports[name]["models"][method]["actual"]["delta_W_krw"],
                       "bottom_up_bootstrap_7d": reports[name]["models"][method]["actual"]["bootstrap"]["7d"],
                       "group_delta_W_krw": transfers[name]["delta_W_krw"], "group_bootstrap_7d": transfers[name]["bootstrap"]["7d"],
                       "transfer_ratio": (transfers[name]["delta_W_krw"] / reports[name]["models"][method]["actual"]["delta_W_krw"]
                                          if reports[name]["models"][method]["actual"]["delta_W_krw"] else None)}
                for name in ("H", "B")}
    result = {"rule_version": 2, "method": method, "rule": "recommend the evaluated model-partition pair when H+B group ΔW is negative and test capacity is feasible",
              "model_definition": spec.model(primitive_model), "transfer_records": transfers, "selection": selection,
              "primitive_model": primitive_model, "champion": spec.champion, "primitive_is_champion": rule["primitive_is_champion"],
              "group_delta_W_total_krw": group_total, "evidence": evidence, "deploy": deploy,
              "deployment": "candidate" if deploy else "original", "candidate_partition": test["partition"],
              "test_capacity_kw": capacity, "test_capacity_valid": rule["test_capacity_valid"],
              "selection_scope": selection.get("instances", []), "selection_choice": selection.get("choice", "explicit")}
    write_json(directory / "decision.json", result)
    return result


def required_forecast_periods(spec, instances) -> list[str]:
    """요청한 instance의 estimation·evaluation에 필요한 site forecast만 반환한다."""
    needed = set()
    for name in instances:
        item = spec.protocol["bundling_search"]["instances"][name]
        needed.update(item["estimation"])
        if item["evaluation"]:
            needed.add(item["evaluation"])
    return [name for name in ("D", "V", "H", "B") if name in needed]


def completed_reports(directory: Path, names) -> dict:
    reports = {}
    for name in names:
        path = Path(directory) / f"{name}.json"
        if path.exists():
            report = read_json(path)
            if report.get("status") == "completed":
                reports[name] = report
    return reports


def recorded_spec(spec, experiment: dict):
    """실험에 저장한 model·budget·period를 후속 transfer와 audit에 사용한다."""
    registry, protocol, config = deepcopy(spec.registry), deepcopy(spec.protocol), deepcopy(spec.config)
    name = experiment["config"]["primitive"]["model"]
    registry["models"][name] = experiment["primitive_model"]
    protocol["periods"] = experiment["periods"]
    protocol["hyperparameters"] = experiment.get("forecast_budgets", protocol["hyperparameters"])
    protocol["output"] = experiment.get("forecast_output", protocol["output"])
    protocol["blending"] = experiment.get("forecast_blending", protocol["blending"])
    protocol["aux_lambda"] = experiment.get("forecast_aux_lambda", protocol["aux_lambda"])
    protocol["bundling_search"] = experiment.get("upstream_bundling", protocol["bundling_search"])
    config.update(experiment.get("forecast_settings", {}))
    # 출처 hash는 보존하고, run identity는 복원한 설정에서 계산한 model cache key로 정한다.
    return replace(spec, registry=registry, protocol=protocol, config=config,
                   settings_source={"kind": "recorded_experiment", "key": experiment.get("key")})


def transfer_from_directory(spec, directory: Path, period: str, method: str | None = None) -> dict:
    """완료한 labelled instance 하나의 group transfer를 독립적으로 실행한다."""
    directory = Path(directory)
    experiment = read_json(directory / "experiment.json")
    frozen_spec = recorded_spec(spec, experiment)
    reports = completed_reports(directory, [period])
    if period not in LABELLED or period not in reports:
        raise ValueError(f"완료한 labelled instance가 필요합니다: {period}")
    report = reports[period]
    selection_path = directory / "selection.json"
    if method is None and selection_path.exists():
        method = read_json(selection_path)["method"]
    if "balance_grid" in experiment["config"]["optimization"] and "finalized_balance" not in report:
        balance = choose_balance(reports, directory)
        report = finalize_instance(report, balance["relative"], load_data(frozen_spec), directory)
        freeze_method({period: report}, directory, method)
    if method is None:
        method, _ = choose_method({period: report})
    if method not in report["models"]:
        raise ValueError(f"{period}에 method 결과가 없습니다: {method}")
    return transfer_method(frozen_spec, report, {"method": method}, directory, experiment["config"])


def read_transfer(directory: Path, period: str, method: str, report: dict | None = None) -> dict:
    candidates = (Path(directory) / f"transfer-{period}-{method}.json", Path(directory) / f"transfer-{period}.json")
    for path in candidates:
        if path.exists():
            value = read_json(path)
            if value["method"] != method:
                continue
            if report is not None and value.get("partition_sha") != file_sha(Path(report["models"][method]["partition"])):
                raise ValueError(f"{period} {method}의 현재 partition에 대한 transfer 결과가 필요합니다.")
            return value
    raise FileNotFoundError(f"{period} {method}의 transfer 결과가 없습니다.")


def decide_from_directory(spec, directory: Path, method: str | None = None) -> dict:
    """저장된 H·B transfer와 test 구성으로 명시적인 추천을 만든다. 학습은 수행하지 않는다."""
    directory = Path(directory)
    experiment = read_json(directory / "experiment.json")
    reports = completed_reports(directory, ["H", "B", "test"])
    if set(reports) != {"H", "B", "test"}:
        raise ValueError("이 추천에 필요한 H·B·test 결과가 없습니다.")
    selection = None
    if method is None:
        path = directory / "selection.json"
        if path.exists():
            selection = read_json(path)
            method = selection["method"]
        else:
            method, scores = choose_method(reports)
            selection = {"method": method, "choice": "ranked", "instances": ["H", "B"], "scores": scores}
    if selection is None:
        selection = {"method": method, "choice": "explicit", "instances": []}
    if any(method not in report["models"] for report in reports.values()):
        raise ValueError(f"{method}의 H·B·test 결과가 필요합니다.")
    transfers = {name: read_transfer(directory, name, method, reports[name]) for name in ("H", "B")}
    if "balance_grid" in experiment["config"]["optimization"] and "finalized_balance" not in reports["test"]:
        balance_path = directory / "balance_selection.json"
        balance = read_json(balance_path) if balance_path.exists() else {
            **select_balance({n: reports[n] for n in LABELLED if n in reports}), "applied_instances": []}
        reports["test"] = finalize_instance(reports["test"], balance["relative"], load_data(recorded_spec(spec, experiment)), directory)
        balance["applied_instances"] = list(dict.fromkeys([*balance.get("applied_instances", balance["instances"]), "test"]))
        write_json(balance_path, balance)
    # 최종 선택의 partition은 후속 exploratory run의 파일 갱신과 분리한다.
    reports = deepcopy(reports)
    source = Path(reports["test"]["models"][method]["partition"])
    snapshot = directory / f"selected-partition-{file_sha(source)[:16]}.csv"
    if not snapshot.exists():
        snapshot.write_bytes(source.read_bytes())
    reports["test"]["models"][method]["partition"] = str(snapshot)
    return decide(recorded_spec(spec, experiment), reports, transfers, selection, directory, experiment["config"])


def execute(spec, instances=None, transfer=False, settings_path="configs/conditional_bundling.json", method: str | None = None):
    cfg = read_json(spec.root / settings_path)
    instances = list(instances or cfg["instances"])
    if any(name not in cfg["instances"] for name in instances):
        raise ValueError(instances)
    configure_determinism(spec.config["runtime"]["torch_threads"])
    data = load_data(spec)
    # bundling 잔차는 제출 모델과 같은 primitive(발전소 단위 XGB+DL 블렌드)를 기준으로 정의한다.
    forecast_periods = required_forecast_periods(spec, instances)
    directory = experiment_dir(spec, data, {}, cfg)
    state_path = directory / "execution.json"
    state = {"status": "running", "pid": os.getpid(), "instances": instances, "forecast_periods": forecast_periods,
             "transfer_requested": transfer,
             "started": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    write_json(state_path, state)
    try:
        forecasts = {name: site_primitive(spec, data, name, cfg["primitive"]) for name in forecast_periods}
        reports = {}
        with threadpool_limits(limits=4):
            for name in instances:
                state["step"] = f"instance-{name}"
                write_json(state_path, state)
                run_instance(spec, data, forecasts, name, cfg, directory)
                reports[name] = read_json(directory / f"{name}.json")
        reports = {name: report for name, report in reports.items() if report.get("status") == "completed"}
        labelled = any(name in reports for name in LABELLED)
        if "balance_grid" in cfg["optimization"] and labelled:
            balance = choose_balance(reports, directory)
            for name in list(reports):
                reports[name] = finalize_instance(reports[name], balance["relative"], data, directory)
        elif "balance_grid" in cfg["optimization"] and (directory / "balance_selection.json").exists():
            balance = read_json(directory / "balance_selection.json")
            for name in reports:
                reports[name] = finalize_instance(reports[name], balance["relative"], data, directory)
            balance["applied_instances"] = list(dict.fromkeys([*balance.get("applied_instances", balance["instances"]), *reports]))
            write_json(directory / "balance_selection.json", balance)
        selection = freeze_method(reports, directory, method) if labelled else None
        transfers = {}
        if transfer:
            for name in (n for n in instances if n in LABELLED):
                state["step"] = f"group-transfer-{name}"
                write_json(state_path, state)
                transfers[name] = transfer_method(spec, reports[name], selection, directory, cfg)
        result = {"directory": str(directory), "completed_instances": list(reports), "forecast_periods": forecast_periods,
                  "selection": selection, "transfers": transfers, "decision": None}
        state.update(status="completed_command", completed_instances=list(reports), transferred_periods=list(transfers),
                     finished=pd.Timestamp.now(tz="Asia/Seoul").isoformat())
        write_json(state_path, state)
        return result
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                     finished=pd.Timestamp.now(tz="Asia/Seoul").isoformat())
        write_json(state_path, state)
        raise


def audit_transfer(result: dict, definition: dict, wide, price) -> dict:
    """저장된 group forecast와 generation·SMP로 transfer 금액을 재계산한다."""
    values = {}
    for label, entry in result["runs"].items():
        manifest = read_json(Path(entry["run"]) / "manifest.json")
        assert manifest["execution_status"] == "completed" and manifest["evaluation_status"] == "evaluated"
        assert manifest["model"] == definition
        forecast = pd.read_csv(Path(entry["run"]) / "validation-forecast.csv")
        forecast.ts = pd.to_datetime(forecast.ts)
        total_w = 0.0
        for group in ("G1", "G2"):
            part = forecast[forecast.group == group].set_index("ts").forecast_kwh
            ids = manifest["cohort"]["site_ids"][group]
            y = wide.reindex(index=part.index, columns=ids).sum(axis=1, min_count=len(ids))
            mask = y.notna()
            k = np.where((part.index.hour >= 11) & (part.index.hour <= 15), 3, 1)
            w = 0.1 * price.reindex(part.index).to_numpy() * k
            assert np.isfinite(w).all()
            total_w += float((np.abs(part[mask] - y[mask]) * w[mask]).sum())
        assert abs(total_w - entry["W_krw"]) < 0.01
        values[label] = total_w
    assert abs(values["candidate"] - values["original"] - result["delta_W_krw"]) < 0.01
    return values


def audit(spec, directory: Path):
    directory = Path(directory)
    experiment = read_json(directory / "experiment.json")
    cfg = experiment["config"]
    spec = recorded_spec(spec, experiment)
    reports = completed_reports(directory, cfg["instances"])
    if not reports:
        raise ValueError("검증할 완료된 instance가 없습니다.")
    sites = pd.read_csv(spec.data_dir / "sites.csv")
    generation = pd.read_parquet(spec.data_dir / "train/generation.parquet")
    valid = np.isfinite(generation.energy) & generation.energy.ge(0)
    first = generation.loc[valid].groupby("site_id").ts.min()
    generation["energy"] = generation.energy.where(valid)
    wide = generation.pivot(index="ts", columns="site_id", values="energy")
    price = pd.read_csv(spec.data_dir / "train/smp.csv")
    price.ts = pd.to_datetime(price.ts)
    price = price.set_index("ts").smp
    evidence = {}
    for name, report in reports.items():
        period = spec.period(name)
        assert report["forecast_hours"] == period.expected_hours
        assert report["cutoff_h"] == period.cutoff_h
        lineage = report["risk_training"]["lineage"]
        assert lineage["last_target_h"] <= period.cutoff_h
        assert report["risk_training"]["steps"] == cfg["risk"]["steps"]
        assert "RTX 2000 Ada" in report["risk_training"]["environment"]["gpu"]["name"]
        analog = report["scenario"]["analog"]
        assert analog["last_analog_hour"] <= period.cutoff_h
        known = sites.site_id.isin(first.index[first.le(period.fit_cutoff)]).to_numpy() if name != "test" else np.ones(len(sites), bool)
        assert int(known.sum()) == report["sites"]
        pool = np.load(directory / f"partitions-{name}.npy")
        assert array_hash(pool) == report["comparison_pool_hash"]
        expected_initial_count = cfg["evaluation"]["candidate_pool_size"] + 1
        assert array_hash(pool[:expected_initial_count]) == report["candidate_pool_hash"]
        assert pool.shape == (len(report["candidate_names"]), int(known.sum()))
        if name != "test":
            assert np.isfinite(report["gaussian_validation"]["mean_nll_per_observed_pv"])
            assert report["gaussian_validation"]["observed_site_hours"] > 0
        cap = sites.loc[known, "capacity_kw"].to_numpy()
        checks = {"hours": period.expected_hours, "sites": int(known.sum()), "cutoff": str(period.fit_cutoff),
                  "risk_steps": report["risk_training"]["steps"], "candidate_hash_valid": True, "models": {}}
        for method, result in report["models"].items():
            if "selected_pool_index" not in result:
                checks["models"][method] = {"selection_complete": False, "verified_scope": "cutoff and candidate pool"}
                continue
            selected = pool[result["selected_pool_index"]]
            partition = pd.read_csv(result["partition"]).set_index("site_id")
            assert len(partition) == 253 and partition.index.is_unique
            assert set(partition.index) == set(sites.site_id)
            group = partition.reindex(sites.site_id).group.to_numpy() == "G1"
            assert np.array_equal(group[known], selected)
            with np.load(result["frontier_partitions"]) as frontier:
                for lam, entry in result["lambda_frontier"].items():
                    x = frontier["partitions"][entry["partition_index"]]
                    balance = (cap[x].sum() - cap[~x].sum()) / cap.sum()
                    assert np.isclose(balance, entry["imbalance"], atol=1e-12)
            if name == "test":
                total_cap = [float(sites.capacity_kw[group].sum()), float(sites.capacity_kw[~group].sum())]
                assert min(total_cap) >= cfg["optimization"]["minimum_final_group_kw"] - 1e-8
                checks["models"][method] = {"deployment_capacity_kw": total_cap, "feasible": True}
                continue
            source = experiment["site_point_forecasts"][name]
            yhat = np.load(spec.artifact_dir / "cache" / source["kind"] / source["key"][:16] / "yhat.npy", mmap_mode="r")[:, known]
            timestamps = pd.date_range(period.start, period.end, freq="h")
            y = wide.reindex(index=timestamps, columns=sites.loc[known, "site_id"]).to_numpy()
            actual_mask = np.isfinite(y).all(axis=1)
            error = y[actual_mask] - yhat[actual_mask]
            hours = timestamps[actual_mask]
            smp = price.reindex(hours).to_numpy()
            assert np.isfinite(smp).all()
            k = np.where((hours.hour >= 11) & (hours.hour <= 15), 3, 1)
            with np.load(result["group_medians"]) as stored_medians:
                medians = stored_medians["selected"][:, actual_mask]       # 사전 시나리오 그룹 중앙값 [그룹, 시각]
            loss = 0.1 * smp * k * (np.abs(error[:, selected].sum(1) - medians[0]) + np.abs(error[:, ~selected].sum(1) - medians[1]))
            raw_w = float(loss.sum())
            stored_w = result["actual"]["selected"]["W_krw"]
            # 발전소 점예측 cache는 FP32이다. 원자료를 FP64로 다시 합한 금액과 1원 이내이어야 한다.
            assert abs(raw_w - stored_w) < 1.0, (name, method, raw_w - stored_w)
            assert int(actual_mask.sum()) == result["actual"]["evaluation_hours"]
            checks["models"][method] = {"raw_generation_recomputed_W_krw": raw_w, "reported_W_krw": stored_w,
                                       "difference_krw": raw_w - stored_w, "valid_hours": int(actual_mask.sum())}
        evidence[name] = checks
    if "balance_grid" in cfg["optimization"] and (directory / "balance_selection.json").exists():
        recorded = read_json(directory / "balance_selection.json")
        scope = recorded.get("instances", list(LABELLED))
        assert recorded["relative"] == select_balance({name: reports[name] for name in scope})["relative"]
        for name in recorded.get("applied_instances", scope):
            report = reports[name]
            assert report["finalized_balance"] == recorded["relative"]
            for method, result in report["models"].items():
                entry = result["lambda_frontier"][str(recorded["relative"])]
                assert result["balance"]["relative"] == recorded["relative"]
                assert result["selected_pool_index"] == entry.get("repaired_pool_index", entry["pool_index"])
    selection = None
    if (directory / "selection.json").exists():
        selection = read_json(directory / "selection.json")
        scope = selection.get("instances", list(LABELLED))
        assert selection["reports_hash"] == {name: digest(reports[name]) for name in scope}
        chosen, table = choose_method({name: reports[name] for name in scope})
        assert selection["method"] in table
        if selection.get("choice", "ranked") == "ranked":
            assert selection["method"] == chosen
    transfers = {}
    primitive_signature = model_signature(spec, cfg["primitive"]["model"])
    for path in sorted(directory.glob("transfer-*.json")):
        result = read_json(path)
        transfers[path.stem] = audit_transfer(result, primitive_signature, wide, price)
    decision = None
    if (directory / "decision.json").exists():
        decision = read_json(directory / "decision.json")
        stored = decision.get("transfer_records") or {name: read_transfer(directory, name, decision["method"]) for name in ("H", "B")}
        values = {name: audit_transfer(stored[name], primitive_signature, wide, price) for name in ("H", "B")}
        group_total = sum(v["candidate"] - v["original"] for v in values.values())
        partition = pd.read_csv(decision["candidate_partition"]).set_index("site_id")
        assert partition.index.is_unique and set(partition.index) == set(sites.site_id)
        group = partition.reindex(sites.site_id).group.to_numpy()
        assert set(group) == {"G1", "G2"}
        capacity = [float(sites.loc[group == g, "capacity_kw"].sum()) for g in ("G1", "G2")]
        verify_decision(decision, group_total, capacity,
                        cfg["optimization"]["minimum_final_group_kw"], cfg["primitive"]["model"], spec.champion)
    output = {"status": "passed", "source": "raw generation + raw SMP + saved site point forecasts / group forecast CSV",
              "instances": evidence, "selection": selection, "group_transfer_recomputed_W_krw": transfers,
              "adoption_consistent": True if decision is not None else None, "decision": decision,
              "scope": {"instances": list(reports), "pending_instances": [name for name in cfg["instances"] if name not in reports],
                        "transfers": list(transfers), "decision_checked": decision is not None}}
    write_json(directory / "verification.json", output)
    return output
