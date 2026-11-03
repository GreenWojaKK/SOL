"""명령행 진입점. 명령마다 필요한 모듈을 불러 실행한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import FOLDS, GRID_LOSSES, INPUTS, LOSSES, season_settings
from .store import _plain


def _print(value) -> None:
    print(json.dumps(_plain(value), ensure_ascii=False, indent=2, default=str), flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m solarvpp", description="태양광 VPP 발전량 예측과 G1·G2 배정")
    parser.add_argument("--config", default="configs/runtime.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("env", help="실행 환경을 출력한다")
    sub.add_parser("audit", help="base·weather cache를 만들고 QA를 보고한다")
    sub.add_parser("cohorts", help="D/V/H panel을 만들고 cohort evidence와 대조한다")
    q = sub.add_parser("queries", help="cutoff별 학습 query 수를 계산한다")
    q.add_argument("--period", required=True, choices=["D", "V", "H", "test"])
    levels = ["smoke", "diagnostic", "noise_floor", "screening", "confirmation", "retrospective", "bundling", "submission"]
    r = sub.add_parser("run", help="한 모델·칸의 학습·예측·후처리·평가 (기본: champion MK 버전, noise_floor 수준은 grid 기준 칸)")
    r.add_argument("--model", help="base MK 버전. 명시한 override는 별도 run 설정으로 기록한다")
    r.add_argument("--recipe", help="protocol의 forecast recipe (weighted-day-energy: XGB forecast-physics + DL forecast-physics-daystate·day-energy·settlement)")
    r.add_argument("--input", choices=list(INPUTS), help="두 expert의 input override. 생략하면 base 설정을 쓴다")
    r.add_argument("--xgb-input", choices=list(INPUTS), help="XGB expert에만 적용하는 input override")
    r.add_argument("--dl-input", choices=list(INPUTS), help="DL expert에만 적용하는 input override")
    r.add_argument("--xgb-weight", type=float, help="weighted-quantile의 XGB 비율 [0, 1]")
    r.add_argument("--site-aux-ratio", type=float, help="발전소 이용률 보조 head의 gradient 비율(0이면 head 없음). 생략하면 base MK의 값을 쓴다")
    r.add_argument("--loss", choices=list(LOSSES), help="DL loss override. 생략하면 base 설정을 쓴다")
    r.add_argument("--aux-ratio", type=float, help="보조항 세기 비율. 생략하면 base MK·recipe의 aux_ratio를 사용한다")
    r.add_argument("--period", required=True, choices=["D", "V", "H", "B", "test", *FOLDS])
    r.add_argument("--level", required=True, choices=levels)
    r.add_argument("--seeds", type=int, nargs="+")
    r.add_argument("--experts", nargs="+", default=["xgb", "dl"], choices=["xgb", "dl"])
    r.add_argument("--use-cache", action="store_true", help="설정을 바꾼 run의 학습 cache도 저장한다(기본은 임시 폴더). 환경 변수 VPP_EXPERIMENT_CACHE=1도 같다")
    r.add_argument("--selection", help="명시한 경우에만 frozen model·settings 일치를 검증한다")
    r.add_argument("--partition", type=Path, help="그룹 배정을 바꿀 partition CSV(site_id, group). 생략하면 original")
    r.add_argument("--time-weight", choices=["none", "settlement"], help="DL pinball·day-energy의 time weight. XGB도 이 값을 기본으로 쓴다")
    r.add_argument("--xgb-time-weight", choices=["none", "settlement"], help="XGB time weight override. --time-weight보다 우선 적용한다")
    r.add_argument("--curtailment", action="store_true", help="blend 뒤에 출력제어 후처리 module을 붙인다(설정은 runtime config의 curtailment)")
    r.add_argument("--curtailment-half-life", type=float, help="출력제어 module의 최근 보정 반감기(일). 주면 module을 붙인다")
    r.add_argument("--season", choices=["spring", "summer", "autumn", "winter"], help="계절 학습의 대상 계절(대상일 인도월 기준)")
    r.add_argument("--season-strategy", choices=["ft", "weighted", "only"],
                   help="계절 학습 방식: ft(Global 모델에서 이어 학습), weighted(대상 계절 가중), only(대상 계절만 학습)")
    r.add_argument("--season-weight", type=float, help="weighted의 대상 계절 가중(기본 3)")
    r.add_argument("--ft-lr", type=float, help="ft의 DL 최대 학습률(기본 0.0001)")
    r.add_argument("--xgb-ft-rounds", type=int, help="ft의 XGB 추가 boosting round(기본 100)")
    r.add_argument("--xgb-ft-learning-rate", type=float, help="ft의 XGB 학습률(기본 0.02)")
    r.add_argument("--season-margin", type=int, help="학습 계절을 앞뒤로 넓힐 개월 수(기본 0, 평가 계절은 그대로)")
    r.add_argument("--season-steps", choices=["scaled", "full"], help="ft·only의 DL step 예산(scaled: T_s, full: Global 예산)")
    r.add_argument("--season-experts", nargs="+", choices=["xgb", "dl"], help="계절 학습을 적용할 expert(기본 둘 다, 나머지는 Global 모델)")
    g = sub.add_parser("grid", help="입력×손실 칸을 순서대로 실행한다")
    g.add_argument("--period", default="V", choices=["D", "V", "H", "B"])
    g.add_argument("--level", default="screening", choices=["smoke", "screening", "confirmation"])
    g.add_argument("--inputs", nargs="+", default=list(INPUTS), choices=list(INPUTS))
    g.add_argument("--losses", nargs="+", default=list(GRID_LOSSES), choices=list(LOSSES))
    g.add_argument("--aux-ratio", type=float)
    g.add_argument("--seeds", type=int, nargs="+")
    a = sub.add_parser("grid-report", help="완료한 screening 결과를 ranking한다. reference·noise floor는 있을 때 함께 표시한다")
    a.add_argument("--period", default="V", choices=["D", "V", "H", "B"])
    a.add_argument("--aux-ratio", type=float)
    a.add_argument("--confirm", nargs="*", default=[], help="confirmation 결과를 함께 정리할 칸 이름")
    cb = sub.add_parser("conditional-bundling", help="조건부 공동 오차·시나리오 최적화·연간 nested 평가")
    cb.add_argument("--instances", nargs="+", choices=["V", "H", "B", "test"])
    cb.add_argument("--settings", default="configs/conditional_bundling.json")
    cb.add_argument("--transfer", action="store_true", help="요청한 labelled instance의 group transfer도 실행한다")
    cb.add_argument("--method", help="group transfer에 사용할 method. 생략하면 요청한 instance scope에서 ranking한다")
    ct = sub.add_parser("conditional-transfer", help="완료한 instance 하나의 group transfer")
    ct.add_argument("--experiment", type=Path)
    ct.add_argument("--period", required=True, choices=["V", "H", "B"])
    ct.add_argument("--method")
    cd = sub.add_parser("conditional-decide", help="저장된 H·B transfer와 test 구성으로 model·partition pair를 추천한다")
    cd.add_argument("--experiment", type=Path)
    cd.add_argument("--method")
    ca = sub.add_parser("conditional-audit", help="원본 generation·SMP로 조건부 실험과 그룹 transfer 금액을 독립 재계산")
    ca.add_argument("--experiment", type=Path, help="실험 디렉터리. 생략하면 산출물 디렉터리의 conditional/latest.json 위치")
    e = sub.add_parser("evaluate", help="저장된 분위수를 다시 채점한다")
    e.add_argument("--run", required=True, type=Path)
    c = sub.add_parser("compare", help="같은 evaluation panel의 두 run을 비교한다")
    c.add_argument("--runs", required=True, nargs=2, type=Path, metavar=("CANDIDATE", "BASELINE"))
    s = sub.add_parser("select", help="같은 evaluation panel의 두 run으로 비교 snapshot을 만든다")
    s.add_argument("--baseline", required=True, type=Path)
    s.add_argument("--finalist", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    vs = sub.add_parser("verify-selection", help="저장된 run과 비교 snapshot의 설정 일치를 검증한다")
    vs.add_argument("--run", required=True, type=Path)
    vs.add_argument("--selection", required=True)
    adoption = sub.add_parser("adopt", help="MK를 명시적으로 champion으로 선택한다. --run은 pipeline이 바뀐 새 MK 정의의 원천이다")
    adoption.add_argument("--model", required=True)
    adoption.add_argument("--run", type=Path)
    sub.add_parser("check-test-inputs", help="test 입력 분포를 학습 마지막 1년과 비교한다")
    m = sub.add_parser("submit", help="제출 파일을 만들고 검사한다")
    m.add_argument("--run", required=True, type=Path)
    m.add_argument("--acknowledge", nargs="*", default=[], help="확인한 분포 표시(channel:kind, wildcard 가능)")
    m.add_argument("--dry-run", action="store_true")
    m.add_argument("--strict-distribution", action="store_true", help="distribution flag에도 acknowledgement를 요구한다")
    m.add_argument("--evidence", type=Path, nargs="*", default=[], help="gate 근거: vppcontract compare 보고서(JSON)")
    ar = sub.add_parser("artifacts", help="run·조건부 실험 기록의 상태(current·superseded·replaced·dead·aborted·running)를 보고한다")
    ar.add_argument("--archive", action="store_true", help="replaced·dead·aborted 기록을 package 산출물 디렉터리의 archive/로 옮기고 index에 적는다")
    ix = sub.add_parser("index-runs", help="완료 run을 계산 identity로 연결한다. 학습하거나 기존 산출물을 변경하지 않는다")
    ix.add_argument("--write", action="store_true", help="run-index/에 기존 run 연결을 저장한다")
    fold_periods = ["V", *FOLDS, "H"]
    fc = sub.add_parser("fold-stratified-partition", help="robust fold test의 fold별 층화 배정을 쓴다. 이미 있는 파일은 덮어쓰지 않고 cohort 배정을 대조한다")
    fc.add_argument("--periods", nargs="+", default=fold_periods, choices=fold_periods)
    fp = sub.add_parser("fold-partition", help="fold cutoff 이전 정보로 배정을 탐색해 partitions/search-<method>-<fold>.csv를 쓴다")
    fp.add_argument("--periods", nargs="+", default=fold_periods, choices=fold_periods)
    fp.add_argument("--settings", default="configs/fold_partition.json")
    fp.add_argument("--methods", nargs="+", choices=["unconditional", "lowrank", "analog"], help="시나리오 방법. 생략하면 설정의 methods")
    fr = sub.add_parser("fold-report", help="robust fold test에서 후보를 champion과 비교해 네 조건을 판정한다")
    target = fr.add_mutually_exclusive_group(required=True)
    target.add_argument("--candidates", nargs="+", choices=list(INPUTS), help="후보 DL 입력 구성(층화 배정)")
    target.add_argument("--partitions", nargs="+", help="후보 배정 이름 label(champion 모델, 배정 파일 <label>-<fold>.csv)")
    fr.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    fr.add_argument("--level", default="confirmation", choices=["screening", "confirmation"],
                    help="비교할 run의 level. screening은 seed 하나의 선별 비교이다")
    args = parser.parse_args(argv)

    from . import experiment
    from .config import load_spec
    spec = load_spec(args.config)
    if args.command == "env":
        from .store import environment
        _print(environment())
    elif args.command == "audit":
        _print(experiment.audit(spec))
    elif args.command == "cohorts":
        _print(experiment.cohorts(spec))
    elif args.command == "queries":
        _print(experiment.query_counts(spec, args.period))
    elif args.command == "run":
        season = None
        if args.season or args.season_strategy:
            if not (args.season and args.season_strategy):
                parser.error("계절 학습은 --season과 --season-strategy를 함께 줍니다.")
            season = season_settings(args.season, args.season_strategy, weight=args.season_weight, ft_lr=args.ft_lr,
                                     xgb_ft_rounds=args.xgb_ft_rounds, xgb_ft_learning_rate=args.xgb_ft_learning_rate,
                                     margin_months=args.season_margin, steps=args.season_steps, experts=args.season_experts)
        print(experiment.run(spec, args.period, args.level, input_name=args.input, loss_name=args.loss, aux_ratio=args.aux_ratio, seeds=args.seeds,
                             experts=tuple(args.experts), selection=args.selection, partition=args.partition, time_weight=args.time_weight,
                             xgb_input_name=args.xgb_input, xgb_time_weight=args.xgb_time_weight, recipe=args.recipe, model=args.model,
                             xgb_weight=args.xgb_weight, dl_input_name=args.dl_input, curtailment=args.curtailment, site_aux_ratio=args.site_aux_ratio,
                             use_cache=args.use_cache,
                             curtailment_half_life=args.curtailment_half_life, season=season))
    elif args.command == "grid":
        from .experiment import run_grid
        for path in run_grid(spec, args.period, args.level, args.inputs, args.losses, args.aux_ratio, args.seeds):
            print(path)
    elif args.command == "grid-report":
        from .experiment import analyze_confirmation, analyze_screening, write_report
        result = {"screening": analyze_screening(spec, args.period, args.aux_ratio)}
        if args.confirm:
            result["confirmation"] = analyze_confirmation(spec, args.confirm, args.period)
        write_report(spec, f"grid-{args.period}", result)
        _print({k: v for k, v in result["screening"].items() if k not in ("cells", "reference", "noise_floor")})
    elif args.command == "conditional-bundling":
        from .bundling.pipeline import execute
        _print(execute(spec, args.instances, args.transfer, args.settings, args.method))
    elif args.command in ("conditional-transfer", "conditional-decide"):
        from .bundling.pipeline import decide_from_directory, transfer_from_directory
        from .store import read_json
        directory = args.experiment or Path(read_json(spec.artifact_dir / "conditional/latest.json")["directory"])
        if args.command == "conditional-transfer":
            _print(transfer_from_directory(spec, directory, args.period, args.method))
        else:
            _print(decide_from_directory(spec, directory, args.method))
    elif args.command == "conditional-audit":
        from .bundling.pipeline import audit
        from .store import read_json
        directory = args.experiment or Path(read_json(spec.artifact_dir / "conditional/latest.json")["directory"])
        result = audit(spec, directory)
        _print({"status": result["status"], "instances": list(result["instances"]),
                "method": (result["selection"] or {}).get("method"), "deploy": (result["decision"] or {}).get("deploy"),
                "scope": result["scope"],
                "verification": str(directory / "verification.json")})
    elif args.command == "evaluate":
        _print(experiment.reevaluate(spec, args.run))
    elif args.command == "compare":
        _print(experiment.compare_runs(spec, *args.runs))
    elif args.command == "select":
        _print(experiment.select(spec, args.baseline, args.finalist, args.out))
    elif args.command == "verify-selection":
        from .validation import verify_run_selection
        _print(verify_run_selection(spec, args.run, args.selection))
    elif args.command == "adopt":
        from .adoption import adopt_model
        _print(adopt_model(spec, args.model, args.run))
    elif args.command == "check-test-inputs":
        from .submission import load_or_check, report_path
        data = experiment.load_data(spec)
        report = load_or_check(spec, data.weather, spec.artifact_dir / "reports")
        _print({"flags": {k: {"station_rounds": v["station_rounds"], "worst": v["worst"]} for k, v in report["flags"].items()},
                "report": str(report_path(spec, data.weather, spec.artifact_dir / "reports"))})
    elif args.command == "submit":
        _print(experiment.submit(spec, args.run, args.acknowledge, args.dry_run, args.strict_distribution, args.evidence))
    elif args.command == "artifacts":
        from .artifacts import audit as audit_artifacts
        _print(audit_artifacts(spec, move=args.archive))
    elif args.command == "index-runs":
        from .store import index_runs
        _print(index_runs(spec.artifact_dir, write=args.write))
    elif args.command == "fold-stratified-partition":
        from .folds import write_partitions
        _print(write_partitions(spec, args.periods))
    elif args.command == "fold-partition":
        from threadpoolctl import threadpool_limits

        from .folds import search_partition, search_settings
        cfg = search_settings(spec, args.settings)
        data = experiment.load_data(spec)
        records = {}
        with threadpool_limits(limits=4):
            for method in args.methods or cfg["methods"]:
                for period in args.periods:
                    record = search_partition(spec, period, method, cfg, data)
                    records[f"{method}@{period}"] = {k: record[k] for k in ("partition", "stratified", "expected_W_krw")} | {
                        "search": {k: v for k, v in record["search"].items() if k != "record"}}
        _print(records)
    elif args.command == "fold-report":
        from .folds import partition_report, fold_report
        if args.partitions:
            _print(partition_report(spec, args.partitions, args.seeds, args.level))
        else:
            _print(fold_report(spec, args.candidates, args.seeds, args.level))
