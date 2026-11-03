"""명령행 진입점. 명령마다 필요한 모듈을 불러 실행한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .store import _plain


def _print(value) -> None:
    print(json.dumps(_plain(value), ensure_ascii=False, indent=2, default=str), flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m solarvpp2", description="집합 forecaster와 risk model로 G1·G2 배정과 제출값을 정한다")
    parser.add_argument("--config", default="configs/solarvpp2/runtime.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("env", help="실행 환경을 출력한다")
    sub.add_parser("models", help="등록한 모델 버전과 기본 모델을 출력한다")
    f = sub.add_parser("forecaster", help="instance 결정 cutoff의 forecaster만 학습한다")
    f.add_argument("--instance", required=True, choices=["V", "fold-2309", "fold-2312", "fold-2403", "H", "B", "test"])
    f.add_argument("--model")
    f.add_argument("--smoke", action="store_true", help="짧은 학습 기간·예산으로 실행 경로만 확인한다")
    i = sub.add_parser("instance", help="forecaster → risk → 배정 결정 → 기간 예측 산출물 → (정답이 있으면) 평가")
    i.add_argument("--instance", required=True, choices=["V", "fold-2309", "fold-2312", "fold-2403", "H", "B", "test"],
                   help="V·H·B·test 또는 robust fold test의 fold(fold-2309·fold-2312·fold-2403)")
    i.add_argument("--model")
    i.add_argument("--smoke", action="store_true", help="짧은 학습 기간·예산으로 실행 경로만 확인한다")
    i.add_argument("--stage", choices=["full", "forecast", "panel"], default="full",
                   help="full은 그래프 전체, forecast는 forecaster와 고정 배정(original·stratified) 평가만, panel은 validation panel 평가를 실행한다")
    i.add_argument("--panel", help="panel 단계의 validation panel 이름(runtime config의 panels, 예: p60-v1)")
    i.add_argument("--xgb-input", help="등록 정의의 XGB 입력 구성을 바꿔 실행한다(run 이름의 MK 뒤에 x, 추천에서 제외)")
    i.add_argument("--dl-input", help="등록 정의의 DL 입력 구성을 바꿔 실행한다(run 이름의 MK 뒤에 x, 추천에서 제외)")
    i.add_argument("--xgb-time-weight", help="등록 정의의 XGB time weight(none, settlement)를 바꿔 실행한다")
    i.add_argument("--dl-time-weight", help="등록 정의의 DL time weight(none, settlement)를 바꿔 실행한다")
    i.add_argument("--dl-loss", help="등록 정의의 DL 손실(pinball, ramp, day-energy 등)을 바꿔 실행한다")
    i.add_argument("--dl-aux-ratio", type=float, help="등록 정의의 DL 보조손실(ramp 등) 비율 aux_ratio를 바꿔 실행한다")
    i.add_argument("--dl-site-aux-ratio", type=float, help="등록 정의의 발전소 이용률 보조 head 세기 비율(dl.site_aux_ratio, 0이면 head 없음)을 바꿔 실행한다")
    i.add_argument("--xgb-weight", type=float, help="등록 정의의 weighted-quantile XGB 비율 [0, 1]을 바꿔 실행한다")
    i.add_argument("--dl-decoder", choices=["bigru", "shared-memory-looped"], help="pooled 시퀀스 decoder 종류(dl.decoder.kind)를 바꿔 실행한다")
    i.add_argument("--dl-decoder-loops", type=int, help="shared-memory-looped decoder의 전체 pass 수 T(dl.decoder.loops)를 바꿔 실행한다")
    i.add_argument("--dl-provider-context", choices=["inverted-attention"],
                   help="시간별 provider 표현에 관측소 사이 context를 더하는 component(dl.provider_context.kind)를 넣어 실행한다")
    i.add_argument("--dl-day-scale", choices=["none", "bounded-log-gain"], help="decoder 뒤의 하루 배율 head(dl.day_scale.kind)를 선택한다")
    i.add_argument("--dl-day-scale-bound", type=float, help="하루 배율의 log 범위 크기(dl.day_scale.max_log_gain)")
    i.add_argument("--dl-day-scale-hidden", type=int, help="하루 배율 head의 hidden 폭(dl.day_scale.hidden)")
    i.add_argument("--dl-plant-routing", choices=["time-shift", "nearby-hours"],
                   help="집합 pooling 전 provider 표현의 temporal routing(dl.plant_routing.kind)을 넣어 실행한다")
    i.add_argument("--dl-plant-hours", type=int, help="nearby-hours가 앞뒤로 읽는 시간(dl.plant_routing.hours)")
    i.add_argument("--dl-group-routing", choices=["nearby-hours"], help="집합 pooling 뒤 sequence의 temporal routing(dl.group_routing.kind)을 넣어 실행한다")
    i.add_argument("--season", choices=["spring", "summer", "autumn", "winter"], help="계절 학습의 대상 계절(대상일 인도월 기준)")
    i.add_argument("--season-strategy", choices=["ft", "weighted", "only"],
                   help="계절 학습 방식: ft(Global 모델에서 이어 학습), weighted(대상 계절 가중), only(대상 계절만 학습)")
    i.add_argument("--season-weight", type=float, help="weighted의 대상 계절 가중(기본 3)")
    i.add_argument("--ft-lr", type=float, help="ft의 DL 최대 학습률(기본 0.0001)")
    i.add_argument("--xgb-ft-rounds", type=int, help="ft의 XGB 추가 boosting round(기본 100)")
    i.add_argument("--xgb-ft-learning-rate", type=float, help="ft의 XGB 학습률(기본 0.02)")
    i.add_argument("--season-margin", type=int, help="학습 계절을 앞뒤로 넓힐 개월 수(기본 0, 평가 계절은 그대로)")
    i.add_argument("--season-steps", choices=["scaled", "full"], help="ft·only의 DL step 예산(scaled: T_s, full: Global 예산)")
    i.add_argument("--season-experts", nargs="+", choices=["xgb", "dl"], help="계절 학습을 적용할 expert(기본 둘 다, 나머지는 Global 모델)")
    i.add_argument("--use-cache", action="store_true", help="설정을 바꾼 실행의 학습 cache도 저장한다(기본은 임시 폴더). 환경 변수 VPP_EXPERIMENT_CACHE=1도 같다")
    i.add_argument("--extract-quantiles", nargs="?", const="stratified", metavar="PARTITION",
                   help="실행 직후 viewer용 분위수를 뽑는다(기본 stratified). 정답이 있는 정규 실행에서만 쓴다")
    i.add_argument("--seeds", type=int, nargs="+", help="등록 정의의 seed 대신 쓸 seed(선별 단계는 1개, run 이름의 MK 뒤에 x, 추천에서 제외)")
    r = sub.add_parser("recommend", help="H·B instance 결과로 test 배정을 추천하고 제출 형식을 검사한다")
    r.add_argument("--model")
    ar = sub.add_parser("artifacts", help="instance 기록의 상태(current·superseded·replaced·dead·aborted·running)를 보고한다")
    ar.add_argument("--archive", action="store_true", help="replaced·dead·aborted 기록을 package 산출물 디렉터리의 archive/로 옮기고 index에 적는다")
    ev = sub.add_parser("evaluate", help="저장된 배정·예측을 재평가하고 추가 진단 후보만 추론한다")
    ev.add_argument("--instance-dir", type=Path, required=True)
    ev.add_argument("--calendar", type=Path)
    ev.add_argument("--bootstrap-blocks", type=int, nargs="+")
    ev.add_argument("--bootstrap-replicates", type=int)
    ev.add_argument("--candidate-pool", type=int)
    pq = sub.add_parser("probabilistic", help="저장 checkpoint에서 공통 expert 분위수 artifact를 추출한다(재학습 없음)")
    pq.add_argument("--instance-dir", type=Path, required=True)
    pq.add_argument("--partition", default="stratified")
    pq.add_argument("--output", type=Path)
    tr = sub.add_parser("transfer", help="H parent checkpoint에서 decoder를 적응시키고 R0 calibration 후 공통 비교 artifact를 생성한다")
    tr.add_argument("--instance-dir", type=Path, required=True)
    tr.add_argument("--baseline-artifact", type=Path, required=True)
    tr.add_argument("--partition", default="stratified")
    tr.add_argument("--open", action="store_true", help="평가 후 Rust viewer로 연다")
    rc = sub.add_parser("recalibrate", help="저장한 R0 예측에서 중앙값·구간을 보정하고 동일 H를 공통 contract로 평가한다")
    rc.add_argument("--transfer-run", type=Path, required=True)
    rc.add_argument("--open", action="store_true", help="평가 후 Rust viewer로 연다")
    args = parser.parse_args(argv)

    from .config import load_spec
    spec = load_spec(args.config)
    if args.command == "env":
        from .store import environment
        _print(environment())
    elif args.command == "models":
        _print({"default": spec.default_model, "models": spec.registry["models"]})
    elif args.command == "forecaster":
        from vppcontract.timeaxis import period

        from .data import load_data
        from .forecast.forecaster import Forecaster
        per = period(args.instance)
        forecaster = Forecaster(load_data(spec), args.model or spec.default_model, per.cutoff_h, per.days, smoke=args.smoke)
        _print({"identity": forecaster.identity(), "details": forecaster.train()})
    elif args.command == "instance":
        from vppcontract.timeaxis import period

        from .instance import run_instance
        from .store import experiment_cache_enabled, temporary_cache
        if args.extract_quantiles and (args.smoke or args.stage == "panel" or not period(args.instance).has_labels):
            raise ValueError("--extract-quantiles는 정답이 있는 기간의 정규 실행에서 씁니다(smoke·panel 단계 제외).")
        overrides = {k: v for k, v in (("xgb.input", args.xgb_input), ("dl.input", args.dl_input), ("xgb.time_weight", args.xgb_time_weight),
                                       ("dl.time_weight", args.dl_time_weight), ("dl.loss", args.dl_loss), ("dl.aux_ratio", args.dl_aux_ratio),
                                       ("dl.site_aux_ratio", args.dl_site_aux_ratio), ("blend.xgb_weight", args.xgb_weight),
                                       ("dl.decoder.kind", args.dl_decoder), ("dl.decoder.loops", args.dl_decoder_loops),
                                       ("dl.provider_context.kind", args.dl_provider_context),
                                       ("dl.day_scale.kind", args.dl_day_scale), ("dl.day_scale.max_log_gain", args.dl_day_scale_bound),
                                       ("dl.day_scale.hidden", args.dl_day_scale_hidden),
                                       ("dl.plant_routing.kind", args.dl_plant_routing), ("dl.plant_routing.hours", args.dl_plant_hours),
                                       ("dl.group_routing.kind", args.dl_group_routing),
                                       ("season.name", args.season), ("season.strategy", args.season_strategy),
                                       ("season.weight", args.season_weight), ("season.ft_lr", args.ft_lr),
                                       ("season.xgb_ft_rounds", args.xgb_ft_rounds), ("season.xgb_ft_learning_rate", args.xgb_ft_learning_rate),
                                       ("season.margin_months", args.season_margin), ("season.steps", args.season_steps),
                                       ("season.experts", args.season_experts),
                                       ("seeds", args.seeds)) if v is not None}
        from .evaluation import evaluate_saved
        use_cache = args.use_cache or experiment_cache_enabled()
        quantiles = None
        with temporary_cache(spec.artifact_dir, bool(overrides) and not use_cache):
            directory = run_instance(spec, args.instance, args.model, args.smoke, args.stage, overrides, args.panel,
                                     need_models=use_cache or bool(args.extract_quantiles))
            evaluation = evaluate_saved(spec, directory) if period(args.instance).has_labels else None
            if args.extract_quantiles:
                from .probabilistic import run as extract_quantiles
                quantiles = extract_quantiles(spec, directory, args.extract_quantiles)
        _print({"directory": str(directory), "evaluation": str(evaluation) if evaluation else None, "quantiles": quantiles})
    elif args.command == "recommend":
        from .instance import recommend
        _print(recommend(spec, args.model))
    elif args.command == "artifacts":
        from .artifacts import audit
        _print(audit(spec, move=args.archive))
    elif args.command == "evaluate":
        from .evaluation import evaluate_saved
        from .store import read_json
        if args.calendar is not None:
            spec.config["paths"]["calendar"] = str(args.calendar.resolve())
        if args.bootstrap_blocks is not None:
            spec.config["evaluation"]["bootstrap_blocks"] = args.bootstrap_blocks
        if args.bootstrap_replicates is not None:
            spec.config["evaluation"]["bootstrap_replicates"] = args.bootstrap_replicates
        if args.candidate_pool is not None:
            source = read_json(args.instance_dir / "instance.json")
            section, key = ("smoke", "candidate_pool") if source.get("smoke") else ("evaluation", "candidate_pool_size")
            spec.config[section][key] = args.candidate_pool
        _print({"prediction": str(args.instance_dir), "evaluation": str(evaluate_saved(spec, args.instance_dir))})
    elif args.command == "probabilistic":
        from .probabilistic import run
        _print(run(spec, args.instance_dir, args.partition, args.output))
    elif args.command == "transfer":
        from .transfer import run
        result = run(spec, args.instance_dir, args.baseline_artifact, args.partition)
        _print(result)
        if args.open:
            from vppcontract.probabilistic import open_viewer
            open_viewer(result["report_directory"])
    elif args.command == "recalibrate":
        from .transfer import recalibrate
        result = recalibrate(spec, args.transfer_run)
        _print(result)
        if args.open:
            from vppcontract.probabilistic import open_viewer
            open_viewer(result["report_directory"])
