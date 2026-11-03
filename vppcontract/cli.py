"""공통 계약층 명령: 산출물 비교, robust fold test, validation panel 비교, 제출 형식 검사, champion 기록."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .champion import read_champion, set_champion
from .evaluate import compare_directories, fold_compare, load_actuals, panel_compare
from .outputs import _plain, read_output, validate_submission, write_json

ROOT = Path(__file__).resolve().parents[1]


def _print(value) -> None:
    print(json.dumps(_plain(value), ensure_ascii=False, indent=2, default=str), flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m vppcontract", description="package 간 비교와 제출 형식 검사")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "participant_dataset")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compare", help="두 산출물의 W와 ΔW(a − b)·대상일 block bootstrap. solarvpp run 디렉터리도 읽는다")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.add_argument("--period", required=True, choices=["D", "V", "fold-2309", "fold-2312", "fold-2403", "H", "B"])
    fo = sub.add_parser("folds", help="robust fold test: fold별 후보·기준 산출물을 공통 평가기로 채점하고 네 조건을 판정한다")
    fo.add_argument("--candidate", nargs="+", required=True, metavar="FOLD=DIR", help="fold별 후보 산출물(계약 형식 또는 solarvpp run)")
    fo.add_argument("--baseline", nargs="+", required=True, metavar="FOLD=DIR", help="fold별 기준 산출물")
    fo.add_argument("--out", type=Path, help="판정 결과 JSON을 쓸 경로")
    pa = sub.add_parser("panel", help="validation panel: 날짜별 구성원으로 후보·기준 산출물을 채점하고 W 변화율과 묶음 bootstrap 구간을 낸다")
    pa.add_argument("--candidate", type=Path, required=True, help="후보 panel 산출물(<run>/outputs/stratified)")
    pa.add_argument("--baseline", type=Path, required=True, help="기준 panel 산출물")
    pa.add_argument("--replicates", type=int, default=2000)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--out", type=Path, help="비교 결과 JSON을 쓸 경로")
    v = sub.add_parser("validate", help="test 제출 디렉터리의 형식을 검사한다")
    v.add_argument("directory", type=Path)
    pr = sub.add_parser("probabilistic", help="저장 분위수 artifact를 검증하고 평가표·viewer 데이터를 재생성한다")
    pr.add_argument("directory", type=Path, help="common.npz, expert 분위수 NPZ, manifest.json이 있는 폴더")
    pr.add_argument("--output", type=Path, help="결과 폴더. 기본값은 <artifact>/report")
    pr.add_argument("--open", action="store_true", help="결과를 만든 뒤 데스크톱 viewer로 연다")
    pr.add_argument("--viewer", type=Path, help="viewer 실행 파일. 기본값은 VPP_PROBABILISTIC_VIEWER 또는 repo의 viewer 빌드")
    vw = sub.add_parser("view", help="이미 만든 확률예측 결과 폴더를 데스크톱 viewer로 연다")
    vw.add_argument("directory", type=Path, help="explorer-data.json이 있는 결과 폴더")
    vw.add_argument("--viewer", type=Path, help="viewer 실행 파일. 기본값은 VPP_PROBABILISTIC_VIEWER 또는 repo의 viewer 빌드")
    ch = sub.add_parser("champion", help="champion 기록을 출력하거나 바꾼다")
    ch.add_argument("--set", help="package:MK 형식(예: solarvpp2:MK3)")
    ch.add_argument("--evidence", help="--set의 근거(비교 보고서 경로 등)")
    args = parser.parse_args(argv)
    if args.command == "compare":
        _print(compare_directories(args.a, args.b, args.period, args.data_dir))
    elif args.command == "folds":
        def pairs(items):
            out = dict(item.split("=", 1) for item in items)
            if len(out) != len(items):
                parser.error("fold 이름이 겹칩니다.")
            return {f: Path(d) for f, d in out.items()}
        result = fold_compare(pairs(args.candidate), pairs(args.baseline), load_actuals(args.data_dir))
        if args.out:
            write_json(args.out, result)
        _print(result)
    elif args.command == "panel":
        result = panel_compare(read_output(args.candidate), read_output(args.baseline), load_actuals(args.data_dir), args.replicates, args.seed)
        result["runs"] = {"candidate": str(args.candidate), "baseline": str(args.baseline)}
        if args.out:
            write_json(args.out, result)
        _print(result)
    elif args.command == "validate":
        _print(validate_submission(args.directory, args.data_dir))
    elif args.command == "probabilistic":
        from .probabilistic import open_viewer, report
        result = report(args.directory, args.output)
        _print(result)
        if args.open:
            open_viewer(result["directory"], args.viewer)
    elif args.command == "view":
        from .probabilistic import open_viewer
        _print({"viewer": str(open_viewer(args.directory, args.viewer)), "directory": str(args.directory.resolve())})
    elif args.command == "champion":
        if args.set:
            if not args.evidence:
                parser.error("--set에는 --evidence가 필요합니다.")
            package, _, model = args.set.partition(":")
            _print(set_champion(package, model, args.evidence))
        else:
            _print(read_champion())
