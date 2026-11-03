# 태양광 VPP predict-and-optimize

태양광 VPP의 forecast와 partition을 개선해 **고정 기준선 정산손실의 10%를 회수하는 것**을 첫 목표로 한다.
253개 발전소의 G1·G2 partition과 다음 날 발전량을 정하고, 최종 그룹 forecast의 정산손실 W로 개선을 판단한다.
같은 그룹·같은 시각에서 부호 있는 오차가 상쇄된 뒤 남은 손실을 줄인다.

| 지표 | 계산 | 용도 |
|---|---|---|
| 고정 기준선 대비 W 변화율 | `(W_candidate − W_baseline) / W_baseline` | 출발점 대비 누적 진척. 첫 목표 −10% |
| Champion 대비 W 변화율 | `(W_challenger − W_champion) / W_champion` | 후보 비교 기준. 음수가 개선이며 낮을수록 좋다 |

W 변화율은 %로 표시한다. 고정 기준선은 출발점의 예측을 보존하고, champion은 후보를 채택할 때 갱신한다.
지표·평가 조건과 기준선의 등록 정보는 [Problem definition](PROBLEM.md#loss-recovery)을 따른다.

`solarvpp`는 forecasting primitive와 발전소 residual scenario를 이용해 배정을 탐색한다.
`solarvpp2`는 발전소 집합을 입력받는 forecaster를 고정하고, 그 예측의 expected absolute error를 학습한 risk model로 배정을 탐색한다.
두 package는 공통 정산식과 산출물 형식으로 결과를 비교한다.

solarvpp2 forecast 후보는 [P60 panel](docs/solarvpp2-design.md#p60-panel)의 같은 조건으로 비교한다.
기존 V/H/B·[robust fold test](docs/solarvpp-reference.md#robust-folds)는 해당 cohort·학습 이력의 진단으로 기록하고, 기존 기간의 10% 목표 추적을 유지한다.

모델은 입력·encoder·decoder·손실·expert 결합 등의 구성으로 선택한다. 등록 이름(MK)과 CLI 기본값은 package별 모델 설정에,
현재 champion은 [champion 기록](configs/champion.json)에 둔다. Run은 실제 설정·배정·prediction artifact를 연결해 재현에 사용한다.
실험 실행·결과 검증·모델 채택은 독립 명령으로 수행하며, 버전 기록은 [개발 규칙](AGENTS.md)을 따른다.

## 설계 문서

| 문서 | 내용 |
|---|---|
| [Problem definition](PROBLEM.md) | 손실 10% 감소 목표와 champion–challenger 기준, task·data·제출 계약, 정산식 |
| [solarvpp 설계](docs/solarvpp-design.md) | 입력·출력, 그룹 forecaster와 배정 탐색의 역할과 연결, 학습 신호와 평가 |
| [solarvpp 구현·실행 참고](docs/solarvpp-reference.md) | Data·query 계약, 입력 구성·손실 수식, 배정 탐색 수식, 기간·fold, 이름·module·cache, recipe |
| [solarvpp2 설계](docs/solarvpp2-design.md) | 입력·출력, 집합 forecaster·risk·partition의 역할과 연결, 학습 신호와 평가 |
| [solarvpp2 구현·실행 참고](docs/solarvpp2-reference.md) | Component 수식·설정·gradient, P60 날짜·mask, CLI 실행·추천·저장 동작, 등록 구성 |
| [solarvpp protocol](docs/solarvpp-protocol.json) | 기간·seed·학습 예산·입력 구성·보조손실·λ·noise floor·블렌드 |
| [기상 schema](docs/weather-schema.json) | 네 파일의 전체 컬럼·단위·대응 target·quality 규칙 |
| [평가 cohort 근거](docs/evidence/cohort-coverage.json) | cutoff 고정 PV 목록과 연간·월별 정답 커버리지 |
| [데이터 특성](docs/data-characteristics.md) | WA 예보 기압 정의 변경, WB revision 빈도, generation 결측 구조 |

## 실행 환경

기존 conda 환경 FAI를 사용한다. 필요한 패키지만 이 환경에서 추가·갱신한다.
DL 장치는 RTX 2000 Ada Laptop의 CUDA GPU이다.

```powershell
$py = Join-Path $env:USERPROFILE 'anaconda3\envs\FAI\python.exe'
& $py -m solarvpp env
```

FAI의 numpy는 conda-forge OpenBLAS(pthreads) build를 쓴다. MKL BLAS는 OpenMP runtime을 불러와
pip torch가 함께 배포하는 `libiomp5md.dll`과 충돌하므로 BLAS를 MKL로 되돌리지 않는다.
`python -m solarvpp`는 torch를 불러오기 전에 `configs/runtime.json`의 `runtime` 값으로 `CUBLAS_WORKSPACE_CONFIG`(cuBLAS 결정성)를 설정하고,
torch를 직접 import하는 스크립트도 이 환경 변수를 먼저 설정한다.

원자료는 `participant_dataset/`, 모델 구성요소는 `solarvpp/`에 둔다. 실행 산출물은 repo 옆 `../sol-artifact/<package>/`에 둔다.

## 실행

```powershell
& $py -m solarvpp audit                                        # base·weather cache 생성과 QA 보고
& $py -m solarvpp cohorts                                      # D/V/H cohort를 evidence와 대조
& $py -m solarvpp index-runs                                   # 기존 완료 run의 계산 identity 연결 미리보기
& $py -m solarvpp index-runs --write                           # 기존 산출물을 유지하고 run-index/만 생성
& $py -m solarvpp run --period D --level smoke                 # 실행 확인
& $py -m solarvpp run --period D --level diagnostic            # D 진단(seed 42)
& $py -m solarvpp run --period V --level noise_floor           # 기준 칸 forecast-solar-diff__pinball, seed 42~51 (solarvpp 설계 5.3절)
& $py -m solarvpp grid --period V --level screening            # 현재 입력 7종 × 손실 6종, seed 42
& $py -m solarvpp grid-report --period V                       # 완료한 후보 전체 ranking, 사용 가능한 reference·noise floor
& $py -m solarvpp run --period V --level confirmation --input forecast-solar-physics --loss fourier   # 한 칸을 seed 42·43·44로 확인
& $py -m solarvpp run --period V --level confirmation --model MK0   # 등록한 모델 버전 실행(생략하면 champion)
& $py -m solarvpp run --period V --level screening --input forecast-solar-physics --loss fourier --aux-ratio 0.3   # 보조항 세기 변경
& $py -m solarvpp run --model MK1 --period H --level smoke --experts dl --aux-ratio 3 --xgb-weight 0.3
& $py -m solarvpp conditional-bundling --instances V           # V만 실행. Site forecast dependency는 D·V
& $py -m solarvpp conditional-bundling --instances H --transfer --method analog  # H 실행과 H group transfer
& $py -m solarvpp conditional-transfer --period B --method analog  # 완료한 B 결과의 transfer를 별도로 실행
& $py -m solarvpp conditional-audit                            # 원자료 정산 재계산·cutoff·후보·선택·용량·재학습 검증
& $py -m solarvpp conditional-decide --method analog           # 저장된 H·B transfer와 test 구성으로 추천
& $py -m solarvpp run --period B --level bundling --partition <csv>   # 그룹 배정을 바꾼 champion 실행
& $py -m solarvpp run --model MK2 --xgb-weight 0.2 --partition <csv> --period B --level bundling   # MK2의 parameter 변경 run
& $py -m solarvpp run --period test --level submission         # 전체 original test 예측(seed 42·43·44)
& $py -m solarvpp check-test-inputs                            # test 입력 분포 점검
& $py -m solarvpp select --baseline <run> --finalist <run> --out <snapshot.json>
& $py -m solarvpp verify-selection --run <run> --selection <snapshot.json>
& $py -m solarvpp adopt --model MKn --run <run>                 # 채택한 pipeline의 run을 새 MK로 등록·champion 선택
& $py -m solarvpp submit --run <test run> --dry-run
& $py -m solarvpp submit --run <test run> --evidence <H 비교.json> <B 비교.json>   # 제출 파일과 기준 MK·parameter·gate 근거 기록
```

- `run`의 기간과 level은 별도로 선택한다. `--selection`을 준 경우에만 snapshot 일치를 검사한다. H·B exposure는 manifest에 기록한다.
- 후보 비교는 `compare --runs <candidate> <baseline>`으로 하며 같은 evaluation panel hash를 요구한다.
  배정이나 package가 다른 비교는 `vppcontract compare`로 같은 평가 조건의 W를 계산한다.
  비교 결과의 W에서 MK0·champion 대비 W 변화율을 계산하고, 기준 run과 후보 run을 함께 제시한다.
- 입력·손실을 생략한 `run`은 등록된 champion을 실행한다. MK·recipe 위에 `aux_ratio`·입력·time weight·blend·배정을 바꾸고 `--experts dl`처럼 expert만 실행할 수 있다.
  `--level noise_floor`의 기본 칸은 grid 기준 칸 forecast-solar-diff__pinball이다. 보조손실 칸의 λ는 200 step warm-up 뒤 gradient 비율로 정한다.
- Distribution flag는 제출 시 warning으로 표시한다. `--strict-distribution`을 선택하면 `--acknowledge`를 요구한다. CSV·cutoff·최종 용량 검사는 유지한다.
- cache는 `sol-artifact/solarvpp/cache/`, run은 `sol-artifact/solarvpp/runs/<period>/<MK>-<level>-<key 앞 10자>/`, 보고서는 `sol-artifact/solarvpp/reports/`, 제출 파일은 `sol-artifact/solarvpp/submissions/`에 쓴다.
- Run·transfer의 key는 실제 expert model cache와 blend·cutoff·배정·평가 조건으로 계산한다. 원본 설정 파일 hash는 provenance로 기록한다.
  `index-runs --write`는 재학습 없이 기존 run을 연결한다. 기존 파일은 유지하고, 이후 같은 계산 요청은 연결된 디렉터리를 재사용한다.
- 공동 오차 모델 학습과 시나리오 실행은 `configs/conditional_bundling.json`을 읽는다. 실험은 `sol-artifact/solarvpp/conditional/<hash>/`에 저장하고 `sol-artifact/solarvpp/conditional/latest.json`에 마지막 실행 위치를 기록한다.
  추천은 해당 실험의 `decision.json`에 기록한다. `conditional-transfer`·`conditional-audit`·`conditional-decide`는 `--experiment <directory>`로 대상을 지정한다.

## solarvpp2와 공통 계약층

solarvpp2는 별도 package이며 solarvpp를 import하지 않는다. 두 package는 공통 계약층 `vppcontract`(시간축·기간, 정산식,
산출물 형식, 공통 평가기, champion 기록)만 공유한다. 설정은 `configs/solarvpp2/`, 산출물은 `sol-artifact/solarvpp2/`,
package를 넘는 champion 기록은 `configs/champion.json`에 둔다.

Hash·artifact index는 `vppcontract.identity`를 공유한다. Package별 계산 payload·저장 helper는 각 `store.py`에 둔다.
새 package는 `ArtifactAdapter`에 기록 파일명·검색 경로·payload 함수·완료 함수를 제공하고 자기 산출물 root에서 index를 사용한다.

```powershell
& $py -m solarvpp2 instance --instance H --smoke   # 짧은 학습 기간·예산으로 실행 경로 확인
& $py -m solarvpp2 instance --instance H           # forecaster → risk → 배정 결정 → H 예측 산출물 → 평가
& $py -m solarvpp2 instance --instance B --stage forecast   # forecaster와 고정 배정(original·층화)만 평가, R·배정 탐색 생략
& $py -m solarvpp2 instance --instance B --stage forecast --xgb-input forecast-physics-daystate
                                                    # 선택한 입력을 report의 parameters에 기록
& $py -m solarvpp2 instance --instance B --stage forecast --seeds 42   # 지정한 seed로 고정 배정의 예측 평가
& $py -m solarvpp2 instance --instance B
& $py -m solarvpp2 instance --instance test        # test 배정 후보(관측소 층화·searched 배정)와 test 예측
& $py -m solarvpp2 instance --instance test --model MK5 --dl-site-aux-ratio 30 --xgb-weight 0.2 --dl-decoder shared-memory-looped --stage panel --panel p60-v1 --seeds 42
                                                    # validation panel(P60-v1): 평가일·완충을 뺀 학습, 날짜별 구성원으로 60일 채점
& $py -m vppcontract panel --candidate <후보 run>\outputs\stratified --baseline <기준 run>\outputs\stratified --out <보고서.json>
                                                    # panel W 변화율(후보/기준 − 1)과 연속 평가일 묶음 bootstrap 구간
& $py -m solarvpp2 evaluate --instance-dir <instance 경로> --bootstrap-replicates 3000
                                                    # 저장된 배정·예측으로 평가만 다시 계산
& $py -m solarvpp2 evaluate --instance-dir <instance 경로> --candidate-pool 250
                                                    # 기존 후보 점수를 재사용하고 추가 후보만 추론
& $py -m solarvpp2 recommend                       # H·B의 ΔW(searched − 층화) 합으로 test 배정 추천, 제출 형식 검사
& $py -m vppcontract compare <a 산출물> <b 산출물 또는 solarvpp run> --period H   # 같은 정답 mask의 W, ΔW(a − b), bootstrap
& $py -m vppcontract validate <제출 디렉터리>
& $py -m vppcontract champion                      # champion 기록 출력. --set solarvpp2:MK3 --evidence <보고서>로 변경
```

- Instance 실행은 `sol-artifact/solarvpp2/instances/<instance>/<MK>-<key>/instance.json`(forecast 단계는 `<MK>-forecast-<key>`,
  panel 단계는 `<MK>-panel-<panel>-<key>`)과
  배정별 `outputs/<배정>/`(`composition.csv`, `forecast.csv`, `manifest.json`)을 남긴다. 추천은 `sol-artifact/solarvpp2/recommendation/<MK>.json`에 기록한다.
- 모델의 resolved inference 설정은 각 instance의 `artifacts.json`에 보존한다. 추가 후보 추론은 이 설정과 저장 weights를 사용한다.
- Evaluation은 `sol-artifact/solarvpp2/evaluations/<key>/evaluation.json`에 별도로 기록한다. Calendar·bootstrap 변경은 저장된 예측으로
  평가만 갱신한다. `evaluate --calendar <calendar JSON>`으로 calendar를 지정할 수 있다. Candidate pool 변경은 배정별 점수 cache를
  사용하며, 추가 후보에 필요한 추론만 실행한다. 기존 instance·모델 cache·예측은 유지한다.
- 설계 9절의 비교 1(기본 배정에서 solarvpp champion과 비교)은 `vppcontract compare <instance의 outputs/original> <solarvpp run>`으로 한다.

### Tree-only·DL-only 확률예측 비교

모델 package는 checkpoint·전처리·고정 배정에서 분위수 artifact를 추출한다.
공통 계약층은 저장 artifact를 검증하고 계절별 편향·폭·coverage 표와 viewer 데이터를 재생성한다.
결과는 repo의 `viewer/`에 있는 데스크톱 viewer로 연다.

```powershell
& $py -m solarvpp2 probabilistic --instance-dir <instance-directory> --partition stratified
& $py -m vppcontract probabilistic <quantile-artifact-directory> --open
& $py -m vppcontract view <report-directory>
cargo build --release --manifest-path viewer/Cargo.toml       # viewer 빌드(처음 한 번, viewer 변경 시)
```

Solarvpp2 artifact의 기본 저장 위치는 `sol-artifact/solarvpp2/reports/probabilistic/<원 run>/stratified/`이다.
공통 평가 결과는 `<artifact>/report/`에 생성하며 `--output`으로 결과 폴더를 지정할 수 있다.
`--open`은 결과를 만든 뒤 viewer를 열고, `view`는 이미 만든 결과 폴더를 연다. Viewer는 별도 창으로 실행되며 CLI는 기다리지 않는다.
Viewer 실행 파일은 `viewer/target/release/probabilistic-viewer.exe`이며, 다른 위치는 `--viewer`나 `VPP_PROBABILISTIC_VIEWER`로 지정한다.
Viewer는 `viewer/rust-toolchain.toml`의 Rust toolchain과 Visual Studio Build Tools(MSVC)로 빌드한다. Viewer를 직접 실행해 결과 폴더를 열 수도 있다.
원 run의 모델 학습과 배정은 유지하며, 상세 계약은 [확률예측 공통 contract](docs/probabilistic-contract.md)를 따른다.

### 기존 checkpoint에서 H 배정으로 적응

`solarvpp2 transfer`는 MK5의 입력·encoder·전처리를 고정하고 decoder/head만 이어 학습한다.
H 이전 R0로 조건부 보정을 학습한 뒤 Tree·B·B+C·T·T+C를 공통 contract로 비교한다.
`transfer`의 보정은 중앙값을 고정한 구간 보정이고, `solarvpp2 recalibrate`는 저장된 transfer run의 R0 예측으로 W 기준 중앙값 shift와 구간 배율을 fit한다.
두 실험은 제출 경로가 없어 MK로 등록하지 않고, 설정은 `configs/solarvpp2/runtime.json`의 `experiments`에 둔다.

```powershell
& $py -m solarvpp2 transfer --instance-dir <H-MK5-parent-run> --baseline-artifact <H-quantile-artifact> --open
& $py -m solarvpp2 recalibrate --transfer-run <transfer-run> --open
```

학습·데이터 분리·보정식·산출물은 [transfer와 calibration](docs/transfer-calibration.md)을 따른다.

<a id="storage"></a>

## Artifact와 cache 관리

Git에는 source·config·설계·test를 보관한다. 원자료와 생성 파일은 repo 밖의 설정된 경로에 두고,
`.gitignore`는 소스 폴더 안에 생긴 tensor·weights도 제외한다. Worktree별로 원자료나 대형 cache를 복제하지 않고
`paths.data_dir`·`paths.artifact_dir`로 사용할 위치를 지정한다.

Run 기록은 실행과 결과의 이력이고, artifact는 그 결과를 다시 사용하기 위해 보존하는 파일이다.
계산 cache는 원자료와 보존 artifact에서 다시 만들 수 있는 중간 결과다. 현재 경로는 다음 역할로 관리한다.

| 역할 | 현재 경로 | 관리 |
|---|---|---|
| 보존 artifact | `cache/models/`, `cache/prep/`, `cache/risk/`, `cache/error_model/` | 등록 정의를 그대로 돌린 실행의 학습 weights·전처리·risk head를 백업한다. 설정을 바꾼 실행은 기본으로 저장하지 않는다. 경로명에 cache가 있어도 일반 cache 정리에서 제외한다 |
| Run·prediction·평가 기록 | `runs/`, `instances/`, `evaluations/`, `submissions/`, `recommendation/` | manifest·`artifacts.json`·배정·prediction·평가를 함께 보존한다 |
| 실험 기록 | solarvpp2의 `experiments/` | 제출 경로가 없는 실험의 설정·계수·비교 artifact를 보존한다 |
| 분석 보고서 | `reports/<주제-날짜>/` | 판단 근거가 되는 표·스크립트·`summary.md`를 보존한다 |
| 재생성 가능한 계산 cache | `cache/base/`, `cache/weather/`, `cache/features/`, `cache/residuals/`, `cache/evaluation_candidates/` | 원자료와 모델 artifact를 보존한 상태에서 용량에 따라 정리한다 |
| 조회 보조 파일 | solarvpp의 `run-index/` | 기록에서 다시 만들 수 있다. solarvpp2는 지정한 run과 evaluation 파일을 직접 읽는다 |

Cache 정리는 실행이 끝난 뒤 위 재생성 항목에 한해 수행한다. `cache/` 전체를 삭제하는 대신 해당 항목을 지정한다.
Model artifact와 원자료는 백업 대상으로 두고, 큰 weather·feature cache는 기본 백업 대상에서 제외한다.
`artifacts --archive`는 run 기록을 옮기는 명령이며 model·계산 cache의 삭제 정책과 별도로 사용한다.

설정을 바꾼 실행은 기본으로 새 학습 cache(model·전처리·feature·risk)를 저장하지 않는다. 저장 공간을 아끼기 위한 repo 기본값이다.
새 파일은 실행 동안 `cache/temporary/`에만 두고 끝나면 지운다. 이미 있는 cache는 그대로 읽으므로, 등록 정의를 먼저 돌려 두면 같은 XGB 모델과 feature를 다시 만들지 않는다.
예측·평가·설정 기록은 항상 남기고, 저장 여부는 실행 기록의 `cache_stored`에 적는다. 실행이 비정상으로 끊겨 `cache/temporary/`가 남으면 지워도 된다.

등록 정의를 그대로 돌린 실행은 항상 저장한다. solarvpp에서 배정만 바꾼 run과 `submission` level도 같은 정의의 제출 경로이므로 저장한다.

저장 공간이 넉넉한 PC는 환경 변수 `VPP_EXPERIMENT_CACHE=1`로 항상 켜 두거나, 실행마다 `--use-cache`를 붙인다. 켜면 설정을 바꾼 실행의 cache도 저장하고 재사용한다.
모델 파일이 나중에 필요한 실행(`transfer`의 parent, 설정을 바꾼 정의의 계절 이어 학습에 쓸 Global)은 켠 상태로 돌린다.
viewer로 볼 실행은 solarvpp2의 `--extract-quantiles [배정]`을 붙인다. 실행 직후 분위수를 뽑아 `reports/probabilistic/`에 둔다.
저장하지 않은 완료 실행은 같은 설정에 cache를 켜거나 `--extract-quantiles`를 붙이면 다시 학습하고, 그렇지 않으면 학습 없이 기록을 재사용한다.

## Weighted-Q + day-energy recipe

`weighted-day-energy`는 XGB forecast-physics와 DL forecast-physics-daystate를 한 run에서 결합한다. XGB는 기존 capacity-weighted quantile loss를 사용하고,
DL은 `day-energy-aux0.1-settlement`(weighted pinball + median trajectory의 k-weighted day-energy)와 observed supervision을 사용한다.
Day-energy의 λ_E는 기존 gradient-ratio 방식으로 정하며 기본 `aux_ratio`는 0.1이다. Blend는 weighted-quantile-w0.5이다.

```powershell
& $py -m solarvpp run --period V --level confirmation --recipe weighted-day-energy
& $py -m solarvpp run --period H --level bundling --recipe weighted-day-energy
& $py -m solarvpp run --period B --level bundling --recipe weighted-day-energy
```

Recipe는 input·loss·weight 기본값을 묶는다. `--recipe weighted-day-energy --aux-ratio 1 --xgb-weight 0.3`처럼 override한다. 개별 설정을 지정할 때는
`--input forecast-physics-daystate --xgb-input forecast-physics --loss day-energy --aux-ratio 0.1 --time-weight settlement --xgb-time-weight none`을 사용한다.
Run manifest에 두 expert의 input과 time weight, recipe, seed별 λ_E와 DayState auxiliary weight를 기록한다.
정의는 [solarvpp 설계의 training objective](docs/solarvpp-reference.md#losses)와 [recipe](docs/solarvpp-reference.md#weighted-day-energy)를 따른다.

