# 고정 H 배정 적응과 조건부 calibration

두 실험은 H 안에서만 동작하고 제출 경로가 없어 MK로 등록하지 않는다. 설정은 `configs/solarvpp2/runtime.json`의
`experiments.transfer`·`experiments.recalibration`에 둔다.

`transfer`는 H cutoff에서 학습된 MK5(site_aux_ratio=30)를 parent로 사용한다. Parent 실행은 모델 파일이 필요하므로 `--use-cache`로 돌린다(또는 `VPP_EXPERIMENT_CACHE=1`). 모델 입력·전처리·집계 계약은 유지하고,
집합 decoder와 quantile head를 고정 H 배정에 적응시킨다. H 이전 R0에서 구간 보정 계수를 학습한 뒤 H에서 다섯 출력을 비교한다.
`recalibrate`는 같은 checkpoint·입력·예측을 재사용하고 R0에서 중앙값과 구간을 보정한다.

## 학습과 보정 데이터

H cutoff는 2024-05-31 17:00 KST, 평가는 대상일 2024-06-01~2025-05-31이다.
학습 객체에 전달하는 generation·status는 cutoff 이후를 NaN·무효로 바꾼다. 원 cache는 유지한다.

최근 12개월(2023-06-01~cutoff)에서 기존 risk held-out 날짜 R0를 뺀 316일로 적응한다.
H 배정의 그룹을 G, 날짜 d까지 편입된 발전소를 U_d라 할 때 target query는 G∩U_d이다.
해당 query의 구성원 실적이 모두 유효한 시각의 합을 정답으로 사용한다.

| 자료 | 날짜 | Query 수 | 유효 group-hours |
|---|---:|---:|---:|
| H 배정 기반 target | 316 | 632 | 7,943 |
| 같은 기간의 기존 sampler replay | 316 | 5,688 | 71,927 |
| 기존 R0의 H 배정 기반 calibration | 50 | 100 | 1,245 |

Calibration 날짜는 봄 8일, 여름·가을·겨울 각각 14일이다. 학습 query에는 R0가 들어가지 않고,
학습·보정의 모든 유효 정답 시각은 cutoff 이하이어야 한다. H의 205개소 전체가 함께 관측된 과거 보정 자료는 12개 주간 시각이므로,
보정은 가용 부분집합에서 학습한 용량 정규화 중앙값 shift와 구간 배율을 완전한 H 그룹에 적용한다.

## 이어 학습

- Seed 42·43·44 각각 같은 seed의 parent checkpoint로 초기화한다.
- Provider encoder·DayState·PV fusion·site auxiliary head와 전처리 state를 고정한다.
- 고정 모듈은 eval mode로 두고, `gru.*`, `head.*` parameter만 갱신한다.
- Batch 16개 중 target 8개, replay 8개를 각 query pool의 섞은 순열로 뽑는다.
- 새 AdamW, learning rate 1e-4, weight decay 1e-4, gradient clip 1.0, cosine decay 500 step을 사용한다.
- 적응 손실은 기존 19분위수 capacity-weighted pinball이며, normalizer는 1:1 표집 분포의 유효 시각 기준 평균 용량이다.
- 마지막 step의 checkpoint를 보존한다. 고정 parameter hash가 일치하고 학습 parameter hash가 달라야 한다.

고정 encoder의 pooled 표현을 미리 계산해 학습에 사용한다. 이 표현은 실행 메모리에서 재사용하고,
보존하는 학습 산출물은 checkpoint·표집 schedule·학습 기록·parent identity다.

## transfer: 중앙값 고정 구간 보정

B(기존 DL)와 T(적응 DL) 각각에 대해 R0의 seed 평균 후처리 분위수로 계수를 fit한다.
중앙값 m=q_0.5를 유지하고 하위·상위 분위수에 양수 배율 s_−, s_+를 적용한다.

`q_cal = m + s_side,season × (q − m)`

목적함수는 용량으로 정규화한 pinball이다. 같은 날의 query·유효 시각에 나누어 그날의 총 weight를 1로 둔다.
단일 배율의 piecewise-linear convex 목적을 weighted breakpoint로 최소화하고, 배율 하한은 1e-6이다.
계절 계수는 `rho = 유효 날짜 수 / (유효 날짜 수 + 20)`으로 해당 계절 fit과 전체 기간 fit을 혼합한다.
마지막에 `[0, 그룹 용량×1h]`로 clipping한다. 중앙값과 W는 보정 전후 동일해야 한다.

Calibration 계수·날짜·query hash를 `calibration.json`에 저장하고 hash를 봉인한 뒤 H 평가를 시작한다.
Seed별 진단 출력에도 동일한 ensemble 보정 계수를 적용한다. Raw 출력은 각 underlying neural model의 후처리 전 값으로 유지한다.

## recalibrate: W 기준 중앙값·구간 보정

`recalibrate`는 완료된 transfer run의 `calibration_B.npz`와 `calibration_T.npz`를 읽는다.
두 자료의 날짜·정답 mask·용량이 같고 원 R0 기록과 일치해야 한다. 유효 정답 시각은 H cutoff 이하이며,
적응 학습 날짜와 겹치면 실행을 중단한다. B·T의 저장 checkpoint와 예측은 그대로 재사용한다.

그룹 용량을 c, 중앙값을 m, 용량 정규화 residual을 `r = (y − m) / c`라 두면 중앙값 shift b의 목적은 다음과 같다.

`min_b Σ (0.1 × SMP × k × c) × |r − b|`

R0의 실제 SMP와 k를 사용한다. 이 식은 clipping 전의 kWh 단위 W와 같으며,
가중 중앙값으로 전체 기간 shift와 계절별 shift를 구한다. 계절 shift는 transfer의 구간 보정과 같은
`rho = 유효 날짜 수 / (유효 날짜 수 + 20)`으로 전체 기간 shift에 수축한다.
해당 계절의 정산 weight 합이 0이면 전체 기간 shift를 사용한다.

이동한 중심 `m + c × b_season`을 고정한 뒤 하단·상단 배율을 용량 정규화 pinball로 fit한다.
폭 학습에서는 날짜마다 총 weight를 1로 두고 계절 배율을 전체 기간 배율에 같은 rho로 수축한다.

`q_cal = m + c × b_season + s_side,season × (q − m)`

최종 출력은 `[0, c×1h]`로 clipping하며 야간을 0으로 둔다. 계절이 빠지면 전체 기간 shift·배율을 사용한다.
Seed별 진단에도 ensemble에서 fit한 같은 계수를 적용한다.
보정 계수와 R0의 가격 weight를 저장하고 hash를 봉인한 뒤 H 정답을 공통 평가기에 전달한다.
이 경로의 q50은 이동하므로 W도 재계산한다.

## 비교와 실행

| Expert ID | 출력 |
|---|---|
| `Tree` | 기존 XGB-only |
| `B` | 기존 DL-only |
| `B_C` | B + R0 calibration |
| `T` | Transfer DL |
| `T_C` | T + R0 calibration |

`B_C`와 `T_C`의 calibration 종류와 parent MK는 manifest의 expert metadata에 기록한다.

다섯 출력은 같은 H cohort·partition·시각·정답 mask·SMP·k를 사용한다.
학습·보정은 `solarvpp2.transfer`, 평가와 viewer 데이터는 [공통 contract](probabilistic-contract.md)가 담당한다.

```powershell
& $py -m solarvpp2 transfer --instance-dir <H-MK5-parent-run> --baseline-artifact <H-quantile-artifact> --open
& $py -m solarvpp2 recalibrate --transfer-run <transfer-run> --open
```

실험은 `sol-artifact/solarvpp2/experiments/transfer-H-<key>/`에 기록한다.
`run.json`은 학습·보정 날짜, 가중치 검증, 실제 패키지와 GPU, 단계 상태를 담는다.
Checkpoint는 기존 `cache/models/`에 별도 key로 보존한다. 비교용 `comparison/`은 공통 분위수 artifact이며,
`comparison/report/`의 CSV·JSON을 Rust viewer로 연다. 원 run·checkpoint와 기본 모델은 유지한다.
`recalibrate` 실행은 `experiments/recalibration-H-<key>/`에 저장한다. `run.json`에 원 transfer run과 checkpoint identity,
R0 입력 hash, 실제 환경, 중앙값 변화량을 기록한다. `comparison/`의 공통 contract는 transfer와 동일하다.
