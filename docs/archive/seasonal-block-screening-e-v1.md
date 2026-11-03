# Seasonal block screening — E v1

이 문서는 robust fold test 이전에 후속 설계로 두었던 E v1 원문을 보존한다. 현재 설계는 [solarvpp 설계 5.7](../solarvpp-design.md#robust-folds)을 따른다.

## 6.1 목적과 기존 평가의 관계

E v1은 forecast 후보의 개선이 계절·연도·cohort가 달라져도 반복되는지 비교하는 탐색 panel이다.
8개 seasonal block의 leave-one-block-out을 주 비교로 두고, pure future→past split 2개로 direction sensitivity를 확인한다.
Forecast 입력·loss·routing·calibration·blend 후보를 비교하며 composition은 P0로 고정한다.

| 평가 | 학습 방향 | 사용 |
|---|---|---|
| 기존 V/H/B | 각 period의 cutoff 이전 → 이후 | 기존 MK0 기준 10% 목표 추적, 시간순 성능 비교, champion 채택 근거 |
| E v1 seasonal blocks | 평가 block을 제외한 과거·미래 → 해당 block | 후보 ranking, 계절·연도별 일관성과 손실 증가 구간 확인 |
| E v1 reverse splits | 미래 구간만 → 과거 block | 학습 방향이 바뀔 때 후보 개선이 유지되는지 확인 |

V/H/B의 기간·cutoff·cohort·정답 mask와 고정 MK0 prediction은 [기존 protocol](../solarvpp-protocol.json)을 유지한다.
E 결과는 별도 탐색 표에 기록하고, 기존 10% 목표의 진척은 V/H/B 표에서 계속 계산한다.
E는 기존 V/H의 historical labels를 다른 train/eval split으로 재사용하므로 exposure를 `previously_exposed`, 용도를 탐색으로 기록한다.
Champion 채택에서는 시간순 평가에 더 높은 증거 가중치를 둔다. E·V·H·B의 W는 panel별로 집계한다.

## 6.2 데이터 범위와 seasonal blocks

날짜는 모두 KST의 delivery day `(ts−1시간).date`이며 양 끝 날짜를 포함한다.
한 delivery day는 당일 01시부터 다음 날 00시까지의 24개 slot이다.
사용 가능한 historical universe U는 **2021-01-01~2025-05-31**이다. Generation의 첫 유효 기록은 2021-01-19 13시이며,
그 이전 날짜와 발전소별 편입 이전 slot은 기존 generation mask로 처리한다.
2025-06-01 이후는 제출용 forecast 구간으로 둔다([원자료 범위](../../participant_dataset/README.md)).

2023-06~2025-05를 두 번의 여름·가을·겨울·봄으로 나눈다. 2021~2022는 training history에 포함한다.
2021년에는 P0 G2의 generation 기록이 없고, 2022년 5월 cohort는 45개소여서 주 탐색 panel은 2023년 여름부터 구성한다.
2023년 작은 G2에서 2025년 전체 253개소까지의 cohort 변화와 2025년 WB revision 증가를 함께 포함한다
([generation·weather 특성](../data-characteristics.md)).

| Block ID | Cycle | 계절 | Eval delivery days | 일수 / 그룹당 시간 | Cohort G1 / G2 |
|---|---|---|---|---:|---:|
| E23-SU | Y23 | 여름 | 2023-06-01~2023-08-31 | 92 / 2,208 | 100 / 6 |
| E23-AU | Y23 | 가을 | 2023-09-01~2023-11-30 | 91 / 2,184 | 111 / 6 |
| E23-WI | Y23 | 겨울 | 2023-12-01~2024-02-29 | 91 / 2,184 | 124 / 12 |
| E24-SP | Y23 | 봄 | 2024-03-01~2024-05-31 | 92 / 2,208 | 127 / 46 |
| E24-SU | Y24 | 여름 | 2024-06-01~2024-08-31 | 92 / 2,208 | 130 / 75 |
| E24-AU | Y24 | 가을 | 2024-09-01~2024-11-30 | 91 / 2,184 | 134 / 118 |
| E24-WI | Y24 | 겨울 | 2024-12-01~2025-02-28 | 90 / 2,160 | 134 / 118 |
| E25-SP | Y24 | 봄 | 2025-03-01~2025-05-31 | 92 / 2,208 | 134 / 119 |

Y23은 2023-06~2024-05, Y24는 2024-06~2025-05이다. 위 8개 eval block은 서로 겹치지 않는다.
Cohort 수는 `sites.csv`와 유효 generation 기록을 기존 cohort 규칙으로 집계한 값이다.
현재 데이터에서는 각 block의 두 그룹 모두 표에 적힌 전체 시간에 정답이 있다.
2024년 12월 중 마지막 발전소가 편입되므로 E24-WI는 252개소, E25-SP는 253개소이다.

## 6.3 Leave-one-block-out의 학습 범위

Eval block E_b의 시작일을 a_b, 종료일을 z_b로 두고, 앞뒤 **7 delivery days**를 gap으로 제외한다.
Gap은 인접 날짜의 연속된 weather episode가 train/eval 양쪽에 걸리는 영향을 줄이기 위한 고정 설계값이다.

\[
T_b=U\setminus[a_b-7\text{일},\;z_b+7\text{일}]
\]

| Block ID | 과거 train delivery days | 미래 train delivery days | Train 일수 |
|---|---|---|---:|
| E23-SU | 2021-01-01~2023-05-24 | 2023-09-08~2025-05-31 | 1,506 |
| E23-AU | 2021-01-01~2023-08-24 | 2023-12-08~2025-05-31 | 1,507 |
| E23-WI | 2021-01-01~2023-11-23 | 2024-03-08~2025-05-31 | 1,507 |
| E24-SP | 2021-01-01~2024-02-22 | 2024-06-08~2025-05-31 | 1,506 |
| E24-SU | 2021-01-01~2024-05-24 | 2024-09-08~2025-05-31 | 1,506 |
| E24-AU | 2021-01-01~2024-08-24 | 2024-12-08~2025-05-31 | 1,507 |
| E24-WI | 2021-01-01~2024-11-23 | 2025-03-08~2025-05-31 | 1,508 |
| E25-SP | 2021-01-01~2025-02-21 | 없음(U 종료) | 1,513 |

Train 일수는 U 안에서 허용한 날짜 수이다. 실제 generation query 수는 해당 날짜의 유효 label과 관측 mask로 정한다.
E25-SP는 U의 마지막 block이므로 같은 제외 규칙을 적용하면 과거 train만 남는다.
각 fold는 자기 eval block과 gap을 제외한다. 다른 fold의 eval 날짜는 해당 fold의 train 범위에 속하면 학습에 사용한다.

## 6.4 Pure future→past direction check

아래 두 split은 6.7절의 shortlist에 적용한다. Eval block·cohort·P0·정답 mask는 대응하는 seasonal block과 같다.
미래 train 범위만 사용하고, MK0·champion·후보를 그 범위에서 각각 학습한다.

| Split ID | Eval delivery days | Train delivery days | 확인할 조건 |
|---|---|---|---|
| R23-SU | 2023-06-01~2023-08-31 | 2023-09-08~2025-05-31 | 이후 연도의 여름을 포함한 학습으로 2023년 여름 예측 |
| R23-WI | 2023-12-01~2024-02-29 | 2024-03-08~2025-05-31 | 이후 겨울·WB revision 증가 구간을 포함한 학습으로 과거 겨울 예측 |

두 reverse split의 회수율은 각각 같은 split의 champion 대비로 계산한다.
두 값이 모두 양수이면 reverse에서도 개선이 유지되는 후보로 표시하고, 한쪽이라도 0 이하이면 해당 계절과 회수율을 direction-sensitive 항목에 남긴다.
Reverse 결과는 별도 두 행으로 보고하며 8-block 합산 W와 개선 block 수에는 넣지 않는다.

## 6.5 Cohort·입력·fit 범위

**Composition과 cohort.** P0(`sites.csv`의 A→G1, B→G2)를 고정한다.
Eval cohort는 각 block 첫 대상일 전일 17시까지 유효 generation이 한 번 이상 있는 PV이며, block 안에서 고정한다.
이 membership 판단에는 block 이전의 기록 존재 여부를 사용하고, fit에 사용하는 값은 6.3·6.4절의 train mask로 제한한다.
Gap 안에서 편입된 PV도 이 cohort 규칙을 만족하면 평가에 포함한다.
Group·시각별로 cohort의 실제 발전량을 합한 뒤 예측과의 absolute error를 계산한다.
Historical cohort의 용량은 표의 구성대로 사용하고, 최종 test composition의 20 MW 조건은 기존 제출 계약에 적용한다.

**학습 query와 site supervision.** Train 날짜에 유효 label이 있는 전체 253개 PV pool을 사용한다.
기존 P0·24-bit 관측 mask별 query 규칙과 site auxiliary의 유효 label·daylight mask를 적용한다.
Eval cohort 밖의 PV도 train 날짜에 label이 있으면 학습에 참여한다. Fold별 train PV 수·query 수·유효 label 수를 기록한다.
0 출력은 유효 label로 유지하며, 2023년의 연속 0 출력 구간에도 같은 규칙을 적용한다.

**Forecast 입력.** 각 대상일의 전날 round 10·17 예보와 그 revision, 정적 정보, solar geometry를 사용한다.
DayState와 temporal routing은 해당 대상일에 제공된 24시간 forecast에서 입력을 만든다.
Eval generation·observed는 채점과 사후 진단에 사용한다.

**Train mask의 적용 범위.** 아래 단계는 모두 같은 train 날짜 집합 T_b를 받는다.

| 단계 | Fit에 사용하는 자료 |
|---|---|
| Generation query·site target | Train 날짜의 유효 generation. Eval·gap target은 missing과 loss mask 0으로 처리 |
| DayState observed·day-energy auxiliary | Train 날짜의 target과 train 날짜로 계산한 target 표준화 통계 |
| Imputation·standardization·ZCA·physics·power curve·query scaler | 각 변환이 원래 사용하는 표본 중 train 날짜의 자료 |
| Gradient calibration·bias/scale/기상 보정 등 학습된 보정 | Train 날짜의 표본. 별도 검증이 필요한 fit은 T_b 안의 inner split 사용 |
| Checkpoint·r·blend weight·window 등 후보 parameter | Screening round 시작 시 후보별로 고정한 값 |

정적 metadata는 공통 `sites.csv`를 사용한다. 모델·보정·전처리는 fold별 train 범위에서 fit한다.
V/H/B 전체로 학습한 checkpoint·보정값을 E fold의 초기값이나 fitted 전처리로 사용하는 경로는 제외한다.
Train 날짜는 명시적인 집합으로 전달한다. 가장 늦은 train 날짜 하나를 cutoff로 전달하면 중간 eval·gap까지 포함되므로,
`ActualView`·query 생성·전처리·auxiliary에 같은 day mask를 적용하는 계약이 필요하다.

## 6.6 기준 모델과 비교 지표

각 screening round의 시작에 다음을 고정한다.

- E v1 block·gap·P0·cohort·정답 mask와 데이터 범위.
- MK0의 등록 pipeline·기본 parameter, 비교 시점 champion의 pipeline·parameter.
- 각 후보의 전체 구성, seed, 학습 budget. 후보별 r·blend·window 등은 모든 block에 같은 값으로 적용한다.

MK0·champion·후보를 **같은 split의 train 범위에서 각각 학습**한다. E의 MK0·champion prediction은 해당 round의 기준으로 보존한다.
기존 V/H/B의 고정 MK0 prediction은 기존 목표 추적에 사용한다.
Forecast 후보 비교에서는 champion도 P0로 학습·평가하며, 채택 당시의 다른 composition 효과는 기존 V/H/B 비교에서 별도로 본다.

W_b는 [기존 정산식](../../PROBLEM.md#loss-recovery)의 최종 blend forecast에 대한 손실이다.
각 block에서 MK0 기준 회수율과 champion 대비 추가 회수율을 계산한다.

\[
r_{0,b}=\frac{W_{MK0,b}-W_{candidate,b}}{W_{MK0,b}},\qquad
r_{C,b}=\frac{W_{champion,b}-W_{candidate,b}}{W_{champion,b}}
\]

주 ranking은 8개 block의 합산 W에 대한 champion 대비 추가 회수율 R_C,E로 정한다.

\[
R_{C,E}=\frac{\sum_{b=1}^{8}W_{champion,b}-\sum_{b=1}^{8}W_{candidate,b}}
{\sum_{b=1}^{8}W_{champion,b}}
\]

| 보고 항목 | 계산·용도 |
|---|---|
| 합산 추가 회수율·회수금액 | R_C,E와 Σ(W_champion−W_candidate). 주 ranking |
| 개선 block 수 | r_C,b > 0인 block 수 / 8 |
| Median·worst block 회수율 | 8개 r_C,b의 median과 최솟값. 작은 cohort의 악화도 함께 확인 |
| 계절별·cycle별 회수율 | 해당 block들의 W를 합산한 뒤 비율 계산. 계절 4개, Y23·Y24 두 cycle |
| MK0 기준 탐색 회수율 | 위 합산식의 champion을 MK0로 바꿔 계산. E 탐색 표에 기록 |
| Expert별 W | XGB·DL·최종 blend를 함께 기록해 입력·모델·blend 효과 확인 |
| 한낮 bias | 표시 시각 11~15시의 Σ(예측−실적)/Σ실적. 양수는 과대예측 |
| 학습·평가 규모 | Train PV·query·label 수, eval cohort·용량·유효 시간, seed별 결과 |

W를 합산한 회수율과 block 회수율의 median을 각각 표시한다. Cohort·SMP 규모 차이는 합산 W에 반영하고,
개선 block 수·median·worst에서 계절별 일관성을 함께 읽는다. 기존 V noise floor는 해당 V 실험의 참고값으로 유지한다.
8개 block이 모두 완료된 후보에 전체 ranking을 부여한다. 미완료 block은 상태를 표시하고, 0 또는 평균값으로 대체하지 않는다.

## 6.7 Screening·shortlist·채택 근거

학습 budget은 기존 기준인 XGB 400 boosting rounds, DL 2,000 optimizer steps, 마지막 checkpoint를 사용한다.
한 후보를 여러 block에서 비교하는 동안 구조·loss·r·blend·학습 budget을 고정한다.

| 단계 | 범위·seed | 후보 처리 |
|---|---|---|
| Screening | 전체 8 blocks, seed 42 | R_C,E > 0인 후보 중 R_C,E 상위 최대 2개를 shortlist로 둠. Block별 악화도 함께 표시 |
| Seed confirmation | Shortlist·MK0·champion의 전체 8 blocks, seed 42·43·44 | 기존 seed-mean → 후처리 → blend → median 순서로 W 재계산 |
| Direction check | 같은 shortlist의 R23-SU·R23-WI, seed 42·43·44 | Reverse 회수율 두 개와 direction-sensitive 계절 기록 |
| 기존 평가와 비교 | 후보 구성을 고정한 V/H/B | 기존 cutoff 학습·평가와 E 결과를 함께 제시하고 사용자가 채택 결정 |

Seed confirmation에서 **R_C,E > 0, 개선 block 수 ≥ 6/8, worst block 회수율 ≥ −2%**를 모두 만족하면
일관된 개선 후보로 표시한다. −2%는 이 탐색에서 허용하는 개별 block의 손실 증가폭이다.
R_C,E는 양수이나 이 조건 일부가 충족되지 않으면 개선·악화 계절을 명시한 trade-off 후보로 표시한다.
R_C,E가 0 이하이면 이 panel에서 추가 회수가 없는 후보로 기록한다.
동일 조건의 후보는 R_C,E, worst block 회수율, median 회수율이 큰 순서로 정렬한다.

Seed별 R_C,E도 함께 표시한다. 결합 예측의 W는 세 seed의 W 평균 대신 실제 결합 prediction에서 계산한다.
새 split·gap·후보 parameter로 비교를 바꾸면 round를 새로 고정하고, 비교 기준과 후보를 같은 조건으로 맞춘다.
새 round에서도 이전 결과와 악화 block을 보고서에서 추적한다.
기존 V/H/B 결과가 엇갈리면 기간별 W·추가 회수율과 E의 계절별 패턴을 함께 제시한다.
Champion pointer와 제출 모델은 기존 명시적 채택 절차로 변경한다.

## 6.8 구현할 작업과 착수 조건

착수 조건은 E v1의 구현·실행 요청이다. 그때 다음 계약을 현재 pipeline에 연결한다.

| 대상 | 필요한 작업 |
|---|---|
| Train/eval split | 현재 cutoff 기반 경로에 명시적인 train day mask를 추가하고, 모든 target·fit 단계에 전달 |
| Fold panel | 위 날짜·gap·cohort·P0를 고정하고, mode별 학습 범위와 평가 범위를 기록 |
| 기준·후보 비교 | Fold별 MK0·champion·후보 prediction과 동일 평가 조건을 연결하고 8-block·reverse 표를 계산 |
| 결과 판정 | 합산 회수율·개선 block 수·worst·seed 결과를 위 shortlist 기준에 연결 |
| 문서·실행 계약 | 이 설계를 solarvpp 실행 설계와 protocol에 반영하고, 기존 V/H/B 날짜·cutoff·기준 prediction 유지 확인 |
