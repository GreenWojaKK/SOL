# solarvpp 설계

<a id="pipeline"></a>

## 1. 목적과 역할

solarvpp는 태양광 발전소를 두 그룹으로 묶고, 각 그룹의 다음 날 발전량을 예측해 정산손실을 줄인다.
예측 모델은 정해진 그룹의 합계 발전량을 직접 예측한다. 그룹을 새로 정하는 일은 발전소 단위 예측의 오차로 따로 탐색하고,
고른 그룹으로 예측 모델을 다시 학습해 확인한다.

첫 목표는 프로젝트의 고정 기준선 대비 정산손실 10% 감소다. 후보 간 개선은 비교 시점의 기준 모델(champion) 대비
W 변화율로 판단하며, 음수가 개선이다. 지표와 기준선의 등록 정보는 [Problem definition](../PROBLEM.md#loss-recovery)에 둔다.

| 부분 | 입력 | 출력 |
|---|---|---|
| Forecaster | 대상일에 사용할 수 있는 예보·발전소 정보, 그룹 구성 | 그룹의 시간별 발전량 quantiles와 제출값 |
| 발전소 단위 예측 | 같은 입력, 구성원이 한 곳인 학습 단위 | 발전소별 예측과 과거 예측 오차 |
| 공동 오차 모델 | 발전소별 과거 오차, 예보 | 발전소들의 오차가 함께 움직이는 시나리오 |
| 배정 탐색 | 오차 시나리오, 정산 가중, 용량 조건 | 예상 벌점이 낮은 G1·G2 배정 |
| 그룹 재학습 비교 | 고른 배정과 기준 배정 | 실제 제출 경로로 잰 W 차이와 추천 |
| Evaluator | 제출 예측, 같은 구성원의 실제 발전량, 실제 SMP·k | 실제 정산손실 W와 기준 대비 변화 |

## 2. 과제 조건과 구현 선택

과제가 정한 입력·시간축·제출 형식·정산식은 [Problem definition](../PROBLEM.md)을 따른다.

- 253개 발전소를 두 그룹에 한 번씩 배정하고, test 기간에 같은 배정을 유지한다. 그룹당 용량은 20,000 kW 이상이다.
- 대상일 예측에는 전날 17시까지 발행된 정보를 사용한다. 실제 발전량과 observed weather는 학습·평가의 정답이다.
- 출력은 그룹별 24개 hourly energy(kWh)다. 표시 시각 01시부터 다음 날 00시까지가 한 대상일이다.

solarvpp의 계산 구조는 **그룹을 정한 뒤 그 그룹으로 forecaster를 학습하는 방식**이다.
배정이 바뀌면 학습 자료와 예측이 함께 바뀌므로, 배정 후보마다 그룹 모델을 다시 학습해서 비교할 수는 없다.
그래서 탐색은 발전소 단위 예측의 오차로 하고, 최종 확인만 그룹 모델로 한다.
같은 모델로 여러 배정을 바로 비교하는 구조는 [solarvpp2](solarvpp2-design.md)다.

입력 구성, 보조손실, blend 비율, 학습 예산, seed, 오차 모델 종류는 이 구조를 실행하는 선택이다.
현재 지원하는 선택과 기본값은 [구현·실행 참고](solarvpp-reference.md)와 설정 파일에 둔다.

## 3. 계산 흐름

```text
선택한 배정 → 그룹 학습 단위 → XGB + DL → quantile 결합 → 그룹 제출값 → W

발전소 단위 예측 → 과거 오차 → 공동 오차 시나리오 → 배정 탐색 → 고른 배정
                                                                   ↓
                                  같은 모델을 그룹 단위로 다시 학습해 기준 배정과 비교 → 추천
```

제출 경로는 첫 줄이다. 둘째 줄은 첫 줄에 넣을 배정을 정하는 경로이며, 그 결과는 첫 줄을 다시 실행해 W로 확인한다.
실행, 결과 검증, 채택은 독립 명령이다.

<a id="data-contract"></a>

## 4. 학습 단위와 정보 분리

학습 단위(portfolio query)는 **대상일과 발전소 집합**이고, 정답은 구성 발전소의 같은 시각 실적을 합한 그룹 발전량이다.
날짜별로 그룹 안에서 24시간 관측 패턴이 같은 발전소를 한 묶음으로 만든다. 그래서 유효한 실적은 정답에 한 번씩만 들어가고,
관측되지 않은 발전소의 발전량을 채워 넣지 않는다. 값 0은 유효한 정답이다.

평가 기간마다 그 기간의 첫 대상일 전날 17시까지 실적이 있는 발전소를 평가 대상(cohort)으로 고정한다.
학습과 전처리에는 이 시점 이전의 실적과 observed weather만 쓴다. 평가 기간 안에 새로 들어온 발전소는 그 기간의 채점에 넣지 않는다.

자료 연결, 결측 규칙, 평가 대상의 수치는 [Data와 query 계약](solarvpp-reference.md#data-contract)에 둔다.

<a id="forecasting"></a>

## 5. Forecaster의 구성

Forecaster는 XGB와 DL 두 expert를 결합해 그룹의 24시간 발전량을 19개 quantile로 표현한다.

<a id="features"></a>

### 5.1 입력

두 expert 모두 두 제공사의 10시·17시 발행 예보, 시간 표현, 발전소 정적 정보를 받는다.
여기에 태양 위치, 제공사 간 차이와 발행 회차 간 변화, 일사량 같은 물리 근사값, 하루 날씨 상태를 입력 구성으로 더한다.
입력 구성의 이름과 각 값의 계산식은 [입력 구성](solarvpp-reference.md#features)에 둔다.

<a id="xgb"></a>

### 5.2 XGB: 그룹 feature에서 발전량 예측

발전소별 feature를 그룹 안에서 용량 가중 평균·표준편차로 집계하고, 그룹 용량·발전소 수·용량 집중도를 더한다.
태양 위치로 맞춘 power curve를 기준으로 그 차이의 quantile을 학습하고, 추론에서 power curve를 더해 발전량으로 되돌린다.
세부 설정은 [XGB 구현](solarvpp-reference.md#xgb)을 따른다.

<a id="dl"></a>

### 5.3 DL: 발전소 표현을 모아 하루 곡선 예측

| 부분 | 역할 | 출력이 쓰이는 곳 |
|---|---|---|
| Provider encoder | 제공사·관측소별 예보를 시간별 표현으로 변환 | 발전소별 fusion |
| DayState | 관측소의 24시간 예보에서 하루 상태를 표현 | 발전소별 fusion과 보조 head |
| 발전소별 fusion | 관측소 표현·태양 위치·발전소 입력을 결합 | 그룹 pooling과 발전소 보조 head |
| 그룹 pooling | 구성원 표현의 용량 가중 평균·표준편차와 그룹 통계 계산 | 24시간 BiGRU |
| BiGRU | 하루 안의 시간 관계 학습 | Quantile head |
| Quantile head | 시간별 그룹 표현을 발전량 quantile로 변환 | 후처리·결합 |

구조 크기와 선택 구성은 [DL 구현](solarvpp-reference.md#dl)에 둔다.

<a id="losses"></a>

### 5.4 학습 신호

XGB는 quantile loss만 쓴다. DL은 그룹 발전량 quantile을 맞추는 것이 기본이고, 아래 보조항을 더할 수 있다.

| 학습 신호 | 맞추는 것 | 쓰이는 곳 |
|---|---|---|
| 19개 quantile의 pinball | 그룹 발전량 분포 | 항상 |
| 시간 변화 보조항 | 중앙값 곡선이 시각마다 변하는 양 | 등록 모델 전체 |
| 하루 발전량 보조항 | 중앙값 곡선의 하루 합 | `weighted-day-energy` recipe |
| DayState 보조항 | 그룹의 하루 발전량, 관측소의 실측 운량 등 | `daystate` 입력 구성 |
| 발전소 이용률 보조항 | 발전소별 발전량 / 설비용량 | `site_aux_ratio`가 0보다 큰 구성 |

보조항의 세기는 warm-up 뒤 main loss와 보조항의 gradient 크기 비율로 정하고 고정한다.
실측 기상은 학습의 보조 정답으로만 쓰고, 추론에는 예보만 쓴다.
손실의 식과 전체 설정값은 [학습 손실 구현](solarvpp-reference.md#losses)과 [세기 계산](solarvpp-reference.md#aux-calibration)에 둔다.

<a id="inference"></a>

### 5.5 Quantile에서 제출값으로

Expert별로 seed 평균을 낸 뒤 용량 범위로 자르고 야간을 0으로 두고 정렬한다. 그다음 같은 quantile끼리 XGB와 DL을 가중 평균하고 다시 정렬한다.
최종 0.5 quantile이 제출값이다. W는 그룹·시간별 절대오차로 계산하므로 중앙값이 제출값으로 맞다.

<a id="bundling"></a>

## 6. 배정 탐색

1. **발전소 단위 예측:** 같은 입력·손실·학습 함수를 구성원 한 곳인 학습 단위에 적용해, cutoff 이전 기간의 발전소별 예측 오차를 만든다.
2. **공동 오차 시나리오:** 발전소들의 오차가 함께 움직이는 분포에서 시나리오를 뽑는다. 예보에 조건을 건 저랭크 Gaussian,
   예보가 비슷했던 과거 날의 오차를 쓰는 analog, 과거 오차의 평균적인 공분산을 쓰는 unconditional을 비교한다.
3. **탐색:** 배정마다 그룹 오차 합의 중앙값을 제출 보정으로 두고 시나리오의 기대 벌점을 계산해 줄인다.
   2차 근사 문제(QUBO)의 해, 기준 배정, 무작위 배정을 시작점으로 발전소를 옮기거나 맞바꾼다. 두 그룹 용량의 차이에는 벌점을 둔다.
4. **그룹 재학습 비교:** 고른 배정과 기준 배정으로 그룹 모델을 같은 조건에서 학습해 W로 비교하고 추천을 기록한다.

탐색의 기대 벌점은 "발전소 예측의 합 + 중앙값 보정"으로 제출한다고 가정한 값이고, 실제 제출은 다시 학습한 그룹 모델의 중앙값이다.
4단계가 이 차이를 실제 W로 확인한다. 식과 설정은 [Conditional bundling](solarvpp-reference.md#bundling)에 둔다.

<a id="evaluation"></a>

## 7. 평가

평가 단위는 **같은 구성원·시각에서 만든 그룹 예측과 그룹 실적**이다. 모든 후보를 같은 평가 대상·시각·정답 mask·SMP·k로 비교한다.
Forecast 변경은 같은 배정에서, 배정 변경은 같은 모델에서 비교하면 각각의 효과를 읽을 수 있다.

- 고정 기준선은 출발점 대비 누적 진척을 추적한다. 후보 비교 기준은 같은 평가 조건의 champion이다.
- 기간 D·V·H·B는 cutoff와 평가 대상이 다르다. 각 결과는 그 cutoff·평가 대상·학습 이력에서의 진단이다.
- MAE·pinball·ramp 같은 지표는 W가 달라진 이유를 진단하는 데 쓴다.

기간과 cutoff, 실행 수준, 지표 정의는 [평가와 선택](solarvpp-reference.md#evaluation)에 둔다.

<a id="robust-folds"></a>

### 7.1 Robust fold test

학습 cutoff를 3개월씩 옮긴 다섯 fold에서 여러 seed로 후보와 champion을 비교해, 개선이 학습 시점·계절·seed가 달라져도 유지되는지 본다.
Fold의 평가 기간은 서로 겹치므로 독립된 표본이 아니다. Fold 정의와 네 조건은 [Robust fold test](solarvpp-reference.md#robust-folds)에 둔다.

<a id="versions"></a>

## 8. 실행과 재현 정보

학습·예측·채점(`run`), 저장 예측의 비교(`compare`), 채택(`adopt`), 제출 파일 생성(`submit`)은 독립 명령이다.
Run에는 실제 사용한 설정과 모델·전처리·배정·예측·평가의 연결을 남기고, 내용이 같은 계산은 저장된 결과를 재사용한다.

| 정보 | 위치 |
|---|---|
| 등록된 제출 pipeline과 champion | [models.json](../configs/models.json) |
| 기간·seed·학습 예산·입력 구성 | [solarvpp protocol](solarvpp-protocol.json), [runtime.json](../configs/runtime.json) |
| 배정 탐색 설정 | [conditional_bundling.json](../configs/conditional_bundling.json) |
| 수식·옵션·이름·module·cache 세부사항 | [구현·실행 참고](solarvpp-reference.md) |
| 명령과 저장 위치 | [README](../README.md) |

MK는 등록된 제출 pipeline의 식별자다. 후보 component나 설정을 바꾼 실행은 기준 MK와 실제 변경값으로 기록한다.
Package 사이에서는 공통 계약층 `vppcontract`만 공유하고 서로 import하지 않는다.
