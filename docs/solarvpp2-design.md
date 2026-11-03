# solarvpp2 설계

## 1. 목적과 역할

solarvpp2는 태양광 발전소를 두 그룹으로 묶고, 각 그룹의 다음 날 발전량을 예측해 정산손실을 줄인다.
예측 모델은 발전소 집합을 입력으로 받아 그룹 발전량을 직접 예측한다. 배정 탐색은 같은 모델에 서로 다른 집합을 넣고,
그 예측에서 예상되는 손실을 비교해 그룹 구성원을 정한다.

첫 목표는 프로젝트의 고정 기준선 대비 정산손실 10% 감소다. 후보 간 개선은 비교 시점의 기준 모델(champion) 대비
W 변화율로 판단하며, 음수가 개선이다. 지표와 기준선의 등록 정보는 [Problem definition](../PROBLEM.md#loss-recovery)에 둔다.

| 부분 | 입력 | 출력 |
|---|---|---|
| Forecaster | 대상일에 사용할 수 있는 예보·발전소 정보, 발전소 집합 | 그룹의 시간별 발전량 quantiles, 제출값, risk용 표현 |
| Risk model | 고정된 forecaster의 출력·표현, 집합 정보 | 그 제출값에서 예상되는 시간별 absolute error |
| Partition optimizer | 후보 집합의 risk, 가격·일자 가중, 용량 조건 | 기간 동안 사용할 G1·G2 배정 |
| Evaluator | 제출 예측, 같은 구성원의 실제 발전량, 실제 SMP·k | 실제 정산손실 W와 기준 대비 변화 |

Forecaster는 발전량을, risk model은 **그 forecaster가 낼 오차의 크기**를 예측한다.
Optimizer가 비교하는 예상 손실과 평가기가 계산하는 실제 W도 구분한다.

## 2. 과제 조건과 구현 선택

과제가 정한 입력·시간축·제출 형식·정산식은 [Problem definition](../PROBLEM.md)을 따른다.

- 253개 발전소를 두 그룹에 한 번씩 배정하고, test 기간에 같은 배정을 유지한다. 그룹당 용량은 20,000 kW 이상이다.
- 대상일 예측에는 전날 17시까지 발행된 정보를 사용한다. 실제 발전량과 observed weather는 학습·평가의 정답이다.
- 출력은 그룹별 24개 hourly energy(kWh)다. 표시 시각 01시부터 다음 날 00시까지가 한 대상일이다.
- 실제 W는 그룹·시간별 오차에 절댓값을 취한 뒤 SMP·시간대 가중을 적용해 합한다.

solarvpp2의 현재 계산 구조는 **집합에 조건부인 forecaster를 학습하고 고정한 뒤 risk와 배정을 학습·선택하는 방식**이다.
집합을 바꾸면 pooling 이후의 예측이 달라진다. 발전소별 point forecast를 고정해 단순 합산하는 구조와는 이 지점에서 다르다.

Encoder·decoder의 종류, 보조손실, blend 비율, sampler, 학습 예산, seed, 평가 날짜는 이 구조를 실행하는 선택이다.
현재 지원하는 선택과 기본값은 [구현·실행 참고](solarvpp2-reference.md)와 설정 파일에 둔다.

## 3. 계산 흐름과 집합 의존성

```text
예보·태양 위치·발전소 속성
    ├─ 발전소별 feature ─────────────────────────────┐
    └─ station/provider 표현 → 발전소 representation ─┤
                                                     ↓ 집합 A로 pooling
                                 집합 feature / 24시간 집합 sequence
                                                     ↓
                                     expert별 quantiles → blend → 제출값
                                                     ↓
                                     출력·집합 표현 → risk → 배정 비교
```

학습은 두 경로로 진행한다. Forecaster training 날짜의 집합 query와 정답으로 예측 모델을 학습한다.
Risk held-out 날짜에서는 고정된 forecaster의 예측과 실현 오차로 risk model을 학습한다.
그 뒤 후보 배정들을 같은 forecaster·risk로 비교하고, 선택한 배정의 그룹 예측을 만든다.

| 계산 | 집합 A와의 관계 | 배정 변경 시 처리 |
|---|---|---|
| Station·provider·발전소 representation | 그룹 구성과 독립 | 대상일별로 계산해 재사용 |
| Pooling·집합 통계 | A에 포함된 발전소로 결정 | 구성원의 기여를 다시 집계 |
| 집합 decoder·quantile head | Pooling 결과에 조건적 | 새 집합에 대해 다시 계산 |
| Risk | 해당 집합의 예측·표현에 조건적 | 새 예측의 예상 오차를 계산 |

이 분리가 현재 incremental search의 근거다. 구성원 간 상호작용을 pooling 이전에 넣는 구현은
배정 변경 시 발전소 representation을 재계산하는 비용까지 함께 다뤄야 한다.

<a id="query-sampler"></a>
<a id="training-days"></a>

## 4. 학습 대상과 정보 분리

학습 query는 **대상일 d, 발전소 집합 A, 24시간 정답 mask**다. 정답은 구성 발전소의 같은 시각 실적을 합한 그룹 발전량이다.
집합은 하루 동안 고정한다. 구성원 중 실적이 없는 발전소가 있으면 해당 시각의 loss를 제외한다.

Sampler는 참조 배정, 무작위·관측소 층화 이분할, 다양한 크기와 station 집중도의 부분집합을 만든다.
여러 집합을 학습하는 목적은 **구성원과 예보가 주어진 집합의 발전량**을 예측하게 하는 것이다.
집합 입력은 membership과 집합 통계이며, G1·G2 이름과 구성원 나열 순서에 의존하지 않는다.

Risk 학습에는 forecaster가 정답을 학습하지 않은 날짜를 사용한다. 그 날짜의 실제 오차를 risk target으로 만들고,
risk를 학습하는 동안 forecaster의 weights는 고정한다. 별도의 평가 panel을 쓰면 그 평가일과 완충 날짜도 forecaster 학습에서 뺀다.

정답에서 만드는 보조 target·power curve·통계는 같은 학습 제외를 적용한다. 예보 입력만으로 맞추는 전처리의 fit 범위는
평가 protocol에서 따로 정한다. P60은 평가일·완충을 입력 전처리에서도 제외하고, risk held-out 날짜의 예보 입력은 사용한다.

현재 full-stage 배정은 평가 기간 시작 전 cutoff의 자료로 결정한다. 미래 test 기간의 예보·가격은 이 배정 계산에 넣지 않는다.
대상일의 발전량 예측은 그날 제출 시점에 사용 가능한 예보를 받는다. 날짜·cohort·mask는
[학습 query](solarvpp2-reference.md#query-sampler)와 [평가 protocol](solarvpp2-reference.md#p60-panel)에 둔다.

<a id="forecaster"></a>

## 5. Forecaster의 구성

Forecaster는 예보 정보 I_d와 집합 A를 받아 24시간의 그룹 발전량 분포를 quantiles로 표현한다.
현재 구현은 XGB와 DL 두 expert를 결합한다.

<a id="xgb"></a>

### 5.1 XGB: 집합 feature에서 발전량 예측

발전소별 예보·물리 feature를 집합 안에서 capacity-weighted mean·std로 집계한다.
집합 용량, 발전소 수, 용량 집중도(HHI), 주간 용량 비율, 시간, 제공사별 결측 정보도 입력한다.
집합 power curve를 기준으로 residual quantiles를 학습하고, 추론에서 power curve를 더해 발전량으로 복원한다.
세부 feature와 학습 설정은 [XGB 구현](solarvpp2-reference.md#xgb)을 따른다.

<a id="dl"></a>

### 5.2 DL: 발전소 표현을 모아 하루 곡선 예측

| 부분 | 역할 | 출력이 사용되는 곳 |
|---|---|---|
| Provider encoder | 제공사·관측소별 예보를 시간별 표현으로 변환 | 발전소별 PV fusion |
| DayState | 관측소의 24시간 예보에서 하루 상태를 표현 | PV fusion과 보조 head |
| PV fusion | 관측소 표현·태양 위치·발전소 입력을 결합 | 집합 pooling과 발전소 보조 head |
| 집합 pooling | 구성원 표현의 용량 가중 평균·표준편차와 집합 통계 계산 | 24시간 decoder |
| Decoder | Pooling된 sequence에서 하루 안의 시간 관계 학습 | Quantile head와 risk model |
| Quantile head | 시간별 집합 표현을 발전량 quantiles로 변환 | 후처리·blend |

DayState의 observed weather 보조 target은 학습 시 표현을 감독한다. 추론의 DayState는 예보에서 계산한다.
발전소 이용률 보조 head는 pooling 이전의 표현이 발전소별 발전량도 설명하도록 학습한다.
**그 보조 head의 발전소 예측을 더해서 최종 그룹 예측을 만드는 것은 아니다.** 그룹 예측은 pooling 이후 decoder·quantile head가 계산한다.

추가 component는 역할에 맞는 위치에 연결한다.

| 기능 | 위치 | 바꾸는 계산 |
|---|---|---|
| 관측소 간 context | Provider encoder 뒤 | 같은 대상일의 여러 관측소 정보를 교환 |
| 예보 시각 선택 | Pooling 전 provider 표현 | 대상 시각을 예측할 때 읽을 주변 시각과 비중 선택 |
| 그룹 시각 선택 | Pooling 뒤 | 그룹 맥락으로 주변 시각의 표현 결합 |
| Decoder 교체 | Pooling 뒤 | 집합 sequence에서 시간 관계를 계산하는 방법 |
| 하루 배율 head | Decoder 뒤 | 하루의 모든 시각·quantile에 같은 양의 배율 적용 |

현재 구현의 종류, tensor shape, 초기화와 gradient 경로는 [DL 구현](solarvpp2-reference.md#dl)에 둔다.

### 5.3 학습 신호와 gradient

Main pinball loss는 그룹 발전량 quantiles를 감독한다. 선택한 보조항은 그 계산에 추가 학습 신호를 준다.

| 학습 신호 | 감독 대상 | Gradient가 닿는 범위 |
|---|---|---|
| Group pinball·median 보조손실 | 그룹 quantiles 또는 그 median의 하루 곡선 | Quantile head·decoder·pooling 이전 표현 |
| 발전소 이용률 보조손실 | 발전소별 정규화 발전량 | Site head·PV fusion·연결된 provider/DayState 표현 |
| DayState 보조손실 | 하루 상태와 energy/observed target | 해당 head와 연결된 DayState 경로 |

보조항의 사용 여부와 세기는 설정에서 정한다. 현재 학습기는 warm-up 뒤 main loss와 보조항의 gradient norm으로
계수를 정하고 이후 고정한다. 정답 mask, unique-day 집계, calibration의 parameter 범위는
[학습 손실 구현](solarvpp2-reference.md#training-objective)에 명시한다.

### 5.4 Quantiles에서 제출값으로

현재 후처리는 expert별 동일 quantile의 seed 평균을 계산한 뒤 capacity로 kWh를 복원한다.
용량 범위 clipping·야간 0 처리·quantile 정렬 후, 같은 quantile끼리 XGB와 DL을 가중 평균하고 다시 정렬한다.
최종 0.5 quantile이 제출값이다. Seed 목록과 expert 결합 비율은 실행 설정이다.

학습된 weights와 후처리를 고정한 상태에서, 예측·배정 탐색은 같은 집합 입력에 같은 제출값을 계산한다.
Ensemble W는 **결합된 예측으로 다시 계산**하며 seed별 W의 평균과 따로 기록한다.

<a id="risk"></a>

## 6. 제출값의 risk

Risk는 집합의 실제 발전량 Y와 고정 forecaster의 제출값 a 사이 absolute error의 조건부 평균이다.

\[
R_{\phi,h}(I_d,A)\approx E[|Y_{A,h}-a_{\theta,h}(I_d,A)|\mid I_d,A].
\]

이 target에는 forecaster의 bias도 포함된다. Risk 입력은 decoder의 시간별 집합 표현, 최종 median과 quantile 폭,
expert별 median, 집합 통계와 시간 정보다. 정산 가격은 risk 이후 partition objective에서 적용한다.

현재 구현은 held-out block마다 그 block을 제외하고 risk head를 학습한다. 배정 objective에서 어떤 block을 읽을 때는
그 block의 오차를 학습하지 않은 head를 사용한다. 전체 held-out 자료로 학습한 별도 head는 평가 기간의 risk 진단에 쓴다.
현재 head와 regression objective는 [Risk 구현](solarvpp2-reference.md#risk)에 둔다.

<a id="partition"></a>

## 7. Partition 선택

Optimizer는 용량 조건을 만족하는 G1·G2 배정에 대해 다음 예상 손실을 비교한다.

\[
\hat J(x)=\sum_{d\in\mathcal D}\omega_d\sum_h\pi_{d,h}
\{R_{\phi,h}(I_d,A_1(x))+R_{\phi,h}(I_d,A_2(x))\}.
\]

𝒟는 배정 비교에 사용하는 날짜, ω는 날짜의 비중, π는 정산 가격·시간대 가중의 추정치다.
이 선택은 배정할 기간의 조건을 과거 자료로 어떻게 대표할지 정한다. 현재 구현은 held-out 날짜와 과거 월·시간별 SMP profile을 사용한다.

현재 search는 참조·층화·무작위 배정에서 시작해 발전소를 이동하거나 교환하는 flip·swap 방식이다.
이동한 발전소의 기여만 집합 통계에 더하고 빼며, 그 이후 예측·risk는 일반 추론과 같은 함수로 계산한다.
가격 근사, 과거 cohort의 용량 조건과 search 설정은 [Partition 구현](solarvpp2-reference.md#partition)에 둔다.

<a id="evaluation"></a>
<a id="diagnostics"></a>

## 8. 평가와 실패 분석

평가 단위는 **같은 구성원·시각에서 만든 그룹 예측과 그룹 실적**이다. 실제 W는 각 그룹·시간별 absolute error에
실제 SMP·k를 적용해 합한다. 날짜나 계절을 먼저 평균하면 다른 날짜의 부호 있는 오차가 상쇄되므로, W는 원래 시간축에서 계산한다.

- 고정 기준선은 출발점 대비 누적 진척과 손실 10% 감소 목표를 추적한다.
- 후보 비교 기준은 같은 평가 조건의 champion 또는 해당 실험에서 지정한 기준 run이다.

Forecast 변경은 같은 배정에서, partition 변경은 같은 forecaster에서 비교하면 각각의 효과를 읽을 수 있다.
최종 후보는 모델·후처리·배정을 묶어 같은 평가 조건의 W로 비교한다. 기준과 후보의 run·평가일·구성원·mask·가격·seed를 기록한다.

진단은 시간대·계절·예보 조건·발전소 이력에 따라 residual과 후보−기준의 손실 차이를 나눠 본다.
일별 원곡선은 날짜마다 달라지는 크기·시간 패턴을,계절 평균은 반복되는 편향을 보여준다.
그룹 간 signed residual과 risk calibration은 함께 틀리는 오차와 예상 손실 추정의 문제를 구분하는 데 사용한다.

<a id="p60-panel"></a>

### 8.1 Forecast validation panel

현재 forecast 후보 비교에는 고정 P60 panel을 사용한다. 평가일을 학습에서 제외하면서 발전소별 다른 날짜·계절 이력을 남겨,
최종 test에 가까운 학습 이력으로 후보를 비교한다. 평가일 앞뒤 자료를 모두 학습에 사용할 수 있다.
새 발전소는 첫 완전한 관측일부터 해당 평가일의 예측과 W에 포함한다.

평가일 60개, 완충 구간, risk held-out 날짜와 전처리 fit의 구분, 날짜별 구성원, 계산 명령은
[P60 protocol](solarvpp2-reference.md#p60-panel)에 둔다. Panel의 기준 run은 같은 panel에서 계산한 W를 사용한다.

V·H·B와 robust fold 결과는 각 cutoff·cohort·학습 이력에서의 진단이다. 현재 CLI의 full-stage 추천은
별도의 H·B 규칙으로 동작하며, 그 구체적인 동작은 [추천 명령](solarvpp2-reference.md#recommendation)에 둔다.

<a id="instances"></a>

## 9. 실행과 산출물

Instance는 학습 cutoff, 발전소 cohort, 예측·평가 기간을 묶는다. 현재 CLI는 다음 실행 범위를 제공한다.

| stage | 실행 범위 |
|---|---|
| forecast | Forecaster 학습과 고정 배정의 예측·평가 |
| panel | 평가일을 학습에서 제외한 forecaster 학습과 날짜별 구성원의 예측·평가 |
| full | Forecaster → risk → partition search → 최종 그룹 예측·평가 |

Run에는 실제 사용한 설정과 모델·전처리·배정·예측·평가의 연결을 남긴다. 저장된 예측만으로 가능한 재평가는 학습과 분리한다.
Probabilistic artifact는 expert별·seed별 quantiles를 공통 형식으로 추출해 [확률예측 viewer](probabilistic-contract.md)에서 사용한다.

명령과 파일 위치는 [구현·실행 참고](solarvpp2-reference.md#instances), 저장 정책은 [README](../README.md#storage)를 따른다.

<a id="versions"></a>

## 10. 구현 선택과 재현 정보

구조의 역할은 위 계산 흐름으로 설명하고, 실행할 component와 수치는 아래 파일에서 선택한다.

| 정보 | 위치 |
|---|---|
| 등록된 제출 pipeline과 CLI 기본 모델 | [models.json](../configs/solarvpp2/models.json) |
| 입력 구성·학습 예산·평가 panel·실험 설정 | [runtime.json](../configs/solarvpp2/runtime.json) |
| 비교 시점의 champion | [champion.json](../configs/champion.json) |
| Component별 옵션·초기화·loss·추론 세부사항 | [구현·실행 참고](solarvpp2-reference.md) |

MK는 등록된 제출 pipeline의 식별자다. 후보 component나 설정을 바꾼 실행은 기준 MK와 실제 변경값으로 기록한다.
제출 경로가 없는 실험은 이름과 설정으로 기록한다. 코드가 계산하는 값의 변경은 component version으로 추적하고,
계산 그래프나 node 계약의 변경을 구분하는 package 정책은 [프로젝트 개발 규칙](../AGENTS.md)에 둔다.

Package 사이에서는 시간축·정산·산출물·평가의 공통 계약층 `vppcontract`를 공유한다.
현재 각 package의 forecaster·risk·partition 구현은 서로 직접 import하지 않고 독립적으로 실행한다.

