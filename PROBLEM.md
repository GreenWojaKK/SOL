# Problem Definition

## 1. Task와 목표

태양광 VPP의 forecast와 partition을 개선해 baseline에서 발생하는 정산손실 W를 회수한다.
첫 목표는 **고정 기준선 정산손실의 10% 감소(W 변화율 −10%)**이며, 후보는 **현재 기준 모델(champion) 대비 W 변화율**로 비교한다.

253개 태양광 발전소를 두 집합자원 G1·G2로 partition하고, 각 집합의 다음 날 hourly generation을 예측해 정산 수익을 최대화한다.
전체 설비용량은 42,655.37 kW다. 결정할 값은 평가 기간에 고정할 partition과 그룹별·시각별 제출 예측값이다.

이 과제는 확보한 발전소를 예측·정산 단위로 편성하는 소규모 전력중개 업무의 toy abstraction이다.
정산은 SMP·시간대 가중치를 적용한 선형 absolute-error penalty를 사용한다.

<a id="loss-recovery"></a>

### 1.1 W 변화율과 champion–challenger

W는 최종 그룹 forecast의 정산손실이며 단위는 원이다. 같은 평가 조건에서 ΔW는 후보 W에서 기준 W를 뺀 값이다.

| 지표 | 계산 | 판단 기준 |
|---|---|---|
| 고정 기준선 대비 W 변화율 | `(W_candidate − W_baseline) / W_baseline` | 출발점 대비 누적 개선. 첫 목표는 −10% |
| Champion 대비 W 변화율 | `(W_challenger − W_champion) / W_champion` | 후보 비교. 음수이면 개선이며 낮을수록 좋다 |

W 변화율은 비율로 계산하고 %로 표시한다. 음수는 손실 감소, 0%는 같은 손실, 양수는 손실 증가이다.
손실 10% 회수는 `W_candidate ≤ 0.90 × W_baseline`에 해당한다.

고정 기준선은 프로젝트 출발점의 예측을 보존한 비교 대상이다. Champion이 바뀌어도 누적 진척의 분모를 유지한다.
Champion은 현재 후보와 비교할 예측 pipeline이며, 채택한 후보로 갱신한다.
Champion 비교에는 비교 시점의 champion 모델과 사용한 parameter·partition·seed·prediction artifact를 명시한다.

같은 발전소 cohort·평가 시각·정답 mask·SMP·k에서 기준과 후보의 W를 계산한다.
비교에는 기준·후보의 W, ΔW와 W 변화율을 제시하고, 고정 기준선과 비교할 때 10% 손실 감소 목표 도달 여부를 함께 기록한다.
Forecast와 partition의 변경 효과는 최종 제출 방식으로 생성한 그룹 forecast의 W로 판단한다.
MAE·pinball·ramp 등의 지표는 W가 달라진 이유를 진단하는 데 사용한다.

<a id="baseline-record"></a>

### 1.2 기준선 기록과 평가 protocol

현재 고정 기준선의 등록 이름은 `solarvpp:MK0`이다. [모델 등록 목록](configs/models.json)의 해당 parameter와
원래 배정(`original`: `sites.csv`의 A→G1, B→G2), seed 42·43·44로 만든 기간별 prediction artifact를 사용한다.
MK는 등록 구성을 찾는 식별자이며, 이 기준선은 V·H·B별 누적 진척을 추적한다.

solarvpp2의 forecast 후보는 [P60 panel](docs/solarvpp2-design.md#p60-panel)의 같은 학습·평가 조건과 seed로 비교한다.
기존 V/H/B와 [robust fold test](docs/solarvpp-reference.md#robust-folds)는 해당 cohort·학습 이력에서의 진단으로 기록한다.
P60의 분모에는 P60에서 계산한 기준 W를 사용하고, 기존 기간의 MK0 baseline과 10% 목표 추적은 그 기간 안에서 유지한다.

### 1.3 Historical snapshots

Historical asset universe는 generation 이력이 있는 발전소가 늘면서 다음과 같이 변한다.
개발 목표는 각 historical snapshot에 반복 적용할 수 있는 forecasting·bundling 방법이다.

| Snapshot | 발전소 수 |
|---|---:|
| 2022년 5월 cutoff | 45 |
| 2023년 5월 cutoff | 106 |
| 2024년 5월 cutoff | 205 |
| 2024년 12월 cutoff | 253 |

## 2. 제공 데이터와 정보 가용 시점

| 데이터 | 단위·키 | 역할 |
|---|---|---|
| Sites | 발전소별 `site_id` | Capacity, 좌표, 참조 배정, provider별 station 매핑과 거리 |
| Generation | `site_id, ts` | 발전소별 실제 hourly energy, kWh |
| WA/WB forecast | provider별 station, `round, ts` | 대상일 전날 10시·17시 발행 weather forecast |
| WA/WB observed | provider별 station, `ts` | 학습 구간의 실제 weather |
| SMP | `ts` | 시간별 가격, 원/kWh; test에는 시간대 가중치 k도 제공 |

Train에는 generation·forecast·observed weather·SMP가 있다. Generation의 실제 시작은 2021-01-19 13:00 KST이고,
마지막 표시 시각은 2025-06-01 00:00 KST다. 발전소마다 generation 이력의 시작과 결측 시각이 다르다.
Weather와 SMP의 시작·행 수·전체 schema는 [참가자 데이터 명세](participant_dataset/README.md)를 따른다.

예측 기준 시점은 대상일 전일 17:00 KST다. `round=10`과 `round=17`은 모두 해당 대상일의 전날 발행 예보다.
각 발전소의 weather는 제공된 `wa_station`, `wb_station` 매핑으로 연결된다. Provider별 변수 구성과 단위는
[weather schema](docs/weather-schema.json)에 정의되어 있다.

Test 대상일은 2025-06-01부터 2026-05-31까지다. 이 기간에는 forecast와 SMP·k가 제공되고,
generation·observed weather의 realized feedback은 없다.

좌표는 최대 2 km 이동 후 소수 4자리로 반올림된 값이다. Station 매핑과 거리는 이동 전 위치 기준으로 제공된다.
좌표로 발전소·사업자·원래 위치를 특정하는 행위는 실격 사유다([데이터 명세](participant_dataset/README.md)).

## 3. 시간과 출력 단위

시각은 Asia/Seoul(KST), 1시간, 우측닫힘이다. `13:00` generation은 12–13시 동안의 energy이며 단위는 kWh다.
하루 출력은 대상일 01시부터 다음 날 00시까지의 24개 값이고, 대상일은 `(ts−1시간).date`로 정한다.

Test의 제출·채점 시간축은 `smp.csv`의 8,760시각이다.

- 첫 표시 시각: 2025-06-01 01:00 KST
- 마지막 표시 시각: 2026-06-01 00:00 KST

설비용량은 kW, 실제·예측 generation은 kWh, SMP는 원/kWh다.

## 4. Partition과 제출 계약

253개 발전소를 각각 G1 또는 G2에 한 번씩 배정하고, test 평가 기간 동안 같은 partition을 유지한다.
각 그룹의 설비용량 합은 20,000 kW 이상이어야 한다.
`sites.csv`의 `current_group` A·B는 참조 배정이며 제출할 partition을 새로 정할 수 있다.

| 파일 | 컬럼 | 데이터 행 |
|---|---|---:|
| `composition.csv` | `site_id, group` | 전체 253개 발전소 |
| `forecast.csv` | `group, ts, forecast_kwh` | G1·G2 × 8,760시각 = 17,520 |

`ts`에는 timezone을 포함하고, `forecast_kwh`는 0 이상이다.
전체 컬럼·행 수·파일 예시는 [제출 형식](participant_dataset/README.md#제출하는-파일)을 따른다.

## 5. 정산식

그룹 g, 시각 h의 실제 generation을 Y_g,h, 제출값을 a_g,h라 두면 다음과 같다.

\[
S=\sum_{g,h}SMP_hY_{g,h}-W,\qquad
W=0.10\sum_{g,h}SMP_hk_h|a_{g,h}-Y_{g,h}|.
\]

k_h는 표시 시각의 시가 11~15이면 3, 나머지는 1이다. SMP는 0 이상이고 0원인 시각도 있다.
매전수익과 S·W의 단위는 원이다.

같은 발전소 pool·평가 시각에서 partition과 예측은 총 매전수익을 바꾸지 않는다.
따라서 정산 수익 S의 최대화는 W의 최소화와 같다.

정산 단위는 그룹·시각이다. 그룹 실적은 구성 발전소의 실적을 합한 값이며, 이 합과 그룹 forecast의 차이에 절댓값을 취한다.
발전소별 forecast를 합산하는 경우 그룹 오차는 부호 있는 발전소 오차의 합이므로, 같은 그룹·같은 시각의 +·−오차가 먼저 상쇄된다.
그룹·시각별로 남은 절대오차에 0.10·SMP·k를 곱해 W를 합산하고, 이 W로 두 기준 대비 W 변화율을 계산한다.

## 6. 구현 설계

- [solarvpp 설계](docs/solarvpp-design.md): 그룹 forecaster와 배정 탐색의 역할과 연결. 수식·설정은 [구현·실행 참고](docs/solarvpp-reference.md).
- [solarvpp2 설계](docs/solarvpp2-design.md): 집합 forecaster와 risk model을 사용하는 계산 그래프.
- [README](README.md): 실행 환경·명령·산출물.
