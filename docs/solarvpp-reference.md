# solarvpp 구현·실행 참고

[설계 본문](solarvpp-design.md)은 forecaster·배정 탐색의 역할과 연결을 설명한다.
이 문서는 현재 구현의 수식·설정·옵션과 CLI의 평가·저장 동작을 기술한다.
실행에는 run에 기록된 resolved 설정을 사용한다.

<a id="pipeline"></a>

## 1. 단계별 예측값과 실행 기본값

Bundling에서 사용하는 예측은 단계에 따라 다음과 같다.

| 단계 | 예측값 | 용도 |
|---|---|---|
| Site primitive | 구성원 1개 query의 최종 blend 점예측 | OOT residual 생성 |
| Scenario 기반 평가 | site 점예측의 그룹 합 + 해당 그룹 residual scenario의 median correction | 배정 탐색·실현 평가 |
| Group-level transfer | 선택 partition으로 재학습한 group forecaster의 최종 blend | 실제 제출 pipeline에서 original과 비교 |

각 명령은 필요한 dependency를 계산하고 내용이 같은 cache를 재사용한다.
Forecasting의 기본 실행 모델은 [models.json](../configs/models.json)의 `champion`이고, 등록 구성은 [6.1절](#versions)에 둔다.
Bundling은 실행 설정의 `primitive.model`을 사용한다. 초기 grid와 개별 recipe의 기본값은 [7절](#recipes)에 둔다.

Partition의 기본값 original은 `sites.csv`의 A→G1, B→G2이다. `--partition <csv>`를 지정하면 해당 배정을
query 구성·학습·평가에 적용한다. original의 cohort 수치와 초기 grid 조건은 해당 절에 별도로 표시한다.

<a id="data-contract"></a>

## 2. Data와 query 계약

데이터 정의는 [참가자 데이터 명세](../participant_dataset/README.md), 전체 기상 컬럼·단위·대응 관계는
[weather schema](weather-schema.json)를 따른다.

### 2.1 Data 연결과 전처리

| 데이터 | 관측 단위·키 | 역할 |
|---|---|---|
| `sites.csv` | `site_id` | 설비용량·위경도·고정 그룹·WA/WB 지점 매핑 |
| `generation.parquet` | `site_id, ts` | 해당 1시간의 발전량 정답, kWh |
| `forecast_wa/wb` | 제공사 station, `round, ts` | 전일 10·17시 발행 예보 |
| `observed_wa/wb` | 제공사 station, `ts` | 해당 지점의 기상 실측 |
| `smp.csv` | `ts` | 정산 가격과 평가 시간축 |

PV의 WA/WB 자료는 `sites.csv.wa_station`, `sites.csv.wb_station`으로 연결한다.
기상 예보와 실측의 대응은 같은 제공사·같은 station·같은 ts에서 만든다.
여러 PV가 공유하는 station의 기상 encoder 출력은 한 번 계산한 뒤 PV로 전달한다.

A는 forecast/observed에 같은 10개 기상 컬럼을 갖는다. B는 공통 10개,
forecast 전용 10개, observed 전용 5개를 갖는다. Schema 파일에 모든 컬럼을 보존하고,
모델 입력마다 원본 컬럼·변환·집계·파생식을 feature manifest에 기록한다.

- A/B 입력은 제공사 prefix를 유지한다.
- 풍속은 m/s로 맞춘다. 풍향은 sin/cos 및 풍속을 곱한 두 성분으로 표현한다.
- B 강설량은 cm→mm로 변환하고 B 변수로 유지한다.
- Cross-provider feature는 기온·상대습도·운량·풍속·풍향 성분에 적용한다.
- 17−10시 revision은 같은 제공사·같은 변수에서 계산한다.
- 태양 위치는 PV 좌표와 구간 중앙 시각으로 계산한다. PV 좌표는 최대 2 km 이동 후 소수 4자리로 반올림된 값이다.
  이동이 태양 위치에 주는 영향(태양시 약 5초, 고도 0.03° 이하)은 1시간 구간에서 무시할 수준이므로 그대로 쓴다.
  PV↔station 매핑과 거리는 이동 전 위치 기준인 파일 값을 쓰고, 좌표로 거리를 다시 계산하지 않는다.
- 강수·강설과 observed 기압은 제공사별 원본 정의를 유지해 해당 branch에 넣는다.
- WA 예보 기압 `wa_ground_press`는 test 구간 중 지점기압으로 정의가 바뀌므로
  forecast 역할 입력에서 원값과 revision을 모두 제외한다([데이터 특성](data-characteristics.md) 1절).
- WA 예보의 `wa_temp_max`·`wa_temp_min`은 99.995% 행에서 `wa_temp`와 같으므로 forecast 입력에서 제외하고,
  schema와 observed에는 보존한다.

중복 키와 정시가 아닌 timestamp 규칙은 forecast와 observed에 같이 적용한다.
중복 키는 컬럼별로 처리한다. 중복 값이 일치하면 하나를 유지하고, 값이 충돌한 컬럼은 해당 키에서 결측으로 둔다.
정시가 아닌 timestamp는 QA 기록에 남기고 정시 join에서 제외한다.
관측 이상값의 범위와 단위는 schema의 quality 규칙을 따르며, 이 범위 규칙은 observed에만 적용한다.
XGB는 결측값을 그대로 처리하고, DL은 학습 구간의 대체 통계와 missing indicator를 사용한다.

유효 발전량은 유한한 0 이상의 값이다. 값 0은 유효 정답이며, 행 누락·NaN·음수는 정답 mask를 0으로 둔다.
기상 학습에는 generation mask를 적용하지 않는다.
발전량 관측 용량 비율로 표본을 자르는 임계값은 두지 않는다.

<a id="portfolio-query"></a>

### 2.2 Portfolio query

모델은 지정한 PV 집합 A의 발전량 분포를 직접 출력한다.

\[
C_A=\sum_{i\in A}capacity_i\times1h,\qquad
Y_{A,t}=\sum_{i\in A}Y_{i,t},\qquad z_{A,t}=Y_{A,t}/C_A
\]

한 query는 하루 24시간 동안 같은 PV 집합을 사용한다. 입력의 기상·태양 위치·설비 정보도
이 A에 대해 계산한다. Query의 용량, PV 수, 용량 집중도와 구성 mask를 함께 전달한다.
모델의 정규화 분위수에 C_A를 곱해 kWh 분위수를 얻는다.

학습 query는 다음 규칙으로 만든다.

1. 날짜별·선택한 partition의 그룹별로 각 PV의 24비트 generation 관측 mask를 만든다.
2. 동일한 mask를 가진 PV를 하나의 query로 묶는다. 하루 내 query 구성은 고정한다.
3. Mask가 1인 시각에 그 query의 실제 발전량 합을 정답으로 사용한다.
4. 24비트가 모두 0인 PV는 해당 날짜의 발전량 학습에서 제외한다.

이 규칙은 유효 발전량 행을 한 query의 정답에 한 번씩 포함한다.
관측되지 않은 PV의 발전량을 합계에 채워 넣는 단계는 두지 않는다.

<a id="cohort"></a>

### 2.3 Evaluation cohort

각 구간의 첫 대상일 전일 17시까지 유효 발전량이 한 번 이상 존재하는 PV를 평가 cohort로 고정한다.
선택한 partition의 G1·G2 배정을 유지한다. Cohort 결정은 이 정보 cutoff까지의 generation만 사용한다.

연간 예측은 이 고정 cohort 전체에 대해 먼저 생성한다. 채점 시에는 해당 cohort의 발전량이
모두 존재하는 그룹·시각에만 정답 mask를 1로 둔다. 이 mask를 모든 후보에 공통으로 적용한다.
기상 입력의 결측이나 모델의 예측값은 채점 mask를 바꾸지 않는다.

original에 적용한 연간 cohort와 채점 시간은 다음과 같다.

| 구간 | 그룹 | PV 수 | 용량 kW | 채점 가능 시간 / 전체 시간 |
|---|---|---:|---:|---:|
| D | G1 | 41 | 3,008.415 | 8,748 / 8,760 |
| D | G2 | 4 | 386.940 | 8,760 / 8,760 |
| V | G1 | 100 | 8,332.545 | 8,784 / 8,784 |
| V | G2 | 6 | 579.485 | 8,784 / 8,784 |
| H | G1 | 130 | 19,267.035 | 8,760 / 8,760 |
| H | G2 | 75 | 13,897.335 | 8,760 / 8,760 |

PV 목록과 월별 채점 시간은 [cohort 근거](evidence/cohort-coverage.json)에 기록한다.
연간 비교의 지표에는 `cohort_id`, PV 수, 용량을 붙이고 `S_cohort`, `W_cohort`로 명명한다.
최종 test는 253개 전체를 선택한 partition의 두 그룹으로 나눠 query로 사용한다. 구성 용량 검사는 이 전체 구성에서 수행한다.

<a id="forecasting"></a>

## 3. Forecasting primitive

두 expert는 선택한 partition의 portfolio query를 학습하고 0.05부터 0.95까지 0.05 간격의 19개 quantile을 출력한다.
각 expert의 입력·time weight와 DL loss는 MK·recipe·명시적 override로 선택한다.

<a id="features"></a>

### 3.1 Input representation

| 이름에 쓰는 말 | 뜻 |
|---|---|
| `raw` | 제공사별 raw 예보(round 10·17) + 시간 표현 + 정적 정보만 쓴다. 명시적 태양 입력 열이 없다 |
| `solar` | 태양 입력 열: 태양 위치, power curve 값, 일출 용량 비율 |
| `diff` | 제공사 간 공통 변수 disagreement + 같은 제공사의 10→17시 revision |
| `physics` | PV physics proxy: clear-sky 일사량, 운량 보정 일사량, 이슬점 차, 온도 감률 |
| `daycontext` | XGB 하루 하늘 상태 요약: query-day의 운량·일사·강수·UV·시정·운고 집계 |
| `daystate` | DL 하루 상태: 관측소별 하루 상태 Z_d와 그 observed 보조 target |

| 입력 구성 | 담는 정보 |
|---|---|
| forecast-raw | 제공사별 raw 예보(round 10·17) + 시간 표현 + 정적 정보 |
| forecast-solar | forecast-raw + 태양 위치 |
| forecast-solar-diff | forecast-solar + disagreement + revision |
| forecast-solar-physics | forecast-solar + PV physics proxy |
| forecast-physics | forecast-solar + disagreement·revision + PV physics proxy |
| forecast-physics-daycontext | forecast-physics + XGB 하루 하늘 상태 요약 |
| forecast-physics-daystate | forecast-physics-daycontext + DL 관측소 하루 상태와 observed 보조학습 |

`daycontext`는 시간 단위 행이라 하루 맥락이 없는 XGB에 query-day의 하늘 상태 요약을 더한다. 운량·운량 보정 일사·강수 확률·UV·시정·운고 채널(두 round와
revision 포함)의 query 평균을 주간 시각에서 cos 천정각 가중 평균·최솟값·최댓값으로 집계해 24시간에 같은 값으로 둔다.
DL은 24시간 BiGRU로 하루 맥락을 이미 보므로 `daycontext`의 영향을 받지 않는다.

야간 0 규칙과 XGB의 power-curve target 변환은 모든 입력 구성에서 고정한다.
forecast-raw는 명시적 태양 입력 열(태양 위치, power-curve 값, 일출 용량 비율)만 제거하고 나머지 파이프라인을 유지한다.
시간 표현(구간 중앙 시각과 대상일 연중일의 sin·cos)과 정적 정보는 모든 구성에 넣는다.
정적 정보는 XGB에서 위도·경도·용량·`wa_dist`·`wb_dist`, DL에서 위도·경도·용량이다.
`daystate`가 없는 구성의 DL에서 비활성 입력 묶음은 0과 mask로 두고 같은 크기의 trunk를 사용한다. `daystate`는 day-state channel만큼 fusion input을 확장한다.

PV physics proxy는 학습하지 않는 고정식이다.

| Proxy | 식 | 계산 단위 |
|---|---|---|
| clear-sky 일사량 G_cs | Haurwitz: 1098·cosθz·exp(−0.057/cosθz) W/m², cosθz ≤ 0이면 0 | PV·시각별 1개, 구간 중앙 |
| 운량 보정 일사량 G | G_cs·(1 − 0.75·c^3.4), c = 운량/100 | 제공사·round별 |
| 이슬점 차 | T − T_d. B는 예보 이슬점, A는 기온·습도에서 Magnus식(a=17.62, b=243.12 °C) | 제공사·round별 |
| 온도 감률 | 1 − 0.004·(T_module − 25), T_module = T + 0.031·G | 제공사·round별 |

A 예보의 습도는 100%를 넘는 값이 있으므로 Magnus식에는 [1, 100]%로 자른 값을 쓴다.
XGB는 proxy를 다른 PV 단위 변수와 같이 용량 가중 평균·표준편차로 집계하고, DL은 x_common에 넣는다.

<a id="xgb"></a>

### 3.2 XGBoost

입력은 query A에 속한 PV들의 PV 단위 변수이며, 입력 구성([3.1절](#features))에 따라 묶음을 고른다.
forecast-solar-diff의 입력은 제공사별 예보·revision·공통변수 차이·태양 위치·시간 표현·정적 정보이고, forecast-physics는 여기에 PV physics proxy를 더한다([3.1절](#features)).
연속 변수는 용량 가중 평균과 표준편차로 집계한다. 풍향 변환은 집계 전에 적용한다.
Query 용량·PV 수·용량 집중도·구성 indicator를 추가한다.
시간 표현은 구간 중앙 시각의 sin·cos와 대상일 연중일의 sin·cos이며 모든 입력 구성에 넣는다.

파워커브는 학습 query의 태양 위치와 정규화 발전량으로 맞춘다.
XGB는 정규화 잔차 `z−power_curve`의 19개 분위수를 학습하고 예측 시 파워커브를 더한다.
이 target 변환은 모든 입력 구성에서 고정한다.
Portfolio query를 처리하는 모델을 공유하고 선택한 partition의 두 그룹에 각각 적용한다.

기본 설정은 `reg:quantileerror`, `hist`, depth 5, learning rate 0.04,
subsample 0.8, colsample 0.9, min_child_weight 20, lambda 5, CPU 8 threads이다.

<a id="dl"></a>

### 3.3 Provider encoder, DayState, BiGRU

Provider encoder와 DayState는 station forecast를 읽고, station-to-PV mapping으로 PV fusion에 연결된다.
`daystate`가 있는 입력 구성은 다음 흐름을 사용한다.

```text
WA/WB station의 10·17시 예보 → provider encoder ─┐
station의 24h forecast ──────→ DayState encoder ─┼→ PV fusion → query pooling → 24h BiGRU → 19 quantiles
PV·태양·시간·physics 정보 ──────────────────────┘                 ↑
                                                        query 용량·구성 정보
```

제공사 encoder는 station·시간별로 계산한다. `sites.csv`의 매핑으로 PV에 연결한다.
PV fusion에는 제공사별 표현 h_A·h_B와 PV 단위 공통 입력 x_common을 전달한다.
x_common은 공통변수 disagreement, 태양 위치, 좌표, 설비용량, 입력 구성에 따른 PV physics proxy이다.
PV physics proxy는 PV 좌표로 계산하는 clear-sky 일사량에 의존하므로 station encoder가 아니라 x_common에 둔다.
Query pooling은 A의 용량 가중 평균·표준편차를 사용한다.
공유 BiGRU를 각 query에 적용하고 G1·G2 출력은 `24×2×19`로 배치한다.

Provider encoder는 station embedding 8과 시간 표현을 받아 MLP 64→32로 변환한다.
기본 크기는 provider 표현 32, PV fusion 64, BiGRU hidden 64·2 layers·dropout 0.1이다.
학습은 AdamW, learning rate 0.001, weight decay 0.0001, gradient clip 1,
batch 16 portfolio-day queries, 2,000 steps, cosine schedule, FP32로 수행한다.
제공사별 연속 입력의 대체·표준화·ZCA(epsilon 0.01)는 학습 구간에서 fit한다.

`daystate`의 DL은 provider별 24h forecast를 DayEncoder로 읽어 station별 day state를 만들고, station-to-PV mapping으로
PV fusion에 전달한다. Day state의 observed auxiliary target은 WA cloud와 WB cloud·UV·ceiling·visibility의 9~17시 평균이며,
target과 standardization에는 cutoff 이전 observed만 사용한다. Generation은 fusion·query pooling·BiGRU를 거쳐 24×19 quantiles로 출력한다.

DayState auxiliary는 query의 daily generation을 맞추는 energy head와 station별 observed head를 사용한다.
두 Huber 항의 가중치는 `dl_daystate_energy_weight`(기본 1.0)와 `dl_daystate_observed_weight`(기본 0.1)로 정한다.
`day-energy`에서는 별도 energy head의 weight를 0으로 두고 median trajectory의 daily energy를 직접 학습한다.
Observed weather는 cutoff 이전의 auxiliary target으로만 사용하고, test inference는 forecast로 DayState를 계산한다.

Temporal routing은 시각 t의 표현이 같은 하루의 다른 시각 표현을 읽어 원래 값에 더하는 선택 구성이다. 모두
`forecast-physics-daystate` 위에 설정을 더한 입력 구성으로 실행하고, 발전소 이용률 보조 head의 세기는 모델의 `site_aux_ratio`를 따른다. 읽는 위치와 가중은 예보 입력과 날씨 맥락으로 정하고, 읽어 오는 값은 원래
표현이다. Station embedding은 읽어 오는 값과 fusion에만 들어간다. 출력 사영을 0으로 초기화하므로 학습 시작 시 출력은 원래와 같다. 기본값인 설정은
`input_key`에서 빠지므로 다른 입력 구성의 key는 바뀌지 않는다.

- `...-time-shift`: query pooling 전 PV별 provider 표현이 참조 시각 τ = t + σ·b + 1.5·σ·tanh(Δ)를 Catmull-Rom 보간으로 읽고 attention으로
  합친다. b는 −1·0·1, σ는 0.5·1·2로 참조점은 9개이고, Δ와 attention logit은 관측소별 prior와 날씨 맥락의 합이다. 날씨 맥락은 두 제공사 예보 입력과
  x_common에서 정적 정보·provider 사용 여부를 뺀 채널이다.
- `...-time-shift-group-nearby-hours`: 위 구성에 더해 query pooling 뒤 sequence가 ±3시간 창의 sliding window attention으로 다른 시각을 읽는다.
  맥락·key는 모든 관측소에 같은 평균 station embedding을 넣고 정적 정보를 0으로 둔 PV 표현의 pooling으로 만든다.
- `...-nearby-hours-3h`, `...-nearby-hours-1h`, `...-nearby-hours-2h`: query pooling 전 PV별 provider 표현이 ±3·±1·±2시간 창의 sliding window attention으로 다른 시각을
  읽는다. Score는 q_t·k_s/√d − λ_t·(s − t)²이고 하루 밖 시각은 가린다. Gate(sigmoid)가 읽어 온 값의 몫을 시각마다 날씨 맥락으로 조절한다.
  날씨 맥락은 두 제공사 예보 입력과 disagreement·태양 위치·하루 상태이다.

<a id="losses"></a>

### 3.4 Training objective

Query별 generation mask M과 용량 C_A를 사용한다.

\[
L_0=\frac{\sum_{A,t,\alpha}M_{A,t}C_A\rho_\alpha(z_{A,t}-q_{A,t,\alpha})}
{|\mathcal Q|\sum_{A,t}M_{A,t}C_A}
\]

야간 출력은 태양 위치 규칙으로 0에 고정하고, generation 학습손실은 주간 유효 정답에 적용한다.
PV별 missing mask는 손실 계산에 사용한다. Query의 구성은 기상 입력과 발전량 target에 동일하게 적용한다.
pinball은 24시간 정답이 불완전한 날에도 유효한 개별 시각을 그대로 사용한다. 선택한 auxiliary loss와 DayState auxiliary를 이 기본손실에 더한다.

DL 손실은 19개 quantile의 pinball이고, 그 위에 중앙값(제출값) 곡선을 직접 맞추는 보조항을 하나 더할 수 있다.
설정값 `dl_loss`가 그 보조항을 고르고, `aux_ratio`가 세기를 정한다([3.5절](#aux-calibration)). 등록 모델은 모두 `ramp`를 쓴다.

| `dl_loss` | 보조항이 맞추는 것 | 쓰이는 곳 |
|---|---|---|
| `ramp` | 한 시각에서 다음 시각으로 넘어가는 변화량 | 등록 모델 전체 |
| `day-energy` | 하루 발전량 합 | `weighted-day-energy` recipe([7.2절](#weighted-day-energy)) |
| `pinball` | 없음. 19개 quantile만 학습한다 | 초기 grid의 기준 칸 |
| `mae` | 시각별 실적과의 절대 차이 | 초기 grid 후보 |
| `mse` | 시각별 실적과의 제곱 차이 | 초기 grid 후보 |
| `fourier` | 하루 곡선의 주기 성분 크기 | 초기 grid 후보 |
| `cosine` | 크기를 뺀 하루 곡선의 모양 | 초기 grid 후보 |

아래 식의 L_0은 pinball이고, L_ramp·L_day·L_MAE·L_MSE·L_FAL·L_cos는 차례로 `ramp`·`day-energy`·`mae`·`mse`·`fourier`·`cosine`의 보조항이다.
보조손실은 DL의 중앙분위수 출력 m(정규화 z 단위)에만 적용하고, XGB의 손실은 바꾸지 않는다.
정산 가중 `-settlement`는 DL의 pinball과 day-energy에 시간 slot의 k(표시 시각 11~15시 3, 그 밖 1)를 곱하며,
칸 이름의 loss 뒤에 붙인다(예: forecast-physics__ramp-aux0.1-settlement). 두 expert의 기본 time weight는 base MK·recipe를 따른다.
`--time-weight`는 양쪽을 바꾸고 `--xgb-time-weight`는 XGB만 override한다.
정산 W의 시각 가중은 0.10·SMP·k이다. Time weight `settlement`를 선택하면 이 중 k를 해당 학습 항에 적용한다.
L_MAE·L_MSE는 pinball과 같은 주간 유효 mask와 용량 가중으로 |m − z|, (m − z)²를 평균한다.
L_MAE는 pinball의 α=0.5 항 가중치를 키우는 것과 같으므로 median 강조 실험이 된다.

Profile 손실(L_ramp, L_FAL, L_cos)은 24시간 정답이 완전한 query-day만 사용한다.
주간 여부는 Y>0이 아니라 태양 위치 기준 mask D로 정하고, 야간은 예측과 정답을 모두 0으로 둔다(m̃ = m·D, z̃ = z·D).
24시간 축은 그대로 유지한다.

\[
L_{ramp}=\frac{1}{23}\sum_{t=2}^{24}\big|(\tilde m_t-\tilde m_{t-1})-(\tilde z_t-\tilde z_{t-1})\big|,\qquad
L_{FAL}=\frac{1}{K}\sum_{k=0}^{12}\big(|X_k|-|Y_k|\big)^2,\qquad
L_{cos}=1-\frac{\langle\tilde m,\tilde z\rangle}{\|\tilde m\|\|\tilde z\|+\epsilon}
\]

X·Y는 m̃·z̃의 24점 실수 FFT이고 K=13이다. k=0 항은 일 합계 차이를 포함한다.
Ramp는 일출·일몰 경계의 변화도 포함한다.
위상까지 포함한 전체 스펙트럼 FCL은 Parseval 정리에 의해 L_cos와 같으므로 cosine 한 칸으로 둔다.
정답 profile이 모두 0인 query-day는 L_cos에서 제외한다.
Profile 손실은 query-day별 값을 query 용량 C_A로 가중 평균한다.

day-energy는 complete query-day 집합 \(\mathcal D_{full}\)에서 DL median output \(m=q_{0.5}\)의 daily sum에 직접 적용한다.
Daylight mask \(D\)와 capacity-normalized target \(z\)를 사용한다.

\[
L_{day}=\frac{\sum_{b\in\mathcal D_{full}} C_b
\left|\sum_h w_h D_{b,h}(m_{b,h}-z_{b,h})\right|}
{\sum_{b\in\mathcal D_{full}} C_b},\qquad
w_h=\begin{cases}k_h & \text{time\_weight=k}\\1 & \text{time\_weight=none}\end{cases}.
\]

Incomplete query-day는 masked pinball로 학습하고, day-energy에는 complete query-day만 사용한다.
Daylight의 signed error를 합한 뒤 absolute value를 취한다. Capacity로 query-day를 가중 평균한다.

위 day-energy 식의 w_h는 해당 loss 안의 `none`/`settlement` time weight이다. Bundling objective의 정산 가중 w_h는 SMP를 포함하며 [4절](#bundling)에서 정의한다.

<a id="aux-calibration"></a>

### 3.5 Auxiliary coefficient calibration

\[
\lambda_k=r\,\frac{\mathbb E\|\nabla L_0\|}{\mathbb E\|\nabla L_k\|}
\]

식의 r은 설정값 `aux_ratio`이다.

1. 같은 input·DayState auxiliary 설정의 칸은 같은 초기화와 batch 순서를 쓴다. 처음 200 step의 generation objective는 pinball이며,
   `daystate` 입력 구성에서는 해당 칸의 DayState auxiliary도 더한다. `-settlement`의 pinball은 weighted pinball이다.
2. 200 step checkpoint에서 schedule의 다음 고정 batch 20개로 전체 parameter gradient norm의 평균을 재고 λ_k를 정한다.
3. 나머지 1,800 step은 pinball + λ_k·L_k에 해당 DayState auxiliary를 더해 학습한다. 총 2,000 step, cosine schedule, gradient clip 1을 사용한다.

pinball 칸과 median auxiliary 칸의 앞 200 step 공유는 input·seed·time weight·DayState auxiliary 설정이 같을 때 성립한다.
day-energy에서는 λ_E를 선택한 time weight의 pinball과 median day-energy의 gradient norm으로 정하고, observed supervision은 warm-up 전후에 유지한다.
λ_k는 입력 구성·seed마다 측정해 run manifest에 기록한다.
λ_k는 기준 L_0와 보조항 L_k의 gradient norm 비율에 맞춰 보조항의 scale을 조정한다.
Screening 기본 `aux_ratio`는 0.1이다. {0.03, 0.1, 0.3}은 초기 탐색 grid이며 `--aux-ratio 1`, `--aux-ratio 3` 등 유한한 nonnegative 값을 직접 지정한다.

<a id="inference"></a>

### 3.6 Inference와 blend

결합 방법은 `weighted-quantile` 하나이다. 모델별로 정렬한 분위수를 같은 분위수끼리 w·Q_XGB + (1−w)·Q_DL로 결합하고 다시 정렬한다.

XGB 비율 w의 실행 기본값은 base MK·recipe의 `xgb_weight`이며 `--xgb-weight`로 [0, 1] 범위에서 바꾼다. 이름 표기의 기준인 protocol의 0.5와 다른 비율은 이름 뒤에
`__weighted-quantile-w0.3`처럼 붙인다. 두 모델의 분위수를 먼저 정렬하므로 결합한 median은 두 모델 median의 가중합이다.

후처리는 seed 평균 → 모델별 clipping·sorting → 모델 간 결합 → sorting → 중앙 분위수 선택이다.
현재 절대오차 정산의 제출값은 최종 0.5 분위수이다. SMP=0인 시각에도 같은 규칙을 적용한다.
가용률 multiplier는 1로 고정한다.

<a id="bundling"></a>

## 4. Conditional bundling

설정은 [conditional_bundling.json](../configs/conditional_bundling.json)을 따른다.
예보와 cutoff 이전 이력으로 joint residual distribution을 만들고, 같은 scenario 집합에서 배정을 비교한다.

<a id="site-primitive"></a>

### 4.1 Site primitive와 OOT residual

Bundling은 설정의 `primitive.model`에 지정한 MK를 site residual 생성과 group-level transfer에 일관되게 적용한다.
`primitive.model`은 [conditional_bundling.json](../configs/conditional_bundling.json)에서 선택하고, MK의 expert 설정은 [models.json](../configs/models.json)을 따른다.
그룹 모델은 배정이 바뀌면 예측이 바뀌어 발전소별 가산 분해가 없으므로, 같은 입력·손실·학습 함수를 발전소-일 query(구성원 1개)에
적용한 발전소 단위 primitive를 쓴다. XGB는 모델의 `xgb_input`·`xgb_time_weight`로 중앙값만 학습하고, DL은 `dl_input`·`dl_loss`·`aux_ratio`·
`dl_time_weight`로 19분위수를 학습한다. `primitive.seeds`(기본 42·43·44)를 평균한 뒤 모델의 `xgb_weight`로 중앙값을 섞는다.
같은 분위수끼리 섞는 블렌드의 중앙값은 두 중앙값의 가중합이다. 발전소 단위 XGB 행렬에는 query-day 하늘 상태 요약(`daycontext`)이 없으므로,
`xgb_input`에 `daycontext`나 `daystate`가 있는 모델은 primitive로 쓰지 않는다.
잔차는 e_i,h = Y_i,h − ŷ_i,h이고, 요청한 instance의 estimation·evaluation에 필요한 기간을 예측해 cache kind `residuals`에 둔다.
각 기간의 site 예측은 그 기간 cutoff 이전 자료로만 학습한 모델로 생성한다.
Primitive 설정이나 구성요소 version이 바뀌면 cache key와 실험 identity도 바뀐다.

<a id="bundling-time"></a>

### 4.2 Instance와 정보 cutoff

instance τ는 cutoff, 발전소 집합 A_τ, 평가기간으로 이루어진다. A_τ는 cutoff까지 유효 발전량이 한 번 이상 있는 발전소이고
([2.3절](#cohort)의 cohort 규칙) V 106개소, H 205개소, B·test 253개소이다. 하나의 평가기간 안에서는 그룹을 고정한다.
과거에 없던 발전소와, 포함된 발전소의 특정 시각 정답 결측은 구분한다. 후자는 발전량 0이나 용량 0이 아니다.

V는 D의 OOT 잔차, H는 V의 OOT 잔차, B는 B cutoff 이전 H의 OOT 잔차를 쓴다.
Test는 H와 B 잔차를 합치고 겹치는 시간에는 더 최근 cutoff 모델의 예측을 쓴다.
공동 오차 모델·대체값·표준화·analog 검색에는 해당 instance cutoff 이전의 자료만 넣는다.
평가시간은 V 8,784시간, H 8,760시간, B 3,792시간 전체이다.
발전소 점예측, 관측 mask, SMP, k와 seed 정책은 비교 모델 사이에서 고정한다.

학습 행의 이력 변수는 대상일 전일 17시까지 알려진 OOT 잔차로 만든다. 같은 날 앞 시각의 잔차는 예측 시점에 알 수 없으므로 넣지 않는다.
예측 구간의 이력 변수는 outer cutoff에서 고정한다. 발전량 0은 유효 관측이며,
발전량 결측에는 likelihood를 적용하지 않는다. 야간 잔차 시나리오는 0이다.

<a id="residual-notation"></a>

### 4.3 Residual과 partition 표기

| 기호 | 정의 | 단위 |
|---|---|---|
| c_i | 발전소 i의 설비용량 | kW |
| e_i,h | Y_i,h − ŷ_i,h, site primitive의 residual | kWh |
| u_i,h | e_i,h / c_i | kWh/kW |
| R_g,h | 그룹 구성원의 e_i,h 합 | kWh |
| m_g,h | 그룹 residual scenario의 median | kWh |
| w_h | 0.10·SMP_h·k_h | 원/kWh |
| C 또는 C_τ | instance의 전체 설비용량 | kW |
| b | 두 그룹의 상대 capacity imbalance | — |

분할은 x_i = 1(G1), x_i = 0(G2), z = 2x − 1, c̃ = c/C_τ이고, 상대 불균형은 b = c̃ᵀz = (C_1 − C_2)/C_τ이다.
발전소 잔차 e_i,h의 그룹 합 R_1,h = Σ_i x_i e_i,h, R_2,h = Σ_i (1 − x_i) e_i,h에서 R_1,h + R_2,h = E_h는 분할과 무관하고

\[
|R_1|+|R_2|=|E_h|+2\min(|R_1|,|R_2|)\,\mathbb 1[R_1R_2<0],\qquad R_1^2+R_2^2=\tfrac12\big(E_h^2+(z^\top e_h)^2\big)
\]

이다. 고정된 site residual의 그룹 합을 직접 채점하면 분할이 바꾸는 것은 그룹 간 상쇄 손실뿐이고, 엄밀한 하한은 W_merge = Σ_h w_h|E_h|이다.
모든 x_i = ½인 연속완화가 이 값에 도달하므로 relaxation으로 더 좁은 하한을 얻지 못한다. 두 번째 항등식이 QUBO의 2차형 근거이다.

이 항등식은 고정된 e의 그룹 합에 대한 관계이다. Median correction을 사용하는 탐색 objective는 [4.5절](#scenario-objective)에 둔다.

<a id="joint-models"></a>

### 4.4 Joint residual models

<a id="lowrank"></a>

#### Lowrank

정규화 잔차 u=e/c에 대해 공유 PV 함수가 μ, L, σ를 예보·태양 위치·설비 정보·과거 이력에서 계산한다.
설비 정보는 용량과 대응 관측소까지의 거리이고, 발전소 좌표는 입력하지 않는다.
공동분포는 u=μ+Lf+σε, f와 ε는 표준정규이고 Σ=LLᵀ+diag(σ²)이다.
PV 수와 site ID에 종속된 출력표를 만들지 않고 같은 함수를 모든 구성원에 적용한다.
관측 PV 부분집합의 Gaussian marginal NLL을 Woodbury 항등식으로 계산한다.
kW당 잔차로 학습하지만 정산은 kWh 그룹 합에서 일어나므로, 시각마다 무작위 분할 4개의 두 그룹 합 S = Σ c_i u_i에 대한
Gaussian NLL을 같은 가중치로 더한다. 그룹 합의 예측분포는 N(Σ c_i μ_i, ‖Σ c_i L_i‖² + Σ c_i² σ_i²)이고, 그룹 용량으로 나눈 척도에서 평가한다.
CUDA에서 사전 고정한 optimizer step을 학습하고 마지막 checkpoint를 사용한다.
조건부 Gaussian의 평균도 학습하므로 오차의 방향과 동시 변동을 함께 표현한다.

<a id="unconditional"></a>

#### Unconditional

Unconditional 비교는 cutoff 이전 잔차의 pairwise 2차 moment를 쓴다. 겹친 표본이 짧은 쌍은 예보 유사도 target
ρ_ij = a + b·sim_ij로 shrink한다. sim_ij는 대응 관측소 17시 예보 운량 시계열 상관의 제공사 평균이며, 같은 관측소 쌍은 1이다.

<a id="analog"></a>

#### Analog

Analog 비교는 대상일 예보 profile과 가까운 cutoff 이전 일자를 고르고, 같은 일자를 모든 PV와 24개 slot에 적용한다.
Analog에 관측된 잔차는 보존하고, 없는 PV는 조건부 저랭크 분포에서 공동 생성한다.

<a id="mixture"></a>

#### Mixture

Mixture는 설정의 구성 방법 시나리오를 같은 비중으로 합친 분포이다. 구성 방법마다 시각당 같은 수의 시나리오를 서로 다른 seed로
뽑아 이어 붙이며, 그 QUBO의 2차 moment는 구성 방법 moment의 평균이다. 비교할 방법 집합은 설정의 `models`가 정한다.

<a id="scenario-objective"></a>

### 4.5 Scenario objective

시간 h의 시나리오 확률은 합이 1이다. 같은 시간·시나리오·발전소 순서를 모든 후보에 사용한다.
Scenario objective는 배정별 그룹 합 분포의 중앙값을 제출값으로 사용한다. 배정마다
시각별 그룹 잔차 합 R_g의 시나리오 중앙값 m_g,h를 제출값 보정으로 보고 기대 절대오차를 계산한다.
목적은 Σ_h w_h·mean_s(|R_1,hs − m_1,h| + |R_2,hs − m_2,h|)/(CΣ_h w_h) + λ_MAE·b², b = (C_1−C_2)/C이다.

<a id="partition-search"></a>

### 4.6 QUBO와 local search

QUBO MIQP 해, original, 무작위 초기해에서 flip·swap 탐색을 수행한다. M은 같은 탐색 scenario의 residual vector를 instance 총 capacity C로
나눈 뒤 시간·scenario weight의 합이 1이 되도록 정규화해 계산한 second moment이다. 두 그룹의 full MSE risk objective는 다음과 같다.

\[
x^\top Mx+(\mathbf1-x)^\top M(\mathbf1-x)
=\tfrac12\mathbf1^\top M\mathbf1+\tfrac12 z^\top Mz.
\]

J_ref,MSE = ½·𝟙ᵀM𝟙는 full objective를 fractional assignment x = ½𝟙에서 평가한 값이다.
전체 residual 합 자체의 MSE는 𝟙ᵀM𝟙이다. QUBO는 constant term J_ref,MSE를 제거한 risk term ½·zᵀMz를 사용하며,
이 risk term의 x = ½𝟙 값은 0이다.

Balance penalty를 더한 QUBO는 Q = ½M + λ_QUBO·c̃c̃ᵀ에 대해 zᵀQz이다.
AAᵀ = ½M, Ã = [A, √λ_QUBO·c̃]로 두면 zᵀQz = ‖Ãᵀz‖²이므로 vector balancing으로 표현한다.
x로 바꾸면 zᵀQz = xᵀ(4Q)x − 4(Q𝟙)ᵀx + 𝟙ᵀQ𝟙이고, 두 그룹을 맞바꿔도 값이 같으므로 용량이 가장 큰 발전소를 G1에 고정한다.
주어진 배정을 MIP start로 고정 thread·seed와 결정적 work limit으로 풀고, 상태·gap·하한을 탐색 기록에 남긴다.
설정에 tabu를 두면 같은 QUBO를 tabu search로도 풀어 Gurobi 해 대비 에너지 차이만 기록하며, 이 기록은 선택에 쓰지 않는다.
QUBO는 2차 moment surrogate로 초기해를 만든다. 최종 해는 QUBO·original·무작위 초기해에서 수행한 탐색 결과 중 scenario objective가 가장 낮은 해로 정한다.
연구 탐색에서는 두 그룹에 구성원이 하나 이상 있어야 하고, 20MW에서 유도한 hard band는 넣지 않는다.

<a id="balance"></a>

### 4.7 Capacity balance와 test 적용

Test instance의 총 capacity C_test = 42,655.37 kW에서 그룹당 20 MW hard constraint는
|b| ≤ β, β = 1 − 2·20,000/C_test ≈ 0.0623인 feasible set을 만든다. 과거 snapshot은 총 capacity가 40 MW보다 작을 수 있으므로,
capacity 균형항은 이 feasible set의 폭을 기준으로 불균형에 비용을 주는 scale-free surrogate로 사용한다.
β는 `minimum_final_group_kw`와 test instance 총 capacity로 계산하고 모든 instance에 같은 값을 쓴다.
b = c̃ᵀz에서 λ > 0인 quadratic penalty λb²는 zᵀ(λc̃c̃ᵀ)z라는 rank-1 항이다.
QUBO에서는 risk matrix에 이 항을 더하고, MAE objective에서는 weighted absolute deviation에 더한다.

`balance_grid`는 objective별 reference scale J_ref로 balance penalty를 정규화한다.

\[
\mathrm{penalty}=\tilde\lambda\,J_{\mathrm{ref}}\,(b/\beta)^2.
\]

λ̃는 |b| = β에서 penalty가 J_ref의 몇 배인지를 나타내는 dimensionless coefficient이다.
Risk term과 J_ref에 같은 양수 scale을 곱하면 전체 objective도 같은 비율로 변하므로 λ̃를 유지한다.

- J_ref,MAE는 전체 residual 합 분포의 median 기준 weighted absolute deviation이다. 실제 MAE risk term과 동일하게
  instance 총 capacity C와 시간 weight 합으로 정규화하며 `J_merge`에 해당한다.
  Fractional assignment x = ½𝟙에서 두 그룹 MAE의 합도 J_ref,MAE이다.
- J_ref,MSE = ½·𝟙ᵀM𝟙는 [4.6절](#partition-search)의 full MSE objective를 x = ½𝟙에서 평가한 값이다.
- QUBO의 coefficient는 λ_eff,MSE = λ̃·J_ref,MSE/β²이며 λ_QUBO에 적용한다.
  따라서 Q = ½M + λ_eff,MSE·c̃c̃ᵀ이다.
- Local search의 coefficient는 λ_eff,MAE = λ̃·J_ref,MAE/β²이며 λ_MAE에 적용한다.
  Objective는 J_MAE(x) + λ_eff,MAE·b²이고, J_MAE(x)는 각 그룹 residual 합 분포의 median 기준 weighted absolute deviation의
  합을 C와 시간 weight 합으로 정규화한 값이다.

고정 coefficient λ에 대응하는 정규화 coefficient는 objective별로 λ̃ = λ·β²/J_ref이다.
예를 들어 λ = 1, β² ≈ 0.0039, J_ref,MAE = 0.0227이면 λ̃ ≈ 0.0039/0.0227 ≈ 0.17이다.

균형 계수의 선택과 test 적용은 다음 두 방식으로 구분한다.

- `balance_grid`: 위 정규화로 λ̃마다 λ_MAE와 λ_QUBO를 계산한다. 오름차순 λ̃마다 QUBO·original·직전 λ̃ 해·무작위 초기해에서 탐색해
  모든 λ̃ 해를 평가 후보에 넣고, 요청한 labelled instance scope의 모든 방법에서 |b| ≤ β인 가장 작은 λ̃*를 실현 결과 없이 고른다.
  Scope와 적용 instance를 `balance_selection.json`에 기록한다. 전체 V·H·B를 요청하면 그 전체에서 고른다.
  test 단독 실행은 저장된 λ̃*를 적용한다. 아직 선택 기록이 없으면 frontier를 보존하고, 이후 `conditional-decide`에서 선택한다.
  test에서는 각 λ̃ 해의 20MW 충족을 판정하고, 미달하면 복구한 배정과 그 목적·용량을 기록한다.
  각 λ̃ 해는 penalty·불균형 frontier로 보존하며, 이 모드에서는 λ̃ grid가 `lambda_frontier`를 대체한다.
- `fixed_lambda`: λ_MAE = λ_QUBO = λ로 탐색하고, `lambda_frontier`의 다른 λ는 주 배정에서 시작해 `frontier_qubo` 예산으로 다시 풀어
  벌점·불균형 frontier로 보존한다. test에서는 frontier 해를 복구한 뒤 평가용 시나리오의 예상 W가 가장 작은 후보를 쓴다.

기본 설정은 `fixed_lambda = 1`이며, QUBO(MSE 척도)와 local search(MAE 척도)에 같은 coefficient를 적용한다.
정규화 방식의 비교 설정은 [7.3절](#balance-grid-recipe)에 둔다.

<a id="evaluation"></a>

## 5. 평가와 선택

최종 후보의 평가에는 **MK0 대비 W 변화율**, **champion 대비 W 변화율**, 각 기준 대비 ΔW를 먼저 제시한다.
MK0 손실의 10% 감소는 프로젝트의 첫 목표이고, champion 대비 W 변화율은 이번 후보가 이전 최선보다 나아졌는지 판단하는 기준이다.
V·H·B별로 목표 도달 여부를 표시하고, 아래 forecasting·scenario·transfer 지표로 개선의 원인을 진단한다.

<a id="periods"></a>

### 5.1 Period와 cutoff

| 구간 | 시작 KST | 종료 KST | 역할 |
|---|---|---|---|
| 초기 학습 | 실제 데이터 시작 | 2022-05-31 17시 | D 예측 모델 학습 |
| D | 2022-06-01 01시 | 2023-06-01 00시 | OOT 예측, 실험 구조·후처리 점검 |
| V | 2023-06-01 01시 | 2024-06-01 00시 | 후보·손실 계수·블렌딩 선택 |
| H | 2024-06-01 01시 | 2025-06-01 00시 | 연간 exploratory 실행·선택 snapshot 비교, bundling 진단 |
| B | 2024-12-25 01시 | 2025-06-01 00시 | 전체 253개 bundling 진단 |
| test | 2025-06-01 01시 | 2026-06-01 00시 | 전체 253개 최종 제출 |

연간 D/V/H의 평가 시간축은 위 기간으로 고정한다.
모델·전처리·cohort의 정보 cutoff는 첫 대상일 전일 17시이다.
D/V/H의 cutoff는 각각 2022/2023/2024년 5월 31일 17시이다.
Actual 타입의 generation·observed에 이 cutoff mask를 먼저 적용한 뒤 query와 24시간 시퀀스를 만든다.
부분 학습일의 cutoff 이후 observed 입력은 missing으로 처리하고 generation loss를 마스크한다.
학습 시작일은 원자료 시작일을 따른다. 연간 D/V/H 평가에는 완전한 1년의 예측 시간축을 유지한다.
날짜 경계는 `(ts−1시간).date`를 기준으로 묶는다.

학습 cutoff를 3개월씩 바꿔 forecast 후보의 개선이 유지되는지 확인하는 [robust fold test](#robust-folds)는 V·H를 fold로 함께 쓴다.
위 V/H/B의 기간·cutoff·평가 계약과 고정 MK0 baseline은 그대로 유지한다.

<a id="levels"></a>

### 5.2 Execution level

기간은 cutoff·평가 시간축을 정하고, level은 기본 seed·학습 budget·보고 용도를 정한다. `submission`은 test에 쓰고,
그 밖의 level은 D·V·H·B·test에서 선택할 수 있다. H smoke와 B screening도 실행한다.
H·B의 exposure 상태는 `previously_exposed`로 보존한다. 정답이 있는 일반 실행은 `exploratory`, smoke는 `execution_check`,
test는 `inference`로 기록한다. `--selection`을 명시하면 저장한 모델 구성·settings와 대조하고,
H는 `retrospective_confirmation`, 다른 labelled 기간은 `frozen_comparison`으로 기록한다.
선택 snapshot은 선택적인 비교 계약이며, snapshot 없이도 새 입력·손실·`aux_ratio`·blend를 실험한다.

아래 표는 period와 level을 조합한 기본 사용 예이다.

| 수준 | 실행 내용 | 판정에 사용하는 결과 |
|---|---|---|
| 실행 확인 | 작은 데이터·짧은 학습으로 I/O와 CUDA 경로 확인 | 실행 가능 여부 |
| D 진단 | champion을 연간 OOT로 계산하고 OOT 학습 곡선을 기록 | 실험 구조·후처리 점검 |
| V noise floor | 기준 칸 forecast-solar-diff__pinball을 seed 42~51로 실행 | seed에 따른 W_cohort 표준편차와 판정 문턱 |
| V screening | seed 42, 같은 본 학습 예산 | 후보별 연간 ΔS_cohort와 설명 지표 |
| V confirmation | 선별 후보와 비교 기준 칸을 seed 42·43·44로 실행 | 3-seed 결합의 연간 비교 |
| H 확인 | 선택된 구조·설정 고정 후 실행 | 연간 paired 비교와 월별·그룹별 기여 |
| Bundling 진단 | H·B에서 배정별 그룹 모델을 seed 42·43·44로 실행 | 배정 변경의 그룹 단위 transfer([5.5절](#transfer)) |

학습 budget은 XGB 400 boosting rounds, DL 2,000 optimizer steps로 고정한다.
Checkpoint는 마지막 step을 사용한다. V/H 결과로 학습 종료 시점을 정하는 단계는 두지 않는다.
입력 표본 수, batch 수, optimizer steps, 모델별 계산시간을 기록한다.

<a id="scoring"></a>

### 5.3 Forecasting 평가

\[
S_{cohort}=\sum_{g,t}M_{g,t}SMP_tY_{g,t}
-0.10\sum_{g,t}M_{g,t}SMP_tk_t|a_{g,t}-Y_{g,t}|
\]

k는 표시 시각의 시가 11~15이면 3, 나머지는 1이다.
W_cohort는 벌점항 \(0.10\sum_{g,t}M_{g,t}SMP_tk_t|a_{g,t}-Y_{g,t}|\)(원)이다.
수익항은 같은 cohort·mask에서 후보 간 같으므로 ΔS_cohort = −ΔW_cohort이다.
모든 후보는 같은 cohort·generation mask·SMP·k로 비교한다.
차이는 `ΔS_cohort=S_candidate−S_champion` 및 `ΔW_cohort=W_candidate−W_champion`으로 기록한다.
Champion 대비 W 변화율은 `ΔW_cohort / W_champion`이며 음수일수록 개선이다.
MK0 대비 W 변화율은 `(W_candidate − W_MK0) / W_MK0`로 계산하고, 고정한 MK0 prediction을 기준으로 유지한다.
W 변화율과 ΔW는 최종 blend forecast의 W에서 계산한다. 기준 MK0·champion과 후보의 run·parameter·partition을 함께 명시한다.
배정이나 package가 다른 비교는 `vppcontract compare`로 같은 cohort·시각·정답 mask의 W를 계산한다.

| 평가 층 | 기록 항목 |
|---|---|
| 목표 진척 | MK0 대비 W 변화율·ΔW, 10% 손실 감소 목표 도달 여부 |
| Champion–challenger | Champion 대비 W 변화율·ΔW, 기준과 후보의 run |
| 정산 | S_cohort, W_cohort, ΔS·ΔW, 월별·그룹별·가격/시간대별 기여 |
| 발전량 분포 | 19개 평균 pinball, 5–95% coverage·폭, 정렬 전 crossing 비율 |
| 발전량 중심·시간 변화 | median MAE, capacity-normalized RMSE, ramp MAE |
| 커버리지 | 대상 PV·용량·시간, 정답 mask의 월별 유효 개수 |
| 계산 | 학습시간·추론시간·GPU peak memory·캐시 크기 |

발전량 설명 지표는 전체 시간과 주간을 각각 계산한다.
Ramp는 같은 query의 연속된 두 유효 시각에서 계산한다.

Paired moving-block bootstrap은 날짜별 두 그룹의 ΔS를 묶고 7·14일 block, 2,000회, 95% 구간을 기본으로 쓴다.
실행·비교 보고서에는 사용한 기간과 exposure를 기록한다.

<a id="noise-floor"></a>

#### Noise floor와 grid 해석

기준 칸 forecast-solar-diff__pinball을 V에서 seed 42~51로 실행해 XGB·DL·블렌드 W_cohort의 seed 간 표준편차 σ를 잰다.
2√2·σ를 참고 문턱으로 함께 표시한다. `grid-report`는 완료한 후보 전체를 W 순으로 표시한다.
Noise floor가 없어도 raw W·ranking을 만들고, reference가 있을 때 ΔW를 계산한다. 문턱과 구간은 후보 선택의 참고 정보이다.
Confirmation의 기본값은 seed 42·43·44이고 `--seeds`로 바꿀 수 있다.

DL loss 변경은 DL 성분의 ΔW_cohort로, 입력 변경은 XGB·DL·blend의 ΔW_cohort로 진단한다.
최종 후보 비교는 blend의 champion 대비 W 변화율을 기준으로 한다.
입력 효과는 손실 6종 평균에서 forecast-solar 위 2×2(`diff`·`physics`) 주효과와 forecast-raw→forecast-solar 차이로, 손실 효과는 입력 5종 평균으로 본다.
모든 칸은 같은 evaluation panel에서 비교한다.

<a id="scenario-evaluation"></a>

### 5.4 Scenario 기반 배정 평가

평가 결과를 읽기 전에 공통 후보 배정 집합을 고정한다. 집합은 original과 상대 용량 차이 10% 이내의 무작위 배정 200개이며,
좌표 기반 배정은 넣지 않는다. 이 범위는 순위 진단용이고 최적화의 허용 영역에는 적용하지 않는다.
선택용 시나리오와 별도 seed의 평가용 시나리오를 쓰고, 실제 평가는 모든 유효 시각을 포함한다.

실현 평가의 제출값은 발전소 점예측의 합에 평가용 시나리오에서 미리 구한 그룹 중앙값 m_g,h를 더한 값이다.
같은 방법 안에서 original과 선택 배정 모두 이 규칙을 쓰며, 보정 없는 발전소 점예측 합의 결과도 함께 기록한다.
중간 단계는 판정 관문을 두지 않고 효과 크기와 구간을 기록한다. 기록 항목은 선택 배정의 original 대비 실현 ΔW와 7·14일 bootstrap 구간,
무작위 분포 안의 백분위, 고정 후보 집합의 순위 상관, 그룹 합 오차의 90% 구간 coverage·quantile loss, 저랭크 분포의 평가 주간 NLL이다.

방법의 기본 ranking은 요청한 labelled instance scope에서 계산한다. 점수는 방법별 선택 배정의 상대 실현 ΔW(ΔW / W_original)를
instance 용량으로 가중 평균한 값이다. `selection.json`에 scope·점수·report hash를 기록한다.
`--method`로 원하는 방법을 직접 선택할 수도 있다. 전체 V·H·B를 요청하면 B와 H가 V보다 용량 비례로 크게 반영된다.

<a id="transfer"></a>

### 5.5 Group-level transfer와 추천

`conditional-transfer`는 지정한 labelled instance의 선택 배정과 original을 같은 primitive·seed로 그룹 재학습해 비교한다.
`conditional-decide`는 저장된 H·B transfer와 test 구성으로 model·partition pair를 추천한다.
그 모델 기준 H·B 합산 ΔW가 음수이고 test 구성이 그룹당 20 MW를 만족하면 candidate를, 그 외에는 original을 추천한다.
당시 champion 일치 여부는 context로 기록한다. 이 명령은 추천을 저장하며 registry 변경은 `adopt`, 제출 파일 생성은 `submit`으로 수행한다.
H·B의 방법 선택 사용 여부와 비교 scope를 보고서에 함께 기록한다.
추천한 model·partition pair는 최종 group forecast의 W로 MK0·champion 대비 W 변화율을 계산한다.
위 original 비교는 같은 primitive에서 partition 변경의 효과를 보여준다.

<a id="adoption"></a>

### 5.6 Comparison snapshot, adoption, submission

Champion–challenger 비교는 최종 그룹 forecast의 정산손실이 이전 최선보다 얼마나 줄었는지 판단한다.
비교 시점의 champion run을 고정하고 W 변화율이 음수인 후보를 개선 후보로 본다.
채택 후 다음 비교의 champion을 갱신하며, 누적 진척의 MK0 baseline은 유지한다.

실행·검증·채택은 독립 명령이다.

- `run`은 지정한 설정으로 학습·예측·채점한다.
- `compare`·`evaluate`는 저장된 예측을 채점한다. 같은 evaluation panel의 두 완료 run을 `select`로 snapshot에 기록한다.
- `verify-selection`은 저장된 run이 명시한 snapshot의 모델·settings와 일치하는지 검사한다. 현재 champion 변경과 무관하게 당시 설정으로 검사한다.
- `adopt`만 champion pointer를 변경한다. Noise floor·bootstrap 구간은 비교 보고서에 표시하고, 채택은 사용자가 지정한다.

제출 파일은 `submit`에서 별도로 만든다. 제출 기록에는 test run의 기준 MK와 parameter(`MK2 (xgb_weight=0.2, partition=…)` 형태의
`model_label` 포함)를 적고, `--evidence`로 받은 공통 평가기 비교 보고서(`vppcontract compare`)의 W·ΔW·7일 block bootstrap 구간을
gate 근거로 함께 적는다. Distribution flag는 기본적으로 warning이며 `--strict-distribution`을 선택하면
`--acknowledge`가 없는 사용 채널의 flag를 blocking으로 처리한다. Cutoff·단위·평가 panel·CSV 키·최종 20 MW 조건은 계속 검사한다.

<a id="robust-folds"></a>

### 5.7 Robust fold test

Robust fold test는 forecast 후보가 champion보다 나은 결과가 학습 시점·계절·seed가 달라져도 유지되는지 확인한다.
제출과 같은 조건, 즉 cutoff까지의 자료로 한 번 학습하고 다음 12개월을 예측하는 조건을 학습 cutoff만 바꿔 반복한다.
후보는 champion 위에서 입력 구성 하나를 바꾼 구성이고, 기준은 같은 fold·배정·seed로 학습한 champion이다.

**Fold.** 5.1의 V·H와 같은 규칙으로, 평가 시작 전날 17시를 학습 cutoff로 두고 시작일 01시부터 12개월 뒤 00시까지를 평가한다.
Cutoff를 3개월 간격으로 둔 다섯 fold를 쓴다. Fold 이름과 기간은 두 package가 함께 쓰는 공통 계약층(`vppcontract.timeaxis`의
`FOLD_PERIODS`·`PERIODS`)에 정의하고, `config.FOLDS`는 그 정의에서 V·H를 뺀 fold의 평가 시작일이다.

| Fold | 학습 cutoff KST | 평가 시작 KST | 평가 종료 KST | 평가 cohort |
|---|---|---|---|---:|
| V | 2023-05-31 17시 | 2023-06-01 01시 | 2024-06-01 00시 | 106 |
| fold-2309 | 2023-08-31 17시 | 2023-09-01 01시 | 2024-09-01 00시 | 117 |
| fold-2312 | 2023-11-30 17시 | 2023-12-01 01시 | 2024-12-01 00시 | 136 |
| fold-2403 | 2024-02-29 17시 | 2024-03-01 01시 | 2025-03-01 00시 | 173 |
| H | 2024-05-31 17시 | 2024-06-01 01시 | 2025-06-01 00시 | 205 |

H는 12개월 평가가 원자료 범위(2025-05-31) 안에 들어가는 마지막 cutoff이다. 2022년 cutoff는 cohort가 45개소 안팎이어서
제출 조건과 차이가 크므로 쓰지 않는다. B는 cutoff에서 평가 종료까지가 6개월로 제출과 달라 fold에 넣지 않는다.

**배정.** 학습 query를 그룹 단위로 만들므로 배정은 평가와 학습 모두에 들어간다. Fold마다 그 fold의 cohort, 즉 cutoff까지
유효 generation이 있는 발전소를 공통 계약층의 기준 층화 배정(`vppcontract.partition`)으로 나눈다.
(WA 관측소, WB 관측소) 쌍마다 용량이 큰 발전소부터 두 그룹의 용량이 비슷해지게 넣고, 각 그룹이 cohort 용량의
0.4689(253개소 전체 용량에서 20 MW에 해당하는 비율) 이상이 되도록 고친다. Cohort 밖 발전소는 같은 규칙으로 따로 나누며
학습과 평가에 쓰이지 않는다. `fold-stratified-partition`이 이 배정을 artifact의 `partitions/stratified-<fold>.csv`로 쓰고,
이미 있는 파일은 덮어쓰지 않고 cohort 발전소의 배정이 같은지 대조한다. V·H의 cohort 배정은 stratified-V·H와 같다.
original은 그룹 용량 대역을 벗어나므로 이 test에 쓰지 않는다.

**실행.** Champion과 후보를 fold마다 seed 42~46으로 학습한다. XGB는 두 구성에서 같으므로 fold·seed마다 한 번 학습해 공유한다.

```powershell
& $py -m solarvpp run --model MK4 --partition <artifact_dir>\partitions\stratified-fold-2309.csv --period fold-2309 --level confirmation --seeds 42 43 44 45 46
& $py -m solarvpp run --model MK4 --dl-input <후보 입력 구성> --partition <artifact_dir>\partitions\stratified-fold-2309.csv --period fold-2309 --level confirmation --seeds 42 43 44 45 46
```

**판정.** W는 5.3의 cohort W를 5-seed 결합 예측의 최종 blend forecast로 계산한다. `fold-report --candidates <후보 입력 구성>`이
fold마다 champion과 후보의 완료 run을 찾아 다음 네 조건을 계산하고 `reports/folds-*.json`에 저장한다. 네 조건을 모두 만족하면 robust 개선으로 판정한다.
네 조건의 판정 함수(`judge`)와 계절 규칙은 공통 계약층(`vppcontract.evaluate`, `vppcontract.timeaxis`)에 두어 solarvpp2 후보도 같은 규칙으로
판정한다. Package 간 비교는 `python -m vppcontract folds`가 맡는다([solarvpp2 참고](solarvpp2-reference.md#instances)).

1. 다섯 fold의 W 합이 champion보다 1% 이상 낮다.
2. 다섯 fold 중 네 개 이상에서 W가 champion보다 낮다.
3. Seed 다섯 개 각각에서, 그 seed 하나의 예측으로 구한 다섯 fold W 합이 모두 champion보다 낮다.
4. 다섯 fold를 합친 계절별(3~5월, 6~8월, 9~11월, 12~2월) W가 champion보다 2% 넘게 높은 계절이 없다.

불확실성은 fold와 seed 사이에서 실제로 나타난 차이로 읽고 block bootstrap 구간은 쓰지 않는다. Fold의 평가 기간이 겹치므로
fold들은 서로 다른 학습이지만 독립된 평가 표본은 아니다. 1%는 fold 다섯 개와 seed 다섯 개를 평균한 학습 노이즈(대략 ±0.4%)보다
큰 개선 폭으로 정한다. Fold W 합은 cohort가 큰 최근 fold에 더 큰 비중을 준다.

<a id="configuration"></a>

## 6. 설정, version, 기록

<a id="versions"></a>

### 6.1 MK와 parameter

| 버전 | XGB | DL | 블렌드 |
|---|---|---|---|
| MK0 | forecast-physics | forecast-physics, ramp-aux0.1 | weighted-quantile-w0.5 |
| MK1 | forecast-physics | forecast-physics-daystate, ramp-aux0.1 | weighted-quantile-w0.5 |
| MK2 | forecast-physics | forecast-physics-daystate, ramp-aux0.1 | weighted-quantile-w0.4 |
| MK4 | forecast-physics | forecast-physics-daystate, ramp-aux0.1, 발전소 이용률 보조 head(`site_aux_ratio` = 10) | weighted-quantile-w0.2 |

현재 champion은 [models.json](../configs/models.json)의 `champion`이 가리킨다.

MK 버전은 pipeline 하나와 그 기본 parameter(두 expert의 입력·time weight, DL 손실·`aux_ratio`, 블렌드 방법·비율)를 묶은 모델 정의이다.
정의는 `configs/models.json`에 두고, 한 번 등록한 버전의 정의는 유지한다. `champion`은 명시적으로 선택한 기본 모델이다.
Parameter·MK·package와 prediction artifact를 함께 기록해 W 변화율의 근거와 재사용할 계산을 구분한다.

- parameter만 바꾸면 같은 MK이다. `run --model MK2 --xgb-weight 0.2 --partition <csv>`처럼 MK 위에 parameter를 바꿔 실행하고,
  `--input`은 두 expert를, `--xgb-input`·`--dl-input`은 해당 expert를 바꾼다. `--input`은 발전소 보조 head 없이 시작하고, `--dl-input`은 기준 MK의
  `site_aux_ratio`를 유지하며, `--site-aux-ratio`로 세기를 바꾼다. 원래 MK 정의는 보존한다.
- 새 MK는 채택해서 제출 경로로 쓰는 pipeline에만 만든다. 후보는 코드를 추가해도 parameter로 실행한다. `adopt --model MKn --run <run>`은 그 pipeline으로 완료한 run의 설정을
  새 MK에 등록하고 champion을 선택한다. 이미 등록된 모델은 `adopt --model MK0`처럼 선택한다. 선택 근거 run과 당시 설정은
  registry의 `adoption`에 기록한다.
- 계산 그래프나 node 계약까지 바뀌면 새 package로 만든다([solarvpp2 설계](solarvpp2-design.md)).

run 폴더 이름은 `<기준 MK 또는 recipe>-<level>-<run key 앞 10자>`이고, 칸 이름은 manifest의 `cell`에 둔다. 후보는 `artifacts` 명령의 MK·parameter·W로 찾는다. manifest의 `model`에는 모델 구성(두 expert 입력·time weight, DL 손실·`aux_ratio`, 발전소 보조 head 세기, 블렌드 비율)을,
`model_version`에는 기준 MK(`--model` 또는 champion)를, `parameters`에는 기준 MK 정의와 다른 parameter와 original이 아닌 배정
(partition 파일 이름과 hash)을 적는다. recipe run은 `model_version`을 비운다. 버전 이름은 run identity에 넣지 않으므로,
같은 계산은 `--model`로 실행하든 입력·손실을 직접 지정하든 같은 run identity를 쓴다. 입력·손실을 생략한 `run`은 champion을 실행한다.

<a id="names"></a>

### 6.2 이름과 설정 출처

이름의 종류와 뜻은 다음과 같다.

| 종류 | 가리키는 것 | 예 |
|---|---|---|
| 배정(partition) | PV의 G1·G2 배정이다. `original`은 `sites.csv`의 A→G1, B→G2 배정이다. 그 밖의 배정은 구성 hash로 구분한다 | original |
| 입력 구성 | expert의 입력 표현이다. `forecast-` 뒤의 말이 담는 정보를 나타낸다([3.1절](#features)) | forecast-physics |
| 손실 | DL expert의 학습 손실이다. 보조항 세기는 `-aux`, 정산 가중은 `-settlement`로 붙인다([3.4절](#losses)·[3.5절](#aux-calibration)) | ramp-aux0.1 |
| 결합 | XGB와 DL 분위수의 결합 방법이다. 기본값과 다른 XGB 비율은 `-w`로 붙인다([3.6절](#inference)) | weighted-quantile-w0.4 |
| MK | pipeline 하나와 그 기본 parameter(두 expert의 입력 구성·time weight, DL의 손실·`aux_ratio`·`site_aux_ratio`, 결합 방법·비율)를 `configs/models.json`에 등록한 정의이다([6.1절](#versions)) | MK2 |
| XGB., DL. | 두 expert의 입력 구성이 다를 때 각 이름 앞에 붙여 어느 expert의 구성인지 나타낸다 | XGB.forecast-physics |

칸 이름은 run을 표시하는 이름으로 manifest와 보고서에 쓴다. 입력 구성과 L을 `__`로 이어 `forecast-physics__ramp-aux0.1`처럼 쓰고,
두 expert의 구성이 다르면 `XGB.{입력}[-settlement]__DL.{입력}[-site-aux{세기}]__{손실}`로 쓰고, 블렌드 비율이 protocol 기본값(0.5)과 다르면 뒤에 `__weighted-quantile-w0.4`처럼 붙인다.
한 번 정한 이름의 정의는 바꾸지 않고, 정의가 바뀌면 새 이름을 붙인다.

데이터 정의는 [참가자 데이터 명세](../participant_dataset/README.md)를 따른다.
실험 설정은 [solarvpp protocol](solarvpp-protocol.json), 전체 기상 컬럼·단위·대응 관계는
[weather schema](weather-schema.json)에 둔다.
설계 편차와 구현 해석은 `configs/runtime.json`의 `deviations`·`interpretations`에 근거와 함께 두고 모든 run manifest에 복사한다.

<a id="modules"></a>

### 6.3 Module과 실행 dependency

`solarvpp`의 모듈 책임은 다음과 같다.

| 모듈 | 책임 |
|---|---|
| `config` | protocol·schema·evidence·실행 config, 입력 구성·손실 정의, 모델 버전(MK) 등록 목록 |
| `store` | content key cache store, 구성요소 버전(`VERSIONS`), 실행 환경 기록 |
| `identity` | Solarvpp 계산 payload와 완료 조건, `vppcontract.identity` adapter |
| `data` | 시간축, 태양 위치, 원자료 계약, base·weather cache |
| `forecast/queries` | 학습 query, 평가 cohort, evaluation panel, `ActualView` |
| `forecast/features` | 입력 구성별 feature, PV physics proxy, 하루 맥락 집계, 대체·표준화·ZCA·power curve |
| `forecast/models` | XGB, DL(하루 상태 확장 포함)과 보조손실, 분위수 후처리·블렌드 |
| `forecast/train` | DL 입력 tensor, λ calibration, 학습·예측 |
| `scoring` | 정산 S·W(그룹 예측과 배정), bootstrap, run 비교 |
| `submission` | 제출 형식 계약, test 입력 점검 |
| `experiment` | run 실행·manifest, 입력×손실 grid와 noise floor 판정 |
| `validation` | 선택적인 comparison snapshot 검증, 저장된 run의 설정 대조 |
| `adoption` | 명시적 MK 등록과 champion pointer 변경 |
| `artifacts` | run·조건부 실험 기록의 상태 판정(current·superseded·replaced·dead·aborted·running)과 archive 이동. 판정 규칙은 `vppcontract.artifacts` |
| `bundling/residuals` | 발전소 단위 primitive와 잔차 |
| `bundling/error_model` | 공동 오차 모델 학습과 오차 시나리오 |
| `bundling/optimize` | 시나리오 목적, QUBO, local search |
| `bundling/partition` | 배정 표현·검증, instance |
| `bundling/pipeline` | 조건부 bundling 실행·방법 선택·transfer·배포 판정·audit |
| `cli` | 명령행 진입점 |

`python -m solarvpp conditional-bundling`은 설정의 전체 instance를 실행한다. `--instances V`는 V만 실행하고 site forecast는 D·V만 만든다.
H 단독은 V·H, B 또는 test 단독은 H·B site forecast를 필요로 한다. `--transfer`는 요청한 labelled instance만 추가로 transfer한다.
완료된 instance와 모델은 내용 기반 key로 재사용한다. 최종 추천은 `conditional-decide`를 명시할 때 만든다.
명령별 사용법은 [README](../README.md#실행)에 둔다.

<a id="cache-records"></a>

### 6.4 Cache와 실행 기록

산출물 root는 `configs/runtime.json`의 `paths.artifact_dir`이다. 아래 `<artifact_dir>`는 이 경로를 가리킨다.
실행 방법과 산출물 위치 안내는 [README](../README.md#실행)를 따른다.

cache key에는 원자료 hash, schema version, original hash, cohort/query 정책, 학습 cutoff, 입력 구성, model 설정, seed, 패키지 버전과
`store.VERSIONS`의 구성요소 버전을 넣는다. 모듈 소스 hash는 key에 넣지 않는다. 구성요소의 출력이 바뀌는 변경을 하면 그 번호를 올리고,
출력이 같은 리팩터링은 번호를 바꾸지 않는다. 각 cache key에는 그 산출물이 쓰는 구성요소의 버전을 모두 넣고,
등록 정의를 벗어난 run은 기본으로 새 학습 cache를 실행 동안의 임시 폴더에만 쓰고, 이미 있는 cache는 읽는다. 배정만 바꾼 run과 `submission` level은 저장한다.
저장 여부는 manifest의 `cache_stored`에 적는다. `--use-cache`나 환경 변수 `VPP_EXPERIMENT_CACHE=1`이면 저장하고, 저장하지 않은 완료 run에 붙이면 같은 계산을 다시 학습한다.
기준은 [README의 저장 관리](../README.md#storage)를 따른다.
코드 출처는 run manifest의 `source_sha`로 기록한다.
공통 hash·index I/O는 `vppcontract.identity`가 담당한다. Solarvpp의 `identity`는 계산 payload, `manifest.json` 위치와
완료 조건을 `ArtifactAdapter`로 제공한다. Index는 package별 산출물 root에 둔다.
Run·transfer는 같은 `calculation_identity`를 사용한다. 실제 실행 expert의 seed별 model cache key, blend, cutoff,
partition hash, evaluation panel, DL runtime, postprocess·scoring version을 계산 key에 넣는다.
일반 평가·confirmation·bundling은 계산이 같으면 같은 key를 쓰고, smoke와 diagnostic은 출력·평가 내용이 달라 별도로 구분한다.
MK·recipe 이름, 경로, 문서 설명, 원본 파일 hash와 selection context는 출처 기록이며 계산 key에 넣지 않는다.
새 manifest의 `calculation`은 key의 내용을, `provenance`는 설정 파일과 recorded experiment의 출처를 기록한다.
Component cache key 계산은 그대로 유지한다. 예를 들어 blend만 바꾸면 expert cache를 재사용하고 최종 run key가 바뀐다.

`index-runs --write`는 기존 완료 run의 manifest에 기록된 model cache key와 평가 조건으로
`<artifact_dir>/run-index/<calculation-key>.json`을 만든다. Index는 기존 디렉터리의 상대경로·기존 run key·manifest SHA를 보관한다.
조회할 때 완료 상태, 출력 파일, manifest SHA와 계산 key를 대조한다. 새 run도 완료 후 이 index에 등록한다.
Migration은 기존 manifest·예측·모델·cache key를 수정하지 않는다. 같은 계산의 기록이 여러 개면 index가 모두 보관한다.
Recorded transfer는 저장된 설정을 복원해 같은 model key 함수를 사용한다. 별도의 문자열 치환이나 `recorded-` key를 만들지 않는다.
모델 간 비교에는 동일한 evaluation-panel hash를 요구한다.
Forecast·observed 원본 캐시는 seed와 실험 간에 공유한다.

`--experiment <directory>`로 transfer·audit·decide 대상을 지정한다. 생략하면 `<artifact_dir>/conditional/latest.json`을 따른다.
`conditional-audit`는 원본 generation·SMP와 저장된 점예측·그룹 예측 CSV로 금액을 독립 재계산한다.
완료한 부분 결과부터 검사하고 pending instance를 표시한다. Selection·transfer·decision은 존재하는 경우에 검사한다.
저장된 primitive definition·period·budget을 쓰며, 현재 champion 변경으로 과거 실험이 invalid해지지 않는다.
Transfer의 model definition은 해당 실험의 primitive와 대조한다. Decision은 기록 당시 context와 transfer snapshot으로 재계산한다.
최종 선택 partition은 별도 snapshot으로 보존한다.
검증 결과는 해당 실험의 `verification.json`, 명령 실행과 실패 상태는 `execution.json`에 기록한다.
NLL checkpoint는 `<artifact_dir>/cache/error_model/`, 시나리오 비교 결과와 구성 파일은
`<artifact_dir>/conditional/<hash>/`에 둔다. 전체 시나리오 배열은 실행 중 생성하고 파일로 중복 보관하지 않는다.
모든 CUDA 작업은 그룹 모델 학습과 같은 `<artifact_dir>/.cuda.lock`으로 하나씩 실행한다.

각 run은 `execution_status`, `evaluation_status`, `selection_status`를 따로 기록한다.
실행이 만든 `selection_status`는 `unselected`이며, `champion_at_execution`은 당시 기본 모델을 나타낸다.
채택은 registry의 `adoption` 기록으로 관리한다.
연간 prediction 시간축, 월별 정답 mask, 실제 scorer 출력, cutoff lineage가 갖춰진 결과를 평가 완료로 처리한다.
축소 실행은 실행 확인 상태로 저장한다.

<a id="recipes"></a>

## 7. Experiment recipes

<a id="initial-grid"></a>

### 7.1 초기 input × loss grid

아래는 초기 비교 설계의 기본값이다. 각 단계·후보·`aux_ratio`·blend는 명령에서 독립적으로 선택한다.

| 단계 | 실험 | 바꾸는 요소 | 고정하는 요소 |
|---|---|---|---|
| 1 | noise floor: forecast-solar-diff__pinball, seed 42~51 | seed | original·query·V·학습 예산·weighted-quantile-w0.5 |
| 2 | 입력×손실 screening: forecast-raw·forecast-solar·forecast-solar-diff·forecast-solar-physics·forecast-physics × pinball·mae·mse·ramp·fourier·cosine, 30칸, seed 42 | 입력 표현·DL 보조손실 | 모델 계열·후처리·weighted-quantile-w0.5·r=0.1 |
| 3 | 지정한 칸 confirmation(seed 42·43·44), `aux_ratio` follow-up | 후보 확인·보조항 세기 | 비교에서 명시한 설정 |
| 4 | 조건부 bundling([4절](#bundling)) | G1·G2 배정 | 선택한 예측 primitive |

각 run은 실제 실행한 expert와 선택한 blend 비율의 지표를 기록한다.

forecast-raw~forecast-physics는 진화 순서가 아니라 서로 다른 representation 가설이다.
forecast-solar 위의 2×2(`diff` 유무 × `physics` 유무)로 두 묶음의 효과를 보고, forecast-raw→forecast-solar 차이로 태양 위치의 효과를 본다.

<a id="weighted-day-energy"></a>

### 7.2 Weighted-Q + day-energy

`weighted-day-energy` recipe는 다음 조합을 기본값으로 제공한다.

| 항목 | 설정 |
|---|---|
| XGB expert | forecast-physics, capacity-weighted quantile loss, time weight `none` |
| DL expert | forecast-physics-daystate, day-energy, time weight `settlement`, `aux_ratio` = 0.1 |
| DL objective | weighted pinball + λ_E·median day-energy + λ_W·observed Huber |
| DayState auxiliary | observed weight는 `dl_daystate_observed_weight`(기본 0.1), 별도 energy head weight는 0 |
| Blend | weighted-quantile-w0.5 |
| Run 기록 | `model_inputs`, `model_time_weights`, `recipe`, seed별 λ_E와 `daystate_aux_weights` |

Recipe의 weighted pinball은 \(\sum_{b,h,\alpha}M_{b,h}C_b k_h\rho_\alpha(z_{b,h}-q_{b,h,\alpha})\)를
\(|Q|\sum_{b,h}M_{b,h}C_b k_h\)로 나눈 값이다. day-energy가 median trajectory를 직접 supervise하며 observed auxiliary를 계속 더한다.

기본 loss grid는 pinball~cosine을 유지하고, day-energy는 recipe 또는 명시적인 loss 선택으로 실행한다.
Recipe 실행과 override 예시는 [README](../README.md#weighted-q--day-energy-recipe)를 따른다.

<a id="balance-grid-recipe"></a>

### 7.3 Capacity balance 정규화 비교

`fixed_lambda = 1`을 기준으로, 같은 forecasting primitive·joint residual model·방법 집합·scenario·seed를 사용하고
capacity 균형항만 [4.7절](#balance)의 정규화 방식으로 바꿔 비교한다.

| 항목 | 설정 |
|---|---|
| λ̃ grid | {0.003, 0.01, 0.03, 0.1}: \|b\| = β에서 penalty가 각 objective의 J_ref 대비 0.3~10% |
| 선택 scope | 요청한 labelled instance scope의 모든 방법. 전체 V·H·B를 요청하면 그 전체 |
| 선택 규칙 | Scope 안의 모든 선택 해가 \|b\| ≤ β를 만족하는 가장 작은 λ̃. 선택에는 모형이 만든 배정의 \|b\|를 사용 |
| 적용 | 선택한 λ̃ 하나를 적용 instance·방법·test에 고정하고 scope와 적용 대상을 `balance_selection.json`에 기록 |
| 탐색 시작점 | 각 λ̃의 QUBO 해·original·직전 λ̃ 해·무작위 배정 |
| 비교 기록 | λ̃별 \|b\| 분포, 탐색 objective, 기준 배정과의 Hamming 거리, penalty·불균형 frontier |

방법 ranking은 [5.4절](#scenario-evaluation)에 따라 요청한 labelled instance scope의 상대 실현 ΔW를 capacity 가중 평균한다.
H·B group-level transfer와 test의 그룹당 20 MW 판정은 [5.5절](#transfer)을 따른다.
Feasibility repair를 적용한 test assignment는 그룹 capacity와 objective를 다시 계산해 기록한다.

Grid의 각 λ̃ 해가 주 해가 될 수 있으므로 모든 λ̃에서 QUBO를 같은 예산으로 푼다.
λ̃ 4개 × 방법 3개 × instance 4개를 실행하면 QUBO 48회이며, λ̃당 300 work를 예산으로 둘 수 있다.
Tabu 비교와 group-level transfer는 해당 실행 설정으로 함께 지정한다.

검증 항목은 |b| = β에서 각 objective의 penalty가 λ̃·J_ref인지, full MSE의 x = ½𝟙 값이 J_ref,MSE인지,
현재 선택 scope에서 band를 만족하는 가장 작은 λ̃를 고르는지, test의 20 MW 판정과 repair 후 재평가가 일치하는지이다.

