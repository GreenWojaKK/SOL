# solarvpp2 구현·실행 참고

[설계 본문](solarvpp2-design.md)은 forecaster·risk·partition의 역할과 연결을 설명한다.
이 문서는 현재 구현의 수식·설정·초기화·gradient와 CLI의 평가·저장 동작을 기술한다.
등록 모델의 이름과 수치 예시는 [11절](#versions)에 모으며, 실행에는 run에 기록된 resolved 설정을 사용한다.

## 1. 공통 단위와 정보

집합 A의 용량 C_A는 kW, 시간별 발전량 Y_A와 제출값 a는 kWh, risk R은 예상 absolute error(kWh)다.
실제 정산 가중치는 0.10·SMP·k(원/kWh)이며 W는 그룹·시간별 가중 absolute error의 합(원)이다.
시각은 KST의 우측닫힘 1시간 구간이다. 용량 정규화의 energy 기준은 C_A × 1h다.
데이터와 제출 조건은 [Problem definition](../PROBLEM.md)을 따른다.

## 2. Node별 입력과 출력

Node는 따로 학습하거나 cache하는 계산 단위다.

| node | 입력 | 출력 |
|---|---|---|
| base | sites·generation·SMP 원자료 | 시간축에 정렬한 발전량, 정답 mask, SMP·k, solar geometry, 야간 규칙 |
| weather | forecast·observed 원자료 | provider·round·station·시각 tensor와 결측 mask |
| sampler | base, cutoff, sampler 설정 | query: 대상일 d, 집합 A, 24시간 정답 mask |
| features | base, weather, query | XGB 집합 feature, DL 입력 tensor |
| forecaster | features, training query, 모델 정의 | 임의의 (d, A)에 대한 24시간 quantiles와 제출값 a_θ, risk 입력용 표현 |
| risk | 고정된 forecaster의 출력·표현과 held-out error | 임의의 (d, A)에 대한 시각별 R |
| partition | risk, 목적 일자·가중치, 용량 제약 | 배정 x |
| output | partition, 같은 forecaster | 기간별 배정 CSV, 그룹 예측 CSV, manifest |

원자료의 중복 키, 정시가 아닌 timestamp, 야간 규칙, cohort, 날짜 경계는
[solarvpp의 data·query 계약](solarvpp-reference.md#data-contract)과 [period·cutoff 정의](solarvpp-reference.md#periods)를 따른다.

## 3. 정보 가용 시점

- 대상일 d의 제출에는 d−1일 17시까지 발행·생성된 정보를 사용한다. 대상일의 round 10·17 forecast도 이 정보에 포함된다.
- 배정에는 평가 기간 첫 대상일의 예측시점까지의 정보를 사용한다. Test 배정의 cutoff는 2025-05-31 17시이며,
  test 기간의 forecast와 SMP는 배정 결정에 사용하지 않는다.
- Generation·observed는 cutoff 이후 값을 mask한 뒤 training query와 target을 만든다. Observed weather는 학습의 보조 target으로만 사용한다.
- 입력 선택과 전처리는 cutoff 이전 자료로 정한다. 대상일별 입력 점검에는 그날 예측시점까지 발행된 forecast를 사용한다.

<a id="query-sampler"></a>

## 4. Training data와 query sampler

<a id="training-days"></a>

### 4.1 Forecaster와 risk의 날짜 분리

Instance의 cohort는 결정 cutoff까지 유효 발전량이 한 번 이상 있는 발전소다. Cutoff 이전 대상일을 forecaster training days와
held-out days로 나눈다. Held-out의 block 길이·주기·위치는 모델 정의의 `heldout`에서 읽는다.

Forecaster는 training days의 query로 학습한다. Held-out days에서는 학습이 끝난 forecaster로 예측하고 실현 오차를 계산해
risk model의 training target을 만든다. 이 과정에서 forecaster의 weights는 고정한다. Risk model의 block별 cross-fitting은 6절을 따른다.

### 4.2 Query 구성과 정답 mask

Query는 (대상일 d, 집합 A, 24시간 정답 mask)다. A는 그날까지 첫 유효 발전량이 나타난 cohort 발전소에서 선택한다.
집합 입력은 구성원 mask와 집합 요약이며, G1·G2 indicator와 site ID를 사용하지 않는다. 구성원의 나열 순서도 예측에 영향을 주지 않는다.

집합 A는 하루 동안 고정한다. 시각 h에서 모든 구성원의 정답이 있으면 mask를 1로 두고 Y_{A,h}를 target으로 사용한다.
구성원 하나라도 정답이 없으면 해당 시각의 mask를 0으로 둔다. 결측 시각에도 A의 구성원은 유지한다.

현재 sampler는 다음 집합 종류를 지원한다.

| 종류 | 구성 |
|---|---|
| original | 원래 두 그룹 각각에서 그날 편입된 발전소 |
| random halves | 무작위 이분할 |
| stratified halves | 같은 WA·WB station 쌍의 발전소를 capacity가 비슷하게 나눈 이분할 |
| uniform subsets | 크기가 다양한 무작위 부분집합 |
| station subsets | station 집중도가 다양한 부분집합 |

종류별 비율·개수와 부분집합 범위는 sampler 설정으로 정한다. 학습 집합은 7절의 용량 조건을 만족하는 후보 배정을 포함하도록 구성한다.
Optimizer는 sampler가 뽑았던 집합 목록에 한정하지 않고 7절의 용량 조건 안에서 탐색한다.

<a id="forecaster"></a>

## 5. Forecaster

Forecaster는 예보 정보 I_d와 집합 A를 받아 Y_A의 24시간 quantiles를 출력한다. 두 expert는 4절의 query와 각 입력 구성에 맞춰 학습한다.

<a id="xgb"></a>

### 5.1 XGB 집합 expert

등록 구성의 XGB 입력은 `forecast-physics`다. 입력 구성은 `xgb.input`으로 선택한다. PV별 feature를 집합 안에서 capacity-weighted mean·std로 집계하고,
집합 capacity·PV 수·HHI, sunup capacity share, 시간 표현, provider·round별 missing capacity share를 더한다.
Target은 정규화 발전량에서 집합 power curve를 뺀 residual의 19개 quantile이다.
Feature와 학습 설정의 정의는 [solarvpp의 XGBoost 정의](solarvpp-reference.md#xgb)를 따른다.

<a id="dl"></a>

### 5.2 DL 집합 expert

DayState를 포함한 DL 경로의 등록 입력은 `forecast-physics-daystate`다. 입력 구성은 `dl.input`으로 선택한다.

```text
station forecast(round 10·17) → provider encoder ─┐
station의 24시간 forecast ───→ DayState encoder ──┼→ PV fusion → 집합 pooling → 24시간 decoder → 19 quantiles
PV별 forecast-physics 입력 ─────────────────────────┘                   ↑
                                                           집합 capacity·PV 수·HHI
```

PV representation은 provider encoder 출력, station별 DayState, PV별 forecast-physics 입력으로 만든다.
DayState는 station의 하루 forecast sequence에서 계산하며, 학습 기간 observed의 하루 평균을 보조 target으로 사용한다.
PV representation은 집합 A와 무관하게 계산한다.

`dl.site_aux_ratio > 0`이면 PV representation 위에 발전소 이용률 보조 head(선형 head 하나)를 둔다. Head는 발전소별 이용률(발전량 / 설비용량)을
학습하고, 그 기울기는 provider encoder·DayState encoder·PV fusion으로 전달된다. 집합 pooling 뒤의 decoder와 quantile head에는
이 항의 기울기 경로가 없다. Head는 학습에만 쓰며, 예측과 배정 탐색은 PV representation → pooling → decode 경로를 그대로 쓴다.
Head는 다른 층을 모두 만든 뒤 붙이므로 나머지 층의 초기값은 보조 head를 끈 같은 seed의 구성과 같다.

Pooling은 구성원 PV representation의 capacity-weighted mean·std다. 두 통계는 representation의 각 차원에서 계산한다.
여기에 집합 capacity·PV 수·HHI를 붙여 pooled sequence [24, 2H + 3]을 만들고, 24시간 decoder와 linear head에 전달한다.
H는 PV representation의 차원이다. 구조 크기와 학습 설정은 [solarvpp의 DL 구조](solarvpp-reference.md#dl)를 따른다.

Decoder는 `solarvpp2/forecast/decoders/`에 decoder별 모듈로 둔다. 모듈은 이름(`dl.decoder.kind`), 구현 version, 기본값, 설정 검사,
생성 함수를 정의하고, 출력은 시각별 집합 representation [24, out_width]이다. 이 representation이 quantile head와 risk 입력으로 간다.

- `bigru`: 2층 양방향 GRU(폭 64). 모델 정의에 `dl.decoder`가 없으면 이 decoder이고, 크기는 runtime hyperparameters의 `dl_gru_*`를 쓴다.

- `shared-memory-looped`: 같은 attention·FFN block을 T번 반복한다. Pooled sequence를 폭 128로 투영하고 시각 embedding을 더한
  X에서, 한 forward에 한 번 shared memory K/V를 만든다. 매 반복은 shared memory와 현재 상태의 K/V를 하나의 softmax로 함께 읽고
  (첫 pass는 둘이 같아 24개, 이후 48개), 원래 입력을 반복마다 다시 더하지 않는다. 4 heads·FFN 256·dropout 0.1, attention·FFN의
  Pre-LN과 반복 출력의 LN을 쓰고, 반복 embedding table은 8행으로 고정한다. Parameter는 head를 빼고 153,728개이며 T에 따라 바뀌지 않는다.

Provider context는 시간별 provider 표현(ProviderEncoder 출력 [대상일, 관측소, 24, 32])에 관측소 사이 context를 더하는 component다.
`solarvpp2/forecast/contexts/`에 종류별 모듈로 두고, 모델 정의에 `dl.provider_context`가 없으면 쓰지 않는다. 그룹 구성과 무관한 PV
representation 계산 안에 있으므로 예측·배정 탐색 경로는 바뀌지 않는다.

- `inverted-attention`: 관측소 하나의 24시간 표현 전체를 폭 64 token으로 투영하고(두 제공사 공유), provider embedding과 이 module
  전용 관측소 embedding을 더한다. 같은 대상일의 WA·WB 관측소 token 전체에 Pre-LN Transformer block 하나(4 heads, FFN 128, dropout 0.1)를
  적용하고, LN 뒤 다시 24 × 32로 투영해 원래 표현에 더한다. 비활성 제공사는 key에서 빼고 residual을 0으로 둔다. 다른 층을 모두 만든 뒤
  붙이므로 기존 층의 초기값은 해당 component를 끈 같은 seed의 구성과 같고, 더하는 parameter는 141,440개다. DayState 보조 손실은 원 입력을 따로 읽으므로 이
  module을 거치지 않는다.

Temporal routing은 시각 t의 표현이 같은 하루의 다른 시각 표현을 읽어 원래 값에 더하는 component다. `solarvpp2/forecast/routing/`에
종류별 모듈로 두고, 모델 정의에 항목이 없으면 쓰지 않는다. 읽는 위치와 가중은 예보 입력과 날씨 맥락으로 정하고, 읽어 오는 값은 원래
표현이다. 출력 사영을 0으로 초기화하고 다른 층을 모두 만든 뒤 붙이므로 기존 층의 초기값과 학습 시작 시 출력은 해당 component를 끈 같은 seed의 구성과 같다.

`dl.plant_routing`은 집합 pooling 전 PV별 provider 표현에 둔다. 그룹 구성과 무관한 PV representation 계산 안에 있으므로 예측·배정 탐색
경로는 바뀌지 않는다. Station embedding은 읽어 오는 값과 fusion에만 들어가고, 읽는 위치와 가중을 정할 때는 쓰지 않는다.

- `time-shift`: 참조 시각 τ = t + σ·b + 1.5·σ·tanh(Δ)를 Catmull-Rom 보간으로 읽고 attention으로 합친다. b는 −1·0·1, 시간 범위 σ는 기본 0.5·1·2로
  참조점은 9개다. Δ와 attention logit은 PV가 속한 관측소별 prior(`region`)와 날씨 맥락(`weather`)의 합이다. 날씨 맥락은 두 제공사 예보 입력과
  fusion 공통 입력 가운데 정적 정보·provider 사용 여부를 뺀 채널이고, attention key는 해당 제공사 예보 입력과 상대 위치로 만든다.
- `nearby-hours`: 시각 t가 앞뒤 W시간 [t − W, t + W] 안의 시각 s를 score q_t·k_s/√d − λ_t·(s − t)²의 attention으로 읽는다. W(`hours`)는 기본 3이고, 하루 밖 시각은 가린다.
  q와 λ_t의 날씨 맥락은 두 제공사 예보 입력과 disagreement·태양 위치·하루 상태이고, k_s는 해당 제공사 예보 입력으로 만든다. Gate(기본 사용)는
  같은 날씨 맥락의 sigmoid로 읽어 온 값의 몫을 시각마다 0~1로 조절한다.

`dl.group_routing`은 집합 pooling 뒤 pooled sequence [24, 2H + 3]에 두고 `decode()` 안에서 계산한다. 종류는 `nearby-hours`이고, 계산은 위와 같으며
W는 기본 3, gate는 기본으로 쓰지 않는다. 맥락(`context`)의 기본은 `neutral`이다. `neutral`은 맥락·key를 모든 관측소에 같은 평균 station embedding을 넣고 정적 정보를
0으로 둔 PV 표현의 pooling으로 만들어, 읽을 시각을 정할 때 관측소·발전소 identity를 보지 않는다. 이 표현은 구성원 합으로 pooling해야 하므로
`fuse()`가 PV representation 뒤에 이어 붙여 폭 2H를 돌려주고, 모든 pooling 경로를 같은 방식으로 거친 뒤 `decode()`가 나눠 읽는다. 발전소 이용률
head는 앞의 H만 읽고, 배정 탐색 저장소의 DL 표현도 폭 2H가 된다. `pooled`는 pooled sequence를 맥락·key로 그대로 쓰고 PV representation의 폭을 유지한다.

하루 배율 head(`dl.day_scale`)는 decoder 뒤에 집합별로 둔다. Decoder의 24시간 표현에서 차원별 평균과
표준편차를 이어 붙이고, LayerNorm → Linear(2·out_width→hidden) → SiLU → Linear(hidden→1)로 log 배율을 계산한다.
기본 BiGRU의 out_width는 128, head의 hidden은 32다.
표준편차는 `sqrt(var + 1e-6)`으로 계산한다. 배율은 `exp(b × tanh(logit))`이며 기본 b는 0.35다.
같은 집합·대상일의 모든 시각·분위수에 배율 하나를 곱한 뒤 기존 expert 후처리와 blend를 적용한다.

Head는 다른 module을 생성한 뒤 마지막에 붙이고, 마지막 선형층의 weight·bias를 0으로 초기화한다. 초기 배율은 1이고
기존 parameter의 초기값과 예측을 보존한다. 첫 backward는 마지막 선형층을 학습하며, 해당 층이 갱신되면 앞 MLP와 decoder에도
배율 경로의 gradient가 흐른다. Quantile head를 통한 기존 gradient 경로는 첫 step부터 유지한다.
Head는 기존 forecast 손실로 함께 학습한다. Site auxiliary와 DayState 보조 head는 각각의 기존 표현을 감독한다.
학습·추론·배정 탐색은 같은 `decode()`에서 현재 집합의 배율을 계산하고, risk 입력용 decoder 표현의 폭은 유지한다.

<a id="training-objective"></a>

### 5.3 Training objective와 기록

`dl.loss`는 main pinball과 median 보조항의 조합을 선택하고, `dl.aux_ratio`는 보조항의 gradient calibration 비율을 정한다.
Loss 정의와 auxiliary coefficient calibration은
[solarvpp의 training objective](solarvpp-reference.md#losses)와 [auxiliary coefficient calibration](solarvpp-reference.md#aux-calibration)을 따른다.

Median을 사용하는 auxiliary loss(ramp·day-energy 등)는 quantiles를 정렬한 뒤 0.5 위치의 값에 적용한다.
Gradient는 sorting 후 그 위치를 차지한 quantile head로 전달된다.

Capacity-weighted loss의 normalizer는 training query 전체에서 한 번 계산한다.

- 시각 단위 항: 해당 항의 가중치 C_A 또는 C_A·k의 유효 시각 평균을 사용한다.
- 하루 단위 항: training query의 C_A 평균을 사용한다.
- Batch의 가중합은 해당 normalizer와 유효 시각 수 또는 query 수의 곱으로 나눈다.

Time weight의 설정값은 `none`과 `settlement`다. `settlement`를 선택하면 표시 시각 11~15시에 k = 3을 적용한다.
XGB는 sample weight에, DL은 pinball과 day-energy의 시각 가중에 곱한다.

`settlement-penalty`는 main pinball에 group median의 정산 가중 absolute error를 보조항으로 더한다.
정렬한 median을 `[0,1]`로 clip하고 `Σ M·C_A·SMP·k·|m−z|`를 학습 query의 고정 가중 평균과 batch 유효 칸 수로 나눈다.
발전소 실적을 합한 group 정답과 group 예측을 비교하며, 발전소별 absolute error를 더하지 않는다.
SMP·k는 학습 loss용 metadata이며 NN 입력과 inference에는 넣지 않는다. Cutoff 이후와 R0/E는 이 학습용 tensor에서 0으로 둔다.
이 항은 `time_weight`와 별개로 실제 SMP·k를 한 번 사용한다. `time_weight=settlement`이면 main pinball에만 k가 추가로 적용된다.
Warm-up 뒤 기존 gradient norm calibration으로 λ를 정한다. DayState energy·observed와 site auxiliary는 유지한다.
Clip 범위 밖의 gradient는 main pinball이 제공한다. 이 보조항은 개별 DL의 group median을 감독하며,
XGB blend·seed ensemble의 최종 W와 같은 학습 목적함수는 아니다.

발전소 이용률 보조항은 `dl.site_aux_ratio > 0`일 때 사용한다.

- 정답은 cutoff까지 유효한 발전량을 설비용량으로 나눈 값이다. 무효 status, 발전소별 야간, cutoff 이후, held-out 대상일은 쓰지 않는다.
  Held-out 대상일을 빼므로 risk 학습용 예측을 만드는 forecaster는 그날의 발전소 정답을 학습하지 않는다.
- 단위는 query가 아니라 batch의 unique 대상일이다. 그날 유효한 발전소 정답을 한 번씩 쓰고, 같은 날 query 수만큼 반복하거나
  query membership으로 제한하지 않는다.
- 손실은 유효 칸의 용량 가중 MAE이고 분모는 batch 유효 칸의 용량 합이다. Warm-up 뒤부터 더한다.
- λ_site는 warm-up 직후 `λ_site = site_aux_ratio · E‖∇L_main‖ / E‖∇L_site‖`로 정해 고정한다. L_main은 학습과 같은 pinball 항(고정 normalizer·
  mask·capacity·slot 가중)이고 다른 보조항을 더하지 않는다. 기울기 크기는 학습 가능한 parameter 전체의 L2 norm이며, 한 손실에
  연결되지 않은 parameter는 0으로 센다. 따라서 `site_aux_ratio`는 calibration batch에서 전체 parameter 기울기 크기의 평균 비율이고,
  매 step 공유 encoder의 기울기 비율을 뜻하지 않는다. Calibration은 RNG 상태를 되돌려 이후 dropout 순서를 유지하고 optimizer를
  진행하지 않는다. 기록에는 `site_aux_ratio`, λ_site, parameter 범위, 두 기울기 크기의 평균, 사용한 batch 수를 남긴다.

학습 총손실은 main, aux_weighted, daystate_energy, daystate_observed, site_weighted의 합이다. 가중 전 값(aux, site)은 기록에만 쓴다.
학습 기록은 50 step마다 main loss, auxiliary loss와 발전소 이용률 항의 가중 전·후 값, DayState의 energy·observed 항, total loss의
평균을 남긴다. Warm-up 시점과 학습 종료 시점에는 총손실에 들어가는 항별 gradient norm을 기록한다.

### 5.4 Inference와 제출값

Seed 목록과 XGB 결합 비율 w는 모델 정의의 `seeds`와 `blend.xgb_weight`에서 읽는다. 후처리 순서는 다음과 같다.

```text
expert별 동일 quantile의 seed 평균
    → capacity를 곱해 kWh로 복원
    → [0, C_A × 1h] clipping·야간 0 처리
    → expert별 quantile sorting
    → 동일 quantile끼리 w·Q_XGB + (1−w)·Q_DL
    → quantile sorting
    → 0.5 quantile을 제출값 a_θ로 선택
```

Forecast stage 평가는 seed별 제출값도 계산한다. Seed s의 XGB·DL 분위수에 위 후처리를 같은 순서로 적용해 결합하고 0.5 분위수를
고른다. 결합 제출값의 W는 seed별 W의 평균이 아니며, 두 값을 따로 기록한다.

예측과 배정 탐색에서는 대상일별 PV representation을 한 번 계산하고, membership에 따라 pooling·decode한다.
Pooling 전에 구성원 간 attention을 적용하면 membership 변경마다 PV representation도 다시 계산해야 하므로,
이 계산 경로에서는 집합과 무관한 PV representation을 유지한다.

<a id="risk"></a>

## 6. Risk model

Risk model의 target은 고정된 forecaster가 실제로 제출하는 값의 absolute error다.

\[
R_{\phi,h}(I_d,A)\approx E\big[\,|Y_{A,h}-a_{\theta,h}(I_d,A)|\;\big|\;I_d,A\big]
\]

Forecaster의 bias도 이 target에 포함된다. 현재 risk 구현은 expected absolute error를 추정하는 objective로
Tweedie deviance(log link, p = 1.5)를 사용한다. Target은 C_A로 정규화하고, 추론 시 C_A를 곱해 R을 kWh로 복원한다.
모든 구성원이 야간인 시각은 R = 0으로 둔다.

MLP head의 입력은 다음과 같다.

- Seed별 DL decoder 출력(BiGRU 또는 shared-memory looped attention의 시각별 집합 representation)을 이어 붙인 값
- 최종 blend median과 quantile 폭, XGB·DL 각각의 median
- 집합 capacity·PV 수·HHI, provider별 station capacity share의 제곱합
- 시간 표현

Training과 objective 계산에 사용할 head는 다음 순서로 만든다.

1. Held-out days에서 risk sampler의 query를 만들고, 같은 forecaster의 제출값과 실현 absolute error를 계산한다.
   Target에는 4절의 정답 mask가 유효한 시각을 사용한다.
2. Held-out block마다 해당 block을 제외한 오차로 head를 학습한다.
3. Partition objective에서 block b의 R을 계산할 때는 b를 학습에서 제외한 head를 사용한다.
4. Held-out block 전체로 별도의 head를 학습하고, 평가 기간의 risk calibration에 사용한다.

모든 risk head의 학습 동안 forecaster는 고정한다. 정산 가격 가중치는 7절의 objective에서 적용한다.

<a id="partition"></a>

## 7. Partition objective와 search

### 7.1 배정과 용량 조건

x ∈ {0,1}^n이며 A_1(x) = {i : x_i = 1}, A_2(x) = {i : x_i = 0}이다. Test에서 각 그룹의 capacity는 20,000 kW 이상이다.
과거 instance에는 같은 최소 capacity share를 적용한다.

\[
\frac{C_{A_g}}{C}\geq\frac{20{,}000}{42{,}655.37}\approx0.4689,
\qquad C=\sum_i c_i.
\]

Optimizer는 binary membership을 탐색하며, R도 해당 집합에 대해 계산한다.

### 7.2 Objective의 날짜와 가중치

결정 cutoff 이전 정보로 다가오는 평가 기간의 expected settlement penalty를 추정한다.

\[
\hat J(x)=\sum_{d\in\mathcal D}\omega_d\sum_h\pi_{d,h}
\Big[R_{\phi,h}(I_d,A_1(x))+R_{\phi,h}(I_d,A_2(x))\Big].
\]

| 항목 | 현재 지원하는 설정 |
|---|---|
| 𝒟 | 결정 cutoff 이전 held-out days |
| ω_d | `calendar_month`: 달력 월마다 가중치 합이 같고 월 안에서 균등. `uniform`: 날짜별 균등 |
| π_{d,h} | 0.10·k_h × cutoff 직전 12개월의 같은 달력 월·시각 평균 SMP |

가격 profile과 R을 곱하는 objective는 I_d가 주어졌을 때 SMP와 absolute error를 독립으로 취급한다.
월·시각별 평균 SMP를 사용하므로 일자별 가격 변동도 이 profile로 근사한다.

### 7.3 Incremental search

현재 optimizer는 설정한 original·stratified·random partition을 시작점으로 flip·swap local search를 수행한다.
Stratified partition은 같은 WA·WB station 쌍의 발전소를 capacity가 비슷하게 나눈 배정이다.

후보 계산은 다음 incremental state를 사용한다.

1. 𝒟의 대상일마다 PV별 기여(XGB 입력·유효 여부·solar geometry·결측, DL representation)를 계산해 저장한다.
2. 현재 배정의 그룹별 가중합을 유지한다. XGB에는 합·제곱합·유효 capacity를, DL에는 S⁽¹⁾·S⁽²⁾를 저장한다.
   집합 capacity·PV 수·capacity 제곱합·station별 capacity도 유지한다.
3. Flip·swap 후보마다 이동한 PV의 기여를 더하고 빼서 후보 가중합을 만든다. 날짜 chunk를 처리할 때는 해당 chunk의 기여만 읽는다.
4. 가중합 이후의 집합 feature, XGB prediction, DL decode, R 계산은 전체 재계산과 같은 함수를 사용한다.
5. 선택한 이동을 현재 가중합에 반영하고, 일정 횟수마다 전체를 다시 합해 누적 오차를 줄인다.

유효 PV가 없는 feature와 야간 판정에는 10⁻⁶ kW를 zero threshold로 사용해 가중합의 부동소수 잔차를 처리한다.
목적값이 유한하지 않은 후보는 제외하고 그 수를 기록한다. 시작 배정의 목적값 또는 한 iteration의 모든 후보 목적값이
유한하지 않으면 search를 중단한다.

<a id="instances"></a>

## 8. Instance와 실행 단계

Instance는 하나의 결정 cutoff, cohort, 평가 기간을 묶는다. Instance의 forecaster는 4절의 날짜 분리로 학습하며,
risk training·partition search·기간 예측에 같은 weights를 사용한다.

아래 시각은 KST이며, 예측·평가 구간의 timestamp는 우측닫힘 1시간 구간의 끝이다.

| instance | 결정 cutoff | 예측·평가 시각 | 발전소 수 |
|---|---|---|---:|
| V | 2023-05-31 17:00 | 2023-06-01 01:00 ~ 2024-06-01 00:00 | 106 |
| H | 2024-05-31 17:00 | 2024-06-01 01:00 ~ 2025-06-01 00:00 | 205 |
| B | 2024-12-24 17:00 | 2024-12-25 01:00 ~ 2025-06-01 00:00 | 253 |
| test | 2025-05-31 17:00 | 2025-06-01 01:00 ~ 2026-06-01 00:00 (제출) | 253 |

`instance --stage`는 실행 범위를 선택한다.

| stage | 실행 범위 | 평가 기록 |
|---|---|---|
| full | forecaster → risk → partition search → output → 평가 | 9절의 배정 비교와 진단 |
| panel | 평가일·완충을 뺀 forecaster 학습 → 날짜별 구성원의 예측·평가 | [P60 protocol](#p60-panel)의 비교 |
| forecast | forecaster 학습 → 고정 배정 original·stratified의 output → 평가 | Blend W, seed별 blend W와 일별 벌점, 같은 후처리의 XGB·DL median W, 계절·휴일 진단 |

Forecast stage는 forecaster 후보 평가에 사용한다. 현재 `recommend` 명령은 full-stage run을 읽는다. Run 이름은 `<MK>-forecast-<key>`다.
Robust fold test의 fold(fold-2309·fold-2312·fold-2403)도 instance 이름으로 받는다. 기간·cutoff는 공통 계약층의 정의(`vppcontract.timeaxis`)를
쓰고, cohort는 위와 같이 cutoff까지 유효 발전량이 있는 발전소다. Fold에서는 forecast stage를 실행한다.
Seed별 blend W는 `instance.json`의 `evaluation.seed_W_krw`에, 결합 예측과 seed별 제출값의 일별 벌점 [대상일, 그룹]은
run의 `daily_penalty.npz`(`<배정>`, `<배정>__s<seed>`)에 둔다.

입력·손실 등의 parameter는 `--xgb-input`, `--dl-input`, `--xgb-time-weight`, `--dl-time-weight`, `--dl-loss`, `--dl-aux-ratio`,
`--dl-site-aux-ratio`, `--xgb-weight`, `--seeds`로 바꾼다. `--dl-aux-ratio`은 ramp 등 기존 보조손실의 비율이고, `--dl-site-aux-ratio`은 발전소 이용률
보조 head의 비율이다.
등록 목록은 유지하고 같은 MK 이름으로 실행하며, report에 실제 모델 정의와 변경값(`parameters`)을 기록한다.
현재 `recommend` 명령은 등록 정의와 다른 parameter run을 자동 추천 대상에서 제외한다.

Forecast 후보는 [P60 panel](#p60-panel)의 같은 학습·평가 조건에서 기준 run과 비교한다.
사용한 seed와 학습 예산은 각 실험의 설정으로 기록한다.

기존 cutoff·cohort에서의 진단을 재현할 때는 V·fold-2309·fold-2312·fold-2403·H 산출물을 `folds`로 비교한다.

```powershell
& $py -m vppcontract folds --candidate V=<run>\outputs\stratified fold-2309=... --baseline V=<기준 run> fold-2309=... --out <보고서.json>
```

`folds`는 fold마다 두 산출물의 발전소 집합과 배정(그룹 이름과 무관)이 같은지 확인하고, 같은 정답 mask로 결합 예측의 W와 계절별 W를
계산한다. Seed별 W는 각 run이 기록한 값(solarvpp `per_seed_W_krw`, solarvpp2 `seed_W_krw`)을 쓰며, run이 기록한 결합 예측 W가
공통 평가기의 W와 같을 때만 쓴다. 네 조건과 판정 자격(다섯 fold·다섯 seed)은 solarvpp 설계 5.7절의 robust fold test와 같다.

<a id="evaluation"></a>

## 9. 평가와 추천

### 9.1 비교 대상

V·H·B에서 같은 시각·같은 정답 mask로 W를 계산한다. 정답 mask는 산출물에 포함된 발전소 전체의 generation이 유효한 시각이며,
비교하는 모든 배정·그룹에 공통으로 적용한다. 각 비교의 ΔW와 7·14일 날짜 block bootstrap 구간을 기록한다.

최종 후보는 모델·parameter·partition을 묶어 champion과 비교한다. 비교 결과의 W에서 다음을 계산해 먼저 제시한다.

| 지표 | 계산 | 용도 |
|---|---|---|
| 고정 기준선 대비 W 변화율 | `(W_candidate − W_baseline) / W_baseline` | 누적 개선과 첫 목표 −10% 도달 여부 |
| Champion 대비 W 변화율 | `(W_challenger − W_champion) / W_champion` | 후보 비교. 음수이면 개선이며 낮을수록 좋다 |

W 변화율은 %로 표시하고, 기준 대비 ΔW와 기준·후보의 prediction artifact를 함께 제시한다.
고정 기준선의 등록 정보는 [Problem definition](../PROBLEM.md#baseline-record)에 둔다. 후보 비교에는 비교 시점의 모델·parameter·partition·run을 명시한다.
아래 비교 1~3은 forecaster와 partition 변경이 최종 W에 기여하는 경로를 진단한다.

| 비교 | 대상 | 목적 |
|---|---|---|
| 1 | 후보 forecaster와 기준 forecaster를 같은 배정으로 예측 | Forecaster 변경의 효과 |
| 2 | 후보 forecaster + stratified partition | 배정 비교의 기준 |
| 3 | 같은 forecaster + risk로 선택한 partition | 비교 2 대비 배정 변경의 효과 |

비교하는 forecaster의 학습 날짜와 held-out 규칙을 함께 기록한다.
Package 간 비교는 공통 계약층 `vppcontract`의 평가기가 수행한다. 평가기는 output과 원자료(generation·SMP)를 읽어
같은 평가 시각의 W·ΔW·날짜 block bootstrap을 계산하며, solarvpp run 디렉터리도 입력으로 받는다.
고정 기준선과 최종 champion–challenger 비교에도 이 평가기의 W를 사용한다.

<a id="diagnostics"></a>

### 9.2 진단

- **계절·휴일:** 배정별로 계절(봄·가을·그 외) × 평일·휴일의 대상일 수, W, k=3 시각의 bias(Σ제출값 / Σ실적 − 1)를 계산한다.
  휴일은 토·일·공휴일·근로자의 날이다. 공휴일 목록은 `configs/calendar.json`에 둔다.
  임시공휴일과 법정 일자가 아닌 선거일은 공고일을 기록하고, 대상일 d의 예측에는 d−2일까지 공고된 항목을 반영한다.
  이 날짜 정보는 진단에만 사용하며, forecast stage에서도 같은 진단을 기록한다.
- **배정 순위:** original, stratified partition, random partition 200개, risk로 선택한 partition에서 Ĵ와 실제 W의 순위 상관을 계산한다.
- **Risk calibration:** 평가 기간의 (d, A, h)에 대해 held-out block 전체로 학습한 head의 R과 실현 absolute error를 구한다.
  Predicted R의 decile별 predicted mean과 realized mean을 비교한다. 상위 decile의 비교를 high-predicted-risk mean calibration으로 기록한다.

<a id="recommendation"></a>

### 9.3 현재 full-stage 추천 명령

현재 full-stage `recommend` 명령은 등록된 H·B 결과로 배정을 추천한다. 비교 1로 original에서 forecaster의 개선을 진단하고,
최종 model·partition 후보는 champion 대비 W 변화율로 비교한다.
W 변화율이 음수인 후보를 개선 후보로 기록한다. 제출 pipeline 채택은 [comparison·adoption 명령](solarvpp-reference.md#adoption)으로 별도 수행한다.
기존 V·H·B 결과의 고정 기준선 비교와 forecast 후보의 P60 비교는 각 평가 조건 안에서 계산한다.

배정은 H·B 합산 ΔW(비교 3 − 비교 2)가 음수이면 risk로 선택한 partition을, 그 밖에는 stratified partition을 test 배정으로 추천한다.
추천한 test 배정은 그룹당 20,000 kW 이상이어야 한다.
이 partition 추천 결과와 같은 forecaster의 최종 forecast를 묶어 위 champion–challenger 비교에 사용한다.

추천에는 현재 등록 정의·component version과 일치하는 완료된 full-stage run을 사용한다. Smoke run은 제외한다.
같은 instance·MK에 해당하는 run이 여러 개이면 가장 늦게 완료된 run을 선택하고, 선택·제외한 run을 추천 기록에 남긴다.
선택한 H·B 산출물은 현재 evaluation 설정으로 평가하고, 해당 evaluation 경로를 추천 기록에 함께 남긴다.

<a id="p60-panel"></a>

### 9.4 Forecast 후보의 validation panel(P60-v1)

최종 test와 가까운 정보 상태에서 forecast 후보를 비교한다. test forecaster는 253개 발전소의 계절별 이력을 학습한 뒤 정답을 보지 않은
대상일을 예측한다. Panel도 평가일만 학습에서 빼고, 평가 대상 발전소의 다른 날짜·계절 이력은 평가일보다 앞인지 뒤인지와 관계없이
학습에 남긴다. 이 protocol은 평가일 앞뒤 자료를 사용해 최종 test의 발전소·계절별 학습 이력을 가깝게 유지한다.
예보 제공사 변경, 발전소 상태 변화, 휴일 출력제어 등은 발생 시기와 모델이 사용한 학습 자료를 함께 기록해 해석한다.

- 기준과 후보는 같은 panel·배정을 사용한다. P60-v1 배정은 253개 전체에 대한 `stratified-test.csv`다. 최초 기준 run의 설정은 [11절](#versions)에 둔다.
- 후보·seed마다 모델을 한 번 학습해 같은 60일을 예측한다. 월별로 다시 학습하지 않으며, 60일 전체가 하나의 고정 panel이다.
- 전체 자료로 학습한 checkpoint를 panel 평가에 재사용하지 않고, 기준 모델도 같은 분리로 다시 학습한다.
  같은 panel·seed·XGB 설정을 쓰는 후보끼리는 새로 학습한 XGB를 공유한다.

**평가일.** KST 대상일이고 양끝을 포함한다. 대상일 d의 출력은 d 01:00부터 다음 날 00:00까지다.

| 월 | 평가일(5일) | 학습 제외(평가일과 앞뒤 1일) |
|---|---|---|
| 2024-06 | 06-03~06-07 | 06-02~06-08 |
| 2024-07 | 07-22~07-26 | 07-21~07-27 |
| 2024-08 | 08-13~08-17 | 08-12~08-18 |
| 2024-09 | 09-07~09-11 | 09-06~09-12 |
| 2024-10 | 10-26~10-30 | 10-25~10-31 |
| 2024-11 | 11-05~11-09 | 11-04~11-10 |
| 2024-12 | 12-12~12-16 | 12-11~12-17 |
| 2025-01 | 01-23~01-27 | 01-22~01-28 |
| 2025-02 | 02-01~02-05 | 01-31~02-06 |
| 2025-03 | 03-22~03-26 | 03-21~03-27 |
| 2025-04 | 04-10~04-14 | 04-09~04-15 |
| 2025-05 | 05-10~05-14 | 05-09~05-15 |

날짜는 예보·달력·용량·관측 가능 여부로 고르고, 실제 발전량과 기존 모델의 오차·성적은 쓰지 않는다. 모든 월을 포함하고, 계절마다
예보 운량의 낮음·중간·높음 구간과 제공사 불일치가 큰 날을 넣으며, 추가로 빠지는 학습 이력이 적도록 둔다. 앞뒤 1일 완충은 바로 붙은
날짜의 label을 학습에서 피하면서 이력 손실을 제한한다.

**학습 날짜.**

```text
학습 범위 = 2021-01-19 ~ 2025-05-31 17:00 KST
R0        = risk용 held-out 날짜, heldout_mask(day, 7, 7, 3) (test forecaster도 뺀다)
E         = 평가일 60일 + 앞뒤 1일 완충 = 84일
학습 날짜 = 학습 범위 − (R0 ∪ E)
```

E 84일 중 50일이 R0와 겹치므로 test 학습보다 34일이 더 빠진다. XGB·DL 학습, 보조 target, target에서 추정하는 통계·power curve·calibration,
정답 기반 전처리 fit에 같은 학습 날짜를 쓴다. 예보 입력만 사용하는 전처리 fit은 E만 제외하고 R0의 입력은 포함한다. R0만 뺀 학습 대비 이력 보존율(표시 9~16시에 유효 실적이 있는 날의 용량 가중 비율)은 전체 93.5%,
봄 95.5%, 여름 89.9%, 가을 94.3%, 겨울 93.4%이고, 발전소별 최저는 전체 88.1%, 여름 75.7%다. 같은 계절 이력이 새로 없어지는 발전소는
없으며, test 학습에도 여름·가을 이력이 없는 용량 3.0%는 그대로다.

**날짜별 평가 구성원.** 원자료의 유효성으로 발전소마다 다음을 고정한다.

```text
entry_day[i] = 24시간 실적이 모두 유효한 첫 대상일 (학습용 label mask를 적용하기 전에 계산)
A[d]         = entry_day[i] ≤ d인 발전소 전체
G[g, d]      = 고정 배정의 그룹 g ∩ A[d]
```

- 평가일마다 A[d]를 적용하고, 월이나 block 시작 시점의 cohort로 고정하지 않는다. 새 발전소도 첫 완전한 관측일이 평가일이면 그날부터 넣는다.
- 유효한 실적이 0인 발전소도 구성원으로 둔다.
- 같은 날의 예측과 실적 합은 같은 G[g, d]로 계산한다. 결측 시각이 있으면 그날 유효 구성원 전체의 공통 시각 mask를 기준·후보·두 그룹에
  똑같이 적용한다. 고정 cohort용 평가기(`group_truth`)를 그대로 쓰지 않고, 날짜별 구성원으로 truth와 forecast를 맞춘다.
- 253개 발전소가 모두 한 번 이상 평가에 들어가고, 평균 평가 용량은 test 전체 용량의 96.2%, 결측으로 빠지는 시각은 0개다.
  2024-06·07 평가일 가운데 8일은 고정 배정의 용량 불균형 \|b\|가 운영 대역 β = 0.0623 밖이다(b ≈ −0.11).

**지표.** 60일의 실제 SMP·k를 쓴다.

\[
W_P = 0.1\sum_{d\in P,h,g} SMP_{d,h}\,k_h\,\Bigl|\hat Y_{g,d,h} - \sum_{i\in G_{g,d}} Y_{i,d,h}\Bigr|,\qquad
\text{W 변화율} = \frac{W_{P,\text{candidate}}}{W_{P,\text{baseline}}} - 1
\]

W 변화율이 음수이면 후보의 W가 작다. 비교마다 달별 5일 묶음 12개를 다시 뽑는 block bootstrap 95% 구간을 함께 기록한다.
95% 구간은 후보와 기준의 paired 손실 차이에서 계산하고, 해당 비교의 추정값과 함께 해석한다.

**실행.** Panel 정의는 runtime 설정의 `panels`에 두고, `instance --instance test --stage panel --panel p60-v1`로 실행한다.
Forecaster는 `excluded_days`(E)를 학습 query, 하루 상태·발전소 이용률 보조 target, 입력 전처리 fit에서 뺀다. Risk용 R0의 정답은 forecaster 학습에서 계속 제외한다.
입력 전처리 fit은 test forecaster와 같이 cutoff까지의 시각을 쓰되 E의 시각만 뺀다. Run은 `<MK>-panel-<panel>-<key>`이고,
`outputs/stratified`(평가일 시각만 담은 계약 형식, manifest의 `panel.days`), seed별·expert별 예측(`panel_forecasts.npz`),
일별 벌점(`daily_penalty.npz`)을 남긴다. 후보와 기준의 비교는 `vppcontract panel --candidate <run>/outputs/stratified --baseline …`이
공통 평가기의 날짜별 구성원 채점(`vppcontract.evaluate.panel_compare`)으로 계산한다.

## 10. Cache와 산출물

### 10.1 Node별 모듈과 cache key

각 cache key에는 상위 node의 key와 자신이 사용하는 component version을 담는다.
Parameter를 바꾼 실행은 기본으로 새 model·전처리·feature·risk cache를 실행 동안의 임시 폴더에만 쓰고, 이미 있는 cache는 읽는다.
저장 여부는 `instance.json`의 `cache_stored`에 적는다. 기준과 옵션은 [README의 저장 관리](../README.md#storage)를 따른다.

| node | 모듈 | component version |
|---|---|---|
| base, weather | `solarvpp2/data.py` | `base`, `weather` |
| sampler | `solarvpp2/sampler.py` | `sampler` |
| features | `solarvpp2/features.py` | `features` |
| forecaster | `solarvpp2/forecast/xgb.py`, `solarvpp2/forecast/dl.py`, `solarvpp2/forecast/decoders/`, `solarvpp2/forecast/contexts/`, `solarvpp2/forecast/routing/`, `solarvpp2/forecast/forecaster.py` | `xgb`, `dl_prep`, `dl`, `postprocess`, BiGRU가 아닌 decoder 모듈과 provider context·temporal routing 모듈의 `VERSION` |
| risk | `solarvpp2/risk.py` | `risk` |
| partition | `solarvpp2/partition.py` | `partition` |
| output | `solarvpp2/instance.py`, `vppcontract/outputs.py` | `instance` |
| evaluation | `solarvpp2/evaluation.py` | `EVALUATION_VERSION` |

### 10.2 Output과 보관

Forecaster·prediction의 계산 payload와 저장 helper는 `solarvpp2/store.py`에 둔다.
Hash는 `vppcontract.identity`를 사용한다. 같은 설정의 instance는 계산 key로 정한 output 경로를 재사용한다.
과거 run의 재평가는 `evaluate --instance-dir <instance 경로>`로 source를 지정한다.

`instance.json`은 실행 당시 모델 정의·계산 설정·초기 평가를 기록하고, `outputs/`는 배정·prediction을 보존한다.
`artifacts.json`에는 모델 추론에 필요한 resolved 설정과 최초 평가의 dependency를 보존한다.

- Inference: 입력 구성·excluded inputs·schema, DL 구조·quantiles·runtime, base·weather key, power curve.
- Model·prep·risk: `instance.json`에 기록된 key로 저장 weights와 전처리 state를 읽는다.
- 최초 후보 점수: 실제 배정의 hash에 점수를 연결하고, 계산 당시 truth key와 output hash를 함께 기록한다.

추가 후보 추론은 artifact에 보존한 설정으로 Spec을 복원한다. 평가 명령의 calendar·bootstrap·candidate pool은
평가 설정으로 사용한다. Source의 model 설정은 현재 runtime 설정과 독립적으로 적용한다.

| 변경 | 실행 |
|---|---|
| Calendar | 저장된 partition·prediction으로 계절·휴일 진단을 다시 계산한다 |
| Bootstrap | 저장된 예측의 daily W로 ΔW와 bootstrap 구간을 다시 계산한다 |
| Candidate pool 증가 | 기존 후보 점수를 재사용하고, 점수가 없는 배정만 저장된 forecaster·risk로 추론한다 |
| Candidate pool 감소·같은 설정 재요청 | 저장된 후보 점수 또는 완료 evaluation을 재사용한다 |

후보 점수 cache의 key는 source run, inference 설정, truth·output, 배정으로 정한다.
최초 평가의 점수는 truth·output dependency가 일치할 때 가져온다.
Calendar·bootstrap·pool 크기는 개별 후보 점수 key에 영향을 주지 않는다.

Evaluation은 `<artifact_dir>/evaluations/<key>/evaluation.json`에 기록한다.
Key는 source run, inference 설정, truth·output hash, 실제 평가 설정과 version으로 정한다.
같은 key의 완료 파일을 직접 읽는다. Instance 명령은 prediction 경로와 evaluation 경로를 함께 반환한다.
Risk calibration과 expert별 W는 최초 실행의 진단으로 `instance.json`에 보존한다.
`calibration_bins`는 최초 risk calibration에 적용하고 재평가 설정에서는 제외한다.

기간마다 배정 CSV(`site_id, group`), 그룹 예측 CSV(`group, ts, forecast_kwh`), manifest를 남긴다.
Manifest에는 package, MK, 모델 정의, node별 component version, cutoff, held-out block, seed, 실제 사용한 패키지 version과 GPU를 기록한다.

산출물은 repo 밖 root에 둔다. 경로는 `configs/solarvpp2/runtime.json`의 `paths.artifact_dir`로 정하며,
기본값은 `../sol-artifact/solarvpp2`다. Root에는 현재 등록 정의·parameter·component version으로 재생성할 수 있는 instance 기록을 둔다.
등록하지 않은 MK, 코드에서 제거한 parameter, 중단된 실행의 기록은 `python -m solarvpp2 artifacts --archive`로
root의 `archive/`에 옮긴다. 판정 규칙은 `vppcontract.artifacts`를 따르며, 추천 기록이 가리키는 instance는 유지한다.

Champion pointer는 `configs/champion.json`에 package와 모델 식별자를 함께 기록한다.
Run·artifact·component version은 어떤 계산으로 손실을 회수했는지 추적하고, 같은 계산과 평가를 재사용하도록 관리한다.

### 10.3 Expert별 probabilistic artifact 추출

`solarvpp2 probabilistic`은 정답이 있는 기간의 완료된 instance(forecast·full 단계)에서 저장된 설정·전처리·checkpoint와
partition을 복원한다. 고정 cohort·배정과 기존 seed 평균·후처리 순서로 XGB·DL 분위수 artifact를 추출한다.
Panel 단계는 날짜별 구성원을 사용하므로 대상에서 제외한다. Cache miss는 학습으로 대체하지 않고 오류로 반환한다.

`expert_quantiles=True`는 `Prediction`에 expert별 최종·원출력과 seed별 분위수를 추가한다. XGB는 야간 추론을 건너뛰므로
야간 원출력은 0이다. Model key와 기존 제출 계산은 유지한다.

출력 위치는 `reports/probabilistic/<원 run>/<배정>/`이다. `common.npz`, expert별 NPZ, `composition.csv`, `manifest.json`을
[확률예측 공통 contract](probabilistic-contract.md)에 맞춰 보존한다. 원 run의 expert W가 있으면 이를 대조하고, full 단계처럼
expert W가 없는 run은 blend median으로 원 예측을 대조한다. 추출 identity는 `solarvpp2.probabilistic.VERSION`과 원 artifact로 정한다.

채점·표·viewer 데이터 생성은 `vppcontract probabilistic <artifact-directory>`로 실행하고, 결과는 repo의 `viewer/` 데스크톱 viewer로 연다.

<a id="versions"></a>

## 11. 등록 구성과 실행 설정

등록 이름은 실행 구성을 찾는 식별자다. 현재 [모델 등록 목록](../configs/solarvpp2/models.json)의 구성은 다음과 같다.

| 구분 | 현재 기록 |
|---|---|
| CLI 기본 등록 구성 | `default = MK3`: XGB + DayState DL, ramp, 발전소 이용률 head 없음 |
| 발전소 이용률 head를 쓰는 등록 구성 | `MK5`: `dl.site_aux_ratio = 10` |
| 두 등록 구성의 기본 실행값 | Seed 42·43·44, XGB 비율 0.4, ramp `aux_ratio = 0.1`, time weight `none`, held-out 7주 중 1주 |
| P60-v1 최초 기준 run의 구성 | `MK5`에 `site_aux_ratio = 30`, XGB 비율 0.2를 적용. 비교할 seed와 prediction artifact는 해당 run에서 확인 |

`dl.site_aux_ratio`가 없거나 0이면 발전소 이용률 head를 만들지 않는다. 아래 component는 설정으로 선택하며,
구체적인 조합은 run의 모델 정의와 변경값에 기록한다.

| 기능 | 설정 예시 (명령행) | DL model key |
|---|---|---|
| Looped decoder | `dl.decoder.kind = shared-memory-looped`(`--dl-decoder`), T = 4는 `dl.decoder.loops`(`--dl-decoder-loops`)로 변경 | BiGRU가 아닐 때만 설정과 구현 version이 들어간다 |
| Provider context | `dl.provider_context.kind = inverted-attention`(`--dl-provider-context`). `dl.decoder`와 함께 설정할 수 있다 | 있을 때만 설정과 구현 version이 들어간다 |
| 하루 배율 head | `dl.day_scale.kind = bounded-log-gain`(`--dl-day-scale`), hidden 32(`--dl-day-scale-hidden`), `max_log_gain` 0.35(`--dl-day-scale-bound`). 추가 parameter 8,769개 | 활성화할 때만 설정과 구현 version이 들어간다. 없거나 `kind = none`이면 기존 key와 계산을 유지한다 |
| Time shift | `dl.plant_routing.kind = time-shift`(`--dl-plant-routing time-shift`) | 있을 때만 설정과 구현 version이 들어간다 |
| Time shift와 그룹 nearby hours | `dl.plant_routing.kind = time-shift`, `dl.group_routing.kind = nearby-hours`(`--dl-plant-routing time-shift --dl-group-routing nearby-hours`). 그룹 쪽 맥락은 `neutral`이다 | 항목마다 있을 때만 설정과 구현 version이 들어간다 |
| Nearby hours | `dl.plant_routing.kind = nearby-hours`(`--dl-plant-routing nearby-hours`), 앞뒤 3시간은 `dl.plant_routing.hours`(`--dl-plant-hours`)로 변경. Gate를 쓴다 | 있을 때만 설정과 구현 version이 들어간다 |
| 정산 보조손실 | `dl.loss = settlement-penalty`, `dl.aux_ratio = 0.3`(`--dl-loss`, `--dl-aux-ratio`). 새 head나 inference 입력은 없다 | Loss 정의와 `aux_ratio`가 들어간다 |

제출 경로가 없는 실험은 MK로 등록하지 않는다. `transfer`와 `recalibrate`는 MK5(site_aux_ratio=30)의 H checkpoint를 쓰는 실험이고,
설정은 runtime의 `experiments`에, 계약은 [transfer·calibration 설계](transfer-calibration.md)에 둔다.

등록 모델의 정의는 [모델 등록 목록](../configs/solarvpp2/models.json)에,
hyperparameter·입력 구성·실행 설정은 [runtime 설정](../configs/solarvpp2/runtime.json)에 둔다.
입력 구성, DL loss, blend, pooling, sampler, held-out 규칙, risk objective·입력·정규화·cross-fitting,
partition의 𝒟·ω·π·search·시작점, 평가 기준 배정·추천 규칙은 해당 node의 component와 parameter로 관리한다.

모델 번호·component version·package 구분은 [프로젝트 개발 규칙](../AGENTS.md)을 따른다.
계산에 영향을 주는 component 설정과 구현 version은 해당 node key에 넣고, 실행 당시 값을 artifact에 보존한다.
