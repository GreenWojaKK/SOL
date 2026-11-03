# 확률예측 artifact와 공통 평가

각 모델 package는 checkpoint·전처리·고정 배정으로 분위수를 추출하고, 공통 artifact를 한 폴더에 저장한다.
`vppcontract probabilistic`은 이 폴더만 읽어 검증·채점·viewer 데이터를 생성하고, repo의 `viewer/` 데스크톱 viewer가 그 결과를 연다.

## Artifact 계약 v1

| 파일 | 내용 |
|---|---|
| `manifest.json` | `contract_version=1`, `instance`, `experts`(ID → `label`), 선택적인 `artifact_hashes`와 모델·cutoff·cohort provenance |
| `common.npz` | `hours`, `dates`, `ts`, `groups`, `alphas`, `truth`, `complete`, `daylight`, `common_daylight`, `capacity_kw`, `seeds`, `smp`, `k` |
| `<expert>_quantiles.npz` | `quantiles`, `raw`: `[시각, 그룹, 분위수]`; `seed_quantiles`, `seed_raw`: `[seed, 시각, 그룹, 분위수]` |

`groups`는 G1·G2 순서다. `hours`는 공통 시간축의 오름차순 index이고 각 대상일은 24시각을 모두 포함한다.
`dates`는 대상일 `YYYY-MM-DD`, `ts`는 공통 시간축의 KST timestamp 문자열이다. 날짜가 떨어져 있는 artifact도 같은 규칙으로 읽는다.
실적·분위수는 kWh, 용량은 kW, SMP는 원/kWh다. `alphas`는 오름차순이며 0.05·0.1·0.25·0.5·0.75·0.9·0.95를 포함한다.

`complete[시각]`는 고정 cohort 전체의 실적 유효 mask다. `daylight[시각, 그룹]`는 태양 위치 기반 주간 mask이고,
`common_daylight`는 두 그룹 주간 mask의 교집합이다. 유효한 실적 0을 포함한다.
`quantiles`는 clipping·야간 처리·분위수 정렬 후 출력이고, `raw`는 동일 kWh 척도의 후처리 전 출력이다.
Seed 평균 후 후처리 순서는 추출 package의 제출 규칙을 따른다.

Expert ID·표시 이름은 `manifest.experts`로 전달한다. 이 항목이 없는 초기 v1 artifact는 `xgb`·`dl` 파일을 읽는다.
`source_instance`와 checkpoint 경로는 provenance로 보존한다. 공통 평가는 저장 artifact로 계산한다.

선택적인 `cloud`, `disagreement`는 `[시각, 그룹]` 예보 조건이다. 이 배열을 제공할 때 `manifest.regime_edges`에
학습 예보로 정한 오름차순 경계를 함께 둔다. 동일값은 같은 구간에 배정한다.

공통 loader는 필수 항목, 시간축·그룹·shape·분위수 순서, 유효 실적, boolean mask, 양의 용량, 최종 분위수의 단조성과 물리 범위를 검사한다.
`artifact_hashes`가 있으면 폴더 내부 파일의 hash도 대조한다.

## 실행과 결과

```powershell
& $py -m vppcontract probabilistic <quantile-artifact-directory>
& $py -m vppcontract probabilistic <quantile-artifact-directory> --output <report-directory>
& $py -m vppcontract probabilistic <quantile-artifact-directory> --open
& $py -m vppcontract view <report-directory>
```

결과 기본 위치는 `<artifact>/report/`이다. 실행할 때마다 저장 artifact로 결과를 재계산한다.

| 결과 | 내용 |
|---|---|
| `seasonal_summary.csv` | 전체·계절·그룹·expert·seed별 편향, MAE, pinball, W, crossing·clipping |
| `intervals.csv` | 50·80·90% coverage·폭·상하단 이탈·경계값 일치 |
| `reliability.csv` | 분위수별 `P(Y<q)`·`P(Y≤q)`와 pinball |
| `hourly_bias.csv`, `weather_regimes.csv` | 계절×시간대 편향과 제공된 예보 조건별 진단 |
| `diagnostics.json` | 위 표 전체와 평가 version |
| `explorer-data.json` | viewer 데이터: manifest, expert 목록, 계절, 날짜·그룹별 24시각 실적·분위수(MWh), seed 평균 표 |
| `report.json` | 입력 파일 hash, 평가 version, 실제 포함 계절, viewer 데이터 경로, 결과 파일 hash |

편향은 `q_0.5−Y`로 정의하며 양수는 과대예측이다. 확률 지표는 `complete & common_daylight` 시각에 대해 계산한다.
W는 전체 `complete` 시각을 사용한다. 구간 폭은 kWh와 `그룹 용량×1h`로 정규화한 값을 기록한다.
Coverage는 양 끝을 포함하고 하단 미만·상단 초과·경계값 일치 비율을 함께 제공한다. Pinball은 지정 분위수별 값과 그 평균을 기록한다.
Viewer의 계절 선택지와 계절별 차트는 유효 표본이 있는 계절만 표시한다.

## Viewer

결과 폴더는 repo의 `viewer/`에 있는 Rust 데스크톱 앱으로 연다. Viewer는 `explorer-data.json`만 읽고 값을 다시 계산하지 않는다.
웹 서버와 브라우저를 쓰지 않는다.

```powershell
cargo build --release --manifest-path viewer/Cargo.toml
viewer\target\release\probabilistic-viewer.exe <report-directory>
```

`probabilistic --open`은 결과를 만든 뒤 viewer를 열고, `view <report-directory>`는 이미 만든 결과 폴더를 연다.
Viewer는 별도 프로세스로 실행하며 CLI는 기다리지 않는다. 실행 파일은 `--viewer`, `VPP_PROBABILISTIC_VIEWER`,
`viewer/target/release/probabilistic-viewer.exe` 순서로 찾고, 없으면 빌드 방법을 안내한다. 경로 없이 실행한 viewer는
결과 폴더 열기(Ctrl+O)로 폴더를 고른다.
화면은 Windows의 앱 색상 설정(밝게·어둡게)을 따르고, 오른쪽 위 버튼으로 밝은 화면과 어두운 화면을 바꾼다.

Viewer는 다음을 표시한다.

| 구역 | 내용 |
|---|---|
| 날짜·그룹별 예측구간 | expert별 24시각 50·80·90% 구간과 중앙값·실제 발전량. 날짜 이동·입력, 그룹 선택, 마우스 위치의 시각별 값 |
| 계절별 비교표 | 계절·그룹으로 고른 중앙값·분위수 성능과 구간 적중률·폭 |
| 계절과 시간대 | 계절별 중앙값 편향·pinball, 구간 수준별 coverage·폭, 분위수 신뢰도, 계절 × 시간대 편향 히트맵 |
| 예보 조건별 비교 | `weather_regimes`가 있을 때 운량·제공사 불일치 구간별 표 |

화면 부품은 TPUI 저장소의 `viewer-ui` crate를 쓰며, 그 crate와 GPUI는 `viewer/Cargo.toml`에서 커밋으로 고정한다.
`experts`가 없는 이전 형식의 `explorer-data.json`은 열지 않고, 결과를 다시 만들도록 안내한다.

추출 package의 model identity와 공통 진단 `VERSION`은 각각 관리한다.
