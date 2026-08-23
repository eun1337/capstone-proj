# SARIMA / SARIMAX(S4) SKU 트랙 최종 결론

관련 스크립트: `08_sku_sarima_sarimax.py` / `09_diagnose_sarima_instability.py` / `10_evaluate_sarima.py` / `11_compare_arima_vs_sarima.py`
(전부 `src/ml/day2_statistical_models/`)

참조 파일:
- `data/ml/day2_statistical_models/arima/arima_holdout_2024.parquet` — ARIMA(S0) holdout 예측
- `data/holdout_2024.parquet` — actual lookup 원본(raw qty)
- `data/development_2021_2023.parquet` — MASE scale(lag-1 naive MAE) 계산 원본

---

## 1. SARIMAX(S4) 실행·평가 범위

`08_sku_sarima_sarimax.py`는 SKU마다 SARIMA와 SARIMAX(S4)를 함께 fit(학습/계수 추정)하며,
`forecast_results.parquet`에 두 forecast_source가 각각 2,847,505행씩 존재.

S4는 exog 조합 명명 체계(S0=없음, S1=Economic 3종, S2=COVID 1종, S3=Holiday 3종,
S4=S1+S2+S3 전부 7종, `04_prepare_arimax_full_exog.py` 정의)에서 "Operational-Full"(exog 전부)에 해당한다.

이전 버전에서는 `09_diagnose_sarima_instability.py`/`10_evaluate_sarima.py`/ `11_compare_arima_vs_sarima.py` 세 스크립트 모두 SARIMA와 SARIMAX(S4)를 독립적으로 각각 진단·평가·비교하도록 확장했다.

---

## 2. Override 처리 요약 (model별 독립 판정)

`09_diagnose_sarima_instability.py`가 SARIMA와 SARIMAX(S4)에 각각 동일한 규칙
(literal Inf 또는 `abs(prediction_unclipped) > dev_max_qty*100`)을 독립 적용한 결과:

| model | override_scope | SKU 수 | 행 수 |
|---|---|---|---|
| SARIMA | sku_level | 2 | 310 |
| SARIMA | row_level | 33 | 518 |
| **SARIMA 합계** | | **35** | **828** |
| SARIMAX_S4 | sku_level | 16 | 2,480 |
| SARIMAX_S4 | row_level | 264 | 7,318 |
| **SARIMAX_S4 합계** | | **280** | **9,798** |
| **전체 합계** | | **315** | **10,626** |

SARIMA sku_level 대상 SKU:

| center_id | sku_id |
|---|---|
| A | 18801007235582*BX*1 |
| A | 8803322002439*EA*1 |

flagged SKU의 center_id 분포:

| model | A | B | 합계 |
|---|---|---|---|
| SARIMA | 14 | 21 | 35 |
| SARIMAX_S4 | 101 | 179 | 280 |

development_status=='selected' 전체 모집단(A=11,825 / B=6,546) 대비 flag 비율:

| model | A flag율 | B flag율 |
|---|---|---|
| SARIMA | 0.118% (14/11,825) | 0.321% (21/6,546) |
| SARIMAX_S4 | 0.854% (101/11,825) | 2.735% (179/6,546) |

### 재현 검증 (독립 refit + rolling append 재실행, SARIMA만 수행)

SARIMA 35개 SKU 중 14개를 개별 refit으로 재현 검증했으며 전부 `reproduced_matches_stored=True` (각 SKU 155/155행 완전 일치)
SARIMAX_S4 280개 SKU는 개별 refit 재현 검증을 수행하지 않음(동일 결정론적 탐지 규칙만 적용).

| 구분 | SKU 수 | 결과 |
|---|---|---|
| SARIMA sku_level 2개(수동 최초 발견) | 2 | 155/155 일치 (2/2) |
| SARIMA row_level 중 절대threshold 단계 발견 7개 | 7 | 155/155 일치 (7/7) |
| SARIMA row_level 확장(상대 threshold) 후 무작위 5개 | 5 | 155/155 일치 (5/5) |
| **SARIMA 개별 재현 검증 합계** | **14** | **14/14 (100%)** |

### override_prediction(naive_mean) 값

개별 값이 확인된 SARIMA sku_level 2건:

| center_id | sku_id | override_prediction |
|---|---|---|
| A | 18801007235582*BX*1 | 34.337078651685395 |
| A | 8803322002439*EA*1 | 7.051282051282051 |

나머지(SARIMA row_level 33개, SARIMAX_S4 296개)의 override_prediction 값은 동일 함수(development 기간 실측 qty 평균)로 계산되어 `sarima_status_override.csv`에 저장되어 있음(개별 값은 이 문서에 나열하지 않음).

---

## 3. actual lookup 버그 수정 검증

`08_sku_sarima_sarimax.py`의 actual 계산부(`target_h{h}` 컬럼을 target_week 시점 행에서 재조회하던 부분)를 raw qty 직접 lookup 방식으로 진행

### 수정 전/후 대조

| 지표 | 수정 전 | 수정 후 |
|---|---|---|
| `actual_mismatch_count` (11 audit, SARIMA 대조) | **0** |
| `actual_mismatch_count` (11 audit, SARIMAX_S4 대조) | **0** |
| `actual_censored_excluded` (10 audit, SARIMA 단독 평가) | **0** |
| `actual` 값이 바뀐 행수 (forecast_results.parquet 전체 6,250,223행 기준) | 1,906,797 |

### 샘플 대조 (2건, SARIMA 기준)

| center_id | sku_id | forecast_origin | horizon | target_week | actual | qty(target_week) |
|---|---|---|---|---|---|---|
| A | 8801063311063*EA*1 | 2024-07-01 | 1 | 2024-07-08 | 0.0 | 0.0 |
| A | 18807949200329*BX*1 | 2024-02-26 | 4 | 2024-03-25 | 1.0 | 1.0 |

수정 후 `actual`은 두 샘플 모두 `holdout_2024.parquet`의 `qty(target_week)` 값과 정확히 일치한다.

---

## 4. 최종 ARIMA vs SARIMA vs SARIMAX(S4) 비교 (ALL센터 × h1/h2/h4)

`11_compare_arima_vs_sarima.py` 재실행 결과(actual 버그 수정 후, common key inner join 3,391,697행/모델, Main만:  as_observed_history=True 기준). A/B 센터별 세부값은 `arima_vs_sarima_compare_metrics.csv`에 저장.

| horizon | model | n | WAPE | Bias | RMSE | MAE | MASE | 개선폭(%p, ARIMA 대비) |
|---|---|---|---|---|---|---|---|---|
| h1 | ARIMA | 1,054,753 | 2.2453 | 1.1979 | 3066.1742 | 6.8921 | 4.6824 | — |
| h1 | SARIMA | 1,054,753 | 1.0118 | -0.3255 | 39.4094 | 3.1057 | 1.1214 | +123.35 |
| h1 | SARIMAX_S4 | 1,054,753 | 3.0425 | 1.7482 | 263.5336 | 9.3391 | 2.2572 | -79.72 |
| h2 | ARIMA | 1,034,852 | 1.1156 | 0.0461 | 374.2813 | 3.4330 | 1.7310 | — |
| h2 | SARIMA | 1,034,852 | 0.9574 | -0.4078 | 38.2046 | 2.9462 | 1.0805 | +15.82 |
| h2 | SARIMAX_S4 | 1,034,852 | 2.3783 | 1.0581 | 213.1943 | 7.3187 | 2.0346 | -126.27 |
| h4 | ARIMA | 995,050 | 0.8987 | -0.1838 | 31.8443 | 2.7743 | 1.0667 | — |
| h4 | SARIMA | 995,050 | 1.0104 | -0.3657 | 38.4679 | 3.1192 | 1.1186 | -11.17 |
| h4 | SARIMAX_S4 | 995,050 | 2.4861 | 1.1622 | 237.1889 | 7.6748 | 2.0411 | -158.74 |

**h1/h2는 SARIMA가 ARIMA 대비 개선(ALL WAPE 개선폭 각각 +123.35%p, +15.82%p), 
h4는 SARIMA가 ARIMA 대비 소폭 열위(-11.17%p). SARIMAX(S4)는 h1/h2/h4 전 구간에서 ARIMA 대비 열위(-79.72%p, -126.27%p, -158.74%p)이며, 
SARIMA와 비교해도 전 구간에서 WAPE가 더 높다(h1: 3.0425 vs 1.0118, h2: 2.3783 vs 0.9574, h4: 2.4861 vs 1.0104).**

---

## 5. SARIMAX(S4) 열위 원인 — 데이터 측면 해석
(수치를 근거 해석)

**(1) 불안정 SKU 비율 자체가 SARIMA보다 훨씬 높다.** 
flag된 SKU 수가 SARIMA 35개 대비 SARIMAX_S4 280개로 8배다. 
두 model 다 동일한 `enforce_stationarity=False/enforce_invertibility=False` 설정으로 fit되고 동일한 `refit=False` 재귀 append 구조를 쓰는데, 
SARIMAX(S4)는 여기에 exog 7종의 회귀계수가 추가로 함께 추정된다 — 추정해야 할 파라미터 수가 늘어난 상태로 같은 재귀 필터 구조를 거치므로, 이미 SARIMA에서 확인된 "near-unit-root 계수의 재귀 발산" 경로가 SARIMAX에서 더 자주 발생하는 것으로 관찰된다.

**(2) B센터에서 불안정 비율이 A센터보다 뚜렷하게 높고, 그 격차가 SARIMAX에서 더 커진다.**
모집단 대비 flag 비율이 SARIMA는 A 0.118%/B 0.321%(B가 A의 2.7배)인데, SARIMAX_S4는 A 0.854%/B 2.735%(B가 A의 3.2배)로 격차가 더 벌어진다. 
B센터는 post-regime 관측 이력이 최대 26주로 짧다(원칙 문구 참조) 
— exog 7종의 회귀계수까지 이 짧은 이력 안에서 함께 추정해야 하므로, 관측치 대비 추정할 파라미터 수의 비율이 A보다 불리해지는 구간에서 불안정이 더 많이 나타나는 것과 일치한다.

**(3) override(악성 행 제거) 이후에도 SARIMAX_S4의 "정상" 행 자체가 SARIMA보다 나쁘다.**
override 미적용 행만 놓고 봐도(10 실행 결과, ALL-Main 기준) SARIMA WAPE=0.9925인 반면 SARIMAX_S4 WAPE=2.6465로, 극단치를 걷어낸 뒤에도 3배 가까이 차이난다. 
즉, 이 열위는 소수의 발산 행이 평균을 끌어올린 결과만이 아니라, exog를 반영한 예측 분포 전반이 SARIMA보다 넓게 벗어나 있다는 뜻이다.

**(4) 이 패턴은 이 SKU 트랙에서 처음 관찰된 것이 아니다.** 
ARIMAX 트랙(`04_prepare_arimax_full_exog.py`가 정의하는 S0~S4 블록 비교, 
S4 정의는 이 SKU 트랙의 EXOG_COLS와 동일한 Economic+COVID+Holiday 7종)에도 `07_diagnose_arimax_economic_instability.py`라는 전용 사후진단 스크립트가 이미 존재하며, 그 docstring은 "Economic 변수를 포함한 S1/S4의 극단 예측 원인을 사후진단한다"고 명시한다. 
즉, S4(또는 Economic exog를 포함하는 블록)를 SARIMAX/ARIMAX 필터에 넣었을 때 예측이 극단으로 튀는 현상은, 이 SKU 단위 트랙과는 별개로 진행된 기존 ARIMAX 블록 트랙에서도 독립적으로 관찰되어 이미 전용 진단 스크립트가 만들어져 있었다는 사실과 일치한다.

---

## 6. 최종 산출물 파일 지도

### `data/ml/day2_statistical_models/sarima_sarimax_sku/`

| 파일 | 생성 스크립트 | 내용 |
|---|---|---|
| `forecast_results.parquet` | 08 (actual 컬럼만 이후 패치) | SKU×origin×horizon×model(sarima/sarimax_s4/constant/naive_mean) 예측/실측 원본(6,250,223행) |
| `seasonal_order_selection.csv` | 08 | SKU별 선택된 계절 order 로그 |
| `a_periodogram_diagnosis.csv` | 08 | A센터 periodogram 진단(정보용) |
| `sarima_instability_diagnosis.csv` | 09 | SARIMA+SARIMAX_S4 flag된 315개(model,SKU) 조합의 전체 행 + 병리 진단 요약 컬럼(model 컬럼 포함) |
| `sarima_status_override.csv` | 09 | override 대상 10,626행(model/override_scope/override_prediction 포함) |
| `sarima_eval_audit.csv` | 10 | model별 평가 감사(key_dup/actual_censored_excluded/override 매칭 등) |
| `sarima_eval_coverage.csv` | 10 | model별 단계별 표본 규모(전체/censoring 제외/center/horizon/main-coldstart) |
| `sarima_eval_common_panel.parquet` | 10 | model별 평가에 실제 사용된 전체 행(override 반영 최종 prediction 포함) |
| `sarima_eval_metrics.csv` | 10 | model × A/B/ALL × h1/h2/h4 WAPE/Bias/RMSE/MAE/MASE |
| `sarima_eval_source_breakdown.csv` | 10 | model × forecast_source_final별(sarima/sarimax_s4/constant/naive_mean/override) 성능 분리 |
| `sarima_eval_coldstart_aux.csv` | 10 | model별 has_observed_history=False 보조 결과 |
| `sarima_actual_censoring_summary.csv` | (세션 중 수동 진단, 08 actual 버그 수정 이전 산출) | actual 버그 수정 전 censoring 패턴 기록 — 버그 수정 후 값 기준으로는 더 이상 유효하지 않음 |

### `data/ml/day2_statistical_models/`

| 파일 | 생성 스크립트 | 내용 |
|---|---|---|
| `arima_vs_sarima_compare_audit.csv` | 11 | ARIMA↔SARIMA/SARIMAX_S4 공통 key 감사(model별, inner join 손실/actual 불일치 등) |
| `arima_vs_sarima_compare_metrics.csv` | 11 | horizon × center × model(ARIMA/SARIMA/SARIMAX_S4) × compared_against + WAPE 개선폭 |

### 입력(참조, 이 트랙에서 생성하지 않음)

| 파일 | 비고 |
|---|---|
| `data/ml/day2_statistical_models/arima/arima_holdout_2024.parquet` | ARIMA(S0) holdout 예측 — `02_run_arima_holdout.py` 산출물 |
| `data/holdout_2024.parquet` | actual lookup 원본(raw qty) |
| `data/development_2021_2023.parquet` | MASE scale(lag-1 naive MAE) 계산 원본 |
