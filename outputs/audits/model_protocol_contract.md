# Model Protocol Contract (RF / LSTM / TFT / Informer, LightGBM 향후 integration 대상)

이 문서는 `src/ml/audits/four_family_protocol_audit.py` 실행 결과를 근거로 하며, 실행할
때마다 이 스크립트가 재생성한다(수기로 편집한 내용은 다음 실행 시 덮어써진다). 4개
family(RF/LSTM/TFT/Informer) 모두 아래 protocol을 **동일하게** 따르는 것이 코드
identity(같은 함수/모듈 객체 참조) 기준으로 확인되었다 (`four_family_protocol_audit.csv`
`protocol_pass` 전 항목 PASS).

## 1. 공통으로 확인된 protocol

| 항목 | 공유 방식 | 근거 |
|---|---|---|
| Prediction key | `(center_id, sku_id, week_st, target_date)` | 4개 family 모두 `day3_rf_lightgbm.common.oof.build_oof_frame`을 **동일 함수 객체**로 호출 |
| Horizon | h1/h2/h4, 각각 독립 direct model (재귀 예측 없음) | `HORIZONS=(1,2,4)` 객체 identity 동일(day4 common이 day3 config를 재정의 없이 재사용), 각 trainer가 horizon 1개당 스칼라 target 1개만 다룸 |
| Target 정의 | raw `target_h1/h2/h4` | `TARGET_COLS` 객체 identity 동일 |
| Target transform | `log1p(raw target)` 학습 → `pred_log` → `expm1` → `clip(lower=0)` | 정방향은 각 family 소스에 `log1p` 존재(TFT는 `tft/dataset_adapter.py`에서 수행), 역방향은 4개 family 전부 `common.evaluator.inverse_transform_prediction`을 **동일 함수 객체**로 호출 |
| Fold 정의 (P10/P13) | A센터, Q1~Q4 expanding 4-fold, target_date 기준 purge | 4개 family 모두 `common.folds.generate_expanding_folds`를 **동일 함수 객체**로 호출 |
| target_date purge | `train sample의 target_date < validation_start` | fold 함수 자체가 `train_mask = target_date < val_start`로 계산 - 함수 identity 동일이므로 자동 보장 |
| Train-only preprocessing | 각 family fit()은 train에서만, validation은 transform만 | 소스 패턴 확인 - RF(`fit_transform(train_df` → `.transform(val_df)`), LSTM/Informer(`SequencePreprocessor().fit(train_batch)` → 별도 transform), TFT(`build_training_dataset(train_long...)` → `build_validation_dataset(training_dataset, val_long)`, `TimeSeriesDataSet.from_dataset()`으로 train-fit encoder/scaler 재사용) |
| Evaluator | WAPE/Bias/RMSE/MAE, `sum(y_true)==0`이면 NaN(epsilon 사용 안 함) | 4개 family 전부 `common.evaluator.compute_metrics` **동일 함수 객체** |
| MASE | (center_id, sku_id) 단위 train-only lag-1 naive scale, invalid면 해당 row만 NaN | 4개 family 전부 `common.evaluator.build_mase_scale` **동일 함수 객체** |
| OOF schema | `stage/model_family/config_id/seed/horizon/fold_id/center_id/sku_id/week_st/target_date/y_true/y_pred_log/y_pred/mase_scale` | 4개 family 전부 `common.oof.build_oof_frame`/`validate_oof_frame` **동일 함수 객체**, prediction↔key 1:1 |
| Seed 전달 | 각 family trainer가 `seed`를 명시적 파라미터로 받음 | `inspect.signature(train_and_evaluate_fold)`에 `seed` 파라미터 존재 확인 |

## 2. Family별 정상적인 차이 (protocol mismatch 아님)

- **Feature 표현 방식**: RF는 30개 flat tabular feature(KAN OHE + 27 numeric, `DEMAND_SUMMARY_FEATURES` 3종 포함), DL 3종은 21개 time-varying sequence(known 7 + observed 14) + static 6. DL은 `qty_lag1/rollmean4/rollstd4_filled_log1p`(RF 전용)를 사용하지 않는다 - 의도된 설계.
- **Fold 분할 시점**: RF는 호출부에서 `train_df = dev.loc[fold["train_mask"]]`로 미리 나눈 뒤 trainer에 넘기고, LSTM/TFT/Informer trainer는 `df`(fold 전체 history) + `fold` dict를 받아 내부에서 시퀀스를 만든 뒤 origin 단위로 train/val을 나눈다. DL은 lookback window가 fold 경계를 가로지르므로 구조적으로 필요한 차이이며 purge 의미는 동일하다.
- **Structural NaN 구현**: RF(`rf/preprocessing.py`)와 DL(`common/structural_nan.py`)은 **서로 다른 코드**로 각각 구현돼 있으나, 방법론(KAN 소→중→대→train 전체 median, train-only fit, 기존 non-NaN 값 불변)은 동일하다. RF는 5개 feature(adi/cv2 + demand-summary 3종), DL은 2개 feature(adi/cv2)만 대상으로 한다.
- **TFT의 target transform 위치**: TFT만 `log1p` 적용이 `trainer.py`가 아니라 `tft/dataset_adapter.py`(long dataframe 생성 시점)에서 이루어진다. 최종 동작은 동일.

## 3. 저위험 notes (수정 후보이나 protocol_pass에는 영향 없음)

1. **RF trainer에 NaN target 명시적 가드 없음** - 이번 실측(§4)에서 12개 horizon×fold 전부 `rf_eligible_origins == expected_val_origins`로 target NaN 0건 확인되어 현재 데이터 범위에서 실제 문제 아님.
2. **RF 전용 persistent smoke test 파일 없음** - LSTM/TFT/Informer는 `smoke_test.py`가 있으나 RF는 `benchmark_run_one.py`(compute feasibility 목적)만 있음.

## 4. P10 Common Evaluation Key (실제 key-set intersection, 매 실행마다 재계산)

`common_eval_keys = rf_eligible_keys & dl_lb13_keys & dl_lb26_keys`를 `(center_id, sku_id,
week_st)` 실제 set 연산으로 계산한다(개수 기반 계산 아님). DL 쪽 missing key는
`outputs/audits/dl_coverage_missing_keys.csv`(DL coverage audit 72/72 PASS 시점에 실제
production sequence_builder 경로로 생성됨)를 재사용하고, RF eligible key는 이 스크립트가
직접 만든다.

| horizon | fold | expected_val | rf_eligible | dl_lb13_eligible | dl_lb26_eligible | common_eval | coverage |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 117,607 | 117,607 | 109,832 | 100,763 | 100,763 | 85.68% |
| 1 | 2 | 125,459 | 125,459 | 118,211 | 109,832 | 109,832 | 87.54% |
| 1 | 3 | 132,501 | 132,501 | 126,014 | 118,211 | 118,211 | 89.22% |
| 1 | 4 | 140,625 | 140,625 | 132,987 | 126,014 | 126,014 | 89.61% |
| 2 | 1 | 117,013 | 117,013 | 109,132 | 100,067 | 100,067 | 85.52% |
| 2 | 2 | 124,888 | 124,888 | 117,607 | 109,132 | 109,132 | 87.38% |
| 2 | 3 | 132,010 | 132,010 | 125,459 | 117,607 | 117,607 | 89.09% |
| 2 | 4 | 139,823 | 139,823 | 132,501 | 125,459 | 125,459 | 89.73% |
| 4 | 1 | 115,832 | 115,832 | 107,721 | 98,655 | 98,655 | 85.17% |
| 4 | 2 | 123,722 | 123,722 | 116,442 | 107,721 | 107,721 | 87.07% |
| 4 | 3 | 130,982 | 130,982 | 124,304 | 116,442 | 116,442 | 88.90% |
| 4 | 4 | 138,278 | 138,278 | 131,502 | 124,304 | 124,304 | 89.89% |

검증 결과:
- `dl_lb26_keys ⊆ dl_lb13_keys`: 전체 조합 PASS (`lb26_not_in_lb13` 총 0건)
- `dl_lb26_keys ⊆ rf_eligible_keys`: 전체 조합 PASS (`dl26_not_in_rf` 총 0건)
- common evaluation key duplicate: 0건
- 실제 common key 총 행 수: 1,354,207 (`outputs/audits/common_evaluation_keys.csv`, `.parquet`에 horizon/fold/center_id/sku_id/week_st/target_date로 저장, 향후 HPO/OOF 평가에서 재사용 가능. native coverage는 삭제하지 않고 `rf_eligible_origins`/`dl_lb13_eligible_origins`/`dl_lb26_eligible_origins` 컬럼으로 항상 함께 보존)

## 5. LightGBM Integration Contract (향후 구현 시 반드시 만족)

LightGBM이 구현되면 아래 12개 항목을 만족해야 하며, 만족 여부는
`src.ml.day3_rf_lightgbm.common` 모듈들과의 **identity 비교**로 재검증할 수 있다:

1. Prediction key `(center_id, sku_id, week_st, target_date)` 동일
2. h1/h2/h4 raw target(`target_h1/h2/h4`) 동일, 재귀 예측 금지
3. `common.folds.generate_expanding_folds`를 **동일 함수**로 재사용(P10 A센터 2022 / P13 A센터 2023, Q1~Q4 expanding)
4. target_date purge를 fold 함수 자체에 의존(별도 날짜 재계산 금지)
5. Train-only preprocessing(structural NaN/encoder/scaler 전부 train에서만 fit)
6. `log1p(raw target)`으로 학습
7. `common.evaluator.inverse_transform_prediction`(expm1 + lower clip 0)을 **동일 함수**로 재사용
8. `common.evaluator.compute_metrics`를 **동일 함수**로 재사용(WAPE/Bias/RMSE/MAE, `sum(y_true)==0`이면 NaN)
9. `common.evaluator.build_mase_scale`을 **동일 함수**로 재사용(train-only lag-1 naive scale)
10. `common.oof.build_oof_frame`/`validate_oof_frame`을 **동일 함수**로 재사용
11. trainer 함수가 `seed`를 명시적 파라미터로 받음
12. 이 스크립트의 `build_common_evaluation_keys()` 방식으로 LightGBM eligible key를 계산해 기존 4-family와 실제 key-set intersection이 가능해야 함(원본 row를 임의로 채우거나 native coverage를 숨기지 않음)

LightGBM이 들어오면 4-family audit을 처음부터 다시 하지 않고, 위 12개 항목만 검증하는
"LightGBM vs Frozen Common Protocol" integration audit만 추가로 수행한다.
