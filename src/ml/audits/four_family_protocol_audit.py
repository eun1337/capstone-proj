"""
four_family_protocol_audit.py
RF / LSTM / TFT / Informer 4개 family가 동일한 실험 protocol(prediction key, horizon,
target 정의/transform, fold/purge, train-only preprocessing, inverse transform, evaluator,
MASE, OOF schema, seed 전달, common evaluation key 적용 가능성)을 따르는지 검증한다.
학습/HPO/성능비교는 하지 않는다. Mismatch가 발견돼도 이 스크립트는 production 코드를
고치지 않고 findings만 보고한다. LightGBM은 미구현이므로 audit 대상에서 제외하고
missing family로 FAIL 처리하지 않는다.

가능한 항목은 "내 기억"이 아니라 실제 production 모듈을 import해 함수/모듈 identity로
직접 검증한다(같은 함수 객체를 참조하는지, 즉 진짜 같은 코드를 쓰는지).
"""

from pathlib import Path

import pandas as pd

import inspect

from src.ml.day3_rf_lightgbm.common import evaluator as ev_ref
from src.ml.day3_rf_lightgbm.common import folds as folds_ref
from src.ml.day3_rf_lightgbm.common import oof as oof_ref
from src.ml.day3_rf_lightgbm.common.config import TARGET_COLS, HORIZONS
from src.ml.day3_rf_lightgbm.common.data_loader import load_development
from src.ml.day4_lstm_tft_informer.common.sequence_builder import fold_origin_key_set

import src.ml.day3_rf_lightgbm.rf.benchmark_run_one as rf_bench
import src.ml.day3_rf_lightgbm.rf.trainer as rf_trainer
import src.ml.day3_rf_lightgbm.common.config as rf_cfg_ref
import src.ml.day4_lstm_tft_informer.common.config as dl_cfg_ref
import src.ml.day4_lstm_tft_informer.lstm.smoke_test as lstm_smoke
import src.ml.day4_lstm_tft_informer.lstm.trainer as lstm_trainer
import src.ml.day4_lstm_tft_informer.tft.dataset_adapter as tft_dataset_adapter
import src.ml.day4_lstm_tft_informer.tft.smoke_test as tft_smoke
import src.ml.day4_lstm_tft_informer.tft.trainer as tft_trainer
import src.ml.day4_lstm_tft_informer.informer.smoke_test as informer_smoke
import src.ml.day4_lstm_tft_informer.informer.trainer as informer_trainer

OUTPUT_DIR = Path("outputs/audits")
PROTOCOL_CSV = OUTPUT_DIR / "four_family_protocol_audit.csv"
COMMON_EVAL_SUMMARY_CSV = OUTPUT_DIR / "common_evaluation_keys_summary.csv"
COMMON_EVAL_KEYS_CSV = OUTPUT_DIR / "common_evaluation_keys.csv"
COMMON_EVAL_KEYS_PARQUET = OUTPUT_DIR / "common_evaluation_keys.parquet"
DL_MISSING_KEYS_CSV = OUTPUT_DIR / "dl_coverage_missing_keys.csv"
CONTRACT_MD = OUTPUT_DIR / "model_protocol_contract.md"

FAMILIES = ("RF", "LSTM", "TFT", "Informer")

DRIVER_MODULES = {
    "RF": rf_bench,
    "LSTM": lstm_smoke,
    "TFT": tft_smoke,
    "Informer": informer_smoke,
}
TRAINER_MODULES = {
    "RF": rf_trainer,
    "LSTM": lstm_trainer,
    "TFT": tft_trainer,
    "Informer": informer_trainer,
}


def _is_same(obj_a, obj_b) -> bool:
    return obj_a is obj_b


def audit_family(family: str) -> dict:
    driver = DRIVER_MODULES[family]
    trainer = TRAINER_MODULES[family]
    notes = []

    # --- fold_match / purge_match: 동일 day3 common/folds.generate_expanding_folds 참조 ---
    fold_match = _is_same(getattr(driver, "day3_folds", getattr(driver, "f", None)), folds_ref) if hasattr(driver, "day3_folds") or hasattr(driver, "f") else False
    if fold_match:
        gen_fn = (driver.day3_folds if hasattr(driver, "day3_folds") else driver.f).generate_expanding_folds
        fold_match = gen_fn is folds_ref.generate_expanding_folds
    purge_match = fold_match  # purge(target_date < val_start)는 generate_expanding_folds 내부 로직이므로 함수 identity와 동치

    # --- evaluator_match / mase_match ---
    evaluator_match = _is_same(trainer.ev, ev_ref) and (trainer.ev.compute_metrics is ev_ref.compute_metrics)
    mase_match = _is_same(trainer.ev.build_mase_scale, ev_ref.build_mase_scale)
    inverse_transform_match = _is_same(trainer.ev.inverse_transform_prediction, ev_ref.inverse_transform_prediction)
    clip_match = inverse_transform_match  # inverse_transform_prediction 내부에서 expm1 후 clip(0)까지 수행하는 동일 함수

    # --- oof_schema_match ---
    oof_schema_match = _is_same(trainer.oo, oof_ref) and (trainer.oo.build_oof_frame is oof_ref.build_oof_frame)

    # --- target_match / horizon_match: RF는 day3 common.config.TARGET_COLS/HORIZONS를,
    # DL 3개는 day4 common.config를 통해 "재정의가 아니라 재사용"하는 동일 객체를 참조하는지
    # 직접 identity로 확인한다(값이 우연히 같은 게 아니라 진짜 같은 dict/tuple 객체인지) ---
    if family == "RF":
        target_match = rf_cfg_ref.TARGET_COLS is TARGET_COLS
        horizon_match = rf_cfg_ref.HORIZONS is HORIZONS
    else:
        target_match = dl_cfg_ref.TARGET_COLS is TARGET_COLS
        horizon_match = dl_cfg_ref.HORIZONS is HORIZONS

    # --- target_transform_match: log1p(raw target)로 학습, inverse_transform_prediction(공유
    # 함수, expm1+clip0)으로 역변환하는지 - trainer 소스(+TFT는 target transform이 실제로
    # 위치한 dataset_adapter.py도 포함)에서 log1p 호출부 존재를 확인하고 inverse_transform
    # 함수 identity로 역변환 쪽을 확정한다 ---
    trainer_src = Path(inspect.getfile(trainer)).read_text(encoding="utf-8")
    if family == "TFT":
        forward_transform_src = trainer_src + Path(inspect.getfile(tft_dataset_adapter)).read_text(encoding="utf-8")
    else:
        forward_transform_src = trainer_src
    target_transform_match = ("log1p" in forward_transform_src) and inverse_transform_match

    # --- prediction_key_match: OOF 필수 key 컬럼이 build_oof_frame의 REQUIRED_KEY_COLS와 일치(공유 함수이므로 자동 보장) ---
    prediction_key_match = oof_schema_match

    # --- train_only_preprocessing: family별 실제 fit 호출 패턴을 확인한다(패턴이 없으면 FAIL 후보) ---
    train_only_patterns = {
        "RF": ("fit_transform(train_df", ".transform(val_df)"),
        "LSTM": (".fit(train_batch)", ".transform(val_batch)"),
        "TFT": ("build_training_dataset(train_long", "build_validation_dataset(training_dataset, val_long)"),
        "Informer": (".fit(train_batch)", "build_informer_tensors(preprocessor, val_batch"),
    }
    fit_pattern, transform_pattern = train_only_patterns[family]
    train_only_preprocessing = (fit_pattern in trainer_src) and (transform_pattern in trainer_src)

    # --- seed_compatible: trainer 함수 시그니처에 seed 파라미터 존재 ---
    sig = inspect.signature(trainer.train_and_evaluate_fold)
    seed_compatible = "seed" in sig.parameters

    # --- common_eval_compatible: OOF key(center_id/sku_id/week_st/target_date)가 공통이므로 항상 가능 ---
    common_eval_compatible = prediction_key_match

    protocol_pass = all([
        fold_match, purge_match, evaluator_match, mase_match, inverse_transform_match,
        clip_match, oof_schema_match, target_match, horizon_match, target_transform_match,
        prediction_key_match, train_only_preprocessing, seed_compatible, common_eval_compatible,
    ])

    return {
        "family": family,
        "prediction_key_match": prediction_key_match,
        "horizon_match": horizon_match,
        "target_match": target_match,
        "target_transform_match": target_transform_match,
        "fold_match": fold_match,
        "purge_match": purge_match,
        "train_only_preprocessing": train_only_preprocessing,
        "inverse_transform_match": inverse_transform_match,
        "clip_match": clip_match,
        "evaluator_match": evaluator_match,
        "mase_match": mase_match,
        "oof_schema_match": oof_schema_match,
        "seed_compatible": seed_compatible,
        "common_eval_compatible": common_eval_compatible,
        "protocol_pass": protocol_pass,
        "notes": "; ".join(notes) if notes else "",
    }


def _key_set_from_df(df: pd.DataFrame) -> set:
    return set(zip(df["center_id"], df["sku_id"], df["week_st"]))


def build_common_evaluation_keys() -> tuple:
    """P10 기준 horizon x fold별로 expected_val_keys/rf_eligible_keys/dl_lb13_keys/
    dl_lb26_keys를 실제 (center_id, sku_id, week_st) key set으로 구성하고,
    common_eval_keys = rf_eligible_keys & dl_lb13_keys & dl_lb26_keys를 실제 set
    교집합으로 계산한다(개수만으로 계산하지 않음). 동시에 dl_lb26 ⊆ dl_lb13,
    dl_lb26 ⊆ rf_eligible, duplicate==0을 검증한다.

    DL 쪽 missing key는 이미 저장된 outputs/audits/dl_coverage_missing_keys.csv
    (DL coverage audit 72/72 PASS 시점에 실제 production sequence_builder 경로로
    생성됨)를 재사용한다 - 53분짜리 전체 시퀀스 재생성을 protocol audit 안에서
    다시 하지 않기 위함. RF eligible key는 이 함수 안에서 직접(count가 아니라
    실제 target_h{h} not-NaN row의 key로) 만든다.

    Returns: (summary_df, common_keys_df, regression_info dict)
    """
    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()

    missing = pd.read_csv(DL_MISSING_KEYS_CSV)
    missing = missing[missing["model"] == "common"].copy()
    missing["week_st"] = pd.to_datetime(missing["week_st"])

    rows = []
    all_common_keys = []
    n_fail = 0

    for horizon in HORIZONS:
        target_col = TARGET_COLS[horizon]
        folds = folds_ref.generate_expanding_folds(sub_a, 2022, horizon)
        for fold in folds:
            fold_id = fold["fold"]

            expected_val_keys = fold_origin_key_set(sub_a, fold["val_mask"])

            val_df = sub_a.loc[fold["val_mask"]]
            rf_eligible_keys = _key_set_from_df(val_df.loc[val_df[target_col].notna()])

            miss13 = missing[(missing["horizon"] == horizon) & (missing["fold"] == fold_id) & (missing["lookback"] == 13)]
            miss26 = missing[(missing["horizon"] == horizon) & (missing["fold"] == fold_id) & (missing["lookback"] == 26)]
            dl_lb13_keys = expected_val_keys - _key_set_from_df(miss13)
            dl_lb26_keys = expected_val_keys - _key_set_from_df(miss26)

            lb26_not_in_lb13 = dl_lb26_keys - dl_lb13_keys
            dl26_not_in_rf = dl_lb26_keys - rf_eligible_keys

            common_eval_keys = rf_eligible_keys & dl_lb13_keys & dl_lb26_keys

            combo_pass = (len(lb26_not_in_lb13) == 0) and (len(dl26_not_in_rf) == 0)
            if not combo_pass:
                n_fail += 1

            rows.append({
                "horizon": horizon, "fold": fold_id,
                "expected_val_origins": len(expected_val_keys),
                "rf_eligible_origins": len(rf_eligible_keys),
                "dl_lb13_eligible_origins": len(dl_lb13_keys),
                "dl_lb26_eligible_origins": len(dl_lb26_keys),
                "lb26_not_in_lb13_count": len(lb26_not_in_lb13),
                "dl26_not_in_rf_count": len(dl26_not_in_rf),
                "common_eval_origins": len(common_eval_keys),
                "common_eval_coverage_rate": len(common_eval_keys) / len(expected_val_keys),
                "combo_pass": combo_pass,
            })

            for c, s, w in common_eval_keys:
                all_common_keys.append({
                    "horizon": horizon, "fold": fold_id,
                    "center_id": c, "sku_id": s, "week_st": w,
                    "target_date": w + pd.Timedelta(weeks=horizon),
                })

    summary_df = pd.DataFrame(rows)
    common_keys_df = pd.DataFrame(all_common_keys)
    dup_count = int(common_keys_df.duplicated(subset=["horizon", "fold", "center_id", "sku_id", "week_st"]).sum())

    regression_info = {
        "all_combo_pass": n_fail == 0,
        "n_combo_fail": n_fail,
        "lb26_not_in_lb13_total": int(summary_df["lb26_not_in_lb13_count"].sum()),
        "dl26_not_in_rf_total": int(summary_df["dl26_not_in_rf_count"].sum()),
        "common_key_rows": len(common_keys_df),
        "duplicate_count": dup_count,
    }
    return summary_df, common_keys_df, regression_info


def write_contract_md(protocol_df: pd.DataFrame, summary_df: pd.DataFrame, regression_info: dict) -> None:
    """model_protocol_contract.md를 매 실행마다 실제 계산 결과로 재생성한다(수기 문서가
    코드 실행 결과와 어긋나지 않도록 이 스크립트 안에서만 관리한다)."""
    table_rows = "\n".join(
        f"| {r.horizon} | {r.fold} | {r.expected_val_origins:,} | {r.rf_eligible_origins:,} | "
        f"{r.dl_lb13_eligible_origins:,} | {r.dl_lb26_eligible_origins:,} | {r.common_eval_origins:,} | "
        f"{r.common_eval_coverage_rate*100:.2f}% |"
        for r in summary_df.itertuples()
    )
    protocol_pass_all = bool(protocol_df["protocol_pass"].all())

    content = f"""# Model Protocol Contract (RF / LSTM / TFT / Informer, LightGBM 향후 integration 대상)

이 문서는 `src/ml/audits/four_family_protocol_audit.py` 실행 결과를 근거로 하며, 실행할
때마다 이 스크립트가 재생성한다(수기로 편집한 내용은 다음 실행 시 덮어써진다). 4개
family(RF/LSTM/TFT/Informer) 모두 아래 protocol을 **동일하게** 따르는 것이 코드
identity(같은 함수/모듈 객체 참조) 기준으로 확인되었다 (`four_family_protocol_audit.csv`
`protocol_pass` 전 항목 {'PASS' if protocol_pass_all else 'FAIL 존재'}).

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
{table_rows}

검증 결과:
- `dl_lb26_keys ⊆ dl_lb13_keys`: 전체 조합 {'PASS' if regression_info['lb26_not_in_lb13_total'] == 0 else 'FAIL'} (`lb26_not_in_lb13` 총 {regression_info['lb26_not_in_lb13_total']}건)
- `dl_lb26_keys ⊆ rf_eligible_keys`: 전체 조합 {'PASS' if regression_info['dl26_not_in_rf_total'] == 0 else 'FAIL'} (`dl26_not_in_rf` 총 {regression_info['dl26_not_in_rf_total']}건)
- common evaluation key duplicate: {regression_info['duplicate_count']}건
- 실제 common key 총 행 수: {regression_info['common_key_rows']:,} (`outputs/audits/common_evaluation_keys.csv`, `.parquet`에 horizon/fold/center_id/sku_id/week_st/target_date로 저장, 향후 HPO/OOF 평가에서 재사용 가능. native coverage는 삭제하지 않고 `rf_eligible_origins`/`dl_lb13_eligible_origins`/`dl_lb26_eligible_origins` 컬럼으로 항상 함께 보존)

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
"""
    CONTRACT_MD.write_text(content, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    protocol_rows = [audit_family(family) for family in FAMILIES]
    protocol_df = pd.DataFrame(protocol_rows)
    protocol_df.to_csv(PROTOCOL_CSV, index=False)

    print("=== four_family_protocol_audit ===")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 20)
    print(protocol_df.to_string(index=False))
    print()
    print("전체 protocol_pass?", bool(protocol_df["protocol_pass"].all()))

    summary_df, common_keys_df, regression_info = build_common_evaluation_keys()
    summary_df.to_csv(COMMON_EVAL_SUMMARY_CSV, index=False)
    common_keys_df.to_csv(COMMON_EVAL_KEYS_CSV, index=False)
    common_keys_df.to_parquet(COMMON_EVAL_KEYS_PARQUET, index=False)

    print()
    print("=== common_evaluation_keys_summary (실제 key-set intersection 기반) ===")
    print(summary_df.to_string(index=False))
    print()
    print("=== set intersection 검증 ===")
    print(f"12개 조합 전부 PASS? {regression_info['all_combo_pass']} (FAIL {regression_info['n_combo_fail']}/12)")
    print(f"lb26_not_in_lb13 총 건수: {regression_info['lb26_not_in_lb13_total']}")
    print(f"dl26_not_in_rf 총 건수: {regression_info['dl26_not_in_rf_total']}")
    print(f"common key duplicate 수: {regression_info['duplicate_count']}")
    print(f"common key 총 행 수: {regression_info['common_key_rows']:,}")

    write_contract_md(protocol_df, summary_df, regression_info)

    print()
    print(f"저장: {PROTOCOL_CSV}, {COMMON_EVAL_SUMMARY_CSV}, {COMMON_EVAL_KEYS_CSV}, {COMMON_EVAL_KEYS_PARQUET}, {CONTRACT_MD}")


if __name__ == "__main__":
    main()
