"""
lightgbm_protocol_integration_audit.py
LightGBM(machine_learning/lightgbm/config.py, preprocessing.py, trainer.py,
pipeline/p10/lightgbm_calibration.py, pipeline/p13/lightgbm.py)이
model_protocol_contract.md / four_family_protocol_audit.py가 정의한 Frozen Common
Protocol을 실제 구현/실행 수준에서 만족하는지 확인한다. 기존 4-family protocol audit
결과(outputs/audits/four_family_protocol_audit.csv, common_evaluation_keys*.csv/parquet)는
읽기만 하고 절대 덮어쓰지 않는다 - 5-family 결과는 전부 새 파일명으로 저장한다.

모델 fit은 하지 않는다. LightGBM eligible key는 RF와 동일한 방식(tabular, lookback 없음 -
target_h{h} not-NaN 여부만)으로 학습 없이 계산한다. RF/LSTM/TFT/Informer/common/LightGBM
production 코드는 이 스크립트가 전혀 수정하지 않는다.
"""

import inspect
from pathlib import Path

import pandas as pd

from src.forecasting.common import config as cfg
from src.forecasting.common import evaluator as ev_ref
from src.forecasting.common import folds as folds_ref
from src.forecasting.common import oof as oof_ref
from src.forecasting.common.data_loader import load_development
import src.forecasting.pipeline.p10.lightgbm_calibration as lgbm_p10
import src.forecasting.pipeline.p13.lightgbm as lgbm_p13
import src.forecasting.machine_learning.lightgbm.preprocessing as lgbm_pp
import src.forecasting.machine_learning.lightgbm.trainer as lgbm_trainer
import src.forecasting.machine_learning.rf.preprocessing as rf_pp

OUTPUT_DIR = Path("outputs/audits")
EXISTING_COMMON_KEYS_PARQUET = OUTPUT_DIR / "common_evaluation_keys.parquet"  # 읽기 전용
PROTOCOL_CSV = OUTPUT_DIR / "lightgbm_protocol_integration_audit.csv"
SUMMARY_JSON = OUTPUT_DIR / "lightgbm_protocol_integration_summary.json"
FIVE_FAMILY_KEYS_PARQUET = OUTPUT_DIR / "five_family_common_evaluation_keys.parquet"
FIVE_FAMILY_KEYS_CSV = OUTPUT_DIR / "five_family_common_evaluation_keys.csv"
FIVE_FAMILY_SUMMARY_CSV = OUTPUT_DIR / "five_family_common_evaluation_keys_summary.csv"

HORIZONS = cfg.HORIZONS


# ---------------------------------------------------------------------------
# 1. LightGBM Frozen Common Protocol contract 18개 항목
# ---------------------------------------------------------------------------
def audit_lightgbm_protocol() -> list[dict]:
    rows = []

    def add(item_no, name, status, note=""):
        rows.append({"item": item_no, "name": name, "status": status, "note": note})

    trainer_src = Path(inspect.getfile(lgbm_trainer)).read_text(encoding="utf-8")
    pp_src = Path(inspect.getfile(lgbm_pp)).read_text(encoding="utf-8")

    # 1. RF와 동일한 horizon별 30개 Feature Set
    all_30 = all(len(cfg.get_model_feature_cols(h)) == 30 for h in HORIZONS)
    rf_uses_same_fn = "get_model_feature_cols" in inspect.getsource(rf_pp.RFPreprocessor.fit)
    lgbm_uses_same_fn = "get_model_feature_cols" in inspect.getsource(lgbm_pp.LGBMPreprocessor.fit)
    add(1, "horizon별 30개 Feature Set (RF와 동일 함수 재사용)",
        "PASS" if (all_30 and rf_uses_same_fn and lgbm_uses_same_fn) else "FAIL",
        f"all_30={all_30}, rf_uses_get_model_feature_cols={rf_uses_same_fn}, lgbm_uses_get_model_feature_cols={lgbm_uses_same_fn}")

    # 2. center_id/sku_id는 predictor 아님 (전 horizon)
    excl_ok = all(("center_id" not in cfg.get_model_feature_cols(h)) and ("sku_id" not in cfg.get_model_feature_cols(h)) for h in HORIZONS)
    add(2, "center_id/sku_id predictor 제외(전 horizon)", "PASS" if excl_ok else "FAIL")

    # 3. center_is_B predictor
    center_is_b_ok = all("center_is_B" in cfg.get_model_feature_cols(h) for h in HORIZONS)
    add(3, "center_is_B predictor 포함(전 horizon)", "PASS" if center_is_b_ok else "FAIL")

    # 4. KAN 3종 포함
    kan_ok = all(all(c in cfg.get_model_feature_cols(h) for c in cfg.CATEGORICAL_FEATURES) for h in HORIZONS)
    add(4, "KAN_대/중/소분류 포함(전 horizon)", "PASS" if kan_ok else "FAIL")

    # 5. KAN_CODE 제외
    kan_code_excl = all("KAN_CODE" not in cfg.get_model_feature_cols(h) for h in HORIZONS)
    add(5, "KAN_CODE 제외(전 horizon)", "PASS" if kan_code_excl else "FAIL")

    # 6. h1/h2/h4 direct forecasting (재귀 예측 없음)
    sig = inspect.signature(lgbm_trainer.train_and_evaluate_fold)
    no_recursive_arg = not any(
        "prev" in p.lower() or "recursive" in p.lower() or "prior_pred" in p.lower()
        for p in sig.parameters
    )
    single_target_per_call = "cfg.TARGET_COLS[horizon]" in trainer_src
    add(6, "h1/h2/h4 direct forecasting(horizon당 스칼라 target, 재귀 예측 없음)",
        "PASS" if (no_recursive_arg and single_target_per_call) else "FAIL")

    # 7. common.folds.generate_expanding_folds 재사용 (identity)
    fold_fn_match = (lgbm_p10.day3_folds.generate_expanding_folds is folds_ref.generate_expanding_folds) and \
        (lgbm_p13.day3_folds.generate_expanding_folds is folds_ref.generate_expanding_folds)
    add(7, "common.folds.generate_expanding_folds 동일 함수 재사용", "PASS" if fold_fn_match else "FAIL")

    # 8. target_date purge 동일 (7번 함수 identity에 종속)
    add(8, "target_date purge 동일(fold 함수 내부 로직, identity로 자동 보장)", "PASS" if fold_fn_match else "FAIL")

    # 9. train-only preprocessing
    train_only_pattern = ("fit_transform(train_df" in trainer_src) and (".transform(val_df)" in trainer_src)
    add(9, "preprocessing train-only(fit은 train_df만, val은 transform만)", "PASS" if train_only_pattern else "FAIL")

    # 10. raw target log1p
    log1p_pattern = "np.log1p(train_df[target_col]" in trainer_src
    add(10, "raw target_h{h}를 np.log1p로 학습", "PASS" if log1p_pattern else "FAIL")

    # 11. inverse_transform_prediction 사용
    inv_match = lgbm_trainer.ev.inverse_transform_prediction is ev_ref.inverse_transform_prediction
    add(11, "common.evaluator.inverse_transform_prediction(expm1+clip0) 사용", "PASS" if inv_match else "FAIL")

    # 12. common evaluator 동일 객체
    ev_match = (lgbm_trainer.ev is ev_ref) and (lgbm_trainer.ev.compute_metrics is ev_ref.compute_metrics)
    add(12, "common.evaluator 동일 모듈/함수 객체 사용", "PASS" if ev_match else "FAIL")

    # 13. common MASE scale 동일 함수
    mase_match = lgbm_trainer.ev.build_mase_scale is ev_ref.build_mase_scale
    add(13, "common.evaluator.build_mase_scale 동일 함수 사용", "PASS" if mase_match else "FAIL")

    # 14. common OOF builder 동일 객체
    oof_match = (lgbm_trainer.oo is oof_ref) and (lgbm_trainer.oo.build_oof_frame is oof_ref.build_oof_frame)
    add(14, "common.oof.build_oof_frame 동일 모듈/함수 객체 사용", "PASS" if oof_match else "FAIL")

    # 15. seed 명시적 전달
    seed_ok = "seed" in sig.parameters
    add(15, "trainer 함수가 seed를 명시적 파라미터로 받음", "PASS" if seed_ok else "FAIL")

    # 16. prediction/evaluation key schema 호환 (14번 identity에 의해 자동 보장)
    add(16, "OOF key schema가 기존 protocol과 호환(build_oof_frame 동일 함수 재사용으로 보장)",
        "PASS" if oof_match else "FAIL",
        "row/dedup/정렬 등 실행 결과 기반 검증은 이 audit이 직접 재수행하지 않음(영속 artifact 없음)")

    # 17. structural NaN native missing, coverage 임의로 안 채움
    no_median_impute = ("_fit_hierarchical_median" not in pp_src) and ("_apply_hierarchical_median" not in pp_src)
    add(17, "structural NaN 5종 native missing 유지(median impute 없음), coverage 임의 보충 없음",
        "PASS" if no_median_impute else "FAIL")

    # 18. categorical unknown native missing, row 임의 drop 없음
    transform_src = inspect.getsource(lgbm_pp.LGBMPreprocessor.transform)
    uses_categorical_assignment = "pd.Categorical(" in transform_src
    no_row_filter_calls = not any(
        pattern in transform_src for pattern in ("dropna(", "query(", ".loc[out[")
    )
    add(18, "categorical unknown -> native missing, validation row 임의 drop 없음",
        "PASS" if (uses_categorical_assignment and no_row_filter_calls) else "확인 필요",
        "소스 패턴만 확인(pd.Categorical 사용, dropna/query/loc 필터 없음). "
        "row-level 실측(drop 없음 등)은 이 audit이 직접 재수행하지 않음(영속 artifact 없음)")

    return rows


# ---------------------------------------------------------------------------
# 2. LightGBM eligible key 계산 (모델 fit 없이, target NaN 여부만)
# ---------------------------------------------------------------------------
def _key_set_from_df(df: pd.DataFrame, horizon: int, fold_id: int) -> pd.DataFrame:
    out = df[["center_id", "sku_id", "week_st"]].copy()
    out["horizon"] = horizon
    out["fold"] = fold_id
    out["target_date"] = out["week_st"] + pd.Timedelta(weeks=horizon)
    return out[["horizon", "fold", "center_id", "sku_id", "week_st", "target_date"]]


def build_lightgbm_eligible_keys(sub_a: pd.DataFrame, validation_year: int = 2022) -> tuple[pd.DataFrame, pd.DataFrame]:
    """RF와 동일한 기준(tabular, target_h{h} not-NaN)으로 LightGBM eligible validation
    key를 horizon x fold(기본값 P10/2022, 4-fold)별로 계산한다. 모델 fit 없음."""
    rows_summary = []
    all_keys = []

    for horizon in HORIZONS:
        target_col = cfg.TARGET_COLS[horizon]
        folds = folds_ref.generate_expanding_folds(sub_a, validation_year, horizon)
        for fold in folds:
            fold_id = fold["fold"]
            val_df = sub_a.loc[fold["val_mask"]]
            n_val = len(val_df)
            n_nan_target = int(val_df[target_col].isna().sum())
            eligible_df = val_df.loc[val_df[target_col].notna()]
            n_eligible = len(eligible_df)

            keys_df = _key_set_from_df(eligible_df, horizon, fold_id)
            all_keys.append(keys_df)

            rows_summary.append({
                "horizon": horizon, "fold": fold_id,
                "expected_val_rows": n_val,
                "lightgbm_eligible_rows": n_eligible,
                "target_nan_count": n_nan_target,
                "matches_expected": n_eligible == n_val,
            })

    eligible_keys = pd.concat(all_keys, ignore_index=True)
    summary_df = pd.DataFrame(rows_summary)
    return eligible_keys, summary_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 20)

    print("=" * 100)
    print("[1] LightGBM Frozen Common Protocol contract 18개 항목")
    print("=" * 100)
    protocol_rows = audit_lightgbm_protocol()
    protocol_df = pd.DataFrame(protocol_rows)
    print(protocol_df.to_string(index=False))
    n_fail = int((protocol_df["status"] == "FAIL").sum())
    n_review = int((protocol_df["status"] == "확인 필요").sum())
    protocol_df.to_csv(PROTOCOL_CSV, index=False)
    print(f"\nFAIL={n_fail}, 확인 필요={n_review}  (저장: {PROTOCOL_CSV})")

    print()
    print("=" * 100)
    print("[2] LightGBM eligible key 실측 (모델 fit 없음)")
    print("=" * 100)
    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()
    lgbm_eligible_keys, eligibility_summary = build_lightgbm_eligible_keys(sub_a, validation_year=2022)
    print(eligibility_summary.to_string(index=False))
    all_match = bool(eligibility_summary["matches_expected"].all())
    print(f"\n모든 horizon x fold에서 lightgbm_eligible_rows == expected_val_rows? {all_match}")

    h1f1 = eligibility_summary[(eligibility_summary["horizon"] == 1) & (eligibility_summary["fold"] == 1)].iloc[0]
    print(f"h1 Fold1: expected_val_rows={h1f1['expected_val_rows']}, lightgbm_eligible_rows={h1f1['lightgbm_eligible_rows']} "
          f"(참고: outputs/benchmarks/p12/lightgbm/lightgbm_cheap.json에 동일 P10/A센터/h1/Fold1 "
          f"validation_rows=117,607로 영속 기록됨)")

    dup_in_eligible = int(lgbm_eligible_keys.duplicated(subset=["horizon", "fold", "center_id", "sku_id", "week_st"]).sum())
    print(f"lightgbm_eligible_keys 자체 duplicate = {dup_in_eligible}")

    print()
    print("=" * 100)
    print("[3] 5-family common evaluation key: 기존 4-family key ∩ LightGBM eligible key")
    print("=" * 100)
    existing_4family = pd.read_parquet(EXISTING_COMMON_KEYS_PARQUET)
    print(f"기존 4-family common key row 수 = {len(existing_4family):,} (읽기 전용, 파일 미변경)")
    print(f"LightGBM eligible key row 수 = {len(lgbm_eligible_keys):,}")

    key_cols = ["horizon", "fold", "center_id", "sku_id", "week_st", "target_date"]
    five_family = existing_4family.merge(lgbm_eligible_keys[key_cols], on=key_cols, how="inner")

    dup_5family = int(five_family.duplicated(subset=key_cols).sum())
    n_decrease = len(existing_4family) - len(five_family)
    decrease_rate = n_decrease / len(existing_4family) * 100 if len(existing_4family) else float("nan")

    print(f"새 5-family common key row 수 = {len(five_family):,}")
    print(f"기존 4-family 대비 감소 row 수 = {n_decrease:,}")
    print(f"감소율 = {decrease_rate:.6f}%")
    print(f"5-family key duplicate = {dup_5family}")

    # horizon x fold별 intersection 정상 여부(12개 조합 전부 4-family count == 5-family count인지)
    combo_rows = []
    for horizon in HORIZONS:
        folds = folds_ref.generate_expanding_folds(sub_a, 2022, horizon)
        for fold in folds:
            fold_id = fold["fold"]
            n4 = int(((existing_4family["horizon"] == horizon) & (existing_4family["fold"] == fold_id)).sum())
            n5 = int(((five_family["horizon"] == horizon) & (five_family["fold"] == fold_id)).sum())
            combo_rows.append({
                "horizon": horizon, "fold": fold_id,
                "four_family_count": n4, "five_family_count": n5,
                "decrease": n4 - n5, "intersection_ok": n5 <= n4,
            })
    combo_df = pd.DataFrame(combo_rows)
    print()
    print(combo_df.to_string(index=False))
    combo_all_ok = bool(combo_df["intersection_ok"].all())
    print(f"\n12개 horizon x fold 조합 전부 intersection 정상(5<=4)? {combo_all_ok}")

    five_family.to_parquet(FIVE_FAMILY_KEYS_PARQUET, index=False)
    five_family.to_csv(FIVE_FAMILY_KEYS_CSV, index=False)
    combo_df.rename(columns={
        "four_family_count": "four_family_common_eval_origins",
        "five_family_count": "five_family_common_eval_origins",
    }).to_csv(FIVE_FAMILY_SUMMARY_CSV, index=False)
    print(f"\n저장: {FIVE_FAMILY_KEYS_PARQUET}, {FIVE_FAMILY_KEYS_CSV}, {FIVE_FAMILY_SUMMARY_CSV}")

    print()
    print("=" * 100)
    print("[4] 기존 4-family artifact 무결성 확인(읽기만 했는지)")
    print("=" * 100)
    reread = pd.read_parquet(EXISTING_COMMON_KEYS_PARQUET)
    unchanged = (len(reread) == len(existing_4family)) and reread.equals(existing_4family)
    print(f"기존 4-family artifact({EXISTING_COMMON_KEYS_PARQUET}) 재로드 후 동일? {unchanged}")

    # ---------------------------------------------------------------------
    # 최종 판정
    # ---------------------------------------------------------------------
    print()
    print("=" * 100)
    print("[최종 판정]")
    print("=" * 100)
    overall_pass = (
        n_fail == 0 and n_review == 0 and all_match and dup_in_eligible == 0 and dup_5family == 0
        and combo_all_ok and unchanged and n_decrease == 0
    )
    verdict = "LIGHTGBM INTEGRATION PASS" if overall_pass else "LIGHTGBM INTEGRATION FAIL"
    print(verdict)

    summary = {
        "protocol_n_fail": n_fail,
        "protocol_n_review": n_review,
        "lightgbm_eligible_all_match_expected": all_match,
        "lightgbm_eligible_duplicate": dup_in_eligible,
        "four_family_common_key_count": len(existing_4family),
        "lightgbm_eligible_key_count": len(lgbm_eligible_keys),
        "five_family_common_key_count": len(five_family),
        "decrease_row_count": n_decrease,
        "decrease_rate_pct": decrease_rate,
        "five_family_duplicate": dup_5family,
        "combo_intersection_all_ok": combo_all_ok,
        "existing_4family_artifact_unchanged": unchanged,
        "verdict": verdict,
    }
    import json
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n저장: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
