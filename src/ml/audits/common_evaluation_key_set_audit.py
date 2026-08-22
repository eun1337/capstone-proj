"""
[LEGACY / DEPRECATED - 더 이상 실행할 필요 없음]
common_evaluation_key_set_audit.py

이 스크립트의 실제 key-set intersection 로직은 src/ml/audits/four_family_protocol_audit.py
의 build_common_evaluation_keys()로 완전히 통합되었다(2026-08-22). four_family_protocol_audit.py
한 번 실행으로 four_family_protocol_audit.csv, common_evaluation_keys_summary.csv,
common_evaluation_keys.csv/.parquet, model_protocol_contract.md가 전부 생성되며, 통합 후
regression check(12/12 PASS, lb26_not_in_lb13=0, dl26_not_in_rf=0, common key=1,354,207 rows,
duplicate=0)로 이 스크립트의 결과와 완전히 동일함이 확인되었다. 앞으로는
four_family_protocol_audit.py만 실행하면 되고, 이 파일은 과거 임시 재검증 스크립트로
참고용으로만 남긴다(삭제하지 않음).

--- 아래는 원래 목적(당시 four_family_protocol_audit.py의 count 기반
common_eval_origins = min(rf_eligible_count, dl_lb26_count)를 실제
(center_id, sku_id, week_st) key set 교집합으로 재확정하기 위해 작성됨) ---
"""

from pathlib import Path

import pandas as pd

from src.ml.day3_rf_lightgbm.common import folds as folds_ref
from src.ml.day3_rf_lightgbm.common.config import HORIZONS, TARGET_COLS
from src.ml.day3_rf_lightgbm.common.data_loader import load_development
from src.ml.day4_lstm_tft_informer.common.sequence_builder import fold_origin_key_set

OUTPUT_DIR = Path("outputs/audits")
MISSING_KEYS_CSV = OUTPUT_DIR / "dl_coverage_missing_keys.csv"
OLD_SUMMARY_CSV = OUTPUT_DIR / "common_evaluation_keys_summary.csv"
COMMON_EVAL_KEYS_CSV = OUTPUT_DIR / "common_evaluation_keys.csv"
COMMON_EVAL_KEYS_PARQUET = OUTPUT_DIR / "common_evaluation_keys.parquet"


def _key_set_from_df(df: pd.DataFrame) -> set:
    return set(zip(df["center_id"], df["sku_id"], df["week_st"]))


def main() -> None:
    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()

    missing = pd.read_csv(MISSING_KEYS_CSV)
    missing = missing[missing["model"] == "common"].copy()
    missing["week_st"] = pd.to_datetime(missing["week_st"])

    old_summary = pd.read_csv(OLD_SUMMARY_CSV)

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
            missing_lb13_keys = _key_set_from_df(miss13)
            missing_lb26_keys = _key_set_from_df(miss26)

            dl_lb13_keys = expected_val_keys - missing_lb13_keys
            dl_lb26_keys = expected_val_keys - missing_lb26_keys

            # --- 2. lb26 ⊆ lb13 실제 set 검증 ---
            lb26_not_in_lb13 = dl_lb26_keys - dl_lb13_keys
            # --- 3. lb26 ⊆ RF eligible 실제 set 검증 ---
            dl26_not_in_rf = dl_lb26_keys - rf_eligible_keys

            # --- 4. 실제 교집합으로 common_eval_keys 계산 ---
            common_eval_keys = rf_eligible_keys & dl_lb13_keys & dl_lb26_keys
            common_eval_origins_set_based = len(common_eval_keys)

            # --- 5. 기존 count 기반 값과 비교 ---
            old_row = old_summary[(old_summary["horizon"] == horizon) & (old_summary["fold"] == fold_id)]
            common_eval_origins_count_based = int(old_row["common_eval_origins"].iloc[0])
            count_vs_set_diff = common_eval_origins_count_based - common_eval_origins_set_based

            combo_pass = (len(lb26_not_in_lb13) == 0) and (len(dl26_not_in_rf) == 0) and (count_vs_set_diff == 0)
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
                "common_eval_origins_count_based": common_eval_origins_count_based,
                "common_eval_origins_set_based": common_eval_origins_set_based,
                "count_vs_set_diff": count_vs_set_diff,
                "common_eval_coverage_rate": common_eval_origins_set_based / len(expected_val_keys),
                "combo_pass": combo_pass,
            })

            for c, s, w in common_eval_keys:
                all_common_keys.append({
                    "horizon": horizon, "fold": fold_id,
                    "center_id": c, "sku_id": s, "week_st": w,
                    "target_date": w + pd.Timedelta(weeks=horizon),
                })

    result_df = pd.DataFrame(rows)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 20)
    print(result_df.to_string(index=False))

    common_keys_df = pd.DataFrame(all_common_keys)
    dup_count = int(common_keys_df.duplicated(subset=["horizon", "fold", "center_id", "sku_id", "week_st"]).sum())

    common_keys_df.to_csv(COMMON_EVAL_KEYS_CSV, index=False)
    try:
        common_keys_df.to_parquet(COMMON_EVAL_KEYS_PARQUET, index=False)
        parquet_saved = True
    except Exception as e:
        parquet_saved = False
        print(f"parquet 저장 실패(csv는 저장됨): {e}")

    result_df.to_csv(OLD_SUMMARY_CSV, index=False)

    print()
    print("=" * 100)
    print(f"A. 12개 horizon x fold 실제 set intersection 검증: {'PASS' if n_fail == 0 else 'FAIL'} (FAIL {n_fail}/12)")
    print(f"B. lb26_not_in_lb13 총 건수: {int(result_df['lb26_not_in_lb13_count'].sum())}")
    print(f"C. dl26_not_in_rf 총 건수: {int(result_df['dl26_not_in_rf_count'].sum())}")
    print(f"D. count 방식 vs set-intersection 방식 차이(절대값 합): {int(result_df['count_vs_set_diff'].abs().sum())}")
    print(f"E. common evaluation key duplicate 수: {dup_count}")
    print(f"공통 key 총 행 수: {len(common_keys_df)} (parquet 저장 {'성공' if parquet_saved else '실패, csv만 저장'})")
    print(f"저장: {COMMON_EVAL_KEYS_CSV}" + (f", {COMMON_EVAL_KEYS_PARQUET}" if parquet_saved else ""))
    print(f"갱신: {OLD_SUMMARY_CSV} (set-intersection 기반으로 갱신)")


if __name__ == "__main__":
    main()
