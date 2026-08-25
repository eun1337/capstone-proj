"""
p13_five_family_common_evaluation_keys.py

P13(2023) 전용 5-family(RF/LSTM/TFT/Informer/LightGBM) common evaluation key builder.

기존 P10(2022) eligibility 규칙(four_family_protocol_audit.build_common_evaluation_keys,
lightgbm_protocol_integration_audit.build_lightgbm_eligible_keys)을 코드 그대로 재사용하고
validation_year만 2023으로 전달한다. DL(LSTM/TFT/Informer) sequence coverage missing key는
기존 outputs/audits/dl_coverage_missing_keys.csv가 P10(2022) 전용(week_st가 전부 2022)이라
재사용할 수 없으므로, dl_coverage_audit.compute_common_dl_missing_keys(validation_year=2023)로
새로 계산한다(모델 학습 아님 - structural NaN fit/apply + sequence_builder만 사용).

기존 outputs/audits/five_family_common_evaluation_keys.parquet(P10)은 읽지도 쓰지도 않으며,
산출물은 전부 outputs/audits/p13/ 아래 별도 경로에 저장한다.
"""

from pathlib import Path

import pandas as pd

from src.forecasting.audits.dl_coverage_audit import compute_common_dl_missing_keys
from src.forecasting.audits.four_family_protocol_audit import build_common_evaluation_keys
from src.forecasting.audits.lightgbm_protocol_integration_audit import build_lightgbm_eligible_keys
from src.forecasting.common import folds as folds_ref
from src.forecasting.common.config import HORIZONS
from src.forecasting.common.data_loader import load_development
from src.forecasting.pipeline.p13 import hpo_common as hc

VALIDATION_YEAR = 2023
OUTPUT_DIR = Path("outputs/audits/p13")
DL_MISSING_KEYS_PATH = OUTPUT_DIR / "dl_coverage_missing_keys.csv"
FOUR_FAMILY_KEYS_PARQUET = OUTPUT_DIR / "common_evaluation_keys.parquet"
FOUR_FAMILY_SUMMARY_CSV = OUTPUT_DIR / "common_evaluation_keys_summary.csv"
FIVE_FAMILY_KEYS_PARQUET = OUTPUT_DIR / "five_family_common_evaluation_keys.parquet"
FIVE_FAMILY_KEYS_CSV = OUTPUT_DIR / "five_family_common_evaluation_keys.csv"
FIVE_FAMILY_SUMMARY_CSV = OUTPUT_DIR / "five_family_common_evaluation_keys_summary.csv"

KEY_COLS = ["horizon", "fold", "center_id", "sku_id", "week_st", "target_date"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 20)

    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()

    print("=" * 100)
    print(f"[1/4] DL(LSTM/TFT/Informer) sequence coverage missing key 계산 (validation_year={VALIDATION_YEAR})")
    print("=" * 100)
    missing_df = compute_common_dl_missing_keys(sub_a, VALIDATION_YEAR)
    missing_df.to_csv(DL_MISSING_KEYS_PATH, index=False)
    print(f"missing rows: {len(missing_df):,} -> {DL_MISSING_KEYS_PATH}")

    print()
    print("=" * 100)
    print("[2/4] 4-family(RF/LSTM/TFT/Informer) common evaluation key 계산")
    print("=" * 100)
    summary_df, four_family_keys, regression_info = build_common_evaluation_keys(
        validation_year=VALIDATION_YEAR, missing_keys_path=DL_MISSING_KEYS_PATH,
    )
    print(summary_df.to_string(index=False))
    print(f"\n4-family duplicate: {regression_info['duplicate_count']}, "
          f"4-family common key rows: {regression_info['common_key_rows']:,}, "
          f"combo_all_pass: {regression_info['all_combo_pass']}")
    if regression_info["duplicate_count"] != 0:
        raise ValueError(f"4-family common key duplicate 발견: {regression_info['duplicate_count']}건")
    if not regression_info["all_combo_pass"]:
        raise ValueError(f"4-family combo 검증 실패: n_combo_fail={regression_info['n_combo_fail']}")
    four_family_keys.to_parquet(FOUR_FAMILY_KEYS_PARQUET, index=False)
    summary_df.to_csv(FOUR_FAMILY_SUMMARY_CSV, index=False)

    print()
    print("=" * 100)
    print("[3/4] LightGBM eligible key 계산")
    print("=" * 100)
    lgbm_eligible_keys, lgbm_summary = build_lightgbm_eligible_keys(sub_a, validation_year=VALIDATION_YEAR)
    print(lgbm_summary.to_string(index=False))
    dup_lgbm = int(lgbm_eligible_keys.duplicated(subset=["horizon", "fold", "center_id", "sku_id", "week_st"]).sum())
    print(f"\nLightGBM eligible key duplicate: {dup_lgbm}, rows: {len(lgbm_eligible_keys):,}")
    if dup_lgbm != 0:
        raise ValueError(f"LightGBM eligible key duplicate 발견: {dup_lgbm}건")

    print()
    print("=" * 100)
    print("[4/4] 5-family intersection = 4-family common key ∩ LightGBM eligible key")
    print("=" * 100)
    print(f"4-family common key row 수 = {len(four_family_keys):,}")
    print(f"LightGBM eligible key row 수 = {len(lgbm_eligible_keys):,}")

    five_family = four_family_keys.merge(lgbm_eligible_keys[KEY_COLS], on=KEY_COLS, how="inner")

    dup_5family = int(five_family.duplicated(subset=KEY_COLS).sum())
    if dup_5family != 0:
        raise ValueError(f"5-family common key duplicate 발견: {dup_5family}건")
    if len(five_family) == 0:
        raise ValueError("5-family common key 교집합이 0건임")
    if len(five_family) > min(len(four_family_keys), len(lgbm_eligible_keys)):
        raise ValueError(
            f"merge 결과 row 수가 비정상적으로 증가함(row inflation 의심): "
            f"merged={len(five_family)}, four_family={len(four_family_keys)}, lgbm={len(lgbm_eligible_keys)}"
        )

    print(f"5-family common key row 수 = {len(five_family):,}")
    print(f"5-family common key duplicate = {dup_5family}")

    combo_rows = []
    for horizon in HORIZONS:
        folds = folds_ref.generate_expanding_folds(sub_a, VALIDATION_YEAR, horizon)
        for fold in folds:
            fold_id = fold["fold"]
            n4 = int(((four_family_keys["horizon"] == horizon) & (four_family_keys["fold"] == fold_id)).sum())
            nlg = int(((lgbm_eligible_keys["horizon"] == horizon) & (lgbm_eligible_keys["fold"] == fold_id)).sum())
            n5 = int(((five_family["horizon"] == horizon) & (five_family["fold"] == fold_id)).sum())
            combo_rows.append({
                "horizon": horizon, "fold": fold_id,
                "four_family_common_eval_origins": n4, "lightgbm_eligible_origins": nlg,
                "five_family_common_eval_origins": n5,
                "intersection_ok": (n5 <= n4) and (n5 <= nlg),
            })
    combo_df = pd.DataFrame(combo_rows)
    print()
    print(combo_df.to_string(index=False))
    combo_all_ok = bool(combo_df["intersection_ok"].all())
    print(f"\n{len(combo_rows)}개 horizon x fold 조합 전부 intersection 정상(5<=4, 5<=lgbm)? {combo_all_ok}")
    if not combo_all_ok:
        raise ValueError("5-family intersection이 input보다 커진 조합이 있음")

    for horizon in HORIZONS:
        n = int((five_family["horizon"] == horizon).sum())
        print(f"horizon={horizon}: row 수 = {n:,}")
        if n == 0:
            raise ValueError(f"horizon={horizon}: 5-family common key row가 0건임")

    five_family.to_parquet(FIVE_FAMILY_KEYS_PARQUET, index=False)
    five_family.to_csv(FIVE_FAMILY_KEYS_CSV, index=False)
    combo_df.to_csv(FIVE_FAMILY_SUMMARY_CSV, index=False)
    print(f"\n저장: {FIVE_FAMILY_KEYS_PARQUET}, {FIVE_FAMILY_KEYS_CSV}, {FIVE_FAMILY_SUMMARY_CSV}")

    print()
    print("=" * 100)
    print("[검증] pipeline.p13.hpo_common.check_common_keys_match_p13_period() (h1/h2/h4)")
    print("=" * 100)
    loaded = hc.load_common_eval_keys(FIVE_FAMILY_KEYS_PARQUET)
    for horizon in HORIZONS:
        folds = folds_ref.generate_expanding_folds(sub_a, VALIDATION_YEAR, horizon)
        hc.check_common_keys_match_p13_period(loaded, folds, horizon, FIVE_FAMILY_KEYS_PARQUET)
        print(f"h{horizon}: PASS")


if __name__ == "__main__":
    main()
