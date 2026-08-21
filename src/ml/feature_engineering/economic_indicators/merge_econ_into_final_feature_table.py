"""
merge_econ_into_final_feature_table.py

build_economic_features.py가 만든 feature_table_with_econ.parquet의 경제지표 3개
(ccsi_lag_m1, cpi_y1_prev, cpi_y2_prev_yoy)를 data/final_feature_table.parquet에
재병합한다. add_rollstd_log_features.py가 final_feature_table.parquet을 in-place로
갱신하면서 이 경제지표 재계산분을 반영한 적이 없어(레포에 그 병합 단계가 없었음),
final_feature_table.parquet에는 CPI 원본 재정제 이전의 stale 값(A센터 2021-01-04~
01-25 cpi_y2_prev_yoy NaN 4주)이 그대로 남아 있었다.

(center_id, sku_id, week_st) 기준 1:1 병합이며, 이 3개 컬럼 외 다른 컬럼은 절대
건드리지 않는다. 덮어쓰기 전 원본을 백업해 되돌릴 수 있게 한다.
"""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[4]
ECON_PATH = BASE_DIR / "data" / "ml" / "experiments" / "economic_indicators" / "feature_table_with_econ.parquet"
FINAL_PATH = BASE_DIR / "data" / "final_feature_table.parquet"
BACKUP_PATH = BASE_DIR / "data" / "final_feature_table.parquet.bak_pre_econ_merge"

KEY_COLS = ["center_id", "sku_id", "week_st"]
ECON_COLS = ["ccsi_lag_m1", "cpi_y1_prev", "cpi_y2_prev_yoy"]


def main() -> None:
    econ = pd.read_parquet(ECON_PATH, columns=KEY_COLS + ECON_COLS)
    final_before = pd.read_parquet(FINAL_PATH)

    dup_econ = int(econ.duplicated(KEY_COLS).sum())
    dup_final = int(final_before.duplicated(KEY_COLS).sum())
    assert dup_econ == 0, f"feature_table_with_econ.parquet key 중복 {dup_econ}건"
    assert dup_final == 0, f"final_feature_table.parquet key 중복 {dup_final}건"

    econ_keys = set(map(tuple, econ[KEY_COLS].to_numpy()))
    final_keys = set(map(tuple, final_before[KEY_COLS].to_numpy()))
    only_econ = econ_keys - final_keys
    only_final = final_keys - econ_keys
    assert not only_econ, f"final에 없는 econ key {len(only_econ)}건"
    assert not only_final, f"econ에 없는 final key {len(only_final)}건"

    other_cols = [c for c in final_before.columns if c not in ECON_COLS]
    merged = final_before[other_cols].merge(econ, on=KEY_COLS, how="left", validate="one_to_one")
    merged = merged[final_before.columns]  # 원래 컬럼 순서 유지

    assert len(merged) == len(final_before), f"row 수 변경됨: {len(final_before)} -> {len(merged)}"

    for col in other_cols:
        if col in KEY_COLS:
            continue
        left, right = final_before[col], merged[col]
        left_na, right_na = left.isna().to_numpy(), right.isna().to_numpy()
        na_mismatch = left_na != right_na  # 한쪽만 NaN이면 그 자체로 변경
        both_valid = ~left_na & ~right_na
        value_mismatch = np.zeros(len(left), dtype=bool)
        if pd.api.types.is_float_dtype(left) and pd.api.types.is_float_dtype(right):
            value_mismatch[both_valid] = ~np.isclose(
                left.to_numpy()[both_valid], right.to_numpy()[both_valid], equal_nan=False
            )
        else:
            value_mismatch[both_valid] = left.to_numpy()[both_valid] != right.to_numpy()[both_valid]
        n_diff = int(na_mismatch.sum()) + int(value_mismatch.sum())
        assert n_diff == 0, f"경제피처 외 컬럼 변경 발견: {col} ({n_diff}건)"

    n_na = int(merged[ECON_COLS].isna().sum().sum())
    n_inf = int(np.isinf(merged[ECON_COLS].select_dtypes(include=[np.number]).to_numpy()).sum())
    assert n_na == 0, f"병합 후 경제피처 NaN {n_na}건 남음"
    assert n_inf == 0, f"병합 후 경제피처 inf {n_inf}건"

    changed = int((final_before["cpi_y2_prev_yoy"].isna() & merged["cpi_y2_prev_yoy"].notna()).sum())

    shutil.copy2(FINAL_PATH, BACKUP_PATH)
    merged.to_parquet(FINAL_PATH, index=False)

    print(f"[audit] row 수: {len(final_before):,} -> {len(merged):,} (동일)")
    print(f"[audit] key 불일치: 0건 (econ/final 양쪽 {len(econ_keys):,}개 key 완전 일치)")
    print(f"[audit] 경제피처 외 컬럼 변경: 0건 ({len(other_cols) - len(KEY_COLS)}개 컬럼 전수 검사)")
    print(f"[audit] 병합 후 경제피처 NaN/inf: {n_na}/{n_inf}건")
    print(f"[결과] cpi_y2_prev_yoy가 NaN->값으로 갱신된 행: {changed:,}건")
    print(f"백업: {BACKUP_PATH}")
    print(f"저장 완료: {FINAL_PATH}")


if __name__ == "__main__":
    main()
