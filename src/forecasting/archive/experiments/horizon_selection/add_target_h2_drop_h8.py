"""
add_target_h2_drop_h8.py

설명 : feature_table_with_econ.parquet의 예측 타깃 호라이즌을 {1, 2, 4}주로 정리.
- (center_id, sku_id)별 시계열에서 2주 뒤 값을 shift하여
  target_h2, target_h2_flag, target_h2_qty_log1p 컬럼을 생성함.
- 센터별 데이터 종료일을 초과하는 2주 뒤 타깃은 NaN으로 처리함.
- 더 이상 사용하지 않는 target_h8 계열 3개 컬럼을 제거함.
- 기존 target_h1/h4 값 유지 여부와 새 target_h2의 미래 qty 연결을 검증함.
- 수정 전 파일을 백업한 뒤 결과를 기존 feature_table_with_econ.parquet에 저장함.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SRC_PATH = Path("data/ml/experiments/economic_indicators/feature_table_with_econ.parquet")
BACKUP_PATH = Path("data/ml/experiments/economic_indicators/feature_table_with_econ_backup_20260806.parquet")

GROUP_KEYS = ["center_id", "sku_id"]
WEEK_COL = "week_st"
QTY_COL = "qty"
H = 2
H8_COLS = ["target_h8", "target_h8_flag", "target_h8_qty_log1p"]


def compute_target_h2(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(GROUP_KEYS, sort=False)
    global_end = df.groupby("center_id")[WEEK_COL].transform("max")
    beyond_end = (df[WEEK_COL] + pd.Timedelta(weeks=H)) > global_end

    shifted_qty = grouped[QTY_COL].shift(-H)
    shifted_flag = grouped["sold_flag"].shift(-H)
    shifted_log1p = grouped["qty_log1p"].shift(-H)

    df["target_h2"] = np.where(beyond_end, np.nan, shifted_qty)
    df["target_h2_flag"] = np.where(beyond_end, np.nan, shifted_flag)
    df["target_h2_qty_log1p"] = np.where(beyond_end, np.nan, shifted_log1p)
    return df


def verify(df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    print("=" * 70)
    print("[검증] target_h2 생성 / target_h8 제거 결과")
    print("=" * 70)

    # 1) target_h1/target_h2/target_h4가 실제 미래 qty와 이어지는지 샘플 확인
    sample_sku = (
        df_after.groupby(["center_id", "sku_id"])["week_st"].count().sort_values(ascending=False).index[0]
    )
    center_id, sku_id = sample_sku
    s = df_after[(df_after["center_id"] == center_id) & (df_after["sku_id"] == sku_id)].sort_values("week_st")
    s = s.set_index("week_st")
    checks = []
    for t in s.index[:-4]:
        row = s.loc[t]
        t1, t2, t4 = t + pd.Timedelta(weeks=1), t + pd.Timedelta(weeks=2), t + pd.Timedelta(weeks=4)
        ok1 = pd.isna(row["target_h1"]) or (t1 in s.index and row["target_h1"] == s.loc[t1, "qty"])
        ok2 = pd.isna(row["target_h2"]) or (t2 in s.index and row["target_h2"] == s.loc[t2, "qty"])
        ok4 = pd.isna(row["target_h4"]) or (t4 in s.index and row["target_h4"] == s.loc[t4, "qty"])
        checks.append(ok1 and ok2 and ok4)
    print(f"\n샘플 SKU {center_id}/{sku_id} (n={len(s)}주): target_h1/h2/h4가 실제 미래 qty와 일치 = {all(checks)}")

    # 2) A/B별 target_h2, target_h2_flag 결측 개수
    print("\ncenter별 target_h2 / target_h2_flag 결측 개수:")
    na_counts = df_after.groupby("center_id")[["target_h2", "target_h2_flag"]].apply(lambda g: g.isna().sum())
    print(na_counts)

    # 3) 기존 target_h1/h4 계열 값이 이번 작업으로 변경되지 않았는지 대조
    unchanged_cols = [
        "target_h1", "target_h1_flag", "target_h1_qty_log1p",
        "target_h4", "target_h4_flag", "target_h4_qty_log1p",
    ]
    print("\n기존 target_h1/h4 계열 값 변경 여부 (True=변경없음):")
    for col in unchanged_cols:
        before = df_before[col].to_numpy()
        after = df_after[col].to_numpy()
        same = np.array_equal(before, after, equal_nan=True)
        print(f"  {col}: {same}")

    # 4) target_h8 계열 제거 확인
    remaining_h8 = [c for c in H8_COLS if c in df_after.columns]
    print(f"\ntarget_h8 계열 컬럼 잔존 여부: {remaining_h8 if remaining_h8 else '없음 (정상 제거됨)'}")


def main() -> None:
    shutil.copy2(SRC_PATH, BACKUP_PATH)
    print(f"백업 완료: {BACKUP_PATH}")

    df = pd.read_parquet(SRC_PATH)
    df_before = df.copy()

    df = compute_target_h2(df)
    df = df.drop(columns=H8_COLS)

    verify(df_before, df)

    df.to_parquet(SRC_PATH, index=False)

    print("\n" + "=" * 70)
    print(f"최종 컬럼 수: {len(df.columns)} (기존 {len(df_before.columns)}개 -> +target_h2 3개 -target_h8 3개)")
    print(f"저장 완료: {SRC_PATH}")


if __name__ == "__main__":
    main()
