"""
build_model_data.py

모든 모델이 공통으로 사용하는 development/holdout dataset을 생성한다.
final_feature_table.parquet의 SKU-week row와 기존 feature를 유지하고 날짜 기준으로만 분할한다.

산출물:
- data/development_2021_2023.parquet
- data/holdout_2024.parquet

qty_log1p만 원본의 과거 Hurdle 구조를 제거하기 위해 np.log1p(qty)로 재계산한다.
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"

INPUT_PATH = DATA_DIR / "final_feature_table.parquet"
DEV_OUTPUT_PATH = DATA_DIR / "development_2021_2023.parquet"
HOLDOUT_OUTPUT_PATH = DATA_DIR / "holdout_2024.parquet"

KEY_COLS = ["center_id", "sku_id", "week_st"]

MODEL_COLUMNS = [
    "center_id", "sku_id", "week_st", "qty", "qty_log1p",
    "target_h1", "target_h2", "target_h4",
    "상품명", "규격", "입수", "KAN_CODE", "KAN_대분류", "KAN_중분류", "KAN_소분류",
    "ISO_주차", "평균온도", "총강수량", "existed_before_regime", "월", "분기",
    "is_warmup", "coldstart_flag", "adi_expanding_filled", "cv2_expanding_filled",
    "target_h1_공휴일_W0", "target_h1_공휴일_W-1", "target_h1_공휴일_W+1",
    "target_h2_공휴일_W0", "target_h2_공휴일_W-1", "target_h2_공휴일_W+1",
    "target_h4_공휴일_W0", "target_h4_공휴일_W-1", "target_h4_공휴일_W+1",
    "강수량_호우_count", "covid_flag", "center_is_B", "temp_x_precip",
    "center_temp_inter", "weeks_since_last_active_filled",
    "ccsi_lag_m1", "cpi_y1_prev", "cpi_y2_prev_yoy",
    "qty_lag1_filled_log1p",
    "qty_rollmean_4_filled_log1p",
    "qty_rollstd_4_filled_log1p",
]

DEV_END = pd.Timestamp("2024-01-01")
HOLDOUT_END = pd.Timestamp("2025-01-01")


def load_source() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_PATH)

    missing = [col for col in MODEL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df["week_st"]):
        df["week_st"] = pd.to_datetime(df["week_st"], unit="ms")

    df = df[MODEL_COLUMNS].copy()
    df["qty_log1p"] = np.log1p(df["qty"])

    return df


def validate(df: pd.DataFrame, name: str) -> None:
    duplicates = df.duplicated(KEY_COLS).sum()
    if duplicates:
        raise ValueError(f"{name}: canonical key 중복 {duplicates}건")

    if df["qty_log1p"].isna().any():
        raise ValueError(f"{name}: qty_log1p NaN 존재")

    if not np.allclose(df["qty_log1p"], np.log1p(df["qty"])):
        raise ValueError(f"{name}: qty_log1p 정의 불일치")


def main() -> None:
    df = load_source()

    development = df[df["week_st"] < DEV_END].reset_index(drop=True)
    holdout = df[
        (df["week_st"] >= DEV_END)
        & (df["week_st"] < HOLDOUT_END)
    ].reset_index(drop=True)

    validate(development, "development")
    validate(holdout, "holdout")

    development.to_parquet(DEV_OUTPUT_PATH, index=False)
    holdout.to_parquet(HOLDOUT_OUTPUT_PATH, index=False)

    for name, part in [
        ("development", development),
        ("holdout", holdout),
    ]:
        print(
            f"{name}: "
            f"{len(part):,} rows | "
            f"{part['sku_id'].nunique():,} SKUs | "
            f"{part['week_st'].min().date()} ~ "
            f"{part['week_st'].max().date()}"
        )

    print(f"저장 완료: {DEV_OUTPUT_PATH}")
    print(f"저장 완료: {HOLDOUT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()