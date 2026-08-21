"""
config.py
Day3 RF/LightGBM 공통 설정 - 데이터 경로, horizon, target/feature 정의, fold 날짜 상수, seed, P21 선택 기준.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DEVELOPMENT_PATH = PROJECT_ROOT / "data" / "development_2021_2023.parquet"
HOLDOUT_PATH = PROJECT_ROOT / "data" / "holdout_2024.parquet"

HORIZONS = (1, 2, 4)

TARGET_COLS = {
    1: "target_h1",
    2: "target_h2",
    4: "target_h4",
}

KEY_COLS = (
    "center_id",
    "sku_id",
    "week_st",
)

MODEL_EXCLUDED_ID_COLS = (
    "center_id",
    "sku_id",
    "KAN_CODE",
)

CATEGORICAL_FEATURES = (
    "KAN_대분류",
    "KAN_중분류",
    "KAN_소분류",
)

BASE_FEATURES = (
    "입수",
    "KAN_대분류",
    "KAN_중분류",
    "KAN_소분류",
    "ISO_주차",
    "평균온도",
    "총강수량",
    "existed_before_regime",
    "qty_log1p",
    "월",
    "분기",
    "is_warmup",
    "coldstart_flag",
    "adi_expanding_filled",
    "cv2_expanding_filled",
    "강수량_호우_count",
    "covid_flag",
    "center_is_B",
    "temp_x_precip",
    "center_temp_inter",
    "weeks_since_last_active_filled",
    "ccsi_lag_m1",
    "cpi_y1_prev",
    "cpi_y2_prev_yoy",
)

DEMAND_SUMMARY_FEATURES = (
    "qty_lag1_filled_log1p",
    "qty_rollmean_4_filled_log1p",
    "qty_rollstd_4_filled_log1p",
)

HOLIDAY_FEATURES = {
    1: (
        "target_h1_공휴일_W0",
        "target_h1_공휴일_W-1",
        "target_h1_공휴일_W+1",
    ),
    2: (
        "target_h2_공휴일_W0",
        "target_h2_공휴일_W-1",
        "target_h2_공휴일_W+1",
    ),
    4: (
        "target_h4_공휴일_W0",
        "target_h4_공휴일_W-1",
        "target_h4_공휴일_W+1",
    ),
}


def get_model_feature_cols(horizon: int) -> tuple[str, ...]:
    if horizon not in HORIZONS:
        raise ValueError(f"지원하지 않는 horizon: {horizon!r} (허용값: {HORIZONS})")
    return BASE_FEATURES + DEMAND_SUMMARY_FEATURES + HOLIDAY_FEATURES[horizon]


MAX_DEV_TARGET_DATE = pd.Timestamp("2023-12-31")
MAX_HOLDOUT_TARGET_DATE = pd.Timestamp("2024-12-31")
FINAL_TRAIN_CUTOFF = pd.Timestamp("2024-01-01")
B_HISTORY_START = pd.Timestamp("2023-07-03")

VALID_VALIDATION_YEARS = (2022, 2023)

P13_MODEL_SEED = 42
P13_SAMPLER_SEED = 2026
ROBUSTNESS_SEEDS = (42, 123, 456)

BIAS_GUARDRAIL_ABS_PCT = 20.0
NEAR_TIE_WAPE_PCT_POINT = 1.0
