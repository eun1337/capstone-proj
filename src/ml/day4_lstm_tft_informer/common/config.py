"""
config.py
Day4 LSTM/TFT/Informer 공통 설정. Day3 common의 데이터 경로/horizon/target/fold 날짜 상수/
KAN categorical/holiday 정의를 그대로 재사용하고, DL 시퀀스에 필요한 lookback 후보와
feature role(static/time-varying known/observed) 분류만 추가한다. RF/LGBM 전용
DEMAND_SUMMARY_FEATURES(lag1/rollmean4/rollstd4)는 사용하지 않는다 - qty_log1p는
raw qty에서 runtime에 다시 계산한다(sequence_builder.py).
"""

from src.ml.day3_rf_lightgbm.common.config import (  # noqa: F401
    B_HISTORY_START,
    CATEGORICAL_FEATURES,
    DEVELOPMENT_PATH,
    FINAL_TRAIN_CUTOFF,
    HOLDOUT_PATH,
    HOLIDAY_FEATURES,
    HORIZONS,
    KEY_COLS,
    MAX_DEV_TARGET_DATE,
    MAX_HOLDOUT_TARGET_DATE,
    TARGET_COLS,
    VALID_VALIDATION_YEARS,
)

LOOKBACK_CHOICES = (13, 26)

STATIC_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES  # KAN_대/중/소분류

STATIC_CONTINUOUS_FEATURES = (
    "입수",
    "existed_before_regime",
    "center_is_B",
)

STATIC_FEATURES = STATIC_CATEGORICAL_FEATURES + STATIC_CONTINUOUS_FEATURES

TIME_VARYING_KNOWN_FEATURES = (
    "ISO_주차",
    "월",
    "분기",
    "covid_flag",
)

TIME_VARYING_OBSERVED_FEATURES = (
    "평균온도",
    "총강수량",
    "qty_log1p",
    "is_warmup",
    "coldstart_flag",
    "adi_expanding_filled",
    "cv2_expanding_filled",
    "강수량_호우_count",
    "temp_x_precip",
    "center_temp_inter",
    "weeks_since_last_active_filled",
    "ccsi_lag_m1",
    "cpi_y1_prev",
    "cpi_y2_prev_yoy",
)


def validate_horizon(horizon: int) -> None:
    if horizon not in HORIZONS:
        raise ValueError(f"지원하지 않는 horizon: {horizon!r} (허용값: {HORIZONS})")


def validate_lookback(lookback: int) -> None:
    if lookback not in LOOKBACK_CHOICES:
        raise ValueError(f"지원하지 않는 lookback: {lookback!r} (허용값: {LOOKBACK_CHOICES})")


def get_model_feature_roles(horizon: int) -> dict:
    """horizon별 DL feature role 분류. time_varying_known은 공통 known 4개 +
    해당 horizon의 holiday 3개다."""
    validate_horizon(horizon)
    return {
        "static_categorical": STATIC_CATEGORICAL_FEATURES,
        "static_continuous": STATIC_CONTINUOUS_FEATURES,
        "static": STATIC_FEATURES,
        "time_varying_known": TIME_VARYING_KNOWN_FEATURES + HOLIDAY_FEATURES[horizon],
        "time_varying_observed": TIME_VARYING_OBSERVED_FEATURES,
        "management_key": KEY_COLS + ("target_date",),
    }


def get_all_dl_feature_cols(horizon: int) -> tuple[str, ...]:
    """검증용: 해당 horizon에서 DL이 실제 사용하는 전체 27개 feature(static 6 +
    time-varying known/observed 21) 목록."""
    roles = get_model_feature_roles(horizon)
    return roles["static"] + roles["time_varying_known"] + roles["time_varying_observed"]
