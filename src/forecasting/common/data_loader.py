"""
data_loader.py
Forecasting 공통 feature/target 로더. config.py에 Freeze된 horizon별 30개
feature만 사용하며, 자동 feature discovery는 하지 않는다.
"""

import pandas as pd

from src.forecasting.common.config import (
    DEVELOPMENT_PATH,
    HOLDOUT_PATH,
    HORIZONS,
    KEY_COLS,
    TARGET_COLS,
    get_model_feature_cols,
)


def load_development() -> pd.DataFrame:
    """CV/개발용 development_2021_2023.parquet을 로드한다."""
    return pd.read_parquet(DEVELOPMENT_PATH)


def load_holdout_2024() -> pd.DataFrame:
    """최종 Holdout 전용 holdout_2024.parquet을 로드한다."""
    return pd.read_parquet(HOLDOUT_PATH)


def target_col(horizon: int) -> str:
    if horizon not in HORIZONS:
        raise ValueError(f"지원하지 않는 horizon: {horizon!r} (허용값: {HORIZONS})")
    return TARGET_COLS[horizon]


def get_feature_cols(horizon: int) -> list[str]:
    """config에 Freeze된 horizon별 30개 feature를 그대로 반환한다."""
    return list(get_model_feature_cols(horizon))


def _validate_required_cols(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"필수 컬럼 누락: {missing}")


def prepare_xy(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """horizon별 30개 feature(X), raw target(y), 관리용 key(keys)를 분리해 반환한다.
    y에는 log1p를 적용하지 않는다 - trainer가 fit 직전에 수행한다."""
    feature_cols = get_feature_cols(horizon)
    target = target_col(horizon)
    _validate_required_cols(df, feature_cols + [target] + list(KEY_COLS))

    X = df[feature_cols].copy()
    y = df[target].copy()

    key_cols = list(KEY_COLS) + (["row_id"] if "row_id" in df.columns else [])
    keys = df[key_cols].copy()
    keys["target_date"] = keys["week_st"] + pd.Timedelta(weeks=horizon)

    return X, y, keys
