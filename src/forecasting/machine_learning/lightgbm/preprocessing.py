"""
preprocessing.py

LightGBM 입력 전처리.

구조적 NaN 5개는 LightGBM native missing으로 유지하고,
그 외 numeric NaN/Inf는 fail-fast한다.
categorical feature는 train에서 고정하고 validation/test/holdout에는 transform만 적용한다.
"""

import numpy as np
import pandas as pd

from src.forecasting.common import config as cfg
from src.forecasting.common.input_preprocessing import normalize_model_input_dtypes

ALLOWED_NAN_FEATURES = (
    "adi_expanding_filled",
    "cv2_expanding_filled",
    "qty_lag1_filled_log1p",
    "qty_rollmean_4_filled_log1p",
    "qty_rollstd_4_filled_log1p",
)
CAT_COLS = list(cfg.CATEGORICAL_FEATURES)


def _check_transform_output(out: pd.DataFrame, numeric_cols: list, name: str) -> dict:
    """±Inf와 ALLOWED_NAN_FEATURES 외의 NaN을 fail-fast하고, 허용된 NaN 개수를 반환한다."""
    numeric_values = out[numeric_cols].to_numpy(dtype=float)
    if np.isinf(numeric_values).any():
        bad_cols = [c for c in numeric_cols if np.isinf(out[c].to_numpy(dtype=float)).any()]
        raise ValueError(f"{name}에 ±Inf가 존재함: {bad_cols}")

    unexpected_nan_cols = []
    nan_counts = {}
    for col in numeric_cols:
        n_nan = int(out[col].isna().sum())
        if col in ALLOWED_NAN_FEATURES:
            nan_counts[col] = n_nan
        elif n_nan > 0:
            unexpected_nan_cols.append((col, n_nan))
    if unexpected_nan_cols:
        raise ValueError(f"{name}: 허용되지 않은 feature에 예상하지 못한 NaN 존재: {unexpected_nan_cols}")
    return nan_counts


class LGBMPreprocessor:
    """LightGBM 입력 전처리기. fit(train_df, horizon) 후 transform(df)을 반복 호출한다."""

    def __init__(self):
        self.horizon = None
        self.feature_cols = None
        self.numeric_cols = None
        self._category_levels = None
        self.last_unknown_category_counts_ = None
        self.last_structural_nan_counts_ = None

    def fit(self, train_df: pd.DataFrame, horizon: int) -> "LGBMPreprocessor":
        if horizon not in cfg.HORIZONS:
            raise ValueError(f"지원하지 않는 horizon: {horizon!r} (허용값: {cfg.HORIZONS})")

        feature_cols = list(cfg.get_model_feature_cols(horizon))
        missing = [c for c in feature_cols if c not in train_df.columns]
        if missing:
            raise KeyError(f"train_df에 필수 feature 컬럼 누락: {missing}")

        self.horizon = horizon
        self.feature_cols = feature_cols
        self.numeric_cols = [c for c in feature_cols if c not in CAT_COLS]
        self._category_levels = {
            col: pd.Index(sorted(train_df[col].dropna().unique())) for col in CAT_COLS
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        out = normalize_model_input_dtypes(df[self.feature_cols], self.feature_cols).copy()

        unknown_category_counts = {}
        for col in CAT_COLS:
            categorical = pd.Categorical(out[col], categories=self._category_levels[col])
            n_unknown = int(out[col].notna().sum() - pd.notna(categorical).sum())
            unknown_category_counts[col] = n_unknown
            out[col] = categorical
        self.last_unknown_category_counts_ = unknown_category_counts

        self.last_structural_nan_counts_ = _check_transform_output(out, self.numeric_cols, "transform 결과")
        return out

    def fit_transform(self, train_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        self.fit(train_df, horizon)
        return self.transform(train_df)

    def _check_fitted(self) -> None:
        if self._category_levels is None:
            raise RuntimeError("fit()을 먼저 호출해야 함")

    @property
    def categorical_feature_names(self) -> list:
        self._check_fitted()
        return list(CAT_COLS)

    @property
    def category_counts(self) -> dict:
        self._check_fitted()
        return {col: len(levels) for col, levels in self._category_levels.items()}
