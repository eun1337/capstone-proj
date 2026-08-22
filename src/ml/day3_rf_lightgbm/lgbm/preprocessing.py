"""
preprocessing.py
LightGBM 전용 입력 전처리. Frozen 30개 feature에 dtype 정규화 + 구조적 NaN 5개 train-fit
계층형 median(소->중->대->train 전체, rf/preprocessing.py와 동일 방법론이나 독립 구현)을
적용한다. RF와 달리 LightGBM은 범주형을 native하게 다루므로 OneHotEncoder 대신 KAN 3종을
train에서 관측된 카테고리로 고정한 pandas Categorical로 인코딩한다(OHE sparse matrix가
아니라 DataFrame을 반환). train에서만 fit하고 validation/test/holdout은 transform만 한다.
"""

import numpy as np
import pandas as pd

from src.ml.day3_rf_lightgbm.common import config as cfg
from src.ml.day3_rf_lightgbm.common.input_preprocessing import normalize_model_input_dtypes

NAN_FEATURES = (
    "adi_expanding_filled",
    "cv2_expanding_filled",
    "qty_lag1_filled_log1p",
    "qty_rollmean_4_filled_log1p",
    "qty_rollstd_4_filled_log1p",
)
CAT_COLS = list(cfg.CATEGORICAL_FEATURES)


def _fit_hierarchical_median(train_df: pd.DataFrame, feature: str) -> dict:
    non_nan = train_df.loc[train_df[feature].notna()]
    overall = non_nan[feature].median()
    if pd.isna(overall):
        raise ValueError(f"{feature} has no non-missing values in training data")
    return {
        "sub": non_nan.groupby("KAN_소분류", observed=True)[feature].median(),
        "mid": non_nan.groupby("KAN_중분류", observed=True)[feature].median(),
        "large": non_nan.groupby("KAN_대분류", observed=True)[feature].median(),
        "overall": overall,
    }


def _apply_hierarchical_median(df: pd.DataFrame, feature: str, maps: dict) -> pd.Series:
    values = df[feature].copy()
    for level_col, level_key in (("KAN_소분류", "sub"), ("KAN_중분류", "mid"), ("KAN_대분류", "large")):
        still_nan = values.isna()
        if not still_nan.any():
            break
        candidate = df[level_col].map(maps[level_key])
        fillable = still_nan & candidate.notna()
        values.loc[fillable] = candidate.loc[fillable]
    still_nan = values.isna()
    if still_nan.any():
        values.loc[still_nan] = maps["overall"]
    if values.isna().any():
        raise ValueError(f"{feature}: {int(values.isna().sum())}개 행이 소/중/대/global fallback으로도 해소되지 않음")
    return values


def _check_finite_numeric(df: pd.DataFrame, numeric_cols: list, name: str) -> None:
    if not np.isfinite(df[numeric_cols].to_numpy(dtype=float)).all():
        raise ValueError(f"{name}에 NaN/Inf가 존재함")


class LGBMPreprocessor:
    """LightGBM 입력 전처리기. fit(train_df, horizon) 후 transform(df)을 반복 호출한다.
    transform()은 sparse matrix가 아니라 DataFrame을 반환하며, KAN 3종은 train 카테고리로
    고정된 pandas Categorical이다(LightGBM이 categorical_feature로 native 분기)."""

    def __init__(self):
        self.horizon = None
        self.feature_cols = None
        self.numeric_cols = None
        self._median_maps = None
        self._category_levels = None

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
        self._median_maps = {feature: _fit_hierarchical_median(train_df, feature) for feature in NAN_FEATURES}
        self._category_levels = {
            col: pd.Index(sorted(train_df[col].dropna().unique())) for col in CAT_COLS
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        out = normalize_model_input_dtypes(df[self.feature_cols], self.feature_cols).copy()
        for feature in NAN_FEATURES:
            if feature in self.feature_cols:
                out[feature] = _apply_hierarchical_median(out, feature, self._median_maps[feature])
        for col in CAT_COLS:
            out[col] = pd.Categorical(out[col], categories=self._category_levels[col])
        _check_finite_numeric(out, self.numeric_cols, "transform 결과")
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
