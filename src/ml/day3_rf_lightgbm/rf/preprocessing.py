"""
preprocessing.py
RandomForest 전용 입력 전처리. Frozen 30개 feature에 dtype 정규화 + 구조적 NaN 5개
train-fit 계층형 median(소->중->대->train 전체) + KAN 3종 OneHotEncoder(sparse)를 적용해
sparse matrix를 만든다. train에서만 fit하고 validation/test/holdout은 transform만 한다.
분석용 analyze_nan_hierarchy.py에는 의존하지 않는 독립 구현이다.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

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


def _check_matrix_finite(X, name: str) -> None:
    if not np.isfinite(X.data).all():
        raise ValueError(f"{name}에 NaN/Inf가 존재함")


class RFPreprocessor:
    """RandomForest 입력 전처리기. fit(train_df, horizon) 후 transform(df)을 반복 호출한다."""

    def __init__(self):
        self.horizon = None
        self.feature_cols = None
        self.numeric_cols = None
        self._median_maps = None
        self._column_transformer = None

    def fit(self, train_df: pd.DataFrame, horizon: int) -> "RFPreprocessor":
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

        train_input = self._prepare_input(train_df)
        self._column_transformer = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CAT_COLS),
                ("num", "passthrough", self.numeric_cols),
            ],
            sparse_threshold=1.0,
        )
        self._column_transformer.fit(train_input)
        return self

    def transform(self, df: pd.DataFrame):
        self._check_fitted()
        X = self._column_transformer.transform(self._prepare_input(df))
        _check_matrix_finite(X, "transform 결과")
        return X

    def fit_transform(self, train_df: pd.DataFrame, horizon: int):
        self.fit(train_df, horizon)
        return self.transform(train_df)

    def _prepare_input(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = normalize_model_input_dtypes(df[self.feature_cols], self.feature_cols)
        for feature in NAN_FEATURES:
            if feature in self.feature_cols:
                normalized[feature] = _apply_hierarchical_median(normalized, feature, self._median_maps[feature])
        return normalized

    def _check_fitted(self) -> None:
        if self._column_transformer is None:
            raise RuntimeError("fit()을 먼저 호출해야 함")

    @property
    def ohe_category_counts(self) -> dict:
        self._check_fitted()
        ohe = self._column_transformer.named_transformers_["cat"]
        return {col: len(cats) for col, cats in zip(CAT_COLS, ohe.categories_)}
