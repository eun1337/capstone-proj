"""
structural_nan.py
DL residual structural NaN(adi_expanding_filled/cv2_expanding_filled) 처리. 이미 feature
engineering에서 값이 채워진 row는 절대 변경하지 않고, 여전히 NaN인 residual만 fold
TRAIN row로 fit한 계층형 median(KAN_소분류->중분류->대분류->fold-global)으로 채운다.
validation은 fit에 쓰지 않는다. imputation 이후에도 NaN/Inf가 남으면 fail-fast한다.
"""

import numpy as np
import pandas as pd

RESIDUAL_NAN_FEATURES = ("adi_expanding_filled", "cv2_expanding_filled")


def fit_residual_nan_medians(train_df: pd.DataFrame) -> dict:
    """fold TRAIN row만으로 RESIDUAL_NAN_FEATURES 각각의 소/중/대/전체 median map을 만든다."""
    maps = {}
    for feature in RESIDUAL_NAN_FEATURES:
        non_nan = train_df.loc[train_df[feature].notna()]
        overall = non_nan[feature].median()
        if pd.isna(overall):
            raise ValueError(f"{feature}: fold train에 non-missing 값이 하나도 없음")
        maps[feature] = {
            "sub": non_nan.groupby("KAN_소분류", observed=True)[feature].median(),
            "mid": non_nan.groupby("KAN_중분류", observed=True)[feature].median(),
            "large": non_nan.groupby("KAN_대분류", observed=True)[feature].median(),
            "overall": overall,
        }
    return maps


def _summarize(original_nan_mask: pd.Series, resolved_at: pd.Series) -> dict:
    counts = resolved_at[original_nan_mask].value_counts()
    return {
        "n_before_nan": int(original_nan_mask.sum()),
        "resolved_sub": int(counts.get("sub", 0)),
        "resolved_mid": int(counts.get("mid", 0)),
        "resolved_large": int(counts.get("large", 0)),
        "resolved_global": int(counts.get("global", 0)),
        "unresolved": int(counts.get("unresolved", 0)),
    }


def apply_residual_nan_medians(df: pd.DataFrame, maps: dict) -> tuple[pd.DataFrame, dict]:
    """maps(fit_residual_nan_medians 결과)로 df(train+validation 전체 history)의
    RESIDUAL_NAN_FEATURES residual NaN만 채운 복사본과 feature별 처리 요약을 반환한다.
    기존 non-NaN 값이 바뀌면 즉시 ValueError, imputation 후에도 NaN/Inf가 남으면
    즉시 ValueError."""
    out = df.copy()
    summary = {}

    for feature, m in maps.items():
        original = out[feature]
        original_nan_mask = original.isna()
        values = original.copy()
        resolved_at = pd.Series(np.where(original.notna(), "already_valid", "unresolved"), index=out.index)

        for level_col, level_key in (("KAN_소분류", "sub"), ("KAN_중분류", "mid"), ("KAN_대분류", "large")):
            still_nan = values.isna()
            if not still_nan.any():
                break
            candidate = out[level_col].map(m[level_key])
            fillable = still_nan & candidate.notna()
            values.loc[fillable] = candidate.loc[fillable]
            resolved_at.loc[fillable] = level_key

        still_nan = values.isna()
        if still_nan.any():
            values.loc[still_nan] = m["overall"]
            resolved_at.loc[still_nan] = "global"

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{feature}: imputation 이후에도 NaN/Inf가 남음")

        changed_non_nan = (~original_nan_mask) & (original != values)
        n_changed = int(changed_non_nan.sum())
        if n_changed:
            raise ValueError(f"{feature}: 기존 non-NaN 값이 {n_changed}건 변경됨")

        out[feature] = values
        summary[feature] = {**_summarize(original_nan_mask, resolved_at), "n_non_nan_changed": n_changed}

    return out, summary
