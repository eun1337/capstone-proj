"""
category_fallback.py
계단식 카테고리 평균 fallback(소분류->중분류->대분류->센터전체_같은주->센터전체_widen->
누적평균) helper. archive/feature_engineering/feature_lag_rolling_calendar.py에서
add_rollstd_log_features.py가 필요로 하는 부분만 active source로 분리했다
(나머지 lag/rolling/캘린더 feature 생성 로직은 legacy이므로 archive에 그대로 둔다).
"""

import pandas as pd

WEEK_COL = "week_st"

CATEGORY_HIERARCHY = ["KAN_소분류", "KAN_중분류", "KAN_대분류"]


def add_category_fallback(
    df: pd.DataFrame, feature_cols: list[str], max_widen_weeks: int = 8
) -> pd.DataFrame:
    missing_hierarchy_cols = [c for c in CATEGORY_HIERARCHY if c not in df.columns]
    if missing_hierarchy_cols:
        raise KeyError(
            f"{missing_hierarchy_cols} 컬럼이 입력 데이터에 없음 — 카테고리평균 fallback 불가. "
            "KAN 분류 컬럼이 입력 데이터에 포함되어 있는지 확인 필요."
        )

    for col in feature_cols:
        filled = df[col].copy()
        source = pd.Series(pd.NA, index=df.index, dtype="object")
        needs_fallback = filled.isna()

        for cat_col in CATEGORY_HIERARCHY:
            mask = needs_fallback & filled.isna()
            if not mask.any():
                break
            cat_mean = df.groupby([cat_col, WEEK_COL])[col].transform("mean")
            newly = mask & cat_mean.notna()
            filled.loc[newly] = cat_mean.loc[newly]
            source.loc[newly] = cat_col

        mask = needs_fallback & filled.isna()
        if mask.any():
            week_mean = df.groupby(WEEK_COL)[col].transform("mean")
            newly = mask & week_mean.notna()
            filled.loc[newly] = week_mean.loc[newly]
            source.loc[newly] = "센터전체_같은주"

        mask = needs_fallback & filled.isna()
        if mask.any():
            week_sum = df.groupby(WEEK_COL)[col].sum()
            week_count = df.groupby(WEEK_COL)[col].count()
            for k in range(1, max_widen_weeks + 1):
                mask = needs_fallback & filled.isna()
                if not mask.any():
                    break
                for idx in df.index[mask]:
                    center_week = df.at[idx, WEEK_COL]
                    lo = center_week - pd.Timedelta(weeks=k)
                    hi = center_week
                    window = (week_sum.index >= lo) & (week_sum.index <= hi)
                    total_count = week_count[window].sum()
                    if total_count > 0:
                        filled.at[idx] = week_sum[window].sum() / total_count
                        source.at[idx] = f"센터전체_widen{k}주"

        mask = needs_fallback & filled.isna()
        if mask.any():
            weekly_sum = df.groupby(WEEK_COL)[col].sum().sort_index()
            weekly_count = df.groupby(WEEK_COL)[col].count().sort_index()
            cum_sum = weekly_sum.cumsum()
            cum_count = weekly_count.cumsum()
            expanding_mean = (cum_sum / cum_count).rename("_expanding_mean")
            row_expanding_mean = df.loc[mask, WEEK_COL].map(expanding_mean)
            filled.loc[mask] = row_expanding_mean
            source.loc[mask & filled.notna()] = "누적평균(그 시점까지)"

        df[f"{col}_filled"] = filled
        df[f"{col}_fill_source"] = source
    return df
