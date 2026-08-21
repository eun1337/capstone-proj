"""
Day3 RF/LightGBM 공통 CV fold 생성기 (P10/P13/B robustness).

P10/P13의 validation 구간과 모든 train purge는 horizon별 target_date 기준으로 처리하며,
B robustness는 week_st 기준 1주 step으로 순회하되 target_date 기준으로 미래 누수를 차단한다.
"""

import pandas as pd

from src.ml.day3_rf_lightgbm.common.config import (
    B_HISTORY_START,
    FINAL_TRAIN_CUTOFF,
    HORIZONS,
    MAX_DEV_TARGET_DATE,
    MAX_HOLDOUT_TARGET_DATE,
    VALID_VALIDATION_YEARS,
)

CENTER_COL = "center_id"
WEEK_COL = "week_st"
CENTER_A = "A"
CENTER_B = "B"


def _validate_horizon(horizon: int) -> None:
    if horizon not in HORIZONS:
        raise ValueError(f"지원하지 않는 horizon: {horizon!r} (허용값: {sorted(HORIZONS)})")


def _validate_validation_year(validation_year: int) -> None:
    if validation_year not in VALID_VALIDATION_YEARS:
        raise ValueError(f"지원하지 않는 validation_year: {validation_year!r} (허용값: {sorted(VALID_VALIDATION_YEARS)})")


def _target_date(df: pd.DataFrame, horizon: int) -> pd.Series:
    return df[WEEK_COL] + pd.Timedelta(weeks=horizon)


def _quarter_bounds(year: int, quarter: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=year, month=(quarter - 1) * 3 + 1, day=1)
    end = start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    return start, end


def generate_expanding_folds(df: pd.DataFrame, validation_year: int, horizon: int) -> list[dict]:
    """A센터 단독, validation_year의 Q1~Q4를 validation으로 하는 3개월 expanding 4-fold
    (validation_year=2022 -> P10, 2023 -> P13). 경계는 target_date 기준이며
    target_date<=2023-12-31을 강제한다."""
    _validate_validation_year(validation_year)
    _validate_horizon(horizon)
    a_mask = df[CENTER_COL] == CENTER_A
    target_date = _target_date(df, horizon)

    folds = []
    for quarter in range(1, 5):
        val_start, val_end = _quarter_bounds(validation_year, quarter)
        val_mask = a_mask & target_date.between(val_start, val_end) & (target_date <= MAX_DEV_TARGET_DATE)
        train_mask = a_mask & (target_date < val_start)
        folds.append({
            "fold": quarter,
            "validation_year": validation_year,
            "quarter": quarter,
            "horizon": horizon,
            "val_start": val_start,
            "val_end": val_end,
            "train_mask": train_mask,
            "val_mask": val_mask,
        })
    return folds


def generate_b_walkforward_folds(df: pd.DataFrame, horizon: int,
                                  b_history_start=B_HISTORY_START) -> list[dict]:
    """B센터 post-regime(week_st>=b_history_start) 1주 validation/1주 step expanding
    walk-forward. train은 A+B post-regime 중 target_date<validation target_date인 행만
    포함하며, target_date<=2023-12-31까지만 fold를 만든다. HP/모델군 선택 로직은
    호출부 책임."""
    _validate_horizon(horizon)
    b_history_start = pd.Timestamp(b_history_start)
    a_mask = df[CENTER_COL] == CENTER_A
    b_post_mask = (df[CENTER_COL] == CENTER_B) & (df[WEEK_COL] >= b_history_start)
    target_date = _target_date(df, horizon)

    b_weeks = sorted(df.loc[b_post_mask, WEEK_COL].unique())

    folds = []
    for i, val_week in enumerate(b_weeks):
        val_target_date = val_week + pd.Timedelta(weeks=horizon)
        if val_target_date > MAX_DEV_TARGET_DATE:
            break
        val_mask = b_post_mask & (df[WEEK_COL] == val_week)
        train_mask = (a_mask | b_post_mask) & (target_date < val_target_date)
        folds.append({
            "fold": i + 1,
            "horizon": horizon,
            "val_week": val_week,
            "val_target_date": val_target_date,
            "train_mask": train_mask,
            "val_mask": val_mask,
        })
    return folds


def final_retrain_mask(df: pd.DataFrame, horizon: int,
                        b_history_start=B_HISTORY_START) -> pd.Series:
    """CV fold가 아닌 최종 재학습 대상 필터(A 전체 + B post-regime, target_date<2024-01-01)."""
    _validate_horizon(horizon)
    b_history_start = pd.Timestamp(b_history_start)
    a_mask = df[CENTER_COL] == CENTER_A
    b_post_mask = (df[CENTER_COL] == CENTER_B) & (df[WEEK_COL] >= b_history_start)
    target_date = _target_date(df, horizon)
    return (a_mask | b_post_mask) & (target_date < FINAL_TRAIN_CUTOFF)


def holdout_2024_mask(holdout_df: pd.DataFrame, horizon: int) -> pd.Series:
    """holdout_2024.parquet 전용 필터. target_date가 2025로 넘어가는 연말 우측절단 행만 제외한다."""
    _validate_horizon(horizon)
    target_date = _target_date(holdout_df, horizon)
    return target_date <= MAX_HOLDOUT_TARGET_DATE
