"""
03_prepare_arimax_exog.py

ARIMAX용 주차별 설날·추석 calendar 변수를 준비한다.

기존 KOREAN_LUNAR_HOLIDAYS와 holiday feature 생성 함수를 재사용하여
각 calendar week의 공휴일_W0/W-1/W+1을 생성하고 정렬 상태를 검증한다.
ARIMA/ARIMAX 학습이나 성능평가는 수행하지 않는다.
"""

from pathlib import Path
import sys

import pandas as pd

_FEATURE_ENGINEERING_DIR = Path(__file__).resolve().parents[1] / "feature_engineering"
if str(_FEATURE_ENGINEERING_DIR) not in sys.path:
    sys.path.insert(0, str(_FEATURE_ENGINEERING_DIR))
from feature_external_interaction_concat import add_holiday_calendar_features, WEEK_COL  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[3]
DEV_PATH = BASE_DIR / "data" / "development_2021_2023.parquet"
HOLDOUT_PATH = BASE_DIR / "data" / "holdout_2024.parquet"
OUT_DIR = BASE_DIR / "data" / "ml" / "day2_statistical_models" / "exog"
HOLIDAY_WEEKLY_PATH = OUT_DIR / "statistical_holiday_weekly.parquet"
ALIGNMENT_AUDIT_PATH = OUT_DIR / "statistical_holiday_alignment_audit.csv"

HOLIDAY_COLS = ["공휴일_W0", "공휴일_W-1", "공휴일_W+1"]


def build_statistical_holiday_weekly(week_starts: pd.Series) -> pd.DataFrame:
    """week_st(월요일) 목록 -> (week_st, 공휴일_W0/W-1/W+1) unique table.
    add_holiday_calendar_features(horizons=[0])를 그대로 호출(target_date=week_st)."""
    week_starts = pd.Series(pd.to_datetime(pd.Series(week_starts).unique())).sort_values().reset_index(drop=True)

    non_monday = week_starts[week_starts.dt.weekday != 0]
    if len(non_monday) > 0:
        raise ValueError(f"월요일이 아닌 week_st 존재: {non_monday.tolist()[:5]}")

    df = pd.DataFrame({WEEK_COL: week_starts})
    out = add_holiday_calendar_features(df, horizons=[0])
    out = out.rename(columns={
        "target_h0_공휴일_W0": "공휴일_W0",
        "target_h0_공휴일_W-1": "공휴일_W-1",
        "target_h0_공휴일_W+1": "공휴일_W+1",
    })
    out = out[[WEEK_COL] + HOLIDAY_COLS].sort_values(WEEK_COL).reset_index(drop=True)

    if out[WEEK_COL].duplicated().any():
        raise ValueError("생성된 holiday table에 week_st 중복 존재")
    return out


def get_future_exog_matrix(holiday_weekly: pd.DataFrame, forecast_origin: pd.Timestamp, n_steps: int = 4) -> pd.DataFrame:
    """forecast_origin=t에서 step1..n_steps(=t+1..t+n_steps주)의 Holiday exog를 순서대로
    담은 행렬. ARIMAX가 4-step forecast를 한 번에 생성하므로 h3도 반드시 포함한다."""
    forecast_origin = pd.Timestamp(forecast_origin)
    step_weeks = [forecast_origin + pd.Timedelta(weeks=k) for k in range(1, n_steps + 1)]

    lookup = holiday_weekly.set_index(WEEK_COL)
    missing = [w for w in step_weeks if w not in lookup.index]
    if missing:
        raise ValueError(f"holiday table에 없는 week_st: {[w.date().isoformat() for w in missing]}")

    rows = []
    for step, w in enumerate(step_weeks, start=1):
        r = lookup.loc[w]
        rows.append({"step": step, WEEK_COL: w, **{c: int(r[c]) for c in HOLIDAY_COLS}})
    return pd.DataFrame(rows)


def align_training_holiday(holiday_weekly: pd.DataFrame, week_st_series: pd.Series) -> pd.DataFrame:
    """training/historical 구간 정렬: 주어진 week_st 시리즈에 같은 week_st의 holiday exog를
    1:1로 붙인다(origin=target 동일 주)."""
    left = pd.DataFrame({WEEK_COL: pd.to_datetime(pd.Series(week_st_series).reset_index(drop=True))})
    merged = left.merge(holiday_weekly, on=WEEK_COL, how="left", validate="many_to_one")
    if merged[HOLIDAY_COLS].isna().any().any():
        missing_weeks = merged.loc[merged[HOLIDAY_COLS].isna().any(axis=1), WEEK_COL]
        raise ValueError(f"holiday table에 없는 week_st가 정렬 대상에 존재: {missing_weeks.dt.date.unique().tolist()[:5]}")
    return merged


def load_week_universe() -> pd.Series:
    """development + holdout의 week_st 합집합(read-only)."""
    dev_weeks = pd.read_parquet(DEV_PATH, columns=[WEEK_COL])[WEEK_COL]
    holdout_weeks = pd.read_parquet(HOLDOUT_PATH, columns=[WEEK_COL])[WEEK_COL]
    return pd.concat([dev_weeks, holdout_weeks]).drop_duplicates().sort_values().reset_index(drop=True)


def _audit_row(check: str, passed: bool, detail: str) -> dict:
    return {"check": check, "passed": bool(passed), "detail": detail}


def run_alignment_audit(holiday_weekly: pd.DataFrame, week_universe: pd.Series) -> pd.DataFrame:
    """week uniqueness / Monday / NaN / join duplication을 검증한다."""
    rows = []
    dup = holiday_weekly[WEEK_COL].duplicated().sum()
    rows.append(_audit_row("week_st_unique", dup == 0, f"중복 {dup}건"))

    non_monday = int((holiday_weekly[WEEK_COL].dt.weekday != 0).sum())
    rows.append(_audit_row("all_monday", non_monday == 0, f"월요일 아닌 행 {non_monday}건"))

    n_nan = int(holiday_weekly[HOLIDAY_COLS].isna().sum().sum())
    rows.append(_audit_row("no_unexpected_nan", n_nan == 0, f"Holiday 컬럼 NaN 총 {n_nan}건"))

    left = pd.DataFrame({WEEK_COL: week_universe})
    try:
        merged = left.merge(holiday_weekly, on=WEEK_COL, how="left", validate="one_to_one")
        ok = len(merged) == len(left)
        detail = f"{len(left)}행 -> merge 후 {len(merged)}행"
    except Exception as e:
        ok = False
        detail = f"merge 실패: {type(e).__name__}: {e}"
    rows.append(_audit_row("no_join_duplication", ok, detail))

    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    week_universe = load_week_universe()
    holiday_weekly = build_statistical_holiday_weekly(week_universe)
    holiday_weekly.to_parquet(HOLIDAY_WEEKLY_PATH, index=False)

    audit = run_alignment_audit(holiday_weekly, week_universe)
    audit.to_csv(ALIGNMENT_AUDIT_PATH, index=False)

    n_fail = int((~audit["passed"]).sum())
    print(f"[Holiday table] {holiday_weekly[WEEK_COL].min().date()} ~ {holiday_weekly[WEEK_COL].max().date()}, "
          f"{len(holiday_weekly)}행")
    print(f"[Audit] 총 {len(audit)}건 중 실패 {n_fail}건")
    for _, r in audit.iterrows():
        mark = "OK" if r["passed"] else "FAIL"
        print(f"  [{mark}] {r['check']}: {r['detail']}")

    print(f"저장 완료: {HOLIDAY_WEEKLY_PATH}")
    print(f"저장 완료: {ALIGNMENT_AUDIT_PATH}")


if __name__ == "__main__":
    main()
