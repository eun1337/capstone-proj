"""
04_prepare_arimax_full_exog.py

ARIMAX용 Economic, COVID, Holiday 외생변수를 준비한다.

S1 Economic, S2 COVID, S3 Holiday, S4 Operational-Full block을 구성하고,
forecast origin에서 실제 사용 가능한 시점 규칙에 맞춰
historical exog와 2024 t+1~t+4 future exog를 정렬한다.

시간 정렬, 결측, 중복 및 미래정보 누출 여부를 audit하며 모델 학습은 수행하지 않는다.
"""

from pathlib import Path
import importlib
import sys

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from statistical_utils import CENTER_COL, WEEK_COL, QTY_COL  # noqa: E402

m02 = importlib.import_module("02_run_arima_holdout")
m03 = importlib.import_module("03_prepare_arimax_exog")

CENTERS = m02.CENTERS
ORIGINS = m02.ORIGINS
A_HISTORY_START = m02.A_HISTORY_START
B_HISTORY_START = m02.B_HISTORY_START
HOLDOUT_MAX_WEEK = m02.HOLDOUT_MAX_WEEK
CENTER_START = {"A": A_HISTORY_START, "B": B_HISTORY_START}

HOLIDAY_WEEKLY_PATH = m03.HOLIDAY_WEEKLY_PATH  # 기존 파일 재사용, 재생성하지 않음
HOLIDAY_COLS = m03.HOLIDAY_COLS
get_future_exog_matrix = m03.get_future_exog_matrix
_audit_row = m03._audit_row

BASE_DIR = Path(__file__).resolve().parents[3]
DEV_PATH = BASE_DIR / "data" / "development_2021_2023.parquet"
HOLDOUT_PATH = BASE_DIR / "data" / "holdout_2024.parquet"
OUT_DIR = BASE_DIR / "data" / "ml" / "day2_statistical_models" / "exog"

HISTORICAL_EXOG_PATH = OUT_DIR / "arimax_historical_exog.parquet"
FUTURE_EXOG_PATH = OUT_DIR / "arimax_future_exog_2024.parquet"
AUDIT_PATH = OUT_DIR / "arimax_exog_alignment_audit.csv"

ECONOMIC_COLS = ["ccsi_lag_m1", "cpi_y1_prev", "cpi_y2_prev_yoy"]
COVID_COLS = ["covid_flag"]
EXOG_BLOCKS = {
    "S0": [],
    "S1": ECONOMIC_COLS,
    "S2": COVID_COLS,
    "S3": HOLIDAY_COLS,
    "S4": ECONOMIC_COLS + COVID_COLS + HOLIDAY_COLS,
}
MAIN_BLOCK = "S4"


def load_combined_demand() -> pd.DataFrame:
    cols = [CENTER_COL, WEEK_COL, QTY_COL]
    dev = pd.read_parquet(DEV_PATH, columns=cols)
    hold = pd.read_parquet(HOLDOUT_PATH, columns=cols)
    return pd.concat([dev, hold], ignore_index=True)


def load_calendar_exog(week_universe: pd.Series) -> pd.DataFrame:
    """dev+holdout에 이미 있는 calendar-level economic/covid 컬럼을 그대로 가져온다.
    A/B가 같은 week_st에서 동일 값을 갖는지 검증한 뒤 하나로 합친다."""
    cols = [CENTER_COL, WEEK_COL] + ECONOMIC_COLS + COVID_COLS
    dev = pd.read_parquet(DEV_PATH, columns=cols)
    hold = pd.read_parquet(HOLDOUT_PATH, columns=cols)
    df = pd.concat([dev, hold], ignore_index=True).drop_duplicates([CENTER_COL, WEEK_COL])

    nunique_per_week = df.groupby(WEEK_COL, observed=True)[ECONOMIC_COLS + COVID_COLS].nunique(dropna=False)
    bad_weeks = nunique_per_week[(nunique_per_week > 1).any(axis=1)]
    if len(bad_weeks):
        raise ValueError(f"economic/covid 값이 center별로 다른 week_st 존재: {bad_weeks.index.tolist()[:5]}")

    calendar = (
        df.drop_duplicates(WEEK_COL)[[WEEK_COL] + ECONOMIC_COLS + COVID_COLS]
        .sort_values(WEEK_COL)
        .reset_index(drop=True)
    )
    missing = set(pd.to_datetime(week_universe)) - set(calendar[WEEK_COL])
    if missing:
        raise ValueError(f"calendar exog에 없는 week_st: {sorted(missing)[:5]}")
    return calendar


def build_historical_exog(week_universe: pd.Series, calendar: pd.DataFrame, holiday_weekly: pd.DataFrame) -> pd.DataFrame:
    """A: 2021-01-04~2024-12-30, B: 2023-07-03~2024-12-30 전 구간을 정렬해둔다. 실제 ARIMAX
    fitting cutoff별 슬라이싱(week_st<=cutoff)은 이후 holdout runner가 수행한다."""
    parts = []
    for center_id, start in CENTER_START.items():
        weeks = week_universe[(week_universe >= start) & (week_universe <= HOLDOUT_MAX_WEEK)].sort_values()
        parts.append(pd.DataFrame({CENTER_COL: center_id, WEEK_COL: weeks.to_numpy()}))
    hist = pd.concat(parts, ignore_index=True)

    hist = hist.merge(calendar, on=WEEK_COL, how="left", validate="many_to_one")
    hist = hist.merge(holiday_weekly, on=WEEK_COL, how="left", validate="many_to_one")
    return hist.sort_values([CENTER_COL, WEEK_COL]).reset_index(drop=True)


def get_future_exog_matrix_full(calendar: pd.DataFrame, holiday_weekly: pd.DataFrame,
                                 forecast_origin: pd.Timestamp, n_steps: int = 4) -> pd.DataFrame:
    forecast_origin = pd.Timestamp(forecast_origin)
    origin_row = calendar.loc[calendar[WEEK_COL] == forecast_origin]
    if origin_row.empty:
        raise ValueError(f"calendar exog에 forecast_origin이 없음: {forecast_origin.date()}")
    origin_ccsi = float(origin_row["ccsi_lag_m1"].iloc[0])

    step_weeks = [forecast_origin + pd.Timedelta(weeks=k) for k in range(1, n_steps + 1)]
    calendar_lookup = calendar.set_index(WEEK_COL)
    holiday_lookup = holiday_weekly.set_index(WEEK_COL)

    rows = []
    for step, w in enumerate(step_weeks, start=1):
        available = w in calendar_lookup.index and w in holiday_lookup.index
        if not available:
            rows.append({
                "step": step, WEEK_COL: w, "exog_available": False,
                "ccsi_lag_m1": np.nan, "cpi_y1_prev": np.nan, "cpi_y2_prev_yoy": np.nan, "covid_flag": np.nan,
                "공휴일_W0": np.nan, "공휴일_W-1": np.nan, "공휴일_W+1": np.nan,
            })
            continue
        cal_row = calendar_lookup.loc[w]
        hol_row = holiday_lookup.loc[w]
        rows.append({
            "step": step, WEEK_COL: w, "exog_available": True,
            "ccsi_lag_m1": origin_ccsi,
            "cpi_y1_prev": float(cal_row["cpi_y1_prev"]),
            "cpi_y2_prev_yoy": float(cal_row["cpi_y2_prev_yoy"]),
            "covid_flag": int(cal_row["covid_flag"]),
            "공휴일_W0": int(hol_row["공휴일_W0"]),
            "공휴일_W-1": int(hol_row["공휴일_W-1"]),
            "공휴일_W+1": int(hol_row["공휴일_W+1"]),
        })
    return pd.DataFrame(rows)


def build_future_exog_all_origins(calendar: pd.DataFrame, holiday_weekly: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for center_id in CENTERS:
        for origin in ORIGINS:
            m = get_future_exog_matrix_full(calendar, holiday_weekly, origin)
            m.insert(0, CENTER_COL, center_id)
            m.insert(1, "forecast_origin", origin)
            frames.append(m)
    return pd.concat(frames, ignore_index=True)


def run_audit(historical_exog: pd.DataFrame, future_exog: pd.DataFrame,
              calendar: pd.DataFrame, holiday_weekly: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    rows = []
    avail = future_exog["exog_available"]
    exog_cols = ECONOMIC_COLS + COVID_COLS + HOLIDAY_COLS

    dup_cal = int(calendar[WEEK_COL].duplicated().sum())
    rows.append(_audit_row("calendar_week_st_unique", dup_cal == 0, f"중복 {dup_cal}건"))

    dup_hist = int(historical_exog.duplicated(subset=[CENTER_COL, WEEK_COL]).sum())
    rows.append(_audit_row("historical_week_st_unique", dup_hist == 0, f"중복 {dup_hist}건"))

    step_ok = future_exog.groupby([CENTER_COL, "forecast_origin"])["step"].apply(lambda s: sorted(s.tolist()) == [1, 2, 3, 4])
    bad_steps = int((~step_ok).sum())
    rows.append(_audit_row("future_steps_1to4_complete", bad_steps == 0, f"step 1~4 누락/중복 조합 {bad_steps}건"))

    expected_target = future_exog["forecast_origin"] + future_exog["step"].apply(lambda s: pd.Timedelta(weeks=s))
    bad_target = int((future_exog[WEEK_COL] != expected_target).sum())
    rows.append(_audit_row("target_week_formula", bad_target == 0, f"target_week != forecast_origin+step주 {bad_target}건"))

    n_na_hist = int(historical_exog[exog_cols].isna().to_numpy().sum())
    rows.append(_audit_row("historical_no_unexpected_nan", n_na_hist == 0, f"NaN {n_na_hist}건"))
    n_inf_hist = int(np.isinf(historical_exog[exog_cols].to_numpy(dtype=float)).sum())
    rows.append(_audit_row("historical_no_inf", n_inf_hist == 0, f"inf {n_inf_hist}건"))

    n_unexpected_na_future = int(future_exog.loc[avail, exog_cols].isna().to_numpy().sum())
    n_masked_rows = int((~avail).sum())
    n_na_in_masked = int(future_exog.loc[~avail, exog_cols].isna().to_numpy().sum())
    rows.append(_audit_row(
        "future_no_unexpected_nan", n_unexpected_na_future == 0,
        f"exog_available=True인데 NaN {n_unexpected_na_future}건 "
        f"(exog_available=False {n_masked_rows}행 중 NaN {n_na_in_masked}/{n_masked_rows * len(exog_cols)}은 정상)",
    ))
    n_inf_future = int(np.isinf(future_exog.loc[avail, exog_cols].to_numpy(dtype=float)).sum())
    rows.append(_audit_row("future_no_inf", n_inf_future == 0, f"inf {n_inf_future}건"))

    for center_id, start in CENTER_START.items():
        demand_weeks = set(demand.loc[
            (demand[CENTER_COL] == center_id) & (demand[WEEK_COL] >= start) & (demand[WEEK_COL] <= HOLDOUT_MAX_WEEK),
            WEEK_COL,
        ])
        hist_weeks = set(historical_exog.loc[historical_exog[CENTER_COL] == center_id, WEEK_COL])
        ok = demand_weeks == hist_weeks
        rows.append(_audit_row(
            f"y_week_alignment_{center_id}", ok,
            f"demand={len(demand_weeks)}주 historical_exog={len(hist_weeks)}주 "
            f"차이={len(demand_weeks.symmetric_difference(hist_weeks))}건",
        ))

    calendar_lookup = calendar.set_index(WEEK_COL)
    ccsi_nunique = future_exog.loc[avail].groupby([CENTER_COL, "forecast_origin"])["ccsi_lag_m1"].nunique()
    n_ccsi_multi = int((ccsi_nunique > 1).sum())
    origin_ccsi_expected = future_exog["forecast_origin"].map(calendar_lookup["ccsi_lag_m1"])
    n_ccsi_mismatch = int((future_exog.loc[avail, "ccsi_lag_m1"] != origin_ccsi_expected.loc[avail]).sum())
    rows.append(_audit_row(
        "ccsi_origin_flat", n_ccsi_multi == 0 and n_ccsi_mismatch == 0,
        f"origin당 값 2종 이상 {n_ccsi_multi}건, origin 값과 불일치 {n_ccsi_mismatch}건",
    ))

    for col in ["cpi_y1_prev", "cpi_y2_prev_yoy"]:
        expected = future_exog[WEEK_COL].map(calendar_lookup[col])
        both_nan = future_exog[col].isna() & expected.isna()
        mismatch = avail & ~both_nan & (future_exog[col] != expected)
        rows.append(_audit_row(f"{col}_target_week_match", int(mismatch.sum()) == 0, f"불일치 {int(mismatch.sum())}건"))

    expected_covid = future_exog[WEEK_COL].map(calendar_lookup["covid_flag"])
    mismatch_covid = avail & (future_exog["covid_flag"] != expected_covid)
    rows.append(_audit_row("covid_flag_target_week_match", int(mismatch_covid.sum()) == 0, f"불일치 {int(mismatch_covid.sum())}건"))

    holiday_lookup = holiday_weekly.set_index(WEEK_COL)
    for col in HOLIDAY_COLS:
        expected = future_exog[WEEK_COL].map(holiday_lookup[col])
        mismatch = avail & (future_exog[col] != expected)
        rows.append(_audit_row(f"{col}_target_week_match", int(mismatch.sum()) == 0, f"불일치 {int(mismatch.sum())}건"))

    a_bad = historical_exog[(historical_exog[CENTER_COL] == "A") & (historical_exog[WEEK_COL] < A_HISTORY_START)]
    rows.append(_audit_row("a_history_start_respected", a_bad.empty, f"A센터 {A_HISTORY_START.date()} 이전 행 {len(a_bad)}건"))
    b_bad = historical_exog[(historical_exog[CENTER_COL] == "B") & (historical_exog[WEEK_COL] < B_HISTORY_START)]
    rows.append(_audit_row("b_history_start_respected", b_bad.empty, f"B센터 {B_HISTORY_START.date()} 이전 행 {len(b_bad)}건"))

    leak_2025 = future_exog[avail & (future_exog[WEEK_COL] > HOLDOUT_MAX_WEEK)]
    rows.append(_audit_row("no_2025_leak_in_future", leak_2025.empty, f"exog_available=True인데 {HOLDOUT_MAX_WEEK.date()} 이후 주 {len(leak_2025)}건"))

    missing_hist_cols = set(EXOG_BLOCKS["S4"]) - set(historical_exog.columns)
    rows.append(_audit_row("s4_columns_present_historical", not missing_hist_cols, f"누락 컬럼 {sorted(missing_hist_cols)}"))
    missing_future_cols = set(EXOG_BLOCKS["S4"]) - set(future_exog.columns)
    rows.append(_audit_row("s4_columns_present_future", not missing_future_cols, f"누락 컬럼 {sorted(missing_future_cols)}"))

    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    week_universe = m03.load_week_universe()
    holiday_weekly = pd.read_parquet(HOLIDAY_WEEKLY_PATH)
    calendar = load_calendar_exog(week_universe)
    demand = load_combined_demand()

    historical_exog = build_historical_exog(week_universe, calendar, holiday_weekly)
    historical_exog.to_parquet(HISTORICAL_EXOG_PATH, index=False)

    future_exog = build_future_exog_all_origins(calendar, holiday_weekly)
    future_exog.to_parquet(FUTURE_EXOG_PATH, index=False)

    audit = run_audit(historical_exog, future_exog, calendar, holiday_weekly, demand)
    audit.to_csv(AUDIT_PATH, index=False)

    n_a = int((historical_exog[CENTER_COL] == "A").sum())
    n_b = int((historical_exog[CENTER_COL] == "B").sum())
    n_fail = int((~audit["passed"]).sum())
    print(f"[historical exog] {len(historical_exog):,}행 (A {n_a:,} / B {n_b:,})")
    print(f"[future exog] {len(future_exog):,}행 ({len(CENTERS)}센터 x {len(ORIGINS)}origin x 4step)")
    print(f"[Audit] 총 {len(audit)}건 중 실패 {n_fail}건")
    for _, r in audit.iterrows():
        mark = "OK" if r["passed"] else "FAIL"
        print(f"  [{mark}] {r['check']}: {r['detail']}")

    print(f"저장 완료: {HISTORICAL_EXOG_PATH}")
    print(f"저장 완료: {FUTURE_EXOG_PATH}")
    print(f"저장 완료: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
