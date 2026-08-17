"""
exog_availability_audit.py

Day 1 Step 5: P22 최종 외생변수의 temporal leakage 최소 필수 검증.

검사:
    - forecast origin 정의
    - 공휴일 feature의 target 주차 정렬
    - 날씨 feature의 origin 주차 정렬
    - covid_flag의 week_st 기준 생성 여부
    - CCSI/CPI feature의 과거 시점 참조 여부
    - future realized value 사용 여부
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[4]
FEATURE_TABLE_PATH = BASE_DIR / "data" / "final_feature_table.parquet"
OUT_DIR = BASE_DIR / "data" / "ml" / "experiments" / "day1_audit"

CENTER_COL, WEEK_COL = "center_id", "week_st"
HORIZONS = [1, 2, 4]

sys.path.insert(0, str(BASE_DIR / "src" / "ml" / "feature_engineering"))
sys.path.insert(0, str(BASE_DIR / "src" / "ml" / "feature_engineering" / "economic_indicators"))

from feature_external_interaction_concat import KOREAN_LUNAR_HOLIDAYS  # noqa: E402
from econ_indicator_screening import shift_year_month  # noqa: E402
from build_economic_features import FINAL_SPECS  # noqa: E402

P22_ECONOMIC_FEATURES = {"ccsi_lag_m1", "cpi_y1_prev", "cpi_y2_prev_yoy"}
NEEDED_COLS = [
    CENTER_COL, WEEK_COL, "평균온도", "총강수량", "강수량_호우_count",
    "temp_x_precip", "center_temp_inter", "center_is_B", "covid_flag", "covid_영향여부",
] + [f"target_h{h}_공휴일_{w}" for h in HORIZONS for w in ("W0", "W-1", "W+1")]


def load_table() -> pd.DataFrame:
    df = pd.read_parquet(FEATURE_TABLE_PATH, columns=NEEDED_COLS)
    print(f"[로드] {FEATURE_TABLE_PATH} -> {len(df):,}행")
    return df


def read_source(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def forecast_origin_check() -> dict:
    split_path = BASE_DIR / "src" / "ml" / "feature_engineering" / "split_train_val_test.py"
    src = read_source(split_path)
    qty_weekly_ok = (
        'RAW_QTY_COL = "총판매수량"' in src
        and "group_cols = [RAW_WEEK_COL, RAW_CENTER_COL] + RAW_SKU_PARTS" in src
        and 'agg_map = {c: "sum" for c in SUM_COLS' in src
    )
    weather_origin_merge_ok = (
        "sku_week_weather = load_sku_week_weather" in src
        and "reindexed.merge(sku_week_weather, on=[CENTER_COL, SKU_COL, WEEK_COL]" in src
    )
    return {
        "qty_weekly_aggregation_ok": qty_weekly_ok,
        "weather_origin_merge_ok": weather_origin_merge_ok,
        "status": "PASS" if qty_weekly_ok and weather_origin_merge_ok else "REVIEW",
        "evidence": f"qty week_st 집계={qty_weekly_ok}, weather week_st merge={weather_origin_merge_ok}",
    }


def holiday_check(df: pd.DataFrame) -> list[dict]:
    holiday_dates = pd.Series(KOREAN_LUNAR_HOLIDAYS)
    holiday_weeks = set(holiday_dates - pd.to_timedelta(holiday_dates.dt.weekday, unit="D"))
    rows = []

    for h in HORIZONS:
        target_date = df[WEEK_COL] + pd.Timedelta(weeks=h)
        expected = {
            "W0": target_date.isin(holiday_weeks).astype(int),
            "W-1": (target_date - pd.Timedelta(weeks=1)).isin(holiday_weeks).astype(int),
            "W+1": (target_date + pd.Timedelta(weeks=1)).isin(holiday_weeks).astype(int),
        }
        for window, exp in expected.items():
            col = f"target_h{h}_공휴일_{window}"
            n_mismatch = int((df[col] != exp).sum())
            rows.append({
                "feature": col,
                "group": "holiday",
                "horizon": f"h{h}",
                "reference_period": f"target_date=week_st+{h}주 기준 {window}",
                "future_realized_value_used": False,
                "evidence": f"순수 달력 재계산 mismatch={n_mismatch}건",
                "status": "PASS" if n_mismatch == 0 else "FAIL",
            })
    return rows


def weather_check(df: pd.DataFrame, weather_origin_merge_ok: bool) -> list[dict]:
    base_status = "PASS" if weather_origin_merge_ok else "REVIEW"
    rows = [{
        "feature": col,
        "group": "weather",
        "horizon": "ALL",
        "reference_period": "week_st(origin 주) 관측치",
        "future_realized_value_used": False,
        "evidence": "weather merge key=[center_id, sku_id, week_st]",
        "status": base_status,
    } for col in ["평균온도", "총강수량", "강수량_호우_count"]]

    checks = {
        "temp_x_precip": np.isclose(
            df["temp_x_precip"], df["평균온도"] * df["총강수량"], equal_nan=True
        ),
        "center_temp_inter": np.isclose(
            df["center_temp_inter"], df["center_is_B"] * df["평균온도"], equal_nan=True
        ),
    }
    refs = {
        "temp_x_precip": "평균온도*총강수량(origin 주)",
        "center_temp_inter": "center_is_B*평균온도(origin 주)",
    }
    for col, match in checks.items():
        n_mismatch = int((~match).sum())
        rows.append({
            "feature": col,
            "group": "weather",
            "horizon": "ALL",
            "reference_period": refs[col],
            "future_realized_value_used": False,
            "evidence": f"재계산 mismatch={n_mismatch}건",
            "status": "FAIL" if n_mismatch > 0 else base_status,
        })
    return rows


def covid_check(df: pd.DataFrame) -> list[dict]:
    """week_st 기반 COVID feature 생성 경로와 최종 파생값을 확인."""
    feature_src = read_source(
        BASE_DIR / "src" / "ml" / "feature_engineering" / "feature_external_interaction_concat.py"
    )
    split_src = read_source(
        BASE_DIR / "src" / "ml" / "feature_engineering" / "split_train_val_test.py"
    )

    derived_ok = "covid_flag" in feature_src and "covid_영향여부" in feature_src and "astype(int)" in feature_src
    weekly_ok = "CALENDAR_FIRST_COLS" in split_src and "covid_영향여부" in split_src and "RAW_WEEK_COL" in split_src
    n_mismatch = int((df["covid_flag"] != df["covid_영향여부"].astype(int)).sum())

    status = "FAIL" if n_mismatch > 0 else ("PASS" if derived_ok and weekly_ok else "REVIEW")
    return [{
        "feature": "covid_flag",
        "group": "covid",
        "horizon": "ALL",
        "reference_period": "week_st 기준 covid_영향여부 -> covid_flag",
        "future_realized_value_used": False,
        "evidence": f"week_st 생성 경로={weekly_ok}, flag 파생={derived_ok}, mismatch={n_mismatch}건",
        "status": status,
    }]


def economic_check(df: pd.DataFrame) -> list[dict]:
    specs = {
        col: (indicator, y_off, m_off, use_yoy)
        for col, indicator, y_off, m_off, use_yoy in FINAL_SPECS
        if col in P22_ECONOMIC_FEATURES
    }
    year_months = pd.DataFrame({
        "year": df[WEEK_COL].dt.year,
        "month": df[WEEK_COL].dt.month,
    }).drop_duplicates()

    rows = []
    for col in sorted(P22_ECONOMIC_FEATURES):
        if col not in specs:
            rows.append({
                "feature": col,
                "group": "economy",
                "horizon": "ALL",
                "reference_period": "FINAL_SPECS에서 확인 불가",
                "future_realized_value_used": False,
                "evidence": "P22 feature spec 누락",
                "status": "REVIEW",
            })
            continue

        indicator, y_off, m_off, use_yoy = specs[col]
        n_not_past = 0
        for year, month in zip(year_months["year"], year_months["month"]):
            ref_ym = shift_year_month(year + y_off, month, m_off)
            if ref_ym >= (year, month):
                n_not_past += 1

        rows.append({
            "feature": col,
            "group": "economy",
            "horizon": "ALL",
            "reference_period": f"{indicator}(연오프셋={y_off}, 월오프셋={m_off}, YoY={use_yoy})",
            "future_realized_value_used": n_not_past > 0,
            "evidence": f"origin 연월 이상(당월/미래) 참조={n_not_past}건",
            "status": "PASS" if n_not_past == 0 else "FAIL",
        })
    return rows


def group_gate(audit_df: pd.DataFrame, group: str) -> str:
    sub = audit_df[audit_df["group"] == group]
    if sub.empty:
        return "REVIEW"
    if sub["status"].eq("FAIL").any():
        return "FAIL"
    if sub["status"].eq("REVIEW").any():
        return "REVIEW"
    return "PASS"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_table()
    origin = forecast_origin_check()

    audit_df = pd.DataFrame(
        holiday_check(df)
        + weather_check(df, origin["weather_origin_merge_ok"])
        + covid_check(df)
        + economic_check(df)
    )
    audit_df.to_csv(OUT_DIR / "exog_temporal_audit.csv", index=False, encoding="utf-8-sig")

    gates = {
        "forecast_origin_gate": origin["status"],
        "holiday_gate": group_gate(audit_df, "holiday"),
        "weather_gate": group_gate(audit_df, "weather"),
        "covid_gate": group_gate(audit_df, "covid"),
        "economy_temporal_gate": group_gate(audit_df, "economy"),
    }
    values = list(gates.values())
    gates["exog_temporal_gate"] = "FAIL" if "FAIL" in values else ("REVIEW" if "REVIEW" in values else "PASS")

    pd.DataFrame([{"metric": k, "value": v} for k, v in gates.items()]).to_csv(
        OUT_DIR / "exog_temporal_gate.csv", index=False, encoding="utf-8-sig"
    )

    print(origin["evidence"])
    for k, v in gates.items():
        print(f"{k}: {v}")
    print(f"future_realized_value_used=True: {int(audit_df['future_realized_value_used'].sum())}개")


if __name__ == "__main__":
    main()
