"""
build_economic_features.py

설명: 최종 선정된 경제지표 3개를 전체 feature table에 추가함.
- train 데이터 기반 스크리닝에서 선정된 CCSI/CPI 파생변수 3개를 생성함.
- week_st의 연·월을 기준으로 경제지표를 매핑하고 기존 feature_table_final과 병합함.
- 경제지표 시점 적용이 올바른지 조인 키, 결측치, 리키지 규칙을 검증함.
- 결과를 feature_table_with_econ.parquet으로 저장함.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from econ_indicator_screening import (
    CCSI_PATH,
    CPI_PATH,
    CENTER_COL,
    WEEK_COL,
    load_monthly_series,
    shift_year_month,
    yoy_pct_change,
)

BASE_DIR = Path(__file__).resolve().parents[4]
FEATURE_TABLE_PATH = BASE_DIR / "data" / "ml" / "splits" / "feature_table_final.parquet"

OUT_DIR = BASE_DIR / "data" / "ml" / "experiments" / "economic_indicators"
OUT_PATH = OUT_DIR / "feature_table_with_econ.parquet"

# (컬럼명, 소스지표, 연오프셋, 월오프셋, YoY변환여부)
# 예: cpi_y1_prev -> 기준연도 = 행의 연도-1, 기준월 = 행의 월, 그 기준에서 -1개월(전월)
FINAL_SPECS = [
    ("ccsi_lag_m1", "ccsi", 0, -1, False),
    ("cpi_y1_prev", "cpi", -1, -1, False),
    ("cpi_y2_prev_yoy", "cpi", -2, -1, True),
]


def build_lookup_series() -> dict:
    cpi_raw = load_monthly_series(CPI_PATH, "소비자물가지수")
    ccsi_raw = load_monthly_series(CCSI_PATH, "소비자심리지수")
    return {
        "cpi": cpi_raw,
        "ccsi": ccsi_raw,
        "cpi_yoy": yoy_pct_change(cpi_raw),
        "ccsi_yoy": yoy_pct_change(ccsi_raw),
    }


def build_candidate_features(year_months: pd.DataFrame, series_map: dict) -> pd.DataFrame:
    """확정된 3개 컬럼만 생성(20개 전체 생성하던 스크리닝판 build_candidate_features의
    축소판 — 필요한 (컬럼, 소스, 오프셋) 3개만 순회)."""
    out = {"year": year_months["year"], "month": year_months["month"]}
    for col, indicator, year_off, month_off, use_yoy in FINAL_SPECS:
        key = f"{indicator}_yoy" if use_yoy else indicator
        lookup = series_map[key].to_dict()
        vals = []
        for year, month in zip(year_months["year"], year_months["month"]):
            base_year = year + year_off
            y, m = shift_year_month(base_year, month, month_off)
            vals.append(lookup.get((y, m), np.nan))
        out[col] = vals
    return pd.DataFrame(out)


def verify_join_key(df: pd.DataFrame) -> None:
    """week_st가 걸쳐있는 달이 모호한 주가 있는지 확인 — '월' 컬럼(기존 파이프라인이
    쓰던 calendar month)과 week_st.dt.month가 항상 일치하는지로 검증한다."""
    print("[조인 키 검증] week_st 기준 연/월 vs 기존 '월' 컬럼 일치 여부")
    check = df[[WEEK_COL, "월"]].drop_duplicates(subset=[WEEK_COL]).copy()
    check["calendar_month"] = check[WEEK_COL].dt.month
    mismatch = check[check["월"] != check["calendar_month"]]
    print(f"  고유 week_st 수: {len(check):,}  불일치: {len(mismatch)}건")
    if len(mismatch) > 0:
        raise ValueError(
            f"week_st 기준 연/월과 기존 '월' 컬럼이 불일치하는 주가 {len(mismatch)}건 있음 "
            "— 월초/월말 경계 주 처리 방식을 먼저 확인해야 함(임의로 진행하지 않음).\n"
            f"{mismatch.head(10)}"
        )
    print("  -> 모호한 경계 주 없음. week_st.dt.year/month를 그대로 조인 키로 사용.")


def report_missing_by_center_year(merged: pd.DataFrame) -> dict:
    print()
    print("[결측치 리포트] 신규 컬럼 3개 x 센터 x 연도")
    merged = merged.copy()
    merged["year"] = merged[WEEK_COL].dt.year
    total_na_by_col = {}
    for col, _indicator, year_off, _month_off, use_yoy in FINAL_SPECS:
        print(f"\n  - {col} (연오프셋={year_off}, YoY변환={use_yoy})")
        g = (
            merged.assign(is_na=merged[col].isna())
            .groupby([CENTER_COL, "year"], observed=True)["is_na"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "결측행수", "count": "전체행수"})
        )
        g = g[g["전체행수"] > 0]
        print(g.to_string())
        total_na = int(merged[col].isna().sum())
        total_na_by_col[col] = total_na
        print(f"    총 결측: {total_na:,} / {len(merged):,}")
    return total_na_by_col


def explain_missing_reason(total_na_by_col: dict) -> None:
    print()
    print("[결측 원인 구분]")
    print("  - ccsi_lag_m1, cpi_y1_prev: 당해년도 성분이 없거나(작년 기준) 필요한 최소")
    print("    연월이 2018-12 이상이라 전체 데이터 범위(2021~2024)에서 이론상 결측 0건이어야 함.")
    print("  - cpi_y2_prev_yoy: 재작년(year-2) 동월의 전월을 '전년동월대비 증감률'로 변환하므로,")
    print("    그 기준월이 원본 CPI 시계열의 시작 이전(YoY 계산용 -12개월 시점 포함)으로")
    print("    떨어지는 경우에만 결측이 생긴다 — 리키지 회피와 무관한 '데이터 자체 미보유' 결측.")
    na = total_na_by_col.get("cpi_y2_prev_yoy", 0)
    if na == 0:
        print("    현재 consumer_price_index_national.csv가 2017-12부터 포함되어 있어(재정제 완료),")
        print("    이 데이터 범위(2021~2024)에서 필요한 가장 이른 시점(2018-12 YoY, 기준 2017-12)까지")
        print("    전부 존재 -> 실측 결측 0건.")
    else:
        print(f"    현재 실측 결측 {na:,}건 — 원본 CPI 시계열이 필요한 최소 연월까지 커버하지 못함.")


def sample_leakage_checks(merged: pd.DataFrame, series_map: dict) -> None:
    print()
    print("[리키지 규칙 샘플 대조] 몇 개 (연,월) 샘플을 직접 계산해 파이프라인 결과와 대조")
    samples = [
        (2024, 12, "A"), (2024, 1, "B"), (2021, 1, "A"), (2023, 6, "A"),
    ]
    for year, month, center in samples:
        rows = merged[
            (merged[WEEK_COL].dt.year == year)
            & (merged[WEEK_COL].dt.month == month)
            & (merged[CENTER_COL] == center)
        ]
        if len(rows) == 0:
            print(f"  {center} {year}-{month:02d}: 해당 센터에 이 연/월 데이터 없음(스킵)")
            continue
        row = rows.iloc[0]

        exp_ccsi_lag_m1_ym = shift_year_month(year, month, -1)
        exp_ccsi_lag_m1 = series_map["ccsi"].to_dict().get(exp_ccsi_lag_m1_ym, np.nan)

        exp_cpi_y1_prev_ym = shift_year_month(year - 1, month, -1)
        exp_cpi_y1_prev = series_map["cpi"].to_dict().get(exp_cpi_y1_prev_ym, np.nan)

        exp_cpi_y2_prev_base_ym = shift_year_month(year - 2, month, -1)
        exp_cpi_y2_prev_yoy = series_map["cpi_yoy"].to_dict().get(exp_cpi_y2_prev_base_ym, np.nan)

        def eq(a, b):
            if pd.isna(a) and pd.isna(b):
                return True
            if pd.isna(a) or pd.isna(b):
                return False
            return abs(a - b) < 1e-9

        ok1 = eq(row["ccsi_lag_m1"], exp_ccsi_lag_m1)
        ok2 = eq(row["cpi_y1_prev"], exp_cpi_y1_prev)
        ok3 = eq(row["cpi_y2_prev_yoy"], exp_cpi_y2_prev_yoy)

        print(f"  {center} {year}-{month:02d} (기준월 -> lag_m1 조회월 {exp_ccsi_lag_m1_ym}):")
        print(
            f"    ccsi_lag_m1: 파이프라인={row['ccsi_lag_m1']}  수동계산={exp_ccsi_lag_m1}  "
            f"{'OK' if ok1 else 'MISMATCH'}"
        )
        print(
            f"    cpi_y1_prev: 파이프라인={row['cpi_y1_prev']}  수동계산={exp_cpi_y1_prev}  "
            f"{'OK' if ok2 else 'MISMATCH'}"
        )
        print(
            f"    cpi_y2_prev_yoy: 파이프라인={row['cpi_y2_prev_yoy']}  수동계산={exp_cpi_y2_prev_yoy}  "
            f"{'OK' if ok3 else 'MISMATCH'}"
        )
        if not (ok1 and ok2 and ok3):
            raise AssertionError(f"{center} {year}-{month:02d} 샘플 대조 실패")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] feature_table_final.parquet 전체 로드 (split 필터링 없음, 전 컬럼)")
    df = pd.read_parquet(FEATURE_TABLE_PATH)
    print(f"  전체 행수: {len(df):,}  컬럼수: {df.shape[1]}")
    print(f"  split 분포:\n{df['split'].value_counts().to_string()}")

    verify_join_key(df)

    print("\n[2/5] 연/월 조인 키 생성 (week_st 기준)")
    df["year"] = df[WEEK_COL].dt.year
    df["month"] = df[WEEK_COL].dt.month
    year_months = df[["year", "month"]].drop_duplicates().reset_index(drop=True)
    print(f"  고유 (연,월) 조합: {len(year_months)}개")

    print("\n[3/5] CPI/CCSI 원본+YoY 로드 및 3개 확정 컬럼 계산")
    series_map = build_lookup_series()
    ym_features = build_candidate_features(year_months, series_map)

    print("\n[4/5] feature_table_final과 병합")
    merged = df.merge(ym_features, on=["year", "month"], how="left")
    merged = merged.drop(columns=["year", "month"])
    assert len(merged) == len(df), "병합 후 행수가 달라짐 — 조인 키 중복 의심"

    total_na_by_col = report_missing_by_center_year(merged)
    explain_missing_reason(total_na_by_col)
    sample_leakage_checks(merged, series_map)

    print("\n[5/5] 저장")
    merged.to_parquet(OUT_PATH, index=False)
    print(f"  -> {OUT_PATH} ({len(merged):,}행 x {merged.shape[1]}컬럼, 기존 대비 +3컬럼)")


if __name__ == "__main__":
    main()
