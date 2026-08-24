"""
arima_aggregation_level_eda.py
SARIMAX Top-down baseline을 위한 집계 계층(Aggregation Level) 탐색 EDA.

development_2021_2023.parquet 기준으로 4개 계층(Center / Center×대분류 /
Center×중분류 / Center×소분류)의 주간 총수요 시계열을 만들어 zero-inflation,
시계열 길이, 자기상관(계절성) 특성을 비교한다. 목적은 ARIMA 적합이 안정적으로
수렴하면서 계절성 신호를 과도하게 희석시키지 않는 "Goldilocks Level"을 데이터
근거로 고르는 것.

계층별 시계열 구성 규칙:
    - 그룹(예: center, center+대분류, ...)별로 소속 SKU들의 주간 qty를 합산.
    - 각 그룹의 관측 구간은 그 그룹에 속한 행들의 최초~최종 week_st 사이를
      주간 그리드로 reindex하고 결측 주는 0으로 채운다(집계 후 "그 주에 실제로
      전혀 안 팔림"과 "그룹이 아직 존재하지 않던 기간"을 구분하기 위함).
    - 시계열 길이 = 그 reindex된 주 수(그룹이 관측된 기간의 길이).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

BASE_DIR = Path(__file__).resolve().parents[4]
DEV_PATH = BASE_DIR / "data" / "development_2021_2023.parquet"

WEEK_COL = "week_st"
QTY_COL = "qty"
SHORT_SERIES_THRESHOLD = 52
LAG1 = 1
LAG52 = 52
MIN_LEN_FOR_LAG1 = 10
MIN_LEN_FOR_LAG52 = 104  # lag52를 신뢰성 있게 보려면 최소 2주기(=104주) 이상 요구

def build_group_series(df: pd.DataFrame, group_cols: list[str]) -> dict:
    """group_cols 기준으로 주간 총수요를 합산하고, 그룹별로 자기 존재 구간(첫~끝
    관측 주) 안에서 주간 그리드로 reindex(결측 0)한 시계열 dict를 반환한다."""
    agg = (
        df.groupby(group_cols + [WEEK_COL])[QTY_COL]
        .sum()
        .reset_index()
    )

    series_by_group = {}
    for key, sub in agg.groupby(group_cols):
        sub = sub.sort_values(WEEK_COL)
        full_range = pd.date_range(sub[WEEK_COL].min(), sub[WEEK_COL].max(), freq="7D")
        s = sub.set_index(WEEK_COL)[QTY_COL].reindex(full_range, fill_value=0.0)
        series_by_group[key] = s
    return series_by_group


def acf_significant(s: pd.Series, lag: int, min_len: int) -> bool | None:
    """해당 lag의 표본 ACF가 Bartlett 근사 유의 경계(±1.96/sqrt(n))를 벗어나는지.
    시계열이 너무 짧거나 분산이 0(상수 시계열)이면 None(판정 불가)을 반환."""
    n = len(s)
    if n < min_len or s.std() == 0:
        return None
    try:
        vals = acf(s.values, nlags=lag, fft=True, missing="none")
    except Exception:
        return None
    if len(vals) <= lag:
        return None
    bound = 1.96 / np.sqrt(n)
    return abs(vals[lag]) > bound


def summarize_level(name: str, series_by_group: dict) -> dict:
    zero_ratios = []
    lens = []
    means = []
    lag1_flags = []
    lag52_flags = []

    for s in series_by_group.values():
        zero_ratios.append((s == 0).mean())
        lens.append(len(s))
        means.append(s.mean())
        lag1_flags.append(acf_significant(s, LAG1, MIN_LEN_FOR_LAG1))
        lag52_flags.append(acf_significant(s, LAG52, MIN_LEN_FOR_LAG52))

    zero_ratios = np.array(zero_ratios)
    lens = np.array(lens)
    means = np.array(means)

    lag1_valid = [f for f in lag1_flags if f is not None]
    lag52_valid = [f for f in lag52_flags if f is not None]

    return {
        "Level": name,
        "Group Count": len(series_by_group),
        "Mean Zero Ratio(%)": round(zero_ratios.mean() * 100, 1),
        "Median Zero Ratio(%)": round(np.median(zero_ratios) * 100, 1),
        "Zero Ratio<=10%(%)": round((zero_ratios <= 0.10).mean() * 100, 1),
        "Mean Weekly Demand": round(means.mean(), 2),
        "Short Series(<52w)(%)": round((lens < SHORT_SERIES_THRESHOLD).mean() * 100, 1),
        "Lag-1 유의 비율(%)": round(np.mean(lag1_valid) * 100, 1) if lag1_valid else np.nan,
        "Lag-1 판정가능 n": len(lag1_valid),
        "Lag-52 유의 비율(%)": round(np.mean(lag52_valid) * 100, 1) if lag52_valid else np.nan,
        "Lag-52 판정가능 n": len(lag52_valid),
    }


def main():
    df = pd.read_parquet(
        DEV_PATH,
        columns=["center_id", "sku_id", WEEK_COL, QTY_COL, "KAN_대분류", "KAN_중분류", "KAN_소분류"],
    )

    levels = [
        ("L1: Center", ["center_id"]),
        ("L2: Center x 대분류", ["center_id", "KAN_대분류"]),
        ("L3: Center x 중분류", ["center_id", "KAN_중분류"]),
        ("L4: Center x 소분류", ["center_id", "KAN_소분류"]),
    ]

    rows = []
    for name, cols in levels:
        series_by_group = build_group_series(df, cols)
        rows.append(summarize_level(name, series_by_group))

    report = pd.DataFrame(rows)

    cols_order = [
        "Level", "Group Count",
        "Mean Zero Ratio(%)", "Median Zero Ratio(%)", "Zero Ratio<=10%(%)",
        "Mean Weekly Demand", "Short Series(<52w)(%)",
        "Lag-1 유의 비율(%)", "Lag-1 판정가능 n",
        "Lag-52 유의 비율(%)", "Lag-52 판정가능 n",
    ]
    report = report[cols_order]

    print("=" * 100)
    print("[SARIMAX Top-down Baseline] 집계 계층별 시계열 특성 비교")
    print("=" * 100)

    # Markdown table
    header = "| " + " | ".join(cols_order) + " |"
    sep = "| " + " | ".join(["---"] * len(cols_order)) + " |"
    print(header)
    print(sep)
    for _, r in report.iterrows():
        print("| " + " | ".join(str(r[c]) for c in cols_order) + " |")

    out_path = BASE_DIR / "data" / "ml" / "experiments" / "arima_aggregation_level_eda.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
