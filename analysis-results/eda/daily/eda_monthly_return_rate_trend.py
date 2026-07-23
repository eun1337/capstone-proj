"""
Step 3 후속: 24년 반품률 증가(학습기간 2.65% -> 전체 3.72%)가 특정 시점에 급증한 것인지,
아니면 전반적인 추세인지 확인하기 위한 월별/분기별 반품률 추이.

- 정제 파일(거래 단위)에서 년/월별로 반품률 = sum(반품수량)/sum(총판매수량) 산출.
- 24년 각 월의 반품률을 21-23년 같은 달의 평균과 비교해서(계절성 통제),
  단순 추세인지 특정 월에 튄 것인지 확인.
- 수량 부호 반전이 의심되는 상품(step3_anomalous_product_candidates.csv, 175건)은
  집계 파이프라인과 동일하게 제외하고 계산한다.
"""

import pandas as pd

SRC_PATH = "data/final/cleaned_main_joined_cleaned_for_pred.parquet"
EXCLUDE_PRODUCTS_PATH = "analysis-results/preprocessing/step3_anomalous_product_candidates.csv"
OUT_MONTHLY_PATH = "analysis-results/eda/step3_monthly_return_rate_trend.csv"
OUT_YOY_PATH = "analysis-results/eda/step3_2024_vs_2123_baseline_by_month.csv"

PRODUCT_KEY_COLS = ["바코드", "옵션코드", "상품클러스터"]


def main() -> None:
    print(f"Loading {SRC_PATH} ...")
    df = pd.read_parquet(SRC_PATH)
    print(f"Loaded {len(df):,} rows")

    exclude = pd.read_csv(EXCLUDE_PRODUCTS_PATH, dtype={"바코드": str})[PRODUCT_KEY_COLS]
    before = len(df)
    df = df.merge(exclude.drop_duplicates(), on=PRODUCT_KEY_COLS, how="left", indicator=True)
    df = df[df["_merge"] == "left_only"].drop(columns="_merge")
    print(f"수량 부호 반전 의심 상품 제외: {before - len(df):,}행 제거\n")

    df["판매수량"] = df["수량"].where(df["수량"] > 0, 0)
    df["반품수량_abs"] = (-df["수량"]).where(df["수량"] < 0, 0)

    monthly = (
        df.groupby(["년", "월"])
        .agg(판매수량합=("판매수량", "sum"), 반품수량합=("반품수량_abs", "sum"))
        .reset_index()
    )
    monthly["반품률(%)"] = (monthly["반품수량합"] / monthly["판매수량합"] * 100).round(4)

    quarterly = monthly.copy()
    quarterly["분기"] = ((quarterly["월"] - 1) // 3) + 1
    quarterly = (
        quarterly.groupby(["년", "분기"])
        .agg(판매수량합=("판매수량합", "sum"), 반품수량합=("반품수량합", "sum"))
        .reset_index()
    )
    quarterly["반품률(%)"] = (quarterly["반품수량합"] / quarterly["판매수량합"] * 100).round(4)

    print("=== 월별 반품률 추이 ===")
    print(monthly.to_string(index=False))
    print()
    print("=== 분기별 반품률 추이 ===")
    print(quarterly.to_string(index=False))
    print()

    # 24년 vs 21-23년 동월 평균 (계절성 통제 비교)
    baseline = (
        monthly[monthly["년"].isin([2021, 2022, 2023])]
        .groupby("월")
        .agg(반품수량합=("반품수량합", "sum"), 판매수량합=("판매수량합", "sum"))
    )
    baseline["21-23_동월_반품률(%)"] = (baseline["반품수량합"] / baseline["판매수량합"] * 100).round(4)

    y2024 = monthly[monthly["년"] == 2024].set_index("월")["반품률(%)"].rename("2024_반품률(%)")

    comparison = baseline[["21-23_동월_반품률(%)"]].join(y2024)
    comparison["증감(%p)"] = (comparison["2024_반품률(%)"] - comparison["21-23_동월_반품률(%)"]).round(4)
    comparison = comparison.reset_index()

    print("=== 24년 vs 21-23년 동월 평균 반품률 비교 (계절성 통제) ===")
    print(comparison.to_string(index=False))

    monthly.to_csv(OUT_MONTHLY_PATH, index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT_YOY_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료 -> {OUT_MONTHLY_PATH}")
    print(f"저장 완료 -> {OUT_YOY_PATH}")


if __name__ == "__main__":
    main()
