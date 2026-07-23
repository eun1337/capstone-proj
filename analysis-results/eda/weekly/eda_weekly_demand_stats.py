"""
Step 1: 주 단위 집계 데이터(aggregated_weekly_demand.parquet)에 대한 기술통계량 확인.

- 매출과 반품은 발생 빈도 자체가 크게 다르므로(반품은 전체 행의 75% 이상이 0)
  전체 행 기준으로 같이 통계를 내면 반품 쪽 분포가 0에 눌려 왜곡된다.
  따라서 매출/반품을 각각 "값이 실제로 발생한 행(>0)"만 걸러서 따로 분석한다.
- 순수량/순금액은 매출·반품 둘 다를 반영하는 종합 지표이므로 전체 행 기준으로 유지한다.
- 21-23년(향후 학습에 쓸 기간) 데이터로만 낸 통계와, 21-24년 전체 통계를 같이 낸다.
  21-24년 전체 수치를 그대로 전처리 기준(이상치 컷오프/변환 여부 등)으로 확정하면 24년(예측
  대상 기간) 정보가 전처리 설계에 섞여 들어가는 leakage가 되므로, 실제 기준은 21-23년 쪽을
  우선하고 21-24년 쪽은 "24년 들어 분포가 달라졌는지" 비교하는 참고용으로 쓴다.
- 결과는 매출/반품/순 지표 3개 파일로 나눠 저장하며, 각 파일에 기간(전체 / 21-23) 두 행씩 담는다.
"""

import pandas as pd

SRC_PATH = "data/final/aggregated_weekly_demand.parquet"
SALES_OUT_PATH = "analysis-results/eda/step1_sales_stats.csv"
RETURNS_OUT_PATH = "analysis-results/eda/step1_returns_stats.csv"
NET_OUT_PATH = "analysis-results/eda/step1_net_stats.csv"

SALES_COLS = ["총판매수량", "총판매금액"]
RETURNS_COLS = ["반품수량", "반품금액"]
NET_COLS = ["순수량", "순금액"]

PERCENTILES = [0.25, 0.5, 0.75, 0.9, 0.99]

# 21-23년 학습 기간과 24년 예측 대상 기간의 경계.
# 주시작일=2020-12-28 주는 실제 거래일이 2021-01-02(토)라 21-23년 쪽에 포함된다.
TRAIN_CUTOFF = "2024-01-01"


def describe_columns(df: pd.DataFrame, cols: list, positive_only: bool) -> pd.DataFrame:
    rows = []
    for col in cols:
        series = df.loc[df[col] > 0, col] if positive_only else df[col]

        desc = series.describe(percentiles=PERCENTILES)
        q3 = desc["75%"]
        row = {
            "평균": desc["mean"],
            "표준편차": desc["std"],
            "최소값": desc["min"],
            "Q1(25%)": desc["25%"],
            "중위수(50%)": desc["50%"],
            "Q3(75%)": q3,
            "P90": desc["90%"],
            "P99": desc["99%"],
            "최대값": desc["max"],
        }
        row["평균-중위수"] = row["평균"] - row["중위수(50%)"]
        row["Max/Q3_배수"] = (row["최대값"] / q3) if q3 else "N/A(Q3=0)"
        row["Skewness"] = series.skew()
        row["표본행수"] = len(series)
        row["전체행수"] = len(df)
        row["표본비율(%)"] = round(len(series) / len(df) * 100, 2)
        rows.append(pd.Series(row, name=col))

    result = pd.DataFrame(rows)
    result.index.name = "컬럼명"
    return result.round(3)


def stats_by_period(df: pd.DataFrame, cols: list, positive_only: bool) -> pd.DataFrame:
    train_df = df[df["주시작일"] < TRAIN_CUTOFF]

    full_stats = describe_columns(df, cols, positive_only)
    train_stats = describe_columns(train_df, cols, positive_only)

    full_stats.insert(0, "기간", "전체(21-24)")
    train_stats.insert(0, "기간", "학습기간(21-23)")

    combined = pd.concat([full_stats, train_stats])
    return combined.reset_index().set_index(["컬럼명", "기간"])


def print_section(title: str, stats: pd.DataFrame) -> None:
    print(f"=== {title} ===")
    print(stats.to_string())
    print()


def main() -> None:
    print(f"Loading {SRC_PATH} ...")
    df = pd.read_parquet(SRC_PATH)
    print(f"Loaded {len(df):,} rows (21-24 전체), 학습기간(21-23) {len(df[df['주시작일'] < TRAIN_CUTOFF]):,}행\n")

    sales_stats = stats_by_period(df, SALES_COLS, positive_only=True)
    returns_stats = stats_by_period(df, RETURNS_COLS, positive_only=True)
    net_stats = stats_by_period(df, NET_COLS, positive_only=False)

    print_section("[매출] 기술통계량 (총판매수량 / 총판매금액, 실제 판매가 발생한 행만) - 기간별 비교", sales_stats)
    print_section("[반품] 기술통계량 (반품수량 / 반품금액, 실제 반품이 발생한 행만) - 기간별 비교", returns_stats)
    print_section("[순 지표] 기술통계량 (순수량 / 순금액, 전체 행 기준·참고용) - 기간별 비교", net_stats)

    sales_stats.to_csv(SALES_OUT_PATH, encoding="utf-8-sig")
    returns_stats.to_csv(RETURNS_OUT_PATH, encoding="utf-8-sig")
    net_stats.to_csv(NET_OUT_PATH, encoding="utf-8-sig")
    print(f"저장 완료 -> {SALES_OUT_PATH}")
    print(f"저장 완료 -> {RETURNS_OUT_PATH}")
    print(f"저장 완료 -> {NET_OUT_PATH}")


if __name__ == "__main__":
    main()
