"""
최종 정제 데이터(cleaned_main_joined_cleaned_for_pred.parquet)를 일 단위 / 주 단위로 집계한다.

- 수량이 음수인 행(반품)은 순매출로 상쇄하지 않고 총판매/반품을 별도 컬럼으로 분리 보존한다.
- 센터/시도/시군구(지역) x 바코드/옵션코드/상품클러스터(상품) 조합을 집계 기준으로 사용한다.
- 외부 변수(공휴일, 기상, 거시경제 지표 등)는 항목별 집계 규칙(Max/Mean/Sum)을 적용한다.
- 수량 부호 반전이 의심되는 상품(step3_anomalous_product_candidates.csv, EA+즉석편의식품/
  과자류 등 175건)은 집계 전에 제외한다.
"""

import pandas as pd

SRC_PATH = "data/final/cleaned_main_joined_cleaned_for_pred.parquet"
EXCLUDE_PRODUCTS_PATH = "analysis-results/preprocessing/step3_anomalous_product_candidates.csv"
DAILY_OUT_PATH = "data/final/aggregated_daily_demand.parquet"
WEEKLY_OUT_PATH = "data/final/aggregated_weekly_demand.parquet"

REGION_COLS = ["센터", "시도", "시군구"]
PRODUCT_KEY_COLS = ["바코드", "옵션코드", "상품클러스터"]

# 상품 설명용 컬럼: 상품키 기준으로 소수 불일치가 있어 그룹 내 최초(날짜순) 값을 대표값으로 사용한다.
DESCRIPTIVE_COLS = [
    "상품명",
    "규격",
    "입수",
    "KAN_CODE",
    "KAN_대분류",
    "KAN_중분류",
    "KAN_소분류",
]

MAX_COLS = ["공휴일", "covid_영향여부"]  # 특수일 플래그 -> 기간 내 하루라도 해당하면 1
MEAN_WEATHER_COLS = ["평균온도"]
SUM_WEATHER_COLS = ["총강수량"]
MEAN_ECON_COLS = ["cpi", "경상지수", "불변지수"]


def load_source(path: str = SRC_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["거래일"] = pd.to_datetime(df["거래일"])

    exclude = pd.read_csv(EXCLUDE_PRODUCTS_PATH, dtype={"바코드": str})[PRODUCT_KEY_COLS]
    before = len(df)
    df = df.merge(exclude.drop_duplicates(), on=PRODUCT_KEY_COLS, how="left", indicator=True)
    df = df[df["_merge"] == "left_only"].drop(columns="_merge")
    print(f"수량 부호 반전 의심 상품 제외: {before - len(df):,}행 제거 ({len(exclude):,}개 상품키 기준)")

    return df


def _add_sales_return_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    is_sale = df["수량"] > 0
    is_return = df["수량"] < 0

    df["판매수량"] = df["수량"].where(is_sale, 0)
    df["판매금액"] = df["금액"].where(df["금액"] > 0, 0)
    df["반품수량_abs"] = (-df["수량"]).where(is_return, 0)
    df["반품금액_abs"] = (-df["금액"]).where(df["금액"] < 0, 0)

    df["판매건수_flag"] = is_sale.astype(int)
    df["반품건수_flag"] = is_return.astype(int)
    return df


def _aggregate(df: pd.DataFrame, date_col: str, group_cols: list) -> pd.DataFrame:
    df = df.sort_values("거래일")

    sum_spec = {
        "총판매수량": ("판매수량", "sum"),
        "총판매금액": ("판매금액", "sum"),
        "반품수량": ("반품수량_abs", "sum"),
        "반품금액": ("반품금액_abs", "sum"),
        "총거래건수": ("row_id", "count"),
        "판매건수": ("판매건수_flag", "sum"),
        "반품건수": ("반품건수_flag", "sum"),
    }
    for col in MAX_COLS:
        sum_spec[col] = (col, "max")
    for col in MEAN_WEATHER_COLS:
        sum_spec[col] = (col, "mean")
    for col in SUM_WEATHER_COLS:
        sum_spec[col] = (col, "sum")
    for col in MEAN_ECON_COLS:
        sum_spec[col] = (col, "mean")
    for col in DESCRIPTIVE_COLS:
        sum_spec[col] = (col, "first")

    agg = df.groupby(group_cols, as_index=False).agg(**sum_spec)

    agg["순수량"] = agg["총판매수량"] - agg["반품수량"]
    agg["순금액"] = agg["총판매금액"] - agg["반품금액"]

    ordered_cols = (
        group_cols
        + DESCRIPTIVE_COLS
        + [
            "총판매수량",
            "반품수량",
            "순수량",
            "총판매금액",
            "반품금액",
            "순금액",
            "총거래건수",
            "판매건수",
            "반품건수",
        ]
        + MAX_COLS
        + MEAN_WEATHER_COLS
        + SUM_WEATHER_COLS
        + MEAN_ECON_COLS
    )
    return agg[ordered_cols]


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = _add_sales_return_columns(df)
    group_cols = ["거래일"] + REGION_COLS + PRODUCT_KEY_COLS
    return _aggregate(df, "거래일", group_cols)


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df = _add_sales_return_columns(df)
    # 월요일 기준 주 시작일
    df["주시작일"] = df["거래일"] - pd.to_timedelta(df["거래일"].dt.dayofweek, unit="D")
    group_cols = ["주시작일"] + REGION_COLS + PRODUCT_KEY_COLS
    agg = _aggregate(df, "주시작일", group_cols)

    iso = agg["주시작일"].dt.isocalendar()
    agg.insert(1, "ISO_주차", iso["week"])
    agg.insert(1, "ISO_연도", iso["year"])
    return agg


def print_eda_report(daily: pd.DataFrame, weekly: pd.DataFrame) -> None:
    def stats(df: pd.DataFrame, label: str) -> dict:
        n = len(df)
        zero_sales_ratio = (df["총판매수량"] == 0).mean()
        neg_net_qty_count = (df["순수량"] < 0).sum()
        return_occur_ratio = (df["반품수량"] > 0).mean()
        mean_qty = df["총판매수량"].mean()
        std_qty = df["총판매수량"].std()
        cv_qty = std_qty / mean_qty if mean_qty else float("nan")

        print(f"\n=== {label} 집계 EDA ===")
        print(f"전체 행 수                 : {n:,}")
        print(f"총판매수량=0 비율          : {zero_sales_ratio:.4%}")
        print(f"순수량 음수 건수           : {neg_net_qty_count:,} ({neg_net_qty_count / n:.4%})")
        print(f"반품 발생 비율             : {return_occur_ratio:.4%}")
        print(f"총판매수량 평균/표준편차   : {mean_qty:.3f} / {std_qty:.3f}")
        print(f"총판매수량 변동계수(CV)    : {cv_qty:.3f}")
        return {
            "n": n,
            "zero_sales_ratio": zero_sales_ratio,
            "neg_net_qty_count": neg_net_qty_count,
            "return_occur_ratio": return_occur_ratio,
            "cv_qty": cv_qty,
        }

    daily_stats = stats(daily, "일 단위(Daily)")
    weekly_stats = stats(weekly, "주 단위(Weekly)")

    print("\n=== 일 단위 vs 주 단위 비교 요약 ===")
    print(f"{'지표':<26}{'Daily':>15}{'Weekly':>15}")
    print(f"{'전체 행 수':<26}{daily_stats['n']:>15,}{weekly_stats['n']:>15,}")
    print(
        f"{'총판매수량=0 비율':<26}"
        f"{daily_stats['zero_sales_ratio']:>15.4%}"
        f"{weekly_stats['zero_sales_ratio']:>15.4%}"
    )
    print(
        f"{'반품 발생 비율':<26}"
        f"{daily_stats['return_occur_ratio']:>15.4%}"
        f"{weekly_stats['return_occur_ratio']:>15.4%}"
    )
    print(
        f"{'순수량 음수 건수':<26}"
        f"{daily_stats['neg_net_qty_count']:>15,}"
        f"{weekly_stats['neg_net_qty_count']:>15,}"
    )
    print(
        f"{'총판매수량 CV':<26}"
        f"{daily_stats['cv_qty']:>15.3f}"
        f"{weekly_stats['cv_qty']:>15.3f}"
    )


def main() -> None:
    print(f"Loading source data from {SRC_PATH} ...")
    df = load_source()
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    print("\nAggregating daily ...")
    daily = aggregate_daily(df)
    print(f"Daily aggregated shape: {daily.shape}")

    print("Aggregating weekly ...")
    weekly = aggregate_weekly(df)
    print(f"Weekly aggregated shape: {weekly.shape}")

    daily.to_parquet(DAILY_OUT_PATH, index=False)
    weekly.to_parquet(WEEKLY_OUT_PATH, index=False)
    print(f"\nSaved daily   -> {DAILY_OUT_PATH}")
    print(f"Saved weekly  -> {WEEKLY_OUT_PATH}")

    print_eda_report(daily, weekly)


if __name__ == "__main__":
    main()
