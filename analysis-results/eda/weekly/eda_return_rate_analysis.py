"""
Step 3: 주간 반품률 패턴 및 순수량 < 0 분석 (aggregated_weekly_demand.parquet)

1) 순수량 < 0 (반품이 판매보다 많아 순수량이 음수가 된 주) 분석
   - 건수 및 전체 행 대비 비율
   - 상위 5개 센터 / KAN_대분류 / 상품(바코드+상품명) 요약

2) 반품률(Return Rate) 분포 분석
   - 전체 누적 반품률 = sum(반품수량) / sum(총판매수량)
   - 행별 반품률(총판매수량>0 행 기준) 백분위수 (P50/P75/P90/P95/P99/Max)
   - 반품률 >= 100% 케이스 비중

21-24년 전체와 21-23년(학습기간)을 같이 계산해서 비교한다. 21-23 기준을 우선하되,
24년 들어 반품 패턴이 달라졌는지 참고용으로 함께 본다.
"""

import pandas as pd

SRC_PATH = "data/final/aggregated_weekly_demand.parquet"
OUT_NEG_NET_SUMMARY = "analysis-results/eda/step3_negative_net_summary.csv"
OUT_NEG_NET_BREAKDOWN = "analysis-results/eda/step3_negative_net_top5_breakdown.csv"
OUT_RETURN_RATE_PCTL = "analysis-results/eda/step3_return_rate_percentiles.csv"

TRAIN_CUTOFF = "2024-01-01"
TOP_N = 5


def negative_net_summary(df: pd.DataFrame) -> dict:
    n_rows = len(df)
    neg = df[df["순수량"] < 0]
    n_neg = len(neg)
    return {
        "전체 행 수": n_rows,
        "순수량<0 건수": n_neg,
        "순수량<0 비율(%)": round(n_neg / n_rows * 100, 4) if n_rows else float("nan"),
    }


def top5_breakdown(neg: pd.DataFrame, group_cols, label: str) -> pd.DataFrame:
    n_neg = len(neg)
    top = (
        neg.groupby(group_cols)
        .size()
        .sort_values(ascending=False)
        .head(TOP_N)
        .rename("건수")
        .reset_index()
    )
    top.insert(0, "구분", label)
    top["비율(%)"] = round(top["건수"] / n_neg * 100, 2) if n_neg else 0
    return top


def return_rate_percentiles(df: pd.DataFrame) -> dict:
    sale_rows = df[df["총판매수량"] > 0]
    rate = sale_rows["반품수량"] / sale_rows["총판매수량"] * 100

    overall_rate = df["반품수량"].sum() / df["총판매수량"].sum() * 100
    over_100 = (rate >= 100).sum()

    return {
        "전체 누적 반품률(%)": round(overall_rate, 4),
        "P50(%)": round(rate.quantile(0.50), 4),
        "P75(%)": round(rate.quantile(0.75), 4),
        "P90(%)": round(rate.quantile(0.90), 4),
        "P95(%)": round(rate.quantile(0.95), 4),
        "P99(%)": round(rate.quantile(0.99), 4),
        "Max(%)": round(rate.max(), 4),
        "반품률>=100% 건수": int(over_100),
        "반품률>=100% 비율(%)": round(over_100 / len(sale_rows) * 100, 4),
        "표본행수(총판매수량>0)": len(sale_rows),
    }


def print_kv_table(title: str, stats: dict) -> None:
    print(f"=== {title} ===")
    width = max(len(k) for k in stats)
    for k, v in stats.items():
        print(f"{k:<{width}} : {v:,}" if isinstance(v, int) else f"{k:<{width}} : {v}")
    print()


def main() -> None:
    print(f"Loading {SRC_PATH} ...")
    df = pd.read_parquet(SRC_PATH)
    train = df[df["주시작일"] < TRAIN_CUTOFF]
    print(f"Loaded {len(df):,} rows (21-24 전체), 학습기간(21-23) {len(train):,}행\n")

    periods = {"전체(21-24)": df, "학습기간(21-23)": train}

    # 1) 순수량 < 0 요약
    neg_summary_rows = []
    breakdown_frames = []
    for label, sub in periods.items():
        summary = negative_net_summary(sub)
        summary_row = {"기간": label, **summary}
        neg_summary_rows.append(summary_row)
        print_kv_table(f"[1] 순수량<0 요약 - {label}", summary)

        neg = sub[sub["순수량"] < 0]
        center_top = top5_breakdown(neg, ["센터"], "센터")
        kan_top = top5_breakdown(neg, ["KAN_대분류"], "KAN_대분류")
        product_top = top5_breakdown(neg, ["바코드", "상품명"], "상품(바코드+상품명)")

        for name, top in [("센터", center_top), ("KAN_대분류", kan_top), ("상품", product_top)]:
            print(f"--- 순수량<0 상위 {TOP_N}개 {name} ({label}) ---")
            print(top.drop(columns="구분").to_string(index=False))
            print()
            top.insert(0, "기간", label)
            breakdown_frames.append(top)

    neg_summary_df = pd.DataFrame(neg_summary_rows)
    breakdown_df = pd.concat(breakdown_frames, ignore_index=True)

    # 2) 반품률 분포
    rate_rows = []
    for label, sub in periods.items():
        stats = return_rate_percentiles(sub)
        rate_rows.append({"기간": label, **stats})
        print_kv_table(f"[2] 반품률(Return Rate) 분포 - {label}", stats)

    rate_df = pd.DataFrame(rate_rows)

    neg_summary_df.to_csv(OUT_NEG_NET_SUMMARY, index=False, encoding="utf-8-sig")
    breakdown_df.to_csv(OUT_NEG_NET_BREAKDOWN, index=False, encoding="utf-8-sig")
    rate_df.to_csv(OUT_RETURN_RATE_PCTL, index=False, encoding="utf-8-sig")
    print(f"저장 완료 -> {OUT_NEG_NET_SUMMARY}")
    print(f"저장 완료 -> {OUT_NEG_NET_BREAKDOWN}")
    print(f"저장 완료 -> {OUT_RETURN_RATE_PCTL}")


if __name__ == "__main__":
    main()
