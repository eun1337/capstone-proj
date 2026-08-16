"""
eda_sku_interorder_interval.py

설명:호라이즌 후보(1/2/4/6/8주) 검토를 위한 SKU별 수요 발생 간격 분석.
- 센터별로 sold_flag==1인 주를 기준으로 SKU의 연속 수요 발생 간격을 주 단위로 계산함.
- 수요 발생 주가 5회 이상인 SKU만 분석 대상으로 사용함.
- SKU별 모든 수요 발생 간격을 통합하여 센터별 간격 분포, 최빈값, 중앙값을 계산함.
- 12주 이상 간격은 하나의 구간으로 묶고, 결과를 CSV와 히스토그램으로 저장함.
"""

import pandas as pd

SRC_PATH = "data/ml/experiments/economic_indicators/feature_table_with_econ.parquet"
REPORT_DIR = "data/ml/experiments/horizon_selection/reports"

MIN_ACTIVE_WEEKS = 5
HIST_MAX_BUCKET = 12  # 이 값 이상 간격은 "{HIST_MAX_BUCKET}+"로 묶는다


def compute_intervals(df: pd.DataFrame, center_id: str) -> pd.DataFrame:
    sub = df[(df["center_id"] == center_id) & (df["sold_flag"] == 1)].copy()

    active_counts = sub.groupby("sku_id")["week_st"].size()
    valid_skus = active_counts[active_counts >= MIN_ACTIVE_WEEKS].index
    sub = sub[sub["sku_id"].isin(valid_skus)].sort_values(["sku_id", "week_st"])

    sub["interval_weeks"] = sub.groupby("sku_id")["week_st"].diff().dt.days / 7
    intervals = sub.dropna(subset=["interval_weeks"])
    return intervals[["sku_id", "week_st", "interval_weeks"]], len(valid_skus)


def summarize_distribution(intervals: pd.DataFrame, center_id: str) -> pd.DataFrame:
    vals = intervals["interval_weeks"].round().astype(int)
    bucketed = vals.clip(upper=HIST_MAX_BUCKET)

    counts = bucketed.value_counts().sort_index()
    total = counts.sum()

    rows = []
    for bucket, count in counts.items():
        label = f"{HIST_MAX_BUCKET}+" if bucket == HIST_MAX_BUCKET else str(bucket)
        n_skus = intervals.loc[bucketed == bucket, "sku_id"].nunique()
        rows.append(
            {
                "center_id": center_id,
                "interval_weeks": label,
                "interval_count": int(count),
                "proportion": round(count / total, 4),
                "n_skus_contributing": n_skus,
            }
        )
    return pd.DataFrame(rows)


def plot_histogram(intervals: pd.DataFrame, center_id: str, mode_val: float, median_val: float, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False

    vals = intervals["interval_weeks"].clip(upper=HIST_MAX_BUCKET)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = [b - 0.5 for b in range(1, HIST_MAX_BUCKET + 2)]
    ax.hist(vals, bins=bins, color="#1f77b4", edgecolor="white")
    ax.axvline(mode_val, color="#d62728", linestyle="--", label=f"최빈값={mode_val:.0f}주")
    ax.axvline(median_val, color="#2ca02c", linestyle="--", label=f"중앙값={median_val:.1f}주")
    ax.set_xlabel(f"발주 간격 (주, {HIST_MAX_BUCKET}주 이상은 {HIST_MAX_BUCKET}로 클립)")
    ax.set_ylabel("빈도 (간격 발생 건수)")
    ax.set_title(f"센터 {center_id} SKU 발주 간격 분포 (활성주 {MIN_ACTIVE_WEEKS}회 미만 SKU 제외)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_center(df: pd.DataFrame, center_id: str) -> dict:
    intervals, n_valid_skus = compute_intervals(df, center_id)
    mode_val = intervals["interval_weeks"].round().mode().iloc[0]
    median_val = intervals["interval_weeks"].median()

    dist_df = summarize_distribution(intervals, center_id)
    png_path = f"{REPORT_DIR}/sku_interorder_interval_hist_{center_id}.png"
    plot_histogram(intervals, center_id, mode_val, median_val, png_path)

    return {
        "center_id": center_id,
        "n_valid_skus": n_valid_skus,
        "n_intervals": len(intervals),
        "mode_weeks": float(mode_val),
        "median_weeks": float(median_val),
        "dist_df": dist_df,
    }


def main() -> None:
    df = pd.read_parquet(SRC_PATH, columns=["center_id", "sku_id", "week_st", "sold_flag"])

    results = [run_center(df, "A"), run_center(df, "B")]

    all_dist = pd.concat([r["dist_df"] for r in results], ignore_index=True)
    out_csv = f"{REPORT_DIR}/sku_interorder_interval_distribution.csv"
    all_dist.to_csv(out_csv, index=False)

    print("=" * 70)
    print("[분석 2] SKU 레벨 발주 간격(Inter-order Interval) 요약")
    print("=" * 70)
    for r in results:
        print(f"\n--- 센터 {r['center_id']} (분석대상 SKU {r['n_valid_skus']}개, 간격 표본 {r['n_intervals']}건) ---")
        print(f"  최빈값(mode): {r['mode_weeks']:.0f}주")
        print(f"  중앙값(median): {r['median_weeks']:.1f}주")
        top5 = r["dist_df"].sort_values("interval_count", ascending=False).head(5)
        print("  상위 5개 간격 구간:")
        for _, row in top5.iterrows():
            print(f"    {row['interval_weeks']:>3}주  건수={row['interval_count']:>7}  비율={row['proportion']*100:5.1f}%")


if __name__ == "__main__":
    main()
