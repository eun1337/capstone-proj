"""
eda_center_periodicity.py

설명: 센터별 주간 수요의 주기성을 분석하여 호라이즌 후보(1/2/4/6/8주)를 비교함.
- 센터 A/B의 주간 총수요 시계열을 생성함.
- lag 1~12주의 ACF와 Periodogram을 계산함.
- 후보 호라이즌별 ACF 유의성 및 해당 주기 근방의 spectral power를 비교함.
- A센터 전체, B센터 전체, B센터 2023-05 이후 구간을 각각 분석함.
- 분석 결과를 CSV와 ACF/Periodogram 그래프로 저장함.
"""

import numpy as np
import pandas as pd
from scipy.signal import periodogram

SRC_PATH = "data/ml/experiments/economic_indicators/feature_table_with_econ.parquet"
REPORT_DIR = "data/ml/experiments/horizon_selection/reports"

MAX_LAG = 12
CANDIDATE_HORIZONS = [1, 2, 4, 6, 8]
B_REGIME_CUTOFF = "2023-05-01"
# 호라이즌 후보와 관련된 단기 주기성을 보기 위해
# Periodogram의 1.5~12주 구간에서 가장 강한 주기를 별도로 확인함.
SHORT_PERIOD_RANGE = (1.5, 12.0)


def build_weekly_series(df: pd.DataFrame, center_id: str, start: str | None = None) -> pd.Series:
    sub = df[df["center_id"] == center_id]
    if start is not None:
        sub = sub[sub["week_st"] >= start]
    weekly = sub.groupby("week_st")["qty"].sum().sort_index()
    full_index = pd.date_range(weekly.index.min(), weekly.index.max(), freq="7D")
    return weekly.reindex(full_index, fill_value=0.0)


def compute_acf(series: pd.Series, max_lag: int) -> pd.DataFrame:
    x = series.to_numpy(dtype=float)
    n = len(x)
    x_centered = x - x.mean()
    denom = np.sum(x_centered**2)
    ci_bound = 1.96 / np.sqrt(n)

    rows = []
    for lag in range(1, max_lag + 1):
        num = np.sum(x_centered[lag:] * x_centered[:-lag])
        acf_val = num / denom if denom != 0 else np.nan
        rows.append(
            {
                "lag_weeks": lag,
                "acf": acf_val,
                "ci_95_bound": ci_bound,
                "is_significant": abs(acf_val) > ci_bound,
            }
        )
    return pd.DataFrame(rows)


def compute_periodogram(series: pd.Series) -> pd.DataFrame:
    x = series.to_numpy(dtype=float)
    x_detrended = x - x.mean()
    freqs, power = periodogram(x_detrended, fs=1.0, scaling="spectrum")
    period_weeks = np.divide(1.0, freqs, out=np.full_like(freqs, np.inf), where=freqs != 0)
    result = pd.DataFrame({"freq_cycles_per_week": freqs, "period_weeks": period_weeks, "power": power})
    return result[result["period_weeks"] <= len(series)].sort_values("power", ascending=False)


def plot_acf_periodogram(acf_df: pd.DataFrame, pgram_df: pd.DataFrame, title: str, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    colors = ["#d62728" if sig else "#1f77b4" for sig in acf_df["is_significant"]]
    ax.bar(acf_df["lag_weeks"], acf_df["acf"], color=colors)
    ci = acf_df["ci_95_bound"].iloc[0]
    ax.axhline(ci, color="gray", linestyle="--", linewidth=1)
    ax.axhline(-ci, color="gray", linestyle="--", linewidth=1)
    for h in CANDIDATE_HORIZONS:
        if h <= acf_df["lag_weeks"].max():
            ax.axvline(h, color="green", alpha=0.15, linewidth=6)
    ax.set_xlabel("lag (weeks)")
    ax.set_ylabel("ACF")
    ax.set_title("ACF (red = 95% CI 밖)")
    ax.set_xticks(acf_df["lag_weeks"])

    ax2 = axes[1]
    plot_df = pgram_df[(pgram_df["period_weeks"] >= 2) & (pgram_df["period_weeks"] <= 20)].sort_values(
        "period_weeks"
    )
    ax2.plot(plot_df["period_weeks"], plot_df["power"], marker="o", markersize=3)
    for h in CANDIDATE_HORIZONS:
        ax2.axvline(h, color="green", alpha=0.15, linewidth=6)
    ax2.set_xlabel("period (weeks)")
    ax2.set_ylabel("spectral power")
    ax2.set_title("Periodogram (2~20주 구간)")
    ax2.set_xticks([2, 4, 6, 8, 10, 12, 16, 20])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize_top_peak(acf_df: pd.DataFrame, pgram_df: pd.DataFrame) -> dict:
    sig = acf_df[acf_df["is_significant"]]
    top_acf_lag = None
    if not sig.empty:
        top_acf_lag = int(sig.loc[sig["acf"].abs().idxmax(), "lag_weeks"])

    # 전체 최대 power 주기는 장기 패턴일 수 있으므로 참고용으로 저장함.
    # 호라이즌 검토에는 1.5~12주 범위의 최대 power 주기를 사용함.
    overall_top_period = pgram_df.iloc[0]["period_weeks"] if not pgram_df.empty else None

    lo, hi = SHORT_PERIOD_RANGE
    short_band = pgram_df[(pgram_df["period_weeks"] >= lo) & (pgram_df["period_weeks"] <= hi)]
    short_top_period = short_band.iloc[0]["period_weeks"] if not short_band.empty else None

    candidate_acf = acf_df[acf_df["lag_weeks"].isin(CANDIDATE_HORIZONS)][
        ["lag_weeks", "acf", "is_significant"]
    ].to_dict("records")

    # Periodogram에 정확히 2/4/6/8주 지점이 없을 수 있으므로,
    # 각 후보 주기와 가장 가까운 지점의 power를 찾아 비교함.
    pg_by_period = pgram_df.sort_values("period_weeks").reset_index(drop=True)
    candidate_pgram = []
    for h in CANDIDATE_HORIZONS:
        if h < 2:
            continue
        idx = (pg_by_period["period_weeks"] - h).abs().idxmin()
        candidate_pgram.append(
            {
                "target_period_weeks": h,
                "nearest_bin_period_weeks": round(float(pg_by_period.loc[idx, "period_weeks"]), 3),
                "power": float(pg_by_period.loc[idx, "power"]),
            }
        )

    return {
        "top_significant_acf_lag": top_acf_lag,
        "overall_top_periodogram_period_weeks": round(float(overall_top_period), 2) if overall_top_period else None,
        "short_band_top_periodogram_period_weeks": round(float(short_top_period), 2) if short_top_period else None,
        "candidate_lag_acf": candidate_acf,
        "candidate_pgram": candidate_pgram,
    }


def run_segment(df: pd.DataFrame, center_id: str, label: str, start: str | None = None) -> dict:
    series = build_weekly_series(df, center_id, start=start)
    acf_df = compute_acf(series, MAX_LAG)
    pgram_df = compute_periodogram(series)

    acf_df.insert(0, "segment", label)
    pgram_df = pgram_df.copy()
    pgram_df.insert(0, "segment", label)

    acf_csv = f"{REPORT_DIR}/center_{label}_acf.csv"
    pgram_csv = f"{REPORT_DIR}/center_{label}_periodogram.csv"
    acf_df.to_csv(acf_csv, index=False)
    pgram_df.to_csv(pgram_csv, index=False)

    png_path = f"{REPORT_DIR}/center_acf_periodogram_{label}.png"
    plot_acf_periodogram(acf_df, pgram_df, title=f"센터 {label} 주단위 수요 - ACF & Periodogram (n={len(series)}주)", out_path=png_path)

    summary = summarize_top_peak(acf_df, pgram_df)
    summary["segment"] = label
    summary["n_weeks"] = len(series)
    summary["date_range"] = f"{series.index.min().date()} ~ {series.index.max().date()}"
    return summary


def main() -> None:
    df = pd.read_parquet(SRC_PATH, columns=["center_id", "week_st", "qty"])

    summaries = [
        run_segment(df, "A", "A_full"),
        run_segment(df, "B", "B_full"),
        run_segment(df, "B", "B_from2023-05", start=B_REGIME_CUTOFF),
    ]

    summary_rows = []
    for s in summaries:
        acf_by_lag = {row["lag_weeks"]: row for row in s["candidate_lag_acf"]}
        pgram_by_target = {row["target_period_weeks"]: row for row in s["candidate_pgram"]}
        for h in CANDIDATE_HORIZONS:
            summary_rows.append(
                {
                    "segment": s["segment"],
                    "horizon_weeks": h,
                    "acf": acf_by_lag.get(h, {}).get("acf"),
                    "acf_is_significant": acf_by_lag.get(h, {}).get("is_significant"),
                    "pgram_power": pgram_by_target.get(h, {}).get("power"),
                    "top_significant_acf_lag_overall": s["top_significant_acf_lag"],
                    "top_short_band_period_weeks_overall": s["short_band_top_periodogram_period_weeks"],
                }
            )
    pd.DataFrame(summary_rows).to_csv(f"{REPORT_DIR}/center_acf_periodogram_summary.csv", index=False)

    print("=" * 70)
    print("[분석 1] 센터 레벨 주기성(ACF / Periodogram) 요약")
    print("=" * 70)
    for s in summaries:
        print(f"\n--- {s['segment']} (n={s['n_weeks']}주, 기간={s['date_range']}) ---")
        print(f"  ACF 기준 가장 강한 유의(95% CI 밖) 피크 lag: {s['top_significant_acf_lag']}")
        print(f"  Periodogram 전체 최상위 파워 주기(장기추세 포함, 참고용): {s['overall_top_periodogram_period_weeks']}주")
        print(f"  Periodogram 단기대역(1.5~12주) 내 최상위 파워 주기: {s['short_band_top_periodogram_period_weeks']}주")
        print("  후보 lag(1,2,4,6,8)별 ACF:")
        for row in s["candidate_lag_acf"]:
            mark = "*" if row["is_significant"] else " "
            print(f"    lag={row['lag_weeks']:>2}주  acf={row['acf']:+.3f} {mark}")
        print("  후보 주기(2,4,6,8) 근방 periodogram 파워:")
        powers = [c["power"] for c in s["candidate_pgram"]]
        max_power = max(powers) if powers else None
        for row in s["candidate_pgram"]:
            mark = "<-- 후보 중 최대" if max_power is not None and row["power"] == max_power else ""
            print(f"    target={row['target_period_weeks']:>2}주 (nearest bin={row['nearest_bin_period_weeks']}주)  power={row['power']:.2e} {mark}")

    if summaries[1]["n_weeks"] == summaries[2]["n_weeks"]:
        print(
            "\n[참고] 센터B의 feature_table_with_econ.parquet 상 실제 데이터 시작일은 "
            f"{summaries[1]['date_range'].split(' ~ ')[0]}로, 옛 EDA가 언급한 2023-05 컷오프보다 "
            "이미 늦게 시작합니다. 따라서 'B_full'과 'B_from2023-05'는 사실상 동일한 시계열입니다."
        )


if __name__ == "__main__":
    main()
