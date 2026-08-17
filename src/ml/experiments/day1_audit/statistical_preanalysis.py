"""
statistical_preanalysis.py

Day 1 Step 4: 모델링 전 A/B center-week 수요 시계열 구조 사전분석.

분석:
    - center-week 수요 집계 및 주차 연속성
    - ACF/PACF
    - periodogram
    - ADF/KPSS 및 필요 시 1차 차분
    - A/B 시계열 구조 비교

2024 holdout은 분석에서 제외한다.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

BASE_DIR = Path(__file__).resolve().parents[4]
FEATURE_TABLE_PATH = BASE_DIR / "data" / "final_feature_table.parquet"
OUT_DIR = BASE_DIR / "data" / "ml" / "experiments" / "day1_audit"

CENTER_COL, WEEK_COL, QTY_COL = "center_id", "week_st", "qty"
HOLDOUT_START = pd.Timestamp("2024-01-01")
MAX_LAG_CAP, ALPHA, TOP_N_PEAKS = 52, 0.05, 5


def load_center_week_series(df: pd.DataFrame) -> pd.DataFrame:
    dev = df[df[WEEK_COL] < HOLDOUT_START]
    return dev.groupby([CENTER_COL, WEEK_COL], observed=True)[QTY_COL].sum().reset_index()


def center_week_summary(series_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for center, g in series_df.groupby(CENTER_COL, observed=True):
        g = g.sort_values(WEEK_COL)
        start, end = g[WEEK_COL].min(), g[WEEK_COL].max()
        expected = pd.date_range(start, end, freq="7D")
        missing = expected.difference(g[WEEK_COL])
        qty = g[QTY_COL]
        rows.append({
            "center_id": center,
            "start_week": start,
            "end_week": end,
            "n_weeks": len(g),
            "n_expected_weeks": len(expected),
            "n_missing_weeks": len(missing),
            "calendar_continuous": len(missing) == 0,
            "qty_mean": qty.mean(),
            "qty_median": qty.median(),
            "qty_std": qty.std(),
            "qty_min": qty.min(),
            "qty_max": qty.max(),
            "n_zero_weeks": int((qty == 0).sum()),
            "zero_ratio": round((qty == 0).mean(), 4),
        })
    return pd.DataFrame(rows)


def safe_acf_pacf(qty: np.ndarray) -> pd.DataFrame:
    if len(qty) < 4:
        return pd.DataFrame(columns=["lag", "acf", "acf_ci_low", "acf_ci_high", "pacf", "pacf_ci_low", "pacf_ci_high"])

    max_lag = min(MAX_LAG_CAP, len(qty) - 1)
    acf_vals, acf_ci = acf(qty, nlags=max_lag, alpha=ALPHA, fft=True)

    pacf_lag, pacf_vals, pacf_ci = max_lag, None, None
    while pacf_lag > 0:
        try:
            pacf_vals, pacf_ci = pacf(qty, nlags=pacf_lag, alpha=ALPHA, method="ywm")
            break
        except ValueError:
            pacf_lag -= 1

    out = pd.DataFrame({
        "lag": range(max_lag + 1),
        "acf": acf_vals,
        "acf_ci_low": acf_ci[:, 0],
        "acf_ci_high": acf_ci[:, 1],
    })
    if pacf_vals is not None:
        out = out.merge(pd.DataFrame({
            "lag": range(pacf_lag + 1),
            "pacf": pacf_vals,
            "pacf_ci_low": pacf_ci[:, 0],
            "pacf_ci_high": pacf_ci[:, 1],
        }), on="lag", how="left")
    else:
        out[["pacf", "pacf_ci_low", "pacf_ci_high"]] = np.nan
    return out


def save_acf_pacf_plots(qty: np.ndarray, center: str, max_lag: int):
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(qty, lags=max_lag, alpha=ALPHA, ax=ax, title=f"ACF - center {center}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"acf_{center}.png")
    plt.close(fig)

    pacf_lag = max_lag
    while pacf_lag > 0:
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_pacf(qty, lags=pacf_lag, alpha=ALPHA, method="ywm", ax=ax, title=f"PACF - center {center}")
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"pacf_{center}.png")
            plt.close(fig)
            break
        except ValueError:
            pacf_lag -= 1


def compute_periodogram(qty: np.ndarray) -> pd.DataFrame:
    freqs, power = periodogram(qty, fs=1.0, detrend="constant")
    keep = freqs > 0
    return pd.DataFrame({
        "period_weeks": 1.0 / freqs[keep],
        "power": power[keep],
    }).sort_values("period_weeks").reset_index(drop=True)


def save_periodogram_plot(pgram_df: pd.DataFrame, center: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pgram_df["period_weeks"], pgram_df["power"])
    ax.set(xlabel="period (weeks)", ylabel="power", title=f"Periodogram - center {center}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"periodogram_{center}.png")
    plt.close(fig)


def run_stationarity_test(qty: np.ndarray) -> dict:
    result = {
        "n_obs": len(qty),
        "adf_stat": np.nan,
        "adf_pvalue": np.nan,
        "kpss_stat": np.nan,
        "kpss_pvalue": np.nan,
        "adf_reject_unit_root(alpha=0.05)": None,
        "kpss_reject_stationarity(alpha=0.05)": None,
        "interpretation": "REVIEW",
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            adf_stat, adf_p, *_ = adfuller(qty, autolag="AIC")
            result.update({
                "adf_stat": adf_stat,
                "adf_pvalue": adf_p,
                "adf_reject_unit_root(alpha=0.05)": bool(adf_p < ALPHA),
            })
        except (ValueError, np.linalg.LinAlgError):
            pass

        try:
            kpss_stat, kpss_p, *_ = kpss(qty, regression="c", nlags="auto")
            result.update({
                "kpss_stat": kpss_stat,
                "kpss_pvalue": kpss_p,
                "kpss_reject_stationarity(alpha=0.05)": bool(kpss_p < ALPHA),
            })
        except (ValueError, OverflowError):
            pass

    if pd.isna(result["adf_pvalue"]) or pd.isna(result["kpss_pvalue"]):
        return result

    adf_reject = result["adf_reject_unit_root(alpha=0.05)"]
    kpss_reject = result["kpss_reject_stationarity(alpha=0.05)"]
    if adf_reject and not kpss_reject:
        result["interpretation"] = "stationary_evidence"
    elif not adf_reject and kpss_reject:
        result["interpretation"] = "non_stationary_evidence"
    return result


def stationarity_audit(series_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for center, g in series_df.groupby(CENTER_COL, observed=True):
        qty = g.sort_values(WEEK_COL)[QTY_COL].to_numpy(dtype=float)
        original = run_stationarity_test(qty)
        rows.append({"center_id": center, "series": "original", **original})
        if original["interpretation"] != "stationary_evidence":
            rows.append({"center_id": center, "series": "diff1", **run_stationarity_test(qty[1:] - qty[:-1])})
    return pd.DataFrame(rows)


def top_acf_lags(acf_df: pd.DataFrame, n: int = 5) -> str:
    d = acf_df[acf_df["lag"] > 0].copy()
    d["abs_acf"] = d["acf"].abs()
    return ", ".join(
        f"lag{int(r.lag)}={r.acf:.3f}"
        for r in d.sort_values("abs_acf", ascending=False).head(n).itertuples()
    )


def top_periods(pgram_df: pd.DataFrame, n: int = 3) -> str:
    return ", ".join(
        f"{r.period_weeks:.1f}w(p={r.power:.2e})"
        for r in pgram_df.sort_values("power", ascending=False).head(n).itertuples()
    )


def build_center_comparison(summary_df, acf_by_center, pgram_by_center, station_df) -> pd.DataFrame:
    a = summary_df[summary_df.center_id == "A"].iloc[0]
    b = summary_df[summary_df.center_id == "B"].iloc[0]

    def cv(mean, std):
        return round(std / mean, 4) if mean and mean > 0 else np.nan

    def stat(center, series):
        hit = station_df[(station_df.center_id == center) & (station_df.series == series)]
        return hit["interpretation"].iloc[0] if len(hit) else "N/A"

    return pd.DataFrame([
        {"metric": "observation_period", "A": f"{a.start_week.date()}~{a.end_week.date()}", "B": f"{b.start_week.date()}~{b.end_week.date()}"},
        {"metric": "n_weeks", "A": a.n_weeks, "B": b.n_weeks},
        {"metric": "qty_mean", "A": round(a.qty_mean, 2), "B": round(b.qty_mean, 2)},
        {"metric": "qty_median", "A": round(a.qty_median, 2), "B": round(b.qty_median, 2)},
        {"metric": "qty_std", "A": round(a.qty_std, 2), "B": round(b.qty_std, 2)},
        {"metric": "cv", "A": cv(a.qty_mean, a.qty_std), "B": cv(b.qty_mean, b.qty_std)},
        {"metric": "zero_ratio", "A": a.zero_ratio, "B": b.zero_ratio},
        {"metric": "top_acf_lags", "A": top_acf_lags(acf_by_center["A"]), "B": top_acf_lags(acf_by_center["B"])},
        {"metric": "top_periodogram_peaks", "A": top_periods(pgram_by_center["A"]), "B": top_periods(pgram_by_center["B"])},
        {"metric": "stationarity_original", "A": stat("A", "original"), "B": stat("B", "original")},
        {"metric": "stationarity_diff1", "A": stat("A", "diff1"), "B": stat("B", "diff1")},
        {"metric": "history_length_limited_items", "A": "N/A", "B": f"B는 {b.n_weeks}주로 52주 계절성/장기 패턴 비교가 제한됨"},
    ])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FEATURE_TABLE_PATH, columns=[CENTER_COL, WEEK_COL, QTY_COL])
    series_df = load_center_week_series(df)

    summary_df = center_week_summary(series_df)
    summary_df.to_csv(OUT_DIR / "center_week_summary.csv", index=False, encoding="utf-8-sig")

    acf_rows, acf_by_center, pgram_rows, pgram_by_center = [], {}, [], {}
    for center in ["A", "B"]:
        g = series_df[series_df[CENTER_COL] == center].sort_values(WEEK_COL)
        qty = g[QTY_COL].to_numpy(dtype=float)
        nobs = len(qty)
        max_lag = min(MAX_LAG_CAP, nobs - 1) if nobs > 1 else 0

        acf_df = safe_acf_pacf(qty)
        acf_df.insert(0, CENTER_COL, center)
        acf_rows.append(acf_df)
        acf_by_center[center] = acf_df
        if max_lag > 0:
            save_acf_pacf_plots(qty, center, max_lag)

        pgram_df = compute_periodogram(qty)
        pgram_by_center[center] = pgram_df
        peaks = pgram_df.sort_values("power", ascending=False).head(TOP_N_PEAKS).reset_index(drop=True)
        peaks.insert(0, "rank", range(1, len(peaks) + 1))
        peaks.insert(0, CENTER_COL, center)
        peaks["note"] = "" if nobs >= 52 else f"INSUFFICIENT_HISTORY(관측 {nobs}주 < 52주)"
        pgram_rows.append(peaks)
        if not pgram_df.empty:
            save_periodogram_plot(pgram_df, center)

    pd.concat(acf_rows, ignore_index=True).to_csv(
        OUT_DIR / "acf_pacf.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(pgram_rows, ignore_index=True).to_csv(
        OUT_DIR / "periodogram_peaks.csv", index=False, encoding="utf-8-sig"
    )

    station_df = stationarity_audit(series_df)
    station_df.to_csv(OUT_DIR / "stationarity_tests.csv", index=False, encoding="utf-8-sig")

    comparison_df = build_center_comparison(summary_df, acf_by_center, pgram_by_center, station_df)
    comparison_df.to_csv(OUT_DIR / "center_comparison.csv", index=False, encoding="utf-8-sig")

    history_limit = bool((summary_df["n_weeks"] < 52).any())
    calendar_gap = not summary_df["calendar_continuous"].all()
    station_review = station_df["interpretation"].eq("REVIEW").any()
    status = "REVIEW" if history_limit or calendar_gap or station_review else "PASS"

    for center in ["A", "B"]:
        s = summary_df[summary_df.center_id == center].iloc[0]
        st = station_df[(station_df.center_id == center) & (station_df.series == "original")].iloc[0]
        print(f"[{center}] {s.start_week.date()}~{s.end_week.date()} ({s.n_weeks}주)")
        print(f"- ACF: {top_acf_lags(acf_by_center[center])}")
        print(f"- periodogram: {top_periods(pgram_by_center[center])}")
        print(f"- ADF p={st.adf_pvalue:.4f}, KPSS p={st.kpss_pvalue:.4f}, 판정={st.interpretation}")

    print(f"statistical_preanalysis_status: {status}")


if __name__ == "__main__":
    main()
