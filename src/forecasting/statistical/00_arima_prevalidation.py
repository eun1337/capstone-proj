"""
00_arima_prevalidation.py

ARIMA 본 실행 전 사전검증용 스크립트.

Development 데이터에서 smoke test, 200-SKU 실행 가능성,
short-history 구간 및 미수렴 후보의 fallback 가능성을 점검한다.
MIN_HISTORY=6과 convergence fallback 규칙의 근거를 재현하기 위한 진단이며,
최종 order 선정이나 2024 Holdout 성능평가는 수행하지 않는다.
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from pmdarima import auto_arima

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from common import (  # noqa: E402
    AUTO_ARIMA_KWARGS, FORECAST_N_PERIODS, FORECAST_STEP_TO_HORIZON, MIN_HISTORY,
    extract_valid_fit_candidates, candidate_convergence, candidate_aicc, expm1_clip,
)

BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_PATH = BASE_DIR / "data" / "development_2021_2023.parquet"
OUT_DIR = BASE_DIR / "data" / "ml" / "day2_statistical_models"

CENTER_COL, SKU_COL, WEEK_COL, QTY_COL = "center_id", "sku_id", "week_st", "qty"
CENTERS = ["A", "B"]
REGIME_B_START = pd.Timestamp("2023-07-03")  # B센터 development 편입 시작

N_SAMPLE_PER_CENTER = 100
RANDOM_STATE = 42
SHORT_HISTORY_AUDIT_MAX_WEEKS = 8

SMOKE_TEST_SKUS = [
    ("A", "18801045571949*BX*1"),
    ("A", "1701006181937*BX*1"),
    ("A", "1701006327069*EA*1"),
    ("B", "1078615000233*BX*1"),
    ("B", "18801045202515*CS*1"),
    ("B", "11209012992*EA*1"),
]

SKU_HISTORY_PATH = OUT_DIR / "prevalidation_sku_history_summary.csv"
SMOKE_TEST_PATH = OUT_DIR / "prevalidation_smoke_test.csv"
FEASIBILITY_200_PATH = OUT_DIR / "prevalidation_feasibility_200.csv"
SHORT_HISTORY_PATH = OUT_DIR / "prevalidation_short_history_audit.csv"
SHORT_HISTORY_SUMMARY_PATH = OUT_DIR / "prevalidation_short_history_summary.csv"
FALLBACK_PATH = OUT_DIR / "prevalidation_nonconvergence_fallback.csv"


def load_source() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_PATH, columns=[CENTER_COL, SKU_COL, WEEK_COL, QTY_COL])
    b_min = df.loc[df[CENTER_COL] == "B", WEEK_COL].min()
    if pd.notna(b_min):
        assert b_min >= REGIME_B_START, f"B센터에 regime 시작({REGIME_B_START.date()}) 이전 데이터 존재"
    return df[~((df[CENTER_COL] == "B") & (df[WEEK_COL] < REGIME_B_START))].copy()


def build_sku_history_summary(df: pd.DataFrame) -> pd.DataFrame:
    """(center_id, sku_id)별 history/0-수요 요약 통계."""
    grouped = df.groupby([CENTER_COL, SKU_COL], observed=True)
    summary = grouped.agg(
        first_week=(WEEK_COL, "min"), last_week=(WEEK_COL, "max"),
        history_weeks=(WEEK_COL, "count"), active_weeks=(QTY_COL, lambda s: int((s > 0).sum())),
        total_qty=(QTY_COL, "sum"), n_unique_qty=(QTY_COL, "nunique"),
    ).reset_index()

    expected_span = ((summary["last_week"] - summary["first_week"]).dt.days / 7 + 1).astype(int)
    assert (expected_span == summary["history_weeks"]).all(), "SKU별 first_week~last_week 구간에 결측 주차 존재"

    summary["zero_ratio"] = (summary["history_weeks"] - summary["active_weeks"]) / summary["history_weeks"]
    summary["all_zero_flag"] = summary["total_qty"] == 0
    summary["constant_flag"] = summary["n_unique_qty"] == 1
    return summary


def _fit_forecast(train_log1p: np.ndarray, return_valid_fits: bool = False):
    """auto_arima fit 후 4-step forecast까지 시도. 예외는 호출부에서 처리.
    탐색 중 발생한 warning은 개수만 세고 메시지는 버린다(최종 convergence와는 별개 지표)."""
    kwargs = {**AUTO_ARIMA_KWARGS, "return_valid_fits": return_valid_fits}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ret = auto_arima(train_log1p, **kwargs)
    model = extract_valid_fit_candidates(ret)[0] if return_valid_fits else ret
    fc = np.asarray(model.predict(n_periods=FORECAST_N_PERIODS))
    if not np.isfinite(fc).all():
        raise ValueError(f"forecast에 NaN/inf 포함: {fc.tolist()}")
    forecast = {h: expm1_clip(fc[step])[0] for step, h in FORECAST_STEP_TO_HORIZON.items()}
    return ret, model, forecast, len(caught)


def run_smoke_test(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for center_id, sku_id in SMOKE_TEST_SKUS:
        sub = df[(df[CENTER_COL] == center_id) & (df[SKU_COL] == sku_id)].sort_values(WEEK_COL)
        row = {
            "center_id": center_id, "sku_id": sku_id, "n_obs": len(sub),
            "fit_success": False, "selected_order": None, "final_converged": "Unknown",
            "forecast_success": False, "error": "",
        }
        try:
            qty_log1p = np.log1p(sub[QTY_COL].to_numpy())
            _, model, forecast, _ = _fit_forecast(qty_log1p)
            row.update({
                "fit_success": True, "selected_order": model.order,
                "final_converged": candidate_convergence(model), "forecast_success": True,
                **{f"forecast_{h}": v for h, v in forecast.items()},
            })
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return pd.DataFrame(rows)


def sample_and_run_200(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for center in CENTERS:
        pop = summary[(summary[CENTER_COL] == center) & (~summary["constant_flag"])]
        frames.append(pop.sample(n=N_SAMPLE_PER_CENTER, random_state=RANDOM_STATE)[[CENTER_COL, SKU_COL]])
    sampled = pd.concat(frames, ignore_index=True)

    rows = []
    for spec in sampled.itertuples(index=False):
        sub = df[(df[CENTER_COL] == spec.center_id) & (df[SKU_COL] == spec.sku_id)].sort_values(WEEK_COL)
        row = {
            "center_id": spec.center_id, "sku_id": spec.sku_id, "n_obs": len(sub),
            "fit_success": False, "selected_order": None, "aicc": np.nan,
            "search_warning_count": 0, "final_converged": "Unknown",
            "forecast_success": False, "error": "",
        }
        try:
            qty_log1p = np.log1p(sub[QTY_COL].to_numpy())
            _, model, _, n_warn = _fit_forecast(qty_log1p)
            row.update({
                "fit_success": True, "selected_order": model.order, "aicc": candidate_aicc(model),
                "search_warning_count": n_warn,
                "final_converged": candidate_convergence(model), "forecast_success": True,
            })
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return pd.DataFrame(rows)


def run_short_history_audit(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """constant_flag==False & history_weeks<=SHORT_HISTORY_AUDIT_MAX_WEEKS 전수 검증."""
    targets = summary[(~summary["constant_flag"]) & (summary["history_weeks"] <= SHORT_HISTORY_AUDIT_MAX_WEEKS)]

    rows = []
    for spec in targets.itertuples(index=False):
        sub = df[(df[CENTER_COL] == spec.center_id) & (df[SKU_COL] == spec.sku_id)].sort_values(WEEK_COL)
        row = {
            "center_id": spec.center_id, "sku_id": spec.sku_id, "history_weeks": spec.history_weeks,
            "fit_success": False, "final_converged": "Unknown", "forecast_success": False, "error": "",
        }
        try:
            qty_log1p = np.log1p(sub[QTY_COL].to_numpy())
            _, model, _, _ = _fit_forecast(qty_log1p)
            row.update({"fit_success": True, "final_converged": candidate_convergence(model), "forecast_success": True})
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_short_history(audit: pd.DataFrame) -> pd.DataFrame:
    """history_weeks=1~SHORT_HISTORY_AUDIT_MAX_WEEKS 각각에 대해 fit/forecast/convergence를
    요약한다. MIN_HISTORY=6 경계(5주 이하 vs 6주 이상)가 그대로 드러나도록 각 주차를 개별
    행으로 남기며, 이 결과로 새 threshold를 선택하지 않는다."""
    rows = []
    for hw in range(1, SHORT_HISTORY_AUDIT_MAX_WEEKS + 1):
        sub = audit[audit["history_weeks"] == hw]
        n = len(sub)
        n_fit = int(sub["fit_success"].sum())
        n_fc = int(sub["forecast_success"].sum())
        n_conv = int((sub["final_converged"] == True).sum())  # noqa: E712
        rows.append({
            "history_weeks": hw,
            "n": n,
            "fit_success_n": n_fit,
            "fit_success_rate": n_fit / n if n else np.nan,
            "forecast_success_n": n_fc,
            "forecast_success_rate": n_fc / n if n else np.nan,
            "converged_n": n_conv,
            "convergence_rate": n_conv / n_fit if n_fit else np.nan,
        })
    return pd.DataFrame(rows)


def run_fallback_audit(df: pd.DataFrame, non_converged: pd.DataFrame) -> pd.DataFrame:
    """200-sample 중 fit은 됐지만 미수렴인 SKU에 return_valid_fits=True로 재탐색해,
    같은 stepwise 경로에서 convergence==True인 후보(AICc 최소)가 있는지 확인."""
    rows = []
    for spec in non_converged.itertuples(index=False):
        sub = df[(df[CENTER_COL] == spec.center_id) & (df[SKU_COL] == spec.sku_id)].sort_values(WEEK_COL)
        row = {
            "center_id": spec.center_id, "sku_id": spec.sku_id,
            "original_order": spec.selected_order, "original_aicc": spec.aicc,
            "n_converged_candidates": 0, "fallback_available": False,
            "fallback_order": None, "fallback_aicc": np.nan, "error": "",
        }
        try:
            qty_log1p = np.log1p(sub[QTY_COL].to_numpy())
            ret, _, _, _ = _fit_forecast(qty_log1p, return_valid_fits=True)
            candidates = extract_valid_fit_candidates(ret)
            converged = [(m, candidate_aicc(m)) for m in candidates if candidate_convergence(m) is True]
            row["n_converged_candidates"] = len(converged)
            if converged:
                best, best_aicc = min(converged, key=lambda t: t[1])
                row.update({"fallback_available": True, "fallback_order": best.order, "fallback_aicc": best_aicc})
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame, smoke: pd.DataFrame, feas200: pd.DataFrame,
                   short_hist_summary: pd.DataFrame, fallback: pd.DataFrame) -> None:
    print("=" * 80)
    print("[센터별 SKU 수]")
    for center in CENTERS:
        print(f"  {center}: {int((summary[CENTER_COL] == center).sum()):,}개")

    print(f"\n[6-SKU smoke test] fit 성공 {int(smoke['fit_success'].sum())}/{len(smoke)}, "
          f"forecast 성공 {int(smoke['forecast_success'].sum())}/{len(smoke)}")

    print("\n[200-SKU feasibility] 센터별 fitting 성공률")
    for center in CENTERS:
        sub = feas200[feas200[CENTER_COL] == center]
        n_fit = int(sub["fit_success"].sum())
        print(f"  {center}: {n_fit}/{len(sub)} ({n_fit / len(sub):.1%})")

    n_with_warning = int((feas200["search_warning_count"] > 0).sum())
    n_final_nonconverged = int((feas200["fit_success"] & (feas200["final_converged"] == False)).sum())  # noqa: E712
    print("\n[200-SKU search warning vs 최종 미수렴]")
    print(f"  탐색 중 warning 발생 SKU: {n_with_warning}/{len(feas200)}")
    print(f"  fit 성공 후 최종 미수렴 SKU: {n_final_nonconverged}")

    print(f"\n[short-history(1~{SHORT_HISTORY_AUDIT_MAX_WEEKS}주) 요약] "
          f"(참고: MIN_HISTORY={MIN_HISTORY} 이상부터 ARIMA 후보, 이 결과로 threshold를 재선택하지 않음)")
    print(f"  {'history_weeks':>14} | {'n':>4} | {'fit_n':>5} | {'fit_rate':>8} | {'fc_rate':>8} | {'conv_rate':>9}")
    for _, r in short_hist_summary.iterrows():
        if int(r["history_weeks"]) == MIN_HISTORY:
            print(f"  {'-' * 14}-+-{'-' * 4}-+-{'-' * 5}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 9}  <- MIN_HISTORY={MIN_HISTORY} 경계")
        fit_rate = f"{r['fit_success_rate']:.1%}" if pd.notna(r["fit_success_rate"]) else "n/a"
        fc_rate = f"{r['forecast_success_rate']:.1%}" if pd.notna(r["forecast_success_rate"]) else "n/a"
        conv_rate = f"{r['convergence_rate']:.1%}" if pd.notna(r["convergence_rate"]) else "n/a"
        print(f"  {int(r['history_weeks']):>14} | {int(r['n']):>4} | {int(r['fit_success_n']):>5} | "
              f"{fit_rate:>8} | {fc_rate:>8} | {conv_rate:>9}")

    print("\n[non-convergence fallback]")
    if len(fallback):
        n_avail = int(fallback["fallback_available"].sum())
        print(f"  미수렴 {len(fallback)}건 중 fallback 가능 {n_avail}건")
    else:
        print("  대상 없음(200-sample에 미수렴 없음)")

    print("\n[저장 경로]")
    for p in [SKU_HISTORY_PATH, SMOKE_TEST_PATH, FEASIBILITY_200_PATH,
              SHORT_HISTORY_PATH, SHORT_HISTORY_SUMMARY_PATH, FALLBACK_PATH]:
        print(f"  {p}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_source()
    summary = build_sku_history_summary(df)
    summary.to_csv(SKU_HISTORY_PATH, index=False)

    smoke = run_smoke_test(df)
    smoke.to_csv(SMOKE_TEST_PATH, index=False)

    feas200 = sample_and_run_200(df, summary)
    feas200.to_csv(FEASIBILITY_200_PATH, index=False)

    short_hist = run_short_history_audit(df, summary)
    short_hist.to_csv(SHORT_HISTORY_PATH, index=False)
    short_hist_summary = summarize_short_history(short_hist)
    short_hist_summary.to_csv(SHORT_HISTORY_SUMMARY_PATH, index=False)

    non_converged = feas200[feas200["fit_success"] & (feas200["final_converged"] == False)]  # noqa: E712
    fallback = run_fallback_audit(df, non_converged)
    fallback.to_csv(FALLBACK_PATH, index=False)

    print_summary(summary, smoke, feas200, short_hist_summary, fallback)


if __name__ == "__main__":
    main()
