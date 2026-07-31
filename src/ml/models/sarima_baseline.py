"""
sarima_baseline.py
통계 트랙 Day6: SARIMA(계절성 ARIMA, exog 없음) 검증 — arima_baseline.py 확장.

배경: arimax_exog.py 외생변수 진단에서 no_exog가 전 horizon 최적으로 확정되어 ARIMA/ARIMAX 트랙이 종료됨(팀 확정, 외생변수 재탐색은 ML/DL 트랙으로 이관). 이 파일은 그 다음 단계로 "exog 대신 계절성 자체를 모델 구조에 반영할 때 개선 여부"를 검증한다.

핵심 설계:
    1) exog는 전혀 쓰지 않는다 — arima_baseline.py 구조(SARIMAX 호출에 exog 인자 자체가 없음)를 그대로 확장, 무거운 arimax_exog.py(ablation/계수해석용)는 베이스로 쓰지 않는다.
    2) 계절 주기는 s=52(연간)만 검증한다. s=13(분기)은 CV 폴드 길이(모니터링 해상도 목적으로 채택된 것일 뿐 실제 계절 주기가 아님)를 임의로 차용한 것이라 도메인 근거가 없어 폐기함(확정).
    3) A센터: cv_pooling/folds.py의 OPTION2_FOLDS(3개월 분기 Expanding CV, 4-Window)를 재사용한다. 각 Window마다 ARIMA(seasonal=False)와 SARIMA(seasonal=True, m=52)를 모두 새로 탐색(auto_arima)하고 rolling-origin으로 평가해 동일 조건에서 비교한다. d는 Day4 확정값(A=0) 고정, 계절 차수(P,D,Q)_52만 auto_arima가 탐색한다. Window 1(train 104주=2주기)은 계절 파라미터 수렴 여부를 SARIMAX.fit() 단계의 ConvergenceWarning 캡처로 별도 모니터링한다(표본 부족 위험 구간).
    4) A센터 최종: 4-Window CV로 ARIMA/SARIMA를 비교한 뒤, 2021~2023 전체 (A_FINAL_TRAIN_START/END, folds.py)로 각각 재학습해 2024 holdout(val+test)을 1회만 채점한다(추가 재탐색 없음).
    5) B센터: train 26주로 s=52는커녕 s=13도 표본 부족이라 SARIMA 적용 대상에서 제외한다 — arima_baseline.py의 기존 로직(auto_arima, seasonal=False, d=1 고정)을 그대로 재사용해 ARIMA(0,1,0) 결과를 재확인만 한다.
    6) 산출물은 data/ml/models/sarima/에 저장한다. 콘솔 print만 하던 arima_baseline.py와 달리, 이번엔 ML 트랙(P2)에 전달할 Baseline 성적표가 필요해 CSV로 남긴다:
    - window_detail.csv / cv_summary.csv: A 4-Window CV 원자료/요약(ARIMA vs SARIMA)
    - performance_comparison.csv: 최종 확정 Baseline 성적표(A 2024 holdout ARIMA vs SARIMA, B pool 기존 ARIMA) — ML 트랙 Benchmark 타겟용
    - seasonal_order_selection.csv: 폴드별 선택된 order/seasonal_order + AIC/BIC + 수렴 경고 여부(Window 1 집중 확인 대상)
    7) 지표는 common.compute_metrics(RMSE/MAE/WAPE/MASE/Bias(%))를 그대로 재사용해 arima_baseline.py/arimax_exog.py와 정의를 통일한다.
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from common import MODEL_DIR, compute_metrics

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[3]
FEATURE_TABLE_PATH = BASE_DIR / "data" / "ml" / "splits" / "feature_table_final.parquet"
RESULT_DIR = MODEL_DIR / "sarima"

# A센터 4-Window CV 경계는 cv_pooling/folds.py(OPTION2_FOLDS)를 그대로 재사용한다.
sys.path.insert(0, str(BASE_DIR / "src" / "ml" / "experiments" / "cv_pooling"))
import folds  # noqa: E402

CENTER_COL = "center_id"
WEEK_COL = "week_st"
QTY_COL = "qty"

# Day4에서 확정된 d 고정 재사용(재탐색 없음). p,q(및 계절 P,D,Q)만 auto_arima 탐색.
A_D_FIXED = 0
B_D_FIXED = 1
A_HORIZONS = [1, 4, 8]
B_HORIZONS = [1, 4]
B_EVAL_SPLITS = ["pool"]

A_CV_WINDOWS = folds.OPTION2_FOLDS
A_FINAL_TRAIN_START = folds.A_FINAL_TRAIN_START
A_FINAL_TRAIN_END = folds.A_FINAL_TRAIN_END
A_HOLDOUT_SPLITS = ["val", "test"]

# 연간 계절 주기만 검증(s=13은 도메인 근거 없어 팀 확정으로 폐기).
SEASONAL_M = 52


def load_center_aggregate(df: pd.DataFrame, center: str) -> pd.DataFrame:
    """center 전체 SKU 합산 주간 총수요 집계 시계열(arima_baseline.py와 동일 정의)."""
    sub = df[df[CENTER_COL] == center]
    agg = (
        sub.groupby(WEEK_COL, as_index=False)
        .agg(qty=(QTY_COL, "sum"), split=("split", "first"))
        .sort_values(WEEK_COL)
        .reset_index(drop=True)
    )
    return agg


def adf_kpss_report(series: pd.Series, label: str) -> None:
    print(f"  [{label}] n={len(series)}")
    for name, s in [("원본(level)", series), ("1차 차분(diff1)", series.diff().dropna())]:
        if len(s) < 8:
            print(f"    - {name}: 표본 수 {len(s)}개로 검정 생략(너무 적음)")
            continue
        try:
            adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
        except Exception as e:
            adf_p = np.nan
            print(f"    - {name}: ADF 검정 실패({e})")
        try:
            kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
        except Exception as e:
            kpss_p = np.nan
            print(f"    - {name}: KPSS 검정 실패({e})")
        adf_verdict = "정상성(H0 기각)" if pd.notna(adf_p) and adf_p < 0.05 else "비정상 가능성(H0 기각 못함)"
        kpss_verdict = "정상성(H0 기각 못함)" if pd.notna(kpss_p) and kpss_p >= 0.05 else "비정상 가능성(H0 기각)"
        print(
            f"    - {name}: ADF p={adf_p:.4f}({adf_verdict})  "
            f"KPSS p={kpss_p:.4f}({kpss_verdict})"
        )


def fit_auto_arima(train_vals: np.ndarray, d_fixed: int, seasonal: bool, m: int) -> tuple[tuple, tuple, bool]:
    """d(및 seasonal=True일 때 m)는 고정하고 (p,q)/(P,D,Q)_m만 auto_arima가 탐색한다.
    seasonal=False면 seasonal_order=(0,0,0,0)으로 순수 ARIMA와 동일하게 반환.
    with_intercept도 함께 반환한다 — auto_arima는 필요시 내부적으로 상수항(평균 수준)을
    포함하는데, 이후 raw SARIMAX로 재적합할 때 이 정보를 넘겨주지 않으면 상수항이
    소실되어(trend=None 기본값) order가 (0,0,0)처럼 AR/MA 항이 없는 경우 예측이 통째로
    0에 수렴하는 버그가 생긴다(실측: Window 1에서 발견, WAPE 100%/Bias -100%)."""
    model = auto_arima(
        train_vals,
        d=d_fixed,
        seasonal=seasonal,
        m=m if seasonal else 1,
        max_p=5, max_q=5,
        max_P=2, max_D=1, max_Q=2,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )
    seasonal_order = model.seasonal_order if seasonal else (0, 0, 0, 0)
    return model.order, seasonal_order, model.with_intercept


def rolling_origin_eval(train_vals: np.ndarray, eval_vals: np.ndarray, order: tuple, seasonal_order: tuple,
                         horizons: list[int], trend: str | None) -> tuple[dict, dict]:
    """origin을 한 주씩 전진시키며 각 horizon의 (y_true, y_pred, naive_pred) 쌍을 모은다.
    order/seasonal_order 재탐색 없이 SARIMAX.append(refit=False)로 Kalman 업데이트만 수행(arima_baseline.py와 동일 원칙).
    trend는 fit_auto_arima가 반환한 with_intercept를 그대로 반영한 것('c' 또는 None) —
    auto_arima가 고른 모델의 상수항 유무를 재적합 시에도 그대로 보존하기 위함.
    최초 fit()만 ConvergenceWarning을 캡처해 diagnostics로 반환한다(표본이 얇은 CV Window에서 계절 파라미터 수렴 실패/불안정 여부를 확인하기 위함 — 전역 warnings.filterwarnings("ignore")와 무관하게 catch_warnings 블록 안에서는 정상적으로 잡힌다)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        res = SARIMAX(
            list(train_vals), order=order, seasonal_order=seasonal_order, trend=trend,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
    diagnostics = {
        "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in caught),
        "aic": round(float(res.aic), 3),
        "bic": round(float(res.bic), 3),
        "n_params": len(res.params),
    }

    max_h = max(horizons)
    collected = {h: {"y_true": [], "y_pred": [], "naive_pred": []} for h in horizons}
    n_eval = len(eval_vals)
    last_observed = train_vals[-1] if len(train_vals) > 0 else np.nan

    for i in range(n_eval):
        fc = np.asarray(res.get_forecast(steps=max_h).predicted_mean)
        for h in horizons:
            idx = i + h - 1
            if idx < n_eval:
                collected[h]["y_true"].append(eval_vals[idx])
                collected[h]["y_pred"].append(fc[h - 1])
                collected[h]["naive_pred"].append(last_observed)
        res = res.append([eval_vals[i]], refit=False)
        last_observed = eval_vals[i]

    return collected, diagnostics


def run_variant_eval(train_vals: np.ndarray, eval_vals: np.ndarray, d_fixed: int, horizons: list[int], seasonal: bool, m: int, center: str, variant_label: str) -> tuple[pd.DataFrame, dict]:
    """order 탐색 -> rolling-origin 평가까지 한 variant(arima/sarima)의 전체 과정."""
    order, seasonal_order, with_intercept = fit_auto_arima(train_vals, d_fixed, seasonal, m)
    trend = "c" if with_intercept else None
    collected, diag = rolling_origin_eval(train_vals, eval_vals, order, seasonal_order, horizons, trend)

    rows = []
    for h in horizons:
        n = len(collected[h]["y_true"])
        if n == 0:
            rows.append({"center": center, "variant": variant_label, "horizon": f"h{h}", "n": 0, "RMSE": np.nan, "MAE": np.nan, "WAPE": np.nan, "MASE": np.nan, "Bias(%)": np.nan})
            continue
        m_metrics = compute_metrics(collected[h]["y_true"], collected[h]["y_pred"], collected[h]["naive_pred"])
        rows.append({"center": center, "variant": variant_label, "horizon": f"h{h}", "n": n,
                     **{k: round(v, 3) for k, v in m_metrics.items()}})

    order_row = {
        "center": center, "variant": variant_label,
        "order": str(order), "seasonal_order": str(seasonal_order), "trend": trend,
        **diag,
    }
    return pd.DataFrame(rows), order_row


def run_center_a_cv(agg: pd.DataFrame, d_fixed: int, horizons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A센터 4-Window CV(OPTION2_FOLDS) — Window마다 ARIMA/SARIMA(m=52) 둘 다 새로 탐색·평가하여 동일 조건에서 비교한다."""
    window_rows: list[pd.DataFrame] = []
    order_rows: list[dict] = []

    for f in A_CV_WINDOWS:
        train_df = agg[(agg[WEEK_COL] >= pd.Timestamp(f["train_start"])) & (agg[WEEK_COL] <= pd.Timestamp(f["train_end"]))]
        val_df = agg[(agg[WEEK_COL] >= pd.Timestamp(f["val_start"])) & (agg[WEEK_COL] <= pd.Timestamp(f["val_end"]))]
        print(f"  -- Window {f['fold']}: train {len(train_df)}주 -> val {len(val_df)}주 "f"({f['train_start']}~{f['train_end']} -> {f['val_start']}~{f['val_end']})")

        for seasonal, m, variant in [(False, 1, "arima"), (True, SEASONAL_M, "sarima")]:
            perf_df, order_row = run_variant_eval(
                train_df[QTY_COL].to_numpy(dtype=float), val_df[QTY_COL].to_numpy(dtype=float),
                d_fixed, horizons, seasonal, m, "A", variant,
            )
            perf_df.insert(0, "window", f["fold"])
            window_rows.append(perf_df)
            order_row["window"] = f["fold"]
            order_rows.append(order_row)

            if variant == "sarima" and f["fold"] == 1 and order_row["convergence_warning"]:
                print(f"  ⚠ Window 1 SARIMA 수렴 경고 발생 — order={order_row['order']}, "
                    f"seasonal_order={order_row['seasonal_order']} "
                    f"(train 104주=2주기, 계절 파라미터 표본 부족 가능성)")

    window_df = pd.concat(window_rows, ignore_index=True)
    cv_summary = (
        window_df.groupby(["variant", "horizon"])["WAPE"]
        .agg(WAPE_mean="mean", WAPE_std="std", n_windows="count")
        .reset_index()
        .sort_values(["horizon", "variant"])
    )
    order_df = pd.DataFrame(order_rows)
    return window_df, cv_summary, order_df


def run_center_a_final_holdout(agg: pd.DataFrame, d_fixed: int, horizons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2021~2023 전체로 재학습 후 2024 holdout(val+test)을 ARIMA/SARIMA 각각 1회만
    채점한다(4-Window CV로 비교가 끝난 뒤 재확인 용도, 추가 order 재탐색 없음)."""
    train_df = agg[(agg[WEEK_COL] >= pd.Timestamp(A_FINAL_TRAIN_START)) & (agg[WEEK_COL] <= pd.Timestamp(A_FINAL_TRAIN_END))]
    eval_df = agg[agg["split"].isin(A_HOLDOUT_SPLITS)]
    print(f"  최종 재학습 train {len(train_df)}주(2021~2023) -> holdout eval {len(eval_df)}주(2024 val+test)")

    rows: list[pd.DataFrame] = []
    order_rows: list[dict] = []
    for seasonal, m, variant in [(False, 1, "arima"), (True, SEASONAL_M, "sarima")]:
        perf_df, order_row = run_variant_eval(
            train_df[QTY_COL].to_numpy(dtype=float), eval_df[QTY_COL].to_numpy(dtype=float),
            d_fixed, horizons, seasonal, m, "A", variant,
        )
        perf_df.insert(0, "eval_split", "holdout_2024")
        rows.append(perf_df)
        order_row["eval_split"] = "holdout_2024"
        order_rows.append(order_row)

    return pd.concat(rows, ignore_index=True), pd.DataFrame(order_rows)


def run_center_b(agg: pd.DataFrame, d_fixed: int, horizons: list[int], eval_splits: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """B센터: train 26주로 s=52(2주기 미만)는 물론 s=13도 표본 부족이라 SARIMA 적용
    대상에서 제외한다. arima_baseline.py와 동일하게 auto_arima(seasonal=False, d=1
    고정)로만 재확인한다(기존 ARIMA(0,1,0) 결과 유지가 목적, 하드코딩하지 않고 같은
    코드 경로로 재확인해 재현성을 보장)."""
    train_df = agg[agg["split"] == "train"]
    eval_df = agg[agg["split"].isin(eval_splits)]
    print(f"  train {len(train_df)}주 / eval {len(eval_df)}주(splits={eval_splits}) — "
          f"표본 부족(26주)으로 SARIMA 미적용, 기존 ARIMA 재확인만 진행")

    perf_df, order_row = run_variant_eval(
        train_df[QTY_COL].to_numpy(dtype=float), eval_df[QTY_COL].to_numpy(dtype=float),
        d_fixed, horizons, False, 1, "B", "arima",
    )
    perf_df.insert(0, "eval_split", "pool_2024")
    order_row["eval_split"] = "pool_2024"
    return perf_df, pd.DataFrame([order_row])


def main():
    print("=" * 60)
    print("[Day6 SARIMA] feature_table_final.parquet 로드 및 센터별 (week_st) 집계")
    df = pd.read_parquet(FEATURE_TABLE_PATH, columns=[CENTER_COL, WEEK_COL, QTY_COL, "split"])
    agg_a = load_center_aggregate(df, "A")
    agg_b = load_center_aggregate(df, "B")
    print(f"  A {len(agg_a):,}주 / B {len(agg_b):,}주 집계 완료")

    print()
    print("=" * 60)
    print("[센터 A] ADF/KPSS 정상성 검정(train 구간, 참고용 — d=0은 Day4 확정값 고정)")
    adf_kpss_report(agg_a.loc[agg_a["split"] == "train", QTY_COL], "A train")

    print()
    print("=" * 60)
    print(f"[센터 A] 4-Window CV(분기 Expanding, OPTION2_FOLDS) — ARIMA vs SARIMA(m={SEASONAL_M}) 비교")
    window_df, cv_summary, cv_order_df = run_center_a_cv(agg_a, A_D_FIXED, A_HORIZONS)
    print()
    print("[센터 A] CV 요약(4-Window WAPE 평균/표준편차, variant별)")
    print(cv_summary.to_string(index=False))

    print()
    print("=" * 60)
    print("[센터 A] 최종 재학습(2021~2023) -> 2024 holdout(val+test) 1회 채점")
    holdout_df, holdout_order_df = run_center_a_final_holdout(agg_a, A_D_FIXED, A_HORIZONS)
    print(holdout_df.to_string(index=False))

    print()
    print("=" * 60)
    print("[센터 B] 기존 ARIMA(0,1,0) 재확인(26주 표본 부족 — SARIMA 미적용)")
    adf_kpss_report(agg_b.loc[agg_b["split"] == "train", QTY_COL], "B train")
    b_df, b_order_df = run_center_b(agg_b, B_D_FIXED, B_HORIZONS, B_EVAL_SPLITS)
    print(b_df.to_string(index=False))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    window_detail_path = RESULT_DIR / "window_detail.csv"
    cv_summary_path = RESULT_DIR / "cv_summary.csv"
    perf_path = RESULT_DIR / "performance_comparison.csv"
    order_path = RESULT_DIR / "seasonal_order_selection.csv"

    window_df.to_csv(window_detail_path, index=False, encoding="utf-8-sig")
    cv_summary.to_csv(cv_summary_path, index=False, encoding="utf-8-sig")

    perf_df = pd.concat([holdout_df, b_df], ignore_index=True)
    perf_df.to_csv(perf_path, index=False, encoding="utf-8-sig")

    order_df = pd.concat([cv_order_df, holdout_order_df, b_order_df], ignore_index=True)
    order_df.to_csv(order_path, index=False, encoding="utf-8-sig")

    print()
    print("=" * 60)
    print("[저장 완료]")
    print(f"  A 4-Window CV 원자료      -> {window_detail_path}")
    print(f"  A 4-Window CV 요약        -> {cv_summary_path}")
    print(f"  최종 Baseline 성적표      -> {perf_path}  (ML 트랙 Benchmark 타겟용)")
    print(f"  order/seasonal_order 기록 -> {order_path}  (Window 1 수렴 여부 포함)")


if __name__ == "__main__":
    main()
