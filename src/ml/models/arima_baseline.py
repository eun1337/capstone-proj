"""
arima_baseline.py
통계 트랙: ARIMA Baseline (ML 트랙과 독립된 검증)

feature_table_final.parquet에서 SKU를 모두 합산한 (center_id, week_st) 총수요 집계
시계열 하나로 auto_arima를 학습하고, rolling-origin(매 주 실제값만 반영하는 Kalman
업데이트, order 재탐색 없음) 방식으로 horizon별 예측 오차를 측정한다.

핵심 설계:
    1) 집계 단위: SKU 개별이 아니라 center 전체 총qty 합산. ARIMA를 SKU 수천 개에 개별
       적용하는 것은 비현실적이므로, LightGBM 트랙과 완전히 분리된 "센터 전체 수요
       총량" 단일 시계열로 독립 baseline을 잡는다.
    2) 정상성 검정: A는 학습 구간이 3년(156주)로 충분해 ADF/KPSS로 d를 판단(d=0 기대).
       B는 학습 구간이 2023-07~12 단 26주뿐이라 판단 자체가 통계적으로 불안정함을 리포트에 명시하고, 그럼에도 d=1을 시도
       정책 — 표본이 작아 ADF/KPSS 결과가 엇갈리거나 신뢰도가 낮을 수 있음).
    3) auto_arima는 order(p,q) 탐색에만 쓰고(d는 위 결정으로 고정), 이후 평가는 매
       origin마다 auto_arima를 재탐색하지 않고 고정 order의 SARIMAX.append(refit=False)
       (Kalman 업데이트만, 재추정 없음)로 한 주씩 전진하며 진행한다. 이렇게 해도
       매 스텝 실제값이 반영되므로 진짜 rolling-origin 평가이고, 계산 비용도 감당 가능.
    4) 평가 구간: A=val+test(2024 전체), B=pool(2024 전체, 2023-07~12 학습 이후).
    5) Horizon: A=1/4/8주, B=1/4주(B는 학습 데이터 부족으로 8주는 시도하지 않음).
    6) 지표: common.compute_metrics()를 그대로 재사용해 ML 트랙(Hurdle/Tweedie)과
       RMSE/MAE/WAPE/MASE/Bias(%) 정의를 100% 통일한다. naive_pred(MASE 분모, 1스텝
       지연 단순예측)는 ML 트랙의 "qty=해당 행 자기 주 실측"과 동일한 개념으로, 각
       origin에서 예측을 만들기 직전까지 관측된 마지막 실제값(= 그 시점까지의 마지막
       actual, horizon과 무관하게 그 origin의 모든 h에 공통 재사용)을 사용한다.
       단, 그럼에도 **WAPE/Bias(%) 수치를 ML 트랙과 절대값으로 직접 비교하면 안 된다**
       — 여전히 grain이 다르다(ARIMA=센터 전체 SKU 합산 단일 시계열, ML=SKU×주 개별
       행). 같은 원(총수요)에 대한 상대적 오차율이라는 의미에서만 참고할 것.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

from common import compute_metrics

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[3]
FEATURE_TABLE_PATH = BASE_DIR / "data" / "ml" / "splits" / "feature_table_final.parquet"

CENTER_COL = "center_id"
WEEK_COL = "week_st"
QTY_COL = "qty"

A_D_FIXED = 0
B_D_FIXED = 1
A_HORIZONS = [1, 4, 8]
B_HORIZONS = [1, 4]

A_EVAL_SPLITS = ["val", "test"]
B_EVAL_SPLITS = ["pool"]


def load_center_aggregate(df: pd.DataFrame, center: str) -> pd.DataFrame:
    """center 전체 SKU 합산 주간 총수요 집계 시계열 (연속 주간, 결측 없음 —
    Day1 reindex 설계상 grid_start==stock_week이라 qty에 구조적 NaN이 없음)."""
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
            adf_stat, adf_p = np.nan, np.nan
            print(f"    - {name}: ADF 검정 실패({e})")
        try:
            kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
        except Exception as e:
            kpss_stat, kpss_p = np.nan, np.nan
            print(f"    - {name}: KPSS 검정 실패({e})")
        adf_verdict = "정상성(H0 기각)" if pd.notna(adf_p) and adf_p < 0.05 else "비정상 가능성(H0 기각 못함)"
        kpss_verdict = "정상성(H0 기각 못함)" if pd.notna(kpss_p) and kpss_p >= 0.05 else "비정상 가능성(H0 기각)"
        print(
            f"    - {name}: ADF p={adf_p:.4f}({adf_verdict})  "
            f"KPSS p={kpss_p:.4f}({kpss_verdict})"
        )


def fit_auto_arima(train_vals: np.ndarray, d_fixed: int):
    model = auto_arima(
        train_vals,
        d=d_fixed,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=5,
        max_q=5,
        trace=False,
    )
    return model.order


def rolling_origin_eval(train_vals: np.ndarray, eval_vals: np.ndarray, order: tuple, horizons: list[int]) -> dict:
    """origin을 한 주씩 전진시키며 각 horizon의 (y_true, y_pred, naive_pred) 쌍을 모은다.
    매 스텝 order 재탐색 없이 SARIMAX.append(refit=False)로 Kalman 업데이트만 수행.
    naive_pred는 그 origin에서 예측을 만들기 직전까지 관측된 마지막 실제값(1스텝 지연
    단순예측, ML 트랙의 naive_pred=qty와 동일 개념) — horizon과 무관하게 그 origin의
    모든 h가 공유한다."""
    res = SARIMAX(
        list(train_vals), order=order, enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)

    max_h = max(horizons)
    collected = {h: {"y_true": [], "y_pred": [], "naive_pred": []} for h in horizons}
    n_eval = len(eval_vals)
    last_observed = train_vals[-1] if len(train_vals) > 0 else np.nan

    for i in range(n_eval):
        fc = res.get_forecast(steps=max_h).predicted_mean
        fc = np.asarray(fc)
        for h in horizons:
            idx = i + h - 1
            if idx < n_eval:
                collected[h]["y_true"].append(eval_vals[idx])
                collected[h]["y_pred"].append(fc[h - 1])
                collected[h]["naive_pred"].append(last_observed)
        res = res.append([eval_vals[i]], refit=False)
        last_observed = eval_vals[i]

    return collected


def run_center(df: pd.DataFrame, center: str, d_fixed: int, horizons: list[int], eval_splits: list[str]) -> pd.DataFrame:
    print("=" * 80)
    print(f"[센터 {center}] 집계 시계열 ADF/KPSS 검정")
    agg = load_center_aggregate(df, center)
    train_series = agg.loc[agg["split"] == "train", QTY_COL].reset_index(drop=True)
    eval_series = agg.loc[agg["split"].isin(eval_splits), QTY_COL].reset_index(drop=True)

    if center == "B":
        print(f"  ⚠ B센터 train 구간은 {len(train_series)}주뿐 — 표본 부족으로 ADF/KPSS 결론의 통계적 신뢰도가 낮음."
              f" 아래 검정은 참고용이며, d={d_fixed}는 검정 결론이 아니라 사전 지정 정책값.")
    adf_kpss_report(train_series, f"{center} train")

    print(f"[센터 {center}] auto_arima 학습 (d={d_fixed} 고정)")
    order = fit_auto_arima(train_series.values, d_fixed)
    print(f"  선정된 order(p,d,q) = {order}")

    print(f"[센터 {center}] rolling-origin 평가 (eval n={len(eval_series)}주, horizon={horizons})")
    collected = rolling_origin_eval(train_series.values, eval_series.values, order, horizons)

    rows = []
    for h in horizons:
        n = len(collected[h]["y_true"])
        if n == 0:
            rows.append({"center": center, "horizon": f"h{h}", "n": 0, "RMSE": np.nan, "MAE": np.nan,
                         "WAPE": np.nan, "MASE": np.nan, "Bias(%)": np.nan})
            continue
        m = compute_metrics(collected[h]["y_true"], collected[h]["y_pred"], collected[h]["naive_pred"])
        rows.append({"center": center, "horizon": f"h{h}", "n": n, **{k: round(v, 3) for k, v in m.items()}})
    return pd.DataFrame(rows)


def main():
    df = pd.read_parquet(FEATURE_TABLE_PATH, columns=[CENTER_COL, WEEK_COL, QTY_COL, "split"])

    result_a = run_center(df, "A", A_D_FIXED, A_HORIZONS, A_EVAL_SPLITS)
    result_b = run_center(df, "B", B_D_FIXED, B_HORIZONS, B_EVAL_SPLITS)

    print()
    print("=" * 80)
    print("[ARIMA Baseline 성능표] (A/B 분리, RMSE/MAE/WAPE/MASE/Bias(%), common.compute_metrics 동일 정의)")
    print("  ※ grain 주의: 센터 전체 SKU 합산 단일 시계열 기준 — SKU×주 개별 행 기준인")
    print("    Hurdle/Tweedie 트랙과 절대 수치를 직접 비교하지 말 것(상대적 참고만)")
    report = pd.concat([result_a, result_b], ignore_index=True)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
