"""outputs/model_comparison/stat_pairwise_common_key_metrics_2024.csv,
stat_pairwise_common_key_audit_2024.csv, stat_family_coverage_2024.csv 공용 로더.
(src/forecasting/statistical/11_compare_arima_vs_sarima.py,
12_pairwise_common_key_comparison.py가 생성한 "동일 row(공통 key) 기준" 산출물.)

기존 outputs/model_comparison/stat_model_comparison_2024.csv / stat_common_metrics_2024.csv
(model_comparison_data.py가 읽는, family마다 서로 다른 모집단으로 계산된 산출물)는 이 모듈에서
전혀 읽지 않는다 — 02 통계모델 분석 탭은 이 모듈로만 서빙한다.
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd

OUTPUTS_DIR = Path(__file__).resolve().parents[3] / "data" / "model_analysis" / "model_comparison"

MODEL_LABELS = {
    "ARIMA_S0": "ARIMA",
    "ARIMAX_S1": "ARIMAX-S1",
    "ARIMAX_S2": "ARIMAX-S2",
    "ARIMAX_S3": "ARIMAX-S3",
    "ARIMAX_S4": "ARIMAX-S4",
    "SARIMA": "SARIMA",
    "SARIMAX_S4": "SARIMAX",
}
MODEL_ORDER = list(MODEL_LABELS.keys())

CANONICAL_PAIR = {
    "ARIMA_S0": ("ARIMA_vs_SARIMA", "ARIMA"),
    "SARIMA": ("ARIMA_vs_SARIMA", "SARIMA"),
    "ARIMAX_S1": ("ARIMA_vs_ARIMAX_S1", "ARIMAX_S1"),
    "ARIMAX_S2": ("ARIMA_vs_ARIMAX_S2", "ARIMAX_S2"),
    "ARIMAX_S3": ("ARIMA_vs_ARIMAX_S3", "ARIMAX_S3"),
    "ARIMAX_S4": ("ARIMA_vs_ARIMAX_S4", "ARIMAX_S4"),
    "SARIMAX_S4": ("SARIMA_vs_SARIMAX_S4", "SARIMAX_S4"),
}
PAIR_LABELS = {
    "ARIMA_vs_SARIMA": "ARIMA ↔ SARIMA",
    "ARIMA_vs_SARIMAX_S4": "ARIMA ↔ SARIMAX",
    "ARIMA_vs_ARIMAX_S1": "ARIMA ↔ ARIMAX-S1",
    "ARIMA_vs_ARIMAX_S2": "ARIMA ↔ ARIMAX-S2",
    "ARIMA_vs_ARIMAX_S3": "ARIMA ↔ ARIMAX-S3",
    "ARIMA_vs_ARIMAX_S4": "ARIMA ↔ ARIMAX-S4",
    "SARIMA_vs_SARIMAX_S4": "SARIMA ↔ SARIMAX",
}

EXTREME_WAPE_THRESHOLD = 1000.0


@lru_cache(maxsize=None)
def load_pairwise_metrics() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "stat_pairwise_common_key_metrics_2024.csv")


@lru_cache(maxsize=None)
def load_family_coverage() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "stat_family_coverage_2024.csv")


def canonical_row(model: str, center: str, horizon: int) -> dict | None:
    """model(예: 'ARIMAX_S1')의 대표 pair에서 center×horizon 행 하나를 반환한다.
    partner/population(n_rows/n_sku)을 항상 함께 담아 '이 population 기준'임을 알 수 있게 한다."""
    pair, side = CANONICAL_PAIR[model]
    df = load_pairwise_metrics()
    hit = df[(df["pair"] == pair) & (df["model"] == side) & (df["center_id"] == center) & (df["horizon"] == horizon)]
    if hit.empty:
        return None
    r = hit.iloc[0]
    return {
        "pair": pair, "pair_label": PAIR_LABELS[pair], "compared_against": r["compared_against"],
        "n_rows": int(r["n_rows"]), "n_sku": int(r["n_sku"]),
        "wape": float(r["WAPE"]), "bias": float(r["Bias"]), "mae": float(r["MAE"]), "rmse": float(r["RMSE"]), "mase": float(r["MASE"]),
    }


def is_extreme(model: str) -> bool:
    df = load_pairwise_metrics()
    pair, side = CANONICAL_PAIR[model]
    sub = df[(df["pair"] == pair) & (df["model"] == side) & (df["center_id"] == "ALL")]
    return bool((sub["WAPE"].abs() > EXTREME_WAPE_THRESHOLD).any())


COVERAGE_MODEL_NAME = {
    "ARIMA_S0": "ARIMA", "ARIMAX_S1": "ARIMAX_S1", "ARIMAX_S2": "ARIMAX_S2",
    "ARIMAX_S3": "ARIMAX_S3", "ARIMAX_S4": "ARIMAX_S4", "SARIMA": "SARIMA", "SARIMAX_S4": "SARIMAX_S4",
}


def coverage_row(model: str) -> dict | None:
    df = load_family_coverage()
    hit = df[df["model"] == COVERAGE_MODEL_NAME.get(model, model)]
    if hit.empty:
        return None
    r = hit.iloc[0].where(pd.notna(hit.iloc[0]), None)
    return r.to_dict()


@lru_cache(maxsize=None)
def load_all_family_panel() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "stat_all_family_common_panel_metrics_2024.csv")


def all_family_row(model: str, center: str, horizon: int) -> dict | None:
    df = load_all_family_panel()
    name = COVERAGE_MODEL_NAME.get(model, model)
    hit = df[(df["model"] == name) & (df["center_id"] == center) & (df["horizon"] == horizon)]
    if hit.empty:
        return None
    r = hit.iloc[0]
    return {
        "n_rows": int(r["n_rows"]), "n_sku": int(r["n_sku"]),
        "wape": float(r["WAPE"]), "bias": float(r["Bias"]), "mae": float(r["MAE"]), "rmse": float(r["RMSE"]), "mase": float(r["MASE"]),
    }


def is_extreme_all_family(model: str) -> bool:
    df = load_all_family_panel()
    name = COVERAGE_MODEL_NAME.get(model, model)
    sub = df[(df["model"] == name) & (df["center_id"] == "ALL")]
    return bool((sub["WAPE"].abs() > EXTREME_WAPE_THRESHOLD).any())
