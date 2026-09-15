"""outputs/model_comparison/cross_track_metrics_2024.csv(센터·horizon·scope 집계),
sku_model_comparison_2024.csv(SKU 단위 2024 Holdout 비교) 공용 로더.
04 최종 모델 비교(SARIMA vs Hurdle-LightGBM, 2024 Holdout) 전용 — read-only."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

OUTPUTS_DIR = Path(__file__).resolve().parents[3] / "outputs" / "model_comparison"

STAT_MODEL = "SARIMA_none"
STAT_MODEL_SKU = "SARIMA"
ML_MODEL = "Hurdle-LightGBM"
STAT_LABEL = "SARIMA"
ML_LABEL = "H-LGBM"

SCOPES = ["full_common", "model_fit", "fallback"]

METRIC_COLUMNS = {
    "WAPE": ("stat_wape", "ml_wape"),
    "Bias": ("stat_bias", "ml_bias"),
    "MAE":  ("stat_mae", "ml_mae"),
    "RMSE": ("stat_rmse", "ml_rmse"),
}

QUARTILE_LABELS = {
    "Q1": "저수요 SKU 구간",
    "Q2": "중저수요 SKU 구간",
    "Q3": "중고수요 SKU 구간",
    "Q4": "고수요 SKU 구간",
}
QUARTILE_ORDER = ["Q1", "Q2", "Q3", "Q4"]

@lru_cache(maxsize=None)
def load_cross_track() -> pd.DataFrame:
    df = pd.read_csv(OUTPUTS_DIR / "cross_track_metrics_2024.csv")
    return df[
        (df["stat_model"] == STAT_MODEL)
        & (df["ml_model"] == ML_MODEL)
        & (df["comparison_scope"].isin(SCOPES))
    ].reset_index(drop=True)

@lru_cache(maxsize=None)
def load_sku_compare() -> pd.DataFrame:
    """sku_model_comparison_2024.csv — actual_sum=0(WAPE 정의 불가, winner='tie') 행은
    수요규모 분석 대상에서 제외한다. 이 파일은 comparison_scope 컬럼이 없고 실제로 항상
    full_common 범위로 생성되어 있다(행 수가 cross_track의 full_common n_skus와 정확히 일치함)."""
    df = pd.read_csv(OUTPUTS_DIR / "sku_model_comparison_2024.csv")
    return df[df["actual_sum"] > 0].reset_index(drop=True)

@lru_cache(maxsize=None)
def demand_quartile_edges() -> tuple:
    """전체 유효 SKU-horizon 행의 actual_sum을 4등분(pd.qcut)한 경계값 — 필터가 바뀌어도
    Q1~Q4 정의 자체는 고정되도록 전체 데이터 기준으로 1회만 계산한다."""
    df = load_sku_compare()
    _, bins = pd.qcut(df["actual_sum"], 4, retbins=True, duplicates="drop")
    return tuple(float(b) for b in bins)

def assign_quartile(actual_sum: pd.Series) -> pd.Series:
    edges = list(demand_quartile_edges())
    return pd.cut(actual_sum, bins=edges, labels=QUARTILE_ORDER, include_lowest=True)

def filter_sku_compare(center: str, horizon: str) -> pd.DataFrame:
    df = load_sku_compare()
    if center != "ALL":
        df = df[df["center"] == center]
    if horizon != "ALL":
        df = df[df["horizon"] == int(horizon[1:])]
    return df
