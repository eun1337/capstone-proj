"""outputs/model_comparison/weekly_error_2024.csv, coverage_summary_2024.csv,
stat_model_comparison_2024.csv, cross_track_predictions_2024.parquet 공용 로더.
05 상세 분석/Q&A 전용 — read-only. parquet은 27M행/100MB+라 전체 로드 없이
pyarrow.dataset filter pushdown으로 요청된 SKU·Center·Horizon 행만 읽는다."""

from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

OUTPUTS_DIR = Path(__file__).resolve().parents[3] / "outputs" / "model_comparison"

STAT_MODEL = "SARIMA"
STAT_VARIANT = "none"
ML_MODEL = "Hurdle-LightGBM"
ML_VARIANT = "operational_final"

STAT_KEY_TO_MODEL_VARIANT = {
    "ARIMA_S0": ("ARIMA", "S0"),
    "ARIMAX_S1": ("ARIMAX", "S1"),
    "ARIMAX_S2": ("ARIMAX", "S2"),
    "ARIMAX_S3": ("ARIMAX", "S3"),
    "ARIMAX_S4": ("ARIMAX", "S4"),
    "SARIMA": ("SARIMA", "none"),
    "SARIMAX_S4": ("SARIMAX", "S4"),
}

COVERAGE_SCOPE_LABELS = {"model_fit": "Model-fit", "fallback": "Fallback", "cold_start": "Cold-start"}

def weighted_weekly_combine(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """actual_sum/prediction_sum은 그대로 합산, WAPE는 sum(|err|)=WAPE/100*actual_sum로
    복원한 뒤 합산 그룹 기준으로 재계산 — 여러 center/comparison_scope 행을 하나의
    주차별 WAPE로 합칠 때 쓰는 표준 가중평균(04 WAPE Profile과 동일 원리)."""
    tmp = df.copy()
    tmp["_abs_err"] = tmp["WAPE"] / 100.0 * tmp["actual_sum"]
    g = tmp.groupby(group_cols, as_index=False).agg(
        actual_sum=("actual_sum", "sum"),
        prediction_sum=("prediction_sum", "sum"),
        _abs_err=("_abs_err", "sum"),
    )
    g["WAPE"] = g["_abs_err"] / g["actual_sum"] * 100
    g["Bias"] = (g["prediction_sum"] - g["actual_sum"]) / g["actual_sum"] * 100
    return g.drop(columns=["_abs_err"])

@lru_cache(maxsize=None)
def load_weekly_error() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "weekly_error_2024.csv")

@lru_cache(maxsize=None)
def load_stat_model_comparison() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "stat_model_comparison_2024.csv")

@lru_cache(maxsize=None)
def load_coverage_summary() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "coverage_summary_2024.csv")

@lru_cache(maxsize=None)
def _predictions_dataset():
    return ds.dataset(OUTPUTS_DIR / "cross_track_predictions_2024.parquet", format="parquet")

def query_sku_predictions(sku_id: str, center: str, horizon: int) -> pd.DataFrame:
    """선택 SKU의 SARIMA/Hurdle-LightGBM 주간 actual·prediction·comparison_scope —
    filter pushdown으로 해당 SKU/Center/Horizon 행만 읽어 27M행 전체 로드를 피한다."""
    dataset = _predictions_dataset()
    filt = (
        (ds.field("sku_id") == sku_id)
        & (ds.field("center_id") == center)
        & (ds.field("horizon") == horizon)
        & (ds.field("model").isin([STAT_MODEL, ML_MODEL]))
    )
    tbl = dataset.to_table(
        columns=["sku_id", "center_id", "horizon", "model", "variant", "target_date", "actual", "prediction", "comparison_scope"],
        filter=filt,
    )
    return tbl.to_pandas()
