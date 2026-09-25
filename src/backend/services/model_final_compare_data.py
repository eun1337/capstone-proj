"""outputs/model_comparison/cross_track_metrics_2024.csv(센터·horizon·scope 집계) 공용 로더.
04 최종 모델 비교의 KPI(03 탭에서도 재사용) 전용 — read-only."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

OUTPUTS_DIR = Path(__file__).resolve().parents[3] / "data" / "model_analysis" / "model_comparison"

STAT_MODEL = "SARIMA_none"
ML_MODEL = "Hurdle-LightGBM"

SCOPES = ["full_common", "model_fit", "fallback"]

METRIC_COLUMNS = {
    "WAPE": ("stat_wape", "ml_wape"),
    "Bias": ("stat_bias", "ml_bias"),
    "MAE":  ("stat_mae", "ml_mae"),
    "RMSE": ("stat_rmse", "ml_rmse"),
}

@lru_cache(maxsize=None)
def load_cross_track() -> pd.DataFrame:
    df = pd.read_csv(OUTPUTS_DIR / "cross_track_metrics_2024.csv")
    return df[
        (df["stat_model"] == STAT_MODEL)
        & (df["ml_model"] == ML_MODEL)
        & (df["comparison_scope"].isin(SCOPES))
    ].reset_index(drop=True)
