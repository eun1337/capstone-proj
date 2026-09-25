"""outputs/model_comparison/stat_model_comparison_2024.csv 공용 로더.
02 통계모델 분석의 변수 조합별 효과(stat-variable-effect) 전용 — read-only."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

OUTPUTS_DIR = Path(__file__).resolve().parents[3] / "data" / "model_analysis" / "model_comparison"

STAT_KEY_TO_MODEL_VARIANT = {
    "ARIMA_S0": ("ARIMA", "S0"),
    "ARIMAX_S1": ("ARIMAX", "S1"),
    "ARIMAX_S2": ("ARIMAX", "S2"),
    "ARIMAX_S3": ("ARIMAX", "S3"),
    "ARIMAX_S4": ("ARIMAX", "S4"),
    "SARIMA": ("SARIMA", "none"),
    "SARIMAX_S4": ("SARIMAX", "S4"),
}

@lru_cache(maxsize=None)
def load_stat_model_comparison() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "stat_model_comparison_2024.csv")
