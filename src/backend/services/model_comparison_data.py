"""outputs/model_comparison/*.csv 공용 로더 — 프로세스당 1회만 읽고 캐시한다.
모델링/전처리 산출물은 read-only로만 사용하며 이 파일은 산출물을 생성하지 않는다."""

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

EXTREME_MODELS = {"ARIMAX_S1", "ARIMAX_S4"}

HORIZONS = [1, 2, 4]

@lru_cache(maxsize=None)
def load_stat_common_metrics() -> pd.DataFrame:
    """model x center x horizon 단위 통계모델 공통 지표(WAPE/Bias/MAE/RMSE 등)."""
    return pd.read_csv(OUTPUTS_DIR / "stat_common_metrics_2024.csv")
