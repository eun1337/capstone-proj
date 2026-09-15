"""outputs/model_comparison/ml_dl_model_selection_2023.csv, model_parameter_summary.csv
공용 로더 — 프로세스당 1회만 읽고 캐시한다. 산출물은 read-only로만 사용한다."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

OUTPUTS_DIR = Path(__file__).resolve().parents[3] / "outputs" / "model_comparison"

MODEL_LABELS = {
    "RF": "RF",
    "LightGBM": "LGBM",
    "LSTM": "LSTM",
    "TFT": "TFT",
    "Informer": "Informer",
    "Hurdle-RF": "H-RF",
    "Hurdle-LightGBM": "H-LGBM",
}
MODEL_ORDER = list(MODEL_LABELS.keys())
LABEL_TO_MODEL = {v: k for k, v in MODEL_LABELS.items()}

BIAS_BAND = 20.0

@lru_cache(maxsize=None)
def load_trials_raw() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "ml_dl_model_selection_2023.csv")

@lru_cache(maxsize=None)
def load_trials() -> pd.DataFrame:
    """(model, horizon, trial) 단위로 중복 제거한 trial 요약 + hpo_parameter를 dict 하나로 묶은 hp 컬럼.
    원본은 trial 하나당 hyperparameter 개수만큼 행이 반복되는 long format이라 여기서 한 번만 pivot한다."""
    df = load_trials_raw()
    hp = (
        df.groupby(["model", "horizon", "trial"])
        .apply(lambda g: dict(zip(g["hpo_parameter"], g["parameter_value"])), include_groups=False)
        .rename("hp")
    )
    base = (
        df.drop_duplicates(subset=["model", "horizon", "trial"])
        .set_index(["model", "horizon", "trial"])
        .drop(columns=["hpo_parameter", "parameter_value"])
    )
    return base.join(hp).reset_index()

@lru_cache(maxsize=None)
def load_parameter_summary() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "model_parameter_summary.csv", keep_default_na=False)
