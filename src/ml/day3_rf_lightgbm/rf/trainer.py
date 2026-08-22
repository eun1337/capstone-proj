"""
trainer.py
Random Forest 1-fold 학습/평가. RFPreprocessor + common/evaluator.py + common/oof.py를
재사용해 fold 하나의 train/validation을 학습·평가하고 metrics/OOF/시간 기록을 반환한다.
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.ml.day3_rf_lightgbm.common import config as cfg
from src.ml.day3_rf_lightgbm.common import evaluator as ev
from src.ml.day3_rf_lightgbm.common import oof as oo
from src.ml.day3_rf_lightgbm.rf.config import FIXED_PARAMS
from src.ml.day3_rf_lightgbm.rf.preprocessing import RFPreprocessor


def train_and_evaluate_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    horizon: int,
    *,
    n_estimators: int,
    max_features,
    min_samples_leaf: int,
    stage: str,
    model_family: str,
    config_id: str,
    seed: int,
    fold_id,
) -> dict:
    """한 fold의 train/validation으로 RF를 학습·평가하고 metrics/OOF/시간 기록을 반환한다."""
    t_total_start = time.perf_counter()

    t0 = time.perf_counter()
    preprocessor = RFPreprocessor()
    X_train = preprocessor.fit_transform(train_df, horizon)
    X_val = preprocessor.transform(val_df)
    preprocessing_time_sec = time.perf_counter() - t0

    target_col = cfg.TARGET_COLS[horizon]
    y_train_log = np.log1p(train_df[target_col].to_numpy(dtype=float))

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        **FIXED_PARAMS,
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train_log)
    fit_time_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred_log = model.predict(X_val)
    predict_time_sec = time.perf_counter() - t0

    raw_pred = ev.inverse_transform_prediction(pred_log)

    y_val_true = val_df[target_col].to_numpy(dtype=float)
    val_keys = val_df[["center_id", "sku_id", "week_st"]].copy()
    val_keys["target_date"] = val_keys["week_st"] + pd.Timedelta(weeks=horizon)

    mase_scale = ev.build_mase_scale(train_df[["center_id", "sku_id", "week_st", "qty"]], val_keys)
    metrics = ev.compute_metrics(y_val_true, raw_pred, mase_scale)

    oof_frame = oo.build_oof_frame(
        val_keys, y_val_true, pred_log, raw_pred, mase_scale,
        stage=stage, model_family=model_family, config_id=config_id,
        seed=seed, horizon=horizon, fold_id=fold_id,
    )

    total_time_sec = time.perf_counter() - t_total_start

    return {
        "metrics": metrics,
        "oof": oof_frame,
        "preprocessing_time_sec": preprocessing_time_sec,
        "fit_time_sec": fit_time_sec,
        "predict_time_sec": predict_time_sec,
        "total_time_sec": total_time_sec,
        "n_train": len(train_df),
        "n_val": len(val_df),
    }
