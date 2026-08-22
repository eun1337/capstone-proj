"""
trainer.py
LightGBM 1-fold 학습/평가. LGBMPreprocessor + common/evaluator.py + common/oof.py를 재사용해
fold 하나의 train/validation을 학습·평가하고 metrics/OOF/시간 기록을 반환한다. rf/trainer.py와
동일한 인터페이스(train_and_evaluate_fold 시그니처 규약, ev/oo 모듈 참조, 반환 dict schema)를
따른다 - four_family_protocol_audit.py가 이 모듈의 `ev`/`oo`를
common/evaluator.py·common/oof.py와 동일 객체인지로 검증하므로 import alias를 바꾸면 안 된다.
"""

import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.ml.day3_rf_lightgbm.common import config as cfg
from src.ml.day3_rf_lightgbm.common import evaluator as ev
from src.ml.day3_rf_lightgbm.common import oof as oo
from src.ml.day3_rf_lightgbm.lgbm.config import FIXED_PARAMS
from src.ml.day3_rf_lightgbm.lgbm.preprocessing import LGBMPreprocessor


def train_and_evaluate_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    horizon: int,
    *,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    min_child_samples: int,
    colsample_bytree: float,
    stage: str,
    model_family: str,
    config_id: str,
    seed: int,
    fold_id,
) -> dict:
    """한 fold의 train/validation으로 LightGBM을 학습·평가하고 metrics/OOF/시간 기록을 반환한다."""
    t_total_start = time.perf_counter()

    t0 = time.perf_counter()
    preprocessor = LGBMPreprocessor()
    X_train = preprocessor.fit_transform(train_df, horizon)
    X_val = preprocessor.transform(val_df)
    preprocessing_time_sec = time.perf_counter() - t0

    target_col = cfg.TARGET_COLS[horizon]
    y_train_log = np.log1p(train_df[target_col].to_numpy(dtype=float))

    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        colsample_bytree=colsample_bytree,
        random_state=seed,
        **FIXED_PARAMS,
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train_log, categorical_feature=preprocessor.categorical_feature_names)
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
