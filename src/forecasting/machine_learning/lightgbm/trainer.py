"""
trainer.py

LightGBM 학습/평가.

train_and_evaluate_fold는 한 CV fold의 학습·평가와 OOF 생성을 담당한다.
P10은 validation L2 기준 best_iteration을 사용하고,
P13은 고정 n_estimators 전체 iteration을 사용한다.
fit_final_model은 validation 없이 전체 학습 데이터로 Final model만 fit한다.
"""

import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.forecasting.common import config as cfg
from src.forecasting.common import evaluator as ev
from src.forecasting.common import oof as oo
from src.forecasting.machine_learning.lightgbm.config import FIXED_PARAMS
from src.forecasting.machine_learning.lightgbm.preprocessing import LGBMPreprocessor


def _make_wape_diagnostic_eval(y_val_true_raw: np.ndarray):
    """iteration별 raw-scale WAPE를 diagnostic으로 계산한다."""

    def _wape_eval(y_true_log, y_pred_log):
        try:
            raw_pred = ev.inverse_transform_prediction(y_pred_log)
        except ValueError:
            return "wape", float("nan"), False
        denom = float(np.sum(y_val_true_raw))
        wape = float(np.sum(np.abs(raw_pred - y_val_true_raw)) / denom * 100) if denom != 0 else float("nan")
        return "wape", wape, False

    return _wape_eval


def train_and_evaluate_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    horizon: int,
    *,
    num_leaves: int,
    min_child_samples: int,
    feature_fraction: float,
    bagging_fraction: float,
    bagging_freq: int,
    lambda_l1: float,
    lambda_l2: float,
    stage: str,
    model_family: str,
    config_id: str,
    seed: int,
    fold_id,
    use_best_iteration: bool = True,
    fixed_params: dict | None = None,
) -> dict:
    t_total_start = time.perf_counter()
    resolved_fixed_params = fixed_params if fixed_params is not None else FIXED_PARAMS

    t0 = time.perf_counter()
    preprocessor = LGBMPreprocessor()
    X_train = preprocessor.fit_transform(train_df, horizon)
    X_val = preprocessor.transform(val_df)
    preprocessing_time_sec = time.perf_counter() - t0

    target_col = cfg.TARGET_COLS[horizon]
    y_train_log = np.log1p(train_df[target_col].to_numpy(dtype=float))
    y_val_log = np.log1p(val_df[target_col].to_numpy(dtype=float))
    y_val_true = val_df[target_col].to_numpy(dtype=float)

    model = LGBMRegressor(
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        colsample_bytree=feature_fraction,
        subsample=bagging_fraction,
        subsample_freq=bagging_freq,
        reg_alpha=lambda_l1,
        reg_lambda=lambda_l2,
        random_state=seed,
        **resolved_fixed_params,
    )

    t0 = time.perf_counter()
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        eval_metric=["l2", _make_wape_diagnostic_eval(y_val_true)],
        categorical_feature=preprocessor.categorical_feature_names,
    )
    fit_time_sec = time.perf_counter() - t0

    l2_curve = model.evals_result_["valid_0"]["l2"]
    wape_curve = model.evals_result_["valid_0"]["wape"]
    best_iteration_idx = int(np.argmin(l2_curve))
    best_iteration = best_iteration_idx + 1  # num_iteration은 1-indexed
    best_validation_loss = float(l2_curve[best_iteration_idx])
    wape_min_iteration = int(np.argmin(wape_curve)) + 1
    final_iteration = len(l2_curve)  # EarlyStopping 미사용

    predict_num_iteration = best_iteration if use_best_iteration else None

    t0 = time.perf_counter()
    pred_log = model.predict(X_val, num_iteration=predict_num_iteration)
    predict_time_sec = time.perf_counter() - t0

    raw_pred = ev.inverse_transform_prediction(pred_log)

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
        "best_iteration": best_iteration,
        "best_validation_loss": best_validation_loss,
        "wape_min_iteration": wape_min_iteration,
        "final_iteration": final_iteration,
        "used_iteration": predict_num_iteration if predict_num_iteration is not None else final_iteration,
        "use_best_iteration": use_best_iteration,
        "preprocessing_time_sec": preprocessing_time_sec,
        "fit_time_sec": fit_time_sec,
        "predict_time_sec": predict_time_sec,
        "total_time_sec": total_time_sec,
        "n_train": len(train_df),
        "n_val": len(val_df),
    }


def fit_final_model(
    full_df: pd.DataFrame,
    horizon: int,
    *,
    num_leaves: int,
    min_child_samples: int,
    feature_fraction: float,
    bagging_fraction: float,
    bagging_freq: int,
    lambda_l1: float,
    lambda_l2: float,
    seed: int,
    model_artifact_path,
    fixed_params: dict | None = None,
) -> dict:
    """validation 없이 full_df 전체 학습 후 artifact 저장(best_iteration 선택 없음)."""
    resolved_fixed_params = fixed_params if fixed_params is not None else FIXED_PARAMS
    t0 = time.perf_counter()
    preprocessor = LGBMPreprocessor()
    X = preprocessor.fit_transform(full_df, horizon)
    preprocessing_time_sec = time.perf_counter() - t0

    target_col = cfg.TARGET_COLS[horizon]
    y_log = np.log1p(full_df[target_col].to_numpy(dtype=float))

    model = LGBMRegressor(
        num_leaves=num_leaves, min_child_samples=min_child_samples,
        colsample_bytree=feature_fraction, subsample=bagging_fraction, subsample_freq=bagging_freq,
        reg_alpha=lambda_l1, reg_lambda=lambda_l2, random_state=seed,
        **resolved_fixed_params,
    )

    t0 = time.perf_counter()
    model.fit(X, y_log, categorical_feature=preprocessor.categorical_feature_names)
    fit_time_sec = time.perf_counter() - t0

    joblib.dump({"preprocessor": preprocessor, "model": model}, model_artifact_path)

    return {
        "model_artifact_path": str(model_artifact_path),
        "n_train_rows": len(full_df),
        "n_estimators_used": resolved_fixed_params["n_estimators"],
        "preprocessing_time_sec": preprocessing_time_sec,
        "fit_time_sec": fit_time_sec,
    }
