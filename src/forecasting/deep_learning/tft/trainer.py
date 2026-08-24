"""
trainer.py

TFT 학습/평가.

fold train으로 residual NaN 처리를 fit한 뒤 TimeSeriesDataSet을 구성해
한 CV fold의 학습·평가와 OOF 생성을 수행한다.
EarlyStopping 없이 max_epochs를 모두 학습한 마지막 epoch 모델을 평가한다.
"""

import random
import resource
import sys
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE

from src.forecasting.common import evaluator as ev
from src.forecasting.common import oof as oo
from src.forecasting.deep_learning.common.config import get_model_feature_roles
from src.forecasting.deep_learning.common.sequence_builder import (
    build_sequences,
    fold_origin_key_set,
    split_batch_by_origin_keys,
)
from src.forecasting.deep_learning.common.structural_nan import (
    apply_residual_nan_medians,
    fit_residual_nan_medians,
)
from src.forecasting.deep_learning.tft.config import (
    LSTM_LAYERS,
    OPTIMIZER,
)
from src.forecasting.deep_learning.tft.dataset_adapter import (
    build_tft_long_dataframe,
    build_training_dataset,
    build_validation_dataset,
)

_ACCELERATOR_MAP = {"mps": "mps", "cuda": "gpu", "cpu": "cpu"}


def select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed, workers=True)


def _peak_ram_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def _device_memory_mb(device: str) -> tuple:
    """CUDA peak 또는 MPS 현재 할당 메모리와 해당 지표명을 반환한다."""
    try:
        if device == "cuda":
            return torch.cuda.max_memory_allocated() / (1024 * 1024), "cuda_max_memory_allocated_mb"
        if device == "mps":
            return torch.mps.current_allocated_memory() / (1024 * 1024), "mps_current_allocated_memory_mb"
    except Exception:
        return None, None
    return None, None


def train_and_evaluate_fold(
    df,
    fold: dict,
    horizon: int,
    lookback: int,
    *,
    hidden_size: int,
    hidden_continuous_size: int,
    attention_head_size: int,
    dropout: float,
    learning_rate: float,
    gradient_clip_val: float,
    batch_size: int,
    max_epochs: int,
    seed: int,
    stage: str,
    model_family: str,
    config_id: str,
    checkpoint_dir,
) -> dict:
    """한 CV fold의 TFT를 학습·평가하고 metrics/OOF/시간 기록을 반환한다.
    validation origin의 lookback history를 보존하기 위해 전체 이력에서 시퀀스를 만든 뒤
    fold의 train/validation origin으로 분리한다.
    """
    t_total_start = time.perf_counter()
    device = select_device()
    set_all_seeds(seed)
    fold_id = fold["fold"]

    t0 = time.perf_counter()
    residual_nan_maps = fit_residual_nan_medians(df.loc[fold["train_mask"]])
    df_imputed, residual_nan_summary = apply_residual_nan_medians(df, residual_nan_maps)
    full_batch = build_sequences(df_imputed, horizon, lookback)
    train_keys = fold_origin_key_set(df, fold["train_mask"])
    val_keys = fold_origin_key_set(df, fold["val_mask"])
    train_batch = split_batch_by_origin_keys(full_batch, train_keys)
    val_batch = split_batch_by_origin_keys(full_batch, val_keys)
    n_train_insufficient = sum(1 for k in full_batch.skipped_keys if k in train_keys)
    n_val_insufficient = sum(1 for k in full_batch.skipped_keys if k in val_keys)

    train_long = build_tft_long_dataframe(train_batch, df_imputed, horizon, group_offset=0)
    val_long = build_tft_long_dataframe(val_batch, df_imputed, horizon, group_offset=len(train_batch.target))
    sequence_build_time_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    training_dataset = build_training_dataset(train_long, horizon, lookback)
    validation_dataset = build_validation_dataset(training_dataset, val_long)
    preprocessing_time_sec = time.perf_counter() - t0

    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    val_group_offset = len(train_batch.target)

    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        hidden_size=hidden_size,
        hidden_continuous_size=hidden_continuous_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        lstm_layers=LSTM_LAYERS,
        learning_rate=learning_rate,
        optimizer=OPTIMIZER,
        loss=RMSE(),
        output_size=1,
    )

    observed_cols = set(get_model_feature_roles(horizon)["time_varying_observed"])
    decoder_reals = set(model.hparams.time_varying_reals_decoder)
    leaked_observed = observed_cols & decoder_reals
    if leaked_observed:
        raise AssertionError(f"observed feature가 decoder variable에 포함됨: {sorted(leaked_observed)}")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=_ACCELERATOR_MAP[device],
        devices=1,
        gradient_clip_val=gradient_clip_val,
        callbacks=[],
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    train_time_sec = time.perf_counter() - t0
    epochs_completed = trainer.current_epoch

    # 마지막 epoch의 in-memory model을 평가하며 final_val_loss는 diagnostic으로만 기록한다.
    final_val_loss_tensor = trainer.callback_metrics.get("val_loss")
    final_val_loss = float(final_val_loss_tensor) if final_val_loss_tensor is not None else float("nan")
    trainer.save_checkpoint(str(checkpoint_dir / "final.ckpt"))
    model.eval()

    t0 = time.perf_counter()
    with torch.no_grad():
        prediction_result = model.predict(val_dataloader, mode="prediction", return_index=True)
    predict_time_sec = time.perf_counter() - t0

    raw_pred_log = prediction_result.output.squeeze(-1).cpu().numpy()
    origin_positions = prediction_result.index["origin_id"].to_numpy() - val_group_offset

    n_val = len(val_batch.target)
    if len(origin_positions) != n_val or set(origin_positions.tolist()) != set(range(n_val)):
        raise AssertionError(
            "prediction origin_id가 val_batch의 origin과 정확히 1:1 대응하지 않음: "
            f"n_pred={len(origin_positions)}, n_val={n_val}"
        )
    pred_log = np.empty(n_val, dtype=float)
    pred_log[origin_positions] = raw_pred_log

    raw_pred = ev.inverse_transform_prediction(pred_log)
    y_val_true = val_batch.target

    oof_keys = val_batch.keys[["center_id", "sku_id", "week_st", "target_date"]].copy()
    history_df = df.loc[fold["train_mask"], ["center_id", "sku_id", "week_st", "qty"]]
    mase_scale = ev.build_mase_scale(history_df, oof_keys)
    metrics = ev.compute_metrics(y_val_true, raw_pred, mase_scale)

    oof_frame = oo.build_oof_frame(
        oof_keys, y_val_true, pred_log, raw_pred, mase_scale,
        stage=stage, model_family=model_family, config_id=config_id,
        seed=seed, horizon=horizon, fold_id=fold_id,
    )

    total_time_sec = time.perf_counter() - t_total_start
    device_memory_mb, device_memory_metric = _device_memory_mb(device)

    return {
        "metrics": metrics,
        "oof": oof_frame,
        "residual_nan_summary": residual_nan_summary,
        "n_train_sequences": len(train_batch.target),
        "n_val_sequences": len(val_batch.target),
        "n_train_insufficient_history": n_train_insufficient,
        "n_val_insufficient_history": n_val_insufficient,
        "sequence_build_time_sec": sequence_build_time_sec,
        "preprocessing_time_sec": preprocessing_time_sec,
        "train_time_sec": train_time_sec,
        "predict_time_sec": predict_time_sec,
        "total_time_sec": total_time_sec,
        "epochs_completed": epochs_completed,
        "final_epoch": epochs_completed,
        "final_val_loss": final_val_loss,
        "gradient_clip_val": gradient_clip_val,
        "optimizer": model.hparams.optimizer,
        "device": device,
        "seed": seed,
        "peak_ram_mb": _peak_ram_mb(),
        "device_memory_mb": device_memory_mb,
        "device_memory_metric": device_memory_metric,
    }


def fit_final_model(
    full_df,
    horizon: int,
    lookback: int,
    *,
    hidden_size: int,
    hidden_continuous_size: int,
    attention_head_size: int,
    dropout: float,
    learning_rate: float,
    gradient_clip_val: float,
    batch_size: int,
    max_epochs: int,
    seed: int,
    checkpoint_dir,
) -> dict:
    """validation 없이 전체 이력으로 max_epochs를 학습하고 마지막 epoch artifact를 저장한다."""
    device = select_device()
    set_all_seeds(seed)

    residual_nan_maps = fit_residual_nan_medians(full_df)
    df_imputed, _ = apply_residual_nan_medians(full_df, residual_nan_maps)
    batch = build_sequences(df_imputed, horizon, lookback)
    long_df = build_tft_long_dataframe(batch, df_imputed, horizon, group_offset=0)

    training_dataset = build_training_dataset(long_df, horizon, lookback)
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)

    model = TemporalFusionTransformer.from_dataset(
        training_dataset, hidden_size=hidden_size, hidden_continuous_size=hidden_continuous_size,
        attention_head_size=attention_head_size, dropout=dropout, lstm_layers=LSTM_LAYERS,
        learning_rate=learning_rate, optimizer=OPTIMIZER, loss=RMSE(), output_size=1,
    )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    trainer = L.Trainer(
        max_epochs=max_epochs, accelerator=_ACCELERATOR_MAP[device], devices=1,
        gradient_clip_val=gradient_clip_val, callbacks=[], enable_progress_bar=False, logger=False,
    )
    trainer.fit(model, train_dataloaders=train_dataloader)
    fit_time_sec = time.perf_counter() - t0
    epochs_completed = trainer.current_epoch

    model_artifact_path = checkpoint_dir / "final.ckpt"
    trainer.save_checkpoint(str(model_artifact_path))

    # 2024 holdout에서 동일한 train-fit dataset 설정을 재사용하기 위해 함께 저장한다.
    training_dataset.save(str(checkpoint_dir / "training_dataset.pkl"))

    return {
        "model_artifact_path": str(model_artifact_path),
        "n_train_sequences": len(batch.target),
        "epochs_completed": epochs_completed,
        "fit_time_sec": fit_time_sec,
        "device": device,
    }
