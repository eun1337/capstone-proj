"""
trainer.py
TFT 1-fold 학습/평가. fold TRAIN으로 adi/cv2 residual NaN 계층형 median을 fit해 전체
history에 적용한 뒤 tft/dataset_adapter로 TimeSeriesDataSet을 만들고, pytorch_lightning
Trainer(ModelCheckpoint+EarlyStopping+gradient_clip_val)로 학습한다. day3
common/evaluator, common/oof를 재사용해 metrics/OOF/시간/체크포인트 기록을 반환한다.
"""

import random
import re
import resource
import sys
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE

from src.ml.day3_rf_lightgbm.common import evaluator as ev
from src.ml.day3_rf_lightgbm.common import oof as oo
from src.ml.day4_lstm_tft_informer.common.config import get_model_feature_roles
from src.ml.day4_lstm_tft_informer.common.sequence_builder import (
    build_sequences,
    fold_origin_key_set,
    split_batch_by_origin_keys,
)
from src.ml.day4_lstm_tft_informer.common.structural_nan import (
    apply_residual_nan_medians,
    fit_residual_nan_medians,
)
from src.ml.day4_lstm_tft_informer.tft.config import (
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_MODE,
    EARLY_STOPPING_PATIENCE,
    LSTM_LAYERS,
    OPTIMIZER,
)
from src.ml.day4_lstm_tft_informer.tft.dataset_adapter import (
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
    """CUDA는 실제 peak(max_memory_allocated), MPS는 peak 추적 API가 없어
    current_allocated_memory(호출 시점 값, peak 아님)만 제공한다."""
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
    """fold(day3 common/folds.py 결과 dict) 하나로 TFT를 학습·평가하고 metrics/OOF/
    시간/체크포인트 기록을 반환한다. df는 해당 fold의 train+validation 기간을 모두
    포함하는 이력이어야 한다(LSTM과 동일한 fold 경계/전체 history 원칙)."""
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
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir), filename="{epoch}", monitor="val_loss", mode="min", save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA, mode=EARLY_STOPPING_MODE,
    )

    t0 = time.perf_counter()
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=_ACCELERATOR_MAP[device],
        devices=1,
        gradient_clip_val=gradient_clip_val,
        callbacks=[checkpoint_cb, early_stop_cb],
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    train_time_sec = time.perf_counter() - t0
    epochs_completed = trainer.current_epoch

    best_model_path = checkpoint_cb.best_model_path
    best_val_loss = float(checkpoint_cb.best_model_score)
    match = re.search(r"epoch=(\d+)", best_model_path)
    best_epoch = int(match.group(1)) + 1 if match else -1

    best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    best_model.eval()

    t0 = time.perf_counter()
    with torch.no_grad():
        prediction_result = best_model.predict(val_dataloader, mode="prediction", return_index=True)
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
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "gradient_clip_val": gradient_clip_val,
        "optimizer": model.hparams.optimizer,
        "device": device,
        "seed": seed,
        "peak_ram_mb": _peak_ram_mb(),
        "device_memory_mb": device_memory_mb,
        "device_memory_metric": device_memory_metric,
    }
