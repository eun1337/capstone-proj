"""
trainer.py
LSTM 1-fold 학습/평가. fold TRAIN으로 adi/cv2 residual NaN 계층형 median을 fit해 전체
history에 적용한 뒤 시퀀스를 만들고, common/preprocessing, common/dataset과 day3
common/evaluator, common/oof를 재사용해 fold 하나의 train/validation을 학습·평가하고
metrics/OOF/시간/체크포인트 기록을 반환한다. hidden_size/learning_rate/batch_size/
weight_decay는 호출부가 정하는 HPO 축이며, seed는 python random/numpy/torch(+CUDA)와
DataLoader shuffle generator까지 전부 고정한다(MPS 전용 deterministic 옵션은 강제하지
않음).
"""

import random
import resource
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.ml.day3_rf_lightgbm.common import evaluator as ev
from src.ml.day3_rf_lightgbm.common import oof as oo
from src.ml.day4_lstm_tft_informer.common.dataset import SequenceDataset
from src.ml.day4_lstm_tft_informer.common.preprocessing import SequencePreprocessor
from src.ml.day4_lstm_tft_informer.common.sequence_builder import (
    build_sequences,
    fold_origin_key_set,
    split_batch_by_origin_keys,
)
from src.ml.day4_lstm_tft_informer.common.structural_nan import (
    apply_residual_nan_medians,
    fit_residual_nan_medians,
)
from src.ml.day4_lstm_tft_informer.lstm.config import DROPOUT, NUM_LAYERS
from src.ml.day4_lstm_tft_informer.lstm.model import LSTMForecaster


def select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_all_seeds(seed: int) -> None:
    """Python random/NumPy/torch(+CUDA 사용 시)를 동일 seed로 맞춘다. MPS 전용
    deterministic 옵션은 강제로 켜지 않는다(미지원 연산에서 오류가 날 수 있음)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _peak_ram_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def _device_memory_mb(device: str) -> tuple:
    """디바이스 메모리 스냅샷과 그 지표 이름을 함께 반환한다. CUDA는 실제 peak
    (max_memory_allocated)이지만, MPS는 backend에 peak 추적 API가 없어
    current_allocated_memory(호출 시점 값, peak 아님)만 제공한다 - 이름에 그대로 반영한다."""
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
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    stage: str,
    model_family: str,
    config_id: str,
    checkpoint_path,
) -> dict:
    """fold(day3 common/folds.py 결과 dict, train_mask/val_mask 포함) 하나로 LSTM을
    학습·평가하고 metrics/OOF/시간 기록을 반환한다. df는 해당 fold의 train+validation
    기간을 모두 포함하는 이력이어야 한다 - validation origin의 lookback 히스토리가
    train 기간에 걸쳐 있으므로 시퀀스는 df 전체에서 한 번만 만들고, 이후 fold의
    train_mask/val_mask로 origin을 나눈다(RF/LGBM과 동일한 경계, 시퀀스 특성상 분리
    시점만 다름)."""
    t_total_start = time.perf_counter()
    device = select_device()

    set_all_seeds(seed)
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)
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
    sequence_build_time_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    preprocessor = SequencePreprocessor().fit(train_batch)
    train_transformed = preprocessor.transform(train_batch)
    val_transformed = preprocessor.transform(val_batch)
    preprocessing_time_sec = time.perf_counter() - t0

    train_dataset = SequenceDataset(train_transformed)
    val_dataset = SequenceDataset(val_transformed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=shuffle_generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    n_time_varying = train_batch.time_varying.shape[-1]
    static_cont_dim = train_batch.static_cont.shape[-1]
    vocab_sizes = preprocessor.vocab_sizes()

    model = LSTMForecaster(
        n_time_varying=n_time_varying,
        static_cont_dim=static_cont_dim,
        cat_vocab_sizes=vocab_sizes,
        hidden_size=hidden_size,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    embedding_sizes = model.embedding_sizes

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = -1

    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            tv = batch["time_varying"].to(device)
            sc = batch["static_cont"].to(device)
            scat = batch["static_cat"].to(device)
            y_log = torch.log1p(batch["target"].to(device))

            optimizer.zero_grad()
            pred_log = model(tv, sc, scat)
            loss = loss_fn(pred_log, y_log)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                tv = batch["time_varying"].to(device)
                sc = batch["static_cont"].to(device)
                scat = batch["static_cat"].to(device)
                y_log = torch.log1p(batch["target"].to(device))
                pred_log = model(tv, sc, scat)
                val_losses.append(loss_fn(pred_log, y_log).item() * len(y_log))
        val_loss = sum(val_losses) / len(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
    train_time_sec = time.perf_counter() - t0
    epochs_completed = max_epochs

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    t0 = time.perf_counter()
    pred_log_list = []
    with torch.no_grad():
        for batch in val_loader:
            tv = batch["time_varying"].to(device)
            sc = batch["static_cont"].to(device)
            scat = batch["static_cat"].to(device)
            pred_log_list.append(model(tv, sc, scat).cpu().numpy())
    pred_log = np.concatenate(pred_log_list)
    predict_time_sec = time.perf_counter() - t0

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
        "cat_vocab_sizes": vocab_sizes,
        "cat_embedding_sizes": embedding_sizes,
        "weight_decay": weight_decay,
        "n_train_sequences": len(train_batch.target),
        "n_val_sequences": len(val_batch.target),
        "n_train_insufficient_history": n_train_insufficient,
        "n_val_insufficient_history": n_val_insufficient,
        "input_shapes": {
            "time_varying": tuple(train_dataset.time_varying.shape),
            "static_cont": tuple(train_dataset.static_cont.shape),
            "static_cat": tuple(train_dataset.static_cat.shape),
        },
        "sequence_build_time_sec": sequence_build_time_sec,
        "preprocessing_time_sec": preprocessing_time_sec,
        "train_time_sec": train_time_sec,
        "predict_time_sec": predict_time_sec,
        "total_time_sec": total_time_sec,
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "device": device,
        "seed": seed,
        "peak_ram_mb": _peak_ram_mb(),
        "device_memory_mb": device_memory_mb,
        "device_memory_metric": device_memory_metric,
    }
