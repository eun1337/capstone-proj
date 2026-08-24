"""
trainer.py

Informer 학습/평가.

fold train으로 residual NaN 처리와 전처리를 fit한 뒤 encoder/decoder 입력을 구성해
한 CV fold의 학습·평가와 OOF 생성을 수행한다. P13은 n_heads만 탐색하며,
EarlyStopping 없이 max_epochs를 모두 학습한 마지막 epoch 모델을 평가한다.
"""

import random
import resource
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.forecasting.common import evaluator as ev
from src.forecasting.common import oof as oo
from src.forecasting.deep_learning.common.preprocessing import SequencePreprocessor
from src.forecasting.deep_learning.common.sequence_builder import (
    build_sequences,
    fold_origin_key_set,
    split_batch_by_origin_keys,
)
from src.forecasting.deep_learning.common.structural_nan import (
    apply_residual_nan_medians,
    fit_residual_nan_medians,
)
from src.forecasting.deep_learning.informer.config import (
    D_FF,
    D_LAYERS,
    D_MODEL,
    DROPOUT,
    FACTOR,
    label_len_for,
)
from src.forecasting.deep_learning.informer.dataset import InformerSequenceDataset, build_informer_tensors
from src.forecasting.deep_learning.informer.model import InformerForecaster


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


def _run_epoch_predict(model, loader, device, n_total):
    """sample index를 이용해 원본 batch 순서로 정렬된 pred_log를 반환한다."""
    pred_log = np.empty(n_total, dtype=float)
    filled = np.zeros(n_total, dtype=bool)
    with torch.no_grad():
        for batch in loader:
            idx = batch["idx"].numpy()
            enc = batch["encoder_input"].to(device)
            dv = batch["decoder_value"].to(device)
            dk = batch["decoder_known"].to(device)
            sc = batch["static_cont"].to(device)
            scat = batch["static_cat"].to(device)
            out = model(enc, dv, dk, sc, scat).cpu().numpy()
            pred_log[idx] = out
            filled[idx] = True
    if not filled.all():
        missing = np.where(~filled)[0]
        raise AssertionError(f"prediction idx가 전체 batch를 커버하지 못함: 누락 {len(missing)}건 (예: {missing[:5]})")
    return pred_log


def train_and_evaluate_fold(
    df,
    fold: dict,
    horizon: int,
    lookback: int,
    *,
    e_layers: int,
    n_heads: int,
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
    seed: int,
    stage: str,
    model_family: str,
    config_id: str,
    checkpoint_path,
) -> dict:
    """한 CV fold의 Informer를 학습·평가하고 metrics/OOF/시간 기록을 반환한다.
    validation origin의 lookback history를 보존하기 위해 전체 이력에서 시퀀스를 만든 뒤
    fold의 train/validation origin으로 분리한다.
    """
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
    train_tensors = build_informer_tensors(preprocessor, train_batch, df_imputed, horizon)
    val_tensors = build_informer_tensors(preprocessor, val_batch, df_imputed, horizon)
    label_len = train_tensors["label_len"]
    preprocessing_time_sec = time.perf_counter() - t0

    train_dataset = InformerSequenceDataset(train_tensors)
    val_dataset = InformerSequenceDataset(val_tensors)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=shuffle_generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    n_time_varying = train_batch.time_varying.shape[-1]
    n_known = train_tensors["decoder_known"].shape[-1]
    static_cont_dim = train_batch.static_cont.shape[-1]
    vocab_sizes = preprocessor.vocab_sizes()

    model = InformerForecaster(
        n_time_varying=n_time_varying,
        n_known=n_known,
        static_cont_dim=static_cont_dim,
        cat_vocab_sizes=vocab_sizes,
        e_layers=e_layers,
        n_heads=n_heads,
        d_model=D_MODEL,
        d_ff=D_FF,
        d_layers=D_LAYERS,
        factor=FACTOR,
        dropout=DROPOUT,
    ).to(device)
    embedding_sizes = model.embedding_sizes

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    final_val_loss = float("nan")
    epochs_completed = 0

    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            enc = batch["encoder_input"].to(device)
            dv = batch["decoder_value"].to(device)
            dk = batch["decoder_known"].to(device)
            sc = batch["static_cont"].to(device)
            scat = batch["static_cat"].to(device)
            y_log = torch.log1p(batch["target"].to(device))

            optimizer.zero_grad()
            pred_log = model(enc, dv, dk, sc, scat)
            loss = loss_fn(pred_log, y_log)
            if not torch.isfinite(loss).all():
                raise FloatingPointError(
                    f"training loss가 NaN/Inf임: epoch={epoch}, batch_idx={batch_idx}, loss={loss.item()!r}"
                )
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError(
                        f"gradient가 NaN/Inf임: epoch={epoch}, batch_idx={batch_idx}, parameter={name!r}"
                    )

            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                enc = batch["encoder_input"].to(device)
                dv = batch["decoder_value"].to(device)
                dk = batch["decoder_known"].to(device)
                sc = batch["static_cont"].to(device)
                scat = batch["static_cat"].to(device)
                y_log = torch.log1p(batch["target"].to(device))
                pred_log = model(enc, dv, dk, sc, scat)
                val_losses.append(loss_fn(pred_log, y_log).item() * len(y_log))
        final_val_loss = sum(val_losses) / len(val_dataset)
        epochs_completed = epoch
    train_time_sec = time.perf_counter() - t0

    # EarlyStopping 없이 마지막 epoch의 in-memory weights를 평가한다.
    torch.save(model.state_dict(), checkpoint_path)
    model.eval()

    t0 = time.perf_counter()
    pred_log = _run_epoch_predict(model, val_loader, device, len(val_dataset))
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
        "label_len": label_len,
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
        "e_layers": e_layers,
        "n_heads": n_heads,
        "lookback": lookback,
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
    e_layers: int,
    n_heads: int,
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
    seed: int,
    model_artifact_path,
) -> dict:
    """validation 없이 전체 이력으로 max_epochs를 학습하고 마지막 epoch artifact를 저장한다."""
    device = select_device()
    set_all_seeds(seed)
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)

    residual_nan_maps = fit_residual_nan_medians(full_df)
    df_imputed, _ = apply_residual_nan_medians(full_df, residual_nan_maps)
    batch = build_sequences(df_imputed, horizon, lookback)

    preprocessor = SequencePreprocessor().fit(batch)
    tensors = build_informer_tensors(preprocessor, batch, df_imputed, horizon)
    label_len = tensors["label_len"]

    dataset = InformerSequenceDataset(tensors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=shuffle_generator)

    model = InformerForecaster(
        n_time_varying=batch.time_varying.shape[-1], n_known=tensors["decoder_known"].shape[-1],
        static_cont_dim=batch.static_cont.shape[-1], cat_vocab_sizes=preprocessor.vocab_sizes(),
        e_layers=e_layers, n_heads=n_heads, d_model=D_MODEL, d_ff=D_FF, d_layers=D_LAYERS,
        factor=FACTOR, dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    t0 = time.perf_counter()
    epochs_completed = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for b in loader:
            enc = b["encoder_input"].to(device)
            dv = b["decoder_value"].to(device)
            dk = b["decoder_known"].to(device)
            sc = b["static_cont"].to(device)
            scat = b["static_cat"].to(device)
            y_log = torch.log1p(b["target"].to(device))
            optimizer.zero_grad()
            pred_log = model(enc, dv, dk, sc, scat)
            loss = loss_fn(pred_log, y_log)
            loss.backward()
            optimizer.step()
        epochs_completed = epoch
    fit_time_sec = time.perf_counter() - t0

    torch.save({"model_state_dict": model.state_dict(), "preprocessor": preprocessor,
               "e_layers": e_layers, "n_heads": n_heads, "label_len": label_len}, model_artifact_path)

    return {
        "model_artifact_path": str(model_artifact_path),
        "n_train_sequences": len(dataset),
        "epochs_completed": epochs_completed,
        "label_len": label_len,
        "fit_time_sec": fit_time_sec,
        "device": device,
    }
