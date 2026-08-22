"""
tft_1epoch_stage_profile.py
TFT의 60분 timeout이 어느 단계에서 발생했는지 확인하기 위한 benchmark 전용 stage-by-stage
profiling. tft/trainer.train_and_evaluate_fold의 production 로직(동일 함수들을 동일 순서로
호출)을 그대로 재현하되, max_epochs=1로만 override하고(다른 모든 설정은 이전 1차
benchmark와 동일 - hidden_size/hidden_continuous_size/attention_head_size/dropout/
learning_rate/gradient_clip_val/batch_size는 config.py의 SMOKE_* 참고값, lookback=13),
각 단계가 끝날 때마다 결과를 JSON Lines 파일에 즉시 append+flush한다 - 이후 프로세스가
timeout으로 강제 종료되더라도 이미 끝난 단계의 기록은 남는다. production trainer.py/
dataset_adapter.py는 전혀 수정하지 않는다(같은 함수를 이 스크립트에서 순서대로 호출할 뿐).
성능 metric은 참고용으로만 남기고 판단에 쓰지 않는다.
"""

import argparse
import json
import time
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE

from src.ml.day3_rf_lightgbm.common import folds as day3_folds
from src.ml.day3_rf_lightgbm.common.data_loader import load_development
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
    SMOKE_ATTENTION_HEAD_SIZE,
    SMOKE_DROPOUT,
    SMOKE_GRADIENT_CLIP_VAL,
    SMOKE_HIDDEN_CONTINUOUS_SIZE,
    SMOKE_HIDDEN_SIZE,
    SMOKE_LEARNING_RATE,
    SMOKE_LOOKBACK,
    SMOKE_SEED,
)
from src.ml.day4_lstm_tft_informer.tft.dataset_adapter import (
    build_tft_long_dataframe,
    build_training_dataset,
    build_validation_dataset,
)
from src.ml.day4_lstm_tft_informer.tft.trainer import (
    _ACCELERATOR_MAP,
    _device_memory_mb,
    _peak_ram_mb,
    select_device,
    set_all_seeds,
)

HORIZON = 1
BATCH_SIZE = 128  # tft/config.py의 fixed 값(HPO 축 아님) 그대로


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="stage별 JSON Lines 로그 경로")
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()

    log_path = Path(args.out)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w", encoding="utf-8")

    def log_stage(stage: str, elapsed_sec: float, extra: dict = None) -> None:
        device_memory_mb, device_memory_metric = _device_memory_mb(select_device())
        row = {
            "stage": stage,
            "elapsed_sec": elapsed_sec,
            "cumulative_peak_ram_mb": _peak_ram_mb(),
            "device_memory_mb": device_memory_mb,
            "device_memory_metric": device_memory_metric,
        }
        if extra:
            row.update(extra)
        log_fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        log_fh.flush()
        print(f"[{stage}] {elapsed_sec:.2f}s (누적 peak_ram_mb={row['cumulative_peak_ram_mb']:.1f})", flush=True)

    lookback = SMOKE_LOOKBACK
    seed = SMOKE_SEED
    device = select_device()
    set_all_seeds(seed)

    t0 = time.perf_counter()
    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()
    fold = day3_folds.generate_expanding_folds(sub_a, 2022, HORIZON)[0]
    log_stage("data_load", time.perf_counter() - t0, {
        "n_train_rows": int(fold["train_mask"].sum()), "n_val_rows": int(fold["val_mask"].sum()),
    })

    t0 = time.perf_counter()
    residual_nan_maps = fit_residual_nan_medians(sub_a.loc[fold["train_mask"]])
    df_imputed, _ = apply_residual_nan_medians(sub_a, residual_nan_maps)
    log_stage("structural_nan_fit_apply", time.perf_counter() - t0)

    t0 = time.perf_counter()
    full_batch = build_sequences(df_imputed, HORIZON, lookback)
    train_keys = fold_origin_key_set(sub_a, fold["train_mask"])
    val_keys = fold_origin_key_set(sub_a, fold["val_mask"])
    train_batch = split_batch_by_origin_keys(full_batch, train_keys)
    val_batch = split_batch_by_origin_keys(full_batch, val_keys)
    log_stage("build_sequences", time.perf_counter() - t0, {
        "n_train_sequences": len(train_batch.target), "n_val_sequences": len(val_batch.target),
    })

    t0 = time.perf_counter()
    train_long = build_tft_long_dataframe(train_batch, df_imputed, HORIZON, group_offset=0)
    log_stage("build_tft_long_dataframe_train", time.perf_counter() - t0)

    t0 = time.perf_counter()
    val_long = build_tft_long_dataframe(val_batch, df_imputed, HORIZON, group_offset=len(train_batch.target))
    log_stage("build_tft_long_dataframe_val", time.perf_counter() - t0)

    t0 = time.perf_counter()
    training_dataset = build_training_dataset(train_long, HORIZON, lookback)
    log_stage("build_training_dataset", time.perf_counter() - t0)

    t0 = time.perf_counter()
    validation_dataset = build_validation_dataset(training_dataset, val_long)
    log_stage("build_validation_dataset", time.perf_counter() - t0)

    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    t0 = time.perf_counter()
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        hidden_size=SMOKE_HIDDEN_SIZE, hidden_continuous_size=SMOKE_HIDDEN_CONTINUOUS_SIZE,
        attention_head_size=SMOKE_ATTENTION_HEAD_SIZE, dropout=SMOKE_DROPOUT,
        lstm_layers=LSTM_LAYERS, learning_rate=SMOKE_LEARNING_RATE, optimizer=OPTIMIZER,
        loss=RMSE(), output_size=1,
    )
    log_stage("model_instantiate", time.perf_counter() - t0)

    checkpoint_dir = Path(args.checkpoint_dir)
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
        max_epochs=1,  # benchmark 전용 override
        accelerator=_ACCELERATOR_MAP[device], devices=1,
        gradient_clip_val=SMOKE_GRADIENT_CLIP_VAL,
        callbacks=[checkpoint_cb, early_stop_cb],
        enable_progress_bar=False, logger=False,
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    log_stage("epoch1_train_and_validation", time.perf_counter() - t0, {
        "epochs_completed": trainer.current_epoch,
        "best_val_loss": float(checkpoint_cb.best_model_score) if checkpoint_cb.best_model_score is not None else None,
    })

    t0 = time.perf_counter()
    best_model_path = checkpoint_cb.best_model_path
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    best_model.eval()
    log_stage("checkpoint_reload", time.perf_counter() - t0)

    t0 = time.perf_counter()
    with torch.no_grad():
        prediction_result = best_model.predict(val_dataloader, mode="prediction", return_index=True)
    log_stage("prediction", time.perf_counter() - t0, {
        "n_predictions": int(prediction_result.output.shape[0]),
    })

    log_fh.close()
    print(f"저장: {log_path}")


if __name__ == "__main__":
    main()
