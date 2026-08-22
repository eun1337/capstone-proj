"""
_lstm_compute_bench_worker.py
LSTM Search Space 내부 실제 reference config로 1차 compute benchmark를 재실행하는
worker. 이전 _lstm_bench_worker.py는 hidden_size=16/batch_size=32(SMOKE_* 값)를
써서 실제 확정 HPO Search Space(hidden_size={32,48,64,96,128,256},
batch_size={64,128,256,512,1024}, learning_rate=log 1e-6~1e-3,
weight_decay=log 1e-4~8e-4, lookback={13,26}) 밖의 값이었음이 확인되어, 사용자가
확정한 Search Space 내부 reference(hidden_size=96, batch_size=256, lookback=13,
learning_rate=1e-3, weight_decay=1e-4)로 다시 측정한다. 실제 production
lstm/trainer.train_and_evaluate_fold를 그대로 호출한다(fold generator/target_date
purge/train-only preprocessing/structural NaN/sequence build/model fit/checkpoint/
predict/inverse transform 전부 production 경로). max_epochs는 smoke용이 아니라
config.py의 실제 fixed 값(20)을 그대로 사용한다.
"""

import argparse
import json
from pathlib import Path

from src.ml.day3_rf_lightgbm.common import folds as day3_folds
from src.ml.day3_rf_lightgbm.common.data_loader import load_development
from src.ml.day4_lstm_tft_informer.lstm.config import MAX_EPOCHS
from src.ml.day4_lstm_tft_informer.lstm.trainer import train_and_evaluate_fold

# 사용자가 확정한 Search Space 내부 reference config
LOOKBACK = 13
HIDDEN_SIZE = 96
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    horizon = 1

    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()
    fold = day3_folds.generate_expanding_folds(sub_a, 2022, horizon)[0]

    result = train_and_evaluate_fold(
        sub_a, fold, horizon, LOOKBACK,
        hidden_size=HIDDEN_SIZE, batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, seed=SEED,
        stage="P10", model_family="LSTM", config_id="COMPUTE_BENCH_REF_INSPACE",
        checkpoint_path=args.checkpoint,
    )

    out = {
        "family": "LSTM",
        "configuration": (
            f"lookback={LOOKBACK}, hidden_size={HIDDEN_SIZE}, batch_size={BATCH_SIZE}, "
            f"learning_rate={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}, max_epochs={MAX_EPOCHS}"
        ),
        "n_train": result["n_train_sequences"],
        "n_val": result["n_val_sequences"],
        "sequence_build_time_sec": result["sequence_build_time_sec"],
        "data_prep_time_sec": result["preprocessing_time_sec"],
        "train_time_sec": result["train_time_sec"],
        "predict_time_sec": result["predict_time_sec"],
        "total_time_sec": result["total_time_sec"],
        "epochs_completed": result["epochs_completed"],
        "best_epoch": result["best_epoch"],
        "peak_ram_mb": result["peak_ram_mb"],
        "device": result["device"],
        "device_memory_mb": result["device_memory_mb"],
        "device_memory_metric": result["device_memory_metric"],
        "status": "completed",
        "metrics_not_used_for_decision": result["metrics"],
        "notes": "hidden_size/batch_size/learning_rate/weight_decay/lookback은 사용자가 확정한 실제 Search Space 내부 reference. max_epochs는 실제 fixed 값(20) 그대로 사용",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
