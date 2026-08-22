"""
_informer_compute_bench_worker.py
Informer 1차 compute benchmark worker. 실제 production informer/trainer.train_and_evaluate_fold를
그대로 호출해(fold generator/target_date purge/train-only preprocessing/structural NaN/
encoder-decoder tensor 생성/model fit/checkpoint/predict/inverse transform 전부 production
경로) P10 A센터 전체 h1 Fold1 규모에서 런타임/RAM/device memory만 측정한다.

대표 configuration: e_layers=2, n_heads=8, lookback=13(모두
config.py의 SMOKE_* 값과 동일). batch_size(32)/learning_rate(1e-4)는 HPO 축이 아닌 fixed
값 그대로. max_epochs도 smoke용 값이 아니라 실제 fixed 값(8)을 그대로 쓴다.
"""

import argparse
import json
from pathlib import Path

from src.ml.day3_rf_lightgbm.common import folds as day3_folds
from src.ml.day3_rf_lightgbm.common.data_loader import load_development
from src.ml.day4_lstm_tft_informer.informer.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    SMOKE_E_LAYERS,
    SMOKE_LOOKBACK,
    SMOKE_N_HEADS,
    SMOKE_SEED,
)
from src.ml.day4_lstm_tft_informer.informer.trainer import train_and_evaluate_fold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    horizon = 1
    lookback = SMOKE_LOOKBACK  # 13

    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()
    fold = day3_folds.generate_expanding_folds(sub_a, 2022, horizon)[0]

    result = train_and_evaluate_fold(
        sub_a, fold, horizon, lookback,
        e_layers=SMOKE_E_LAYERS, n_heads=SMOKE_N_HEADS,
        batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
        learning_rate=LEARNING_RATE, seed=SMOKE_SEED,
        stage="P10", model_family="Informer", config_id="COMPUTE_BENCH_REF",
        checkpoint_path=args.checkpoint,
    )

    out = {
        "family": "Informer",
        "configuration": (
            f"lookback={lookback}, e_layers={SMOKE_E_LAYERS}, n_heads={SMOKE_N_HEADS}, "
            f"batch_size={BATCH_SIZE}(fixed), learning_rate={LEARNING_RATE}(fixed), max_epochs={MAX_EPOCHS}"
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
        "notes": "e_layers/n_heads/lookback은 확정한 config.py SMOKE_* 값(성능 기반 선택 아님). batch_size/learning_rate는 HPO 축이 아닌 fixed 값. max_epochs는 실제 fixed 값(8) 그대로 사용",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
