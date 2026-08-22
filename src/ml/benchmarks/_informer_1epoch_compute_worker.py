"""
_informer_1epoch_compute_worker.py
Informer 1-epoch compute scaling 확인용 worker. 실제 production
informer/trainer.train_and_evaluate_fold를 그대로 호출하되 max_epochs만 benchmark
전용으로 1로 override한다(architecture/feature/preprocessing/fold 등은 전혀 변경하지
않음, max_epochs는 애초에 함수 파라미터이므로 이 override 자체가 production 코드
변경이 아니다). e_layers/n_heads/lookback을 인자로 받아 Search Space 내 여러 지점의
1-epoch 비용을 비교할 수 있게 한다. 성능 metric은 참고용으로만 기록하고 판단에 쓰지 않는다.
"""

import argparse
import json
from pathlib import Path

from src.ml.day3_rf_lightgbm.common import folds as day3_folds
from src.ml.day3_rf_lightgbm.common.data_loader import load_development
from src.ml.day4_lstm_tft_informer.informer.config import BATCH_SIZE, LEARNING_RATE, SMOKE_SEED
from src.ml.day4_lstm_tft_informer.informer.trainer import train_and_evaluate_fold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e-layers", type=int, required=True)
    parser.add_argument("--n-heads", type=int, required=True)
    parser.add_argument("--lookback", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    horizon = 1

    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()
    fold = day3_folds.generate_expanding_folds(sub_a, 2022, horizon)[0]

    result = train_and_evaluate_fold(
        sub_a, fold, horizon, args.lookback,
        e_layers=args.e_layers, n_heads=args.n_heads,
        batch_size=BATCH_SIZE, max_epochs=1,
        learning_rate=LEARNING_RATE, seed=SMOKE_SEED,
        stage="P10", model_family="Informer", config_id=f"1EPOCH_e{args.e_layers}_h{args.n_heads}_lb{args.lookback}",
        checkpoint_path=args.checkpoint,
    )

    out = {
        "family": "Informer",
        "configuration": (
            f"lookback={args.lookback}, e_layers={args.e_layers}, n_heads={args.n_heads}, "
            f"batch_size={BATCH_SIZE}(fixed), learning_rate={LEARNING_RATE}(fixed), max_epochs=1(benchmark override)"
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
        "notes": "max_epochs=1은 benchmark 전용 override(production trainer 파라미터 그대로 사용, 코드 변경 아님). compute scaling 확인 목적, 성능 비교 아님",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
