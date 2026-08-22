"""
_tft_compute_bench_worker.py
TFT 1차 compute benchmark worker. 실제 production tft/trainer.train_and_evaluate_fold를
그대로 호출해(fold generator/target_date purge/train-only preprocessing/structural NaN/
long dataframe+TimeSeriesDataSet 생성/model fit/checkpoint/predict/inverse transform
전부 production 경로) P10 A센터 전체 h1 Fold1 규모에서 런타임/RAM/device memory만 측정한다.

대표 configuration: TFT는 hidden_size/hidden_continuous_size/attention_head_size/dropout/
learning_rate/gradient_clip_val의 HPO 축과 탐색 범위가 방법론상 이미 확정되어 있다.
다만 그 Search Space(일부는 연속형 범위라 discrete *_CHOICES 형태로 표현되지 않음)가
tft/config.py에 Optuna 탐색 범위로 아직 코드화되어 있지 않은 상태다. 이번 benchmark는
config.py에 이미 존재하는 SMOKE_* 참고값을 그대로 사용하는데, 이는 Search Space의
"중간" 지점이 아니라 구조적 계산량(hidden_size=16, attention_head_size=1 등 작은 값
위주) 기준으로 가벼운 쪽에 가까운 reference다. batch_size(128)는 애초에 HPO 축이
아니라 고정값이라 그대로 쓴다. max_epochs도 smoke용 2가 아니라 실제 fixed 값(20)을 쓴다.
"""

import argparse
import json
from pathlib import Path

from src.ml.day3_rf_lightgbm.common import folds as day3_folds
from src.ml.day3_rf_lightgbm.common.data_loader import load_development
from src.ml.day4_lstm_tft_informer.tft.config import (
    BATCH_SIZE,
    MAX_EPOCHS,
    SMOKE_ATTENTION_HEAD_SIZE,
    SMOKE_DROPOUT,
    SMOKE_GRADIENT_CLIP_VAL,
    SMOKE_HIDDEN_CONTINUOUS_SIZE,
    SMOKE_HIDDEN_SIZE,
    SMOKE_LEARNING_RATE,
    SMOKE_LOOKBACK,
    SMOKE_SEED,
)
from src.ml.day4_lstm_tft_informer.tft.trainer import train_and_evaluate_fold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()

    horizon = 1
    lookback = SMOKE_LOOKBACK  # 13

    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()
    fold = day3_folds.generate_expanding_folds(sub_a, 2022, horizon)[0]

    result = train_and_evaluate_fold(
        sub_a, fold, horizon, lookback,
        hidden_size=SMOKE_HIDDEN_SIZE, hidden_continuous_size=SMOKE_HIDDEN_CONTINUOUS_SIZE,
        attention_head_size=SMOKE_ATTENTION_HEAD_SIZE, dropout=SMOKE_DROPOUT,
        learning_rate=SMOKE_LEARNING_RATE, gradient_clip_val=SMOKE_GRADIENT_CLIP_VAL,
        batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, seed=SMOKE_SEED,
        stage="P10", model_family="TFT", config_id="COMPUTE_BENCH_REF",
        checkpoint_dir=args.checkpoint_dir,
    )

    out = {
        "family": "TFT",
        "configuration": (
            f"lookback={lookback}, hidden_size={SMOKE_HIDDEN_SIZE}, "
            f"hidden_continuous_size={SMOKE_HIDDEN_CONTINUOUS_SIZE}, "
            f"attention_head_size={SMOKE_ATTENTION_HEAD_SIZE}, dropout={SMOKE_DROPOUT}, "
            f"learning_rate={SMOKE_LEARNING_RATE}, gradient_clip_val={SMOKE_GRADIENT_CLIP_VAL}, "
            f"batch_size={BATCH_SIZE}(fixed), max_epochs={MAX_EPOCHS}"
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
        "notes": (
            "hidden_size/hidden_continuous_size/attention_head_size/dropout/learning_rate/"
            "gradient_clip_val은 HPO 축과 탐색 범위가 방법론상 이미 확정돼 있으나, 그 "
            "Search Space가 tft/config.py에 Optuna 탐색 범위로 아직 코드화되지 않아 "
            "SMOKE_* 참고값을 사용함. 이 SMOKE_* 값은 Search Space의 '중간' 지점이 아니라 "
            "구조적 계산량 기준으로 가벼운 쪽에 가까운 reference임(성능 기반 선택 아님). "
            "batch_size는 애초에 HPO 축이 아닌 fixed 값. max_epochs는 실제 "
            "fixed 값(20) 그대로 사용. sequence_build_time_sec에는 residual NaN + "
            "build_sequences + build_tft_long_dataframe(train+val)이 함께 포함되고, "
            "data_prep_time_sec에는 build_training_dataset+build_validation_dataset이 "
            "포함됨(trainer.py 자체 계측 granularity, production 코드 미변경 원칙상 이 "
            "이상 세분화하지 않음). device_memory_mb(MPS)는 peak가 아니라 predict 종료 "
            "시점의 current allocated memory 스냅샷임"
        ),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
