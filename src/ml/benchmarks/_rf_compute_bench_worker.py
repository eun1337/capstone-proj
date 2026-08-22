"""
_rf_compute_bench_worker.py
RF 1차 compute benchmark worker. 실제 production rf/trainer.train_and_evaluate_fold를
그대로 호출해(fold generator/target_date purge/train-only preprocessing/structural NaN/
model fit/predict/inverse transform 전부 production 경로) P10 A센터 전체 h1 Fold1 규모에서
런타임/RAM만 측정한다. 성능 metric은 기록하되 벤치마크 판단에 쓰지 않는다.
독립 프로세스로 실행해 상위 오케스트레이터가 timeout으로 안전하게 중단할 수 있게 한다.
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from src.ml.day3_rf_lightgbm.common import folds as day3_folds
from src.ml.day3_rf_lightgbm.common.data_loader import load_development
from src.ml.day3_rf_lightgbm.rf.trainer import train_and_evaluate_fold

# 대표 configuration: RF Search Space(N_ESTIMATORS_CHOICES/MAX_FEATURES_CHOICES/
# MIN_SAMPLES_LEAF_CHOICES)에서 각 축의 통계적 중앙값을 그대로 사용한다(성능 기반 선택 아님).
# max_features=1/3, min_samples_leaf=10은 5개 후보의 정확한 중앙값(3번째),
# n_estimators=250은 4개 후보(100,250,500,1000) 중 극단값(100,1000)이 아닌 값.
N_ESTIMATORS = 250
MAX_FEATURES = 1 / 3
MIN_SAMPLES_LEAF = 10


def _peak_ram_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    horizon = 1
    seed = 42

    t_data_start = time.perf_counter()
    dev = load_development()
    fold = day3_folds.generate_expanding_folds(dev, 2022, horizon)[0]
    train_df = dev.loc[fold["train_mask"]]
    val_df = dev.loc[fold["val_mask"]]
    data_load_time_sec = time.perf_counter() - t_data_start

    result = train_and_evaluate_fold(
        train_df, val_df, horizon,
        n_estimators=N_ESTIMATORS, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
        stage="P10", model_family="RF", config_id="COMPUTE_BENCH_REF", seed=seed, fold_id=fold["fold"],
    )

    peak_ram_mb = _peak_ram_mb()

    out = {
        "family": "RF",
        "configuration": f"n_estimators={N_ESTIMATORS}, max_features={MAX_FEATURES:.4f}, min_samples_leaf={MIN_SAMPLES_LEAF}",
        "n_train": result["n_train"],
        "n_val": result["n_val"],
        "data_load_time_sec": data_load_time_sec,
        "data_prep_time_sec": result["preprocessing_time_sec"],
        "train_time_sec": result["fit_time_sec"],
        "predict_time_sec": result["predict_time_sec"],
        "total_time_sec": result["total_time_sec"],
        "epochs_completed": None,
        "best_epoch": None,
        "peak_ram_mb": peak_ram_mb,
        "device": "cpu",
        "device_memory_mb": None,
        "device_memory_metric": None,
        "status": "completed",
        "metrics_not_used_for_decision": result["metrics"],
        "notes": "RF Search Space 각 축 중앙값(성능 기반 선택 아님)",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
