"""
tft_dataprep_profile.py
TFT data preparation 경로(build_sequences -> build_tft_long_dataframe -> build_training_dataset
-> build_validation_dataset)의 단계별 runtime/RAM을 A센터 전체(h1, P10 fold1, lookback=13)
스케일에서 측정한다. 모델/전처리 코드는 이 스크립트에서 변경하지 않는다 - 순수 profiling 전용.
"""

import resource
import sys
import time

from src.forecasting.common import folds
from src.forecasting.common.data_loader import load_development
from src.forecasting.deep_learning.common.sequence_builder import (
    build_sequences,
    fold_origin_key_set,
    split_batch_by_origin_keys,
)
from src.forecasting.deep_learning.common.structural_nan import (
    apply_residual_nan_medians,
    fit_residual_nan_medians,
)
from src.forecasting.deep_learning.tft.dataset_adapter import (
    build_tft_long_dataframe,
    build_training_dataset,
    build_validation_dataset,
)

HORIZON = 1
LOOKBACK = 13
VALIDATION_YEAR = 2022


def _ram_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def _step(name, fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    ram = _ram_mb()
    print(f"[{name}] {elapsed:.2f}s, peak_ram_mb={ram:.1f}", flush=True)
    return result, elapsed, ram


def main() -> None:
    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()
    print(f"A센터 전체 rows={len(sub_a)}, unique SKU={sub_a['sku_id'].nunique()}")

    fold = folds.generate_expanding_folds(sub_a, VALIDATION_YEAR, HORIZON)[0]
    print(f"P10 fold1: train_rows={int(fold['train_mask'].sum())}, val_rows={int(fold['val_mask'].sum())}")

    residual_maps = fit_residual_nan_medians(sub_a.loc[fold["train_mask"]])
    sub_imputed, _ = apply_residual_nan_medians(sub_a, residual_maps)

    full_batch, t_seq, _ = _step("build_sequences", build_sequences, sub_imputed, HORIZON, LOOKBACK)

    train_keys = fold_origin_key_set(sub_a, fold["train_mask"])
    val_keys = fold_origin_key_set(sub_a, fold["val_mask"])
    train_batch = split_batch_by_origin_keys(full_batch, train_keys)
    val_batch = split_batch_by_origin_keys(full_batch, val_keys)
    print(f"n_train_sequences={len(train_batch.target)}, n_val_sequences={len(val_batch.target)}")

    train_long, t_long_train, _ = _step(
        "build_tft_long_dataframe(train)", build_tft_long_dataframe, train_batch, sub_imputed, HORIZON, 0,
    )
    val_long, t_long_val, _ = _step(
        "build_tft_long_dataframe(val)", build_tft_long_dataframe, val_batch, sub_imputed, HORIZON, len(train_batch.target),
    )

    training_dataset, t_train_ds, _ = _step(
        "build_training_dataset", build_training_dataset, train_long, HORIZON, LOOKBACK,
    )
    validation_dataset, t_val_ds, _ = _step(
        "build_validation_dataset", build_validation_dataset, training_dataset, val_long,
    )

    total = t_seq + t_long_train + t_long_val + t_train_ds + t_val_ds
    print()
    print("=== 요약 ===")
    print(f"build_sequences              : {t_seq:.2f}s")
    print(f"build_tft_long_dataframe(tr) : {t_long_train:.2f}s")
    print(f"build_tft_long_dataframe(val): {t_long_val:.2f}s")
    print(f"build_training_dataset       : {t_train_ds:.2f}s")
    print(f"build_validation_dataset     : {t_val_ds:.2f}s")
    print(f"합계                          : {total:.2f}s ({total/60:.1f}분)")
    print(f"len(training_dataset)={len(training_dataset)}, len(validation_dataset)={len(validation_dataset)}")
    print(f"train_long.shape={train_long.shape}, val_long.shape={val_long.shape}")


if __name__ == "__main__":
    main()
