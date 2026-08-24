"""
frozen_checks_dl.py

DL B robustness와 Final retrain workload의 frozen 검증값.

lookback=13 기준 B-center usable fold와 Final retrain sequence 수가
사전에 확정한 audit 결과와 일치하는지 검증하며, 불일치하면 fail-fast한다.
"""

import pandas as pd

from src.forecasting.common import folds as day3_folds
from src.forecasting.deep_learning.common.sequence_builder import (
    build_sequences,
    fold_origin_key_set,
    split_batch_by_origin_keys,
)
from src.forecasting.deep_learning.common.structural_nan import (
    apply_residual_nan_medians,
    fit_residual_nan_medians,
)

LOOKBACK = 13

FROZEN_B_ROBUSTNESS_USABLE_FOLDS = {
    1: {"usable_folds": 13, "first_origin": "2023-09-25", "last_origin": "2023-12-18"},
    2: {"usable_folds": 12, "first_origin": "2023-09-25", "last_origin": "2023-12-11"},
    4: {"usable_folds": 10, "first_origin": "2023-09-25", "last_origin": "2023-11-27"},
}
FROZEN_B_ROBUSTNESS_TOTAL_USABLE_FOLD_HORIZON_EVALS = 35

FROZEN_FINAL_RETRAIN_LB13_SEQUENCES = {1: 1_428_083, 2: 1_408_984, 4: 1_370_946}


def _imputed_full(dev: pd.DataFrame) -> pd.DataFrame:
    residual_maps = fit_residual_nan_medians(dev)
    imputed, _ = apply_residual_nan_medians(dev, residual_maps)
    return imputed


def verify_b_robustness_usable_folds(dev: pd.DataFrame, horizon: int, imputed: pd.DataFrame | None = None) -> None:
    """lookback=13 기준 B-center usable fold 범위가 frozen 값과 일치하는지 검증한다."""
    if imputed is None:
        imputed = _imputed_full(dev)
    folds = day3_folds.generate_b_walkforward_folds(dev, horizon)
    batch = build_sequences(imputed, horizon, LOOKBACK)

    usable_origins = []
    for f in folds:
        val_keys = fold_origin_key_set(dev, f["val_mask"])
        n_val_seq = len(split_batch_by_origin_keys(batch, val_keys).target)
        if n_val_seq > 0:
            usable_origins.append(f["val_week"])
    usable_origins.sort()

    expected = FROZEN_B_ROBUSTNESS_USABLE_FOLDS[horizon]
    actual = {
        "usable_folds": len(usable_origins),
        "first_origin": usable_origins[0].date().isoformat() if usable_origins else None,
        "last_origin": usable_origins[-1].date().isoformat() if usable_origins else None,
    }
    if actual != expected:
        raise RuntimeError(
            f"h{horizon}: B robustness lookback=13 usable fold={actual}가 frozen audit 값 "
            f"{expected}과 다름 - production 데이터/sequence builder가 audit 시점과 달라졌을 "
            f"가능성이 있으므로 B robustness를 진행하지 않는다"
        )


def verify_final_retrain_lb13_sequences(dev: pd.DataFrame, horizon: int, imputed: pd.DataFrame | None = None) -> None:
    """Final retrain의 lookback=13 sequence 수가 frozen 값과 일치하는지 검증한다."""
    if imputed is None:
        imputed = _imputed_full(dev)
    combined_mask = day3_folds.final_retrain_mask(dev, horizon)
    batch = build_sequences(imputed, horizon, LOOKBACK)
    combined_keys = fold_origin_key_set(dev, combined_mask)
    n_seq = len(split_batch_by_origin_keys(batch, combined_keys).target)

    expected = FROZEN_FINAL_RETRAIN_LB13_SEQUENCES[horizon]
    if n_seq != expected:
        raise RuntimeError(
            f"h{horizon}: Final retrain lookback=13 train_sequences={n_seq}가 frozen audit 값 "
            f"{expected}과 다름 - production 데이터/sequence builder가 audit 시점과 달라졌을 "
            f"가능성이 있으므로 Final retrain을 진행하지 않는다"
        )
