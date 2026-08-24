"""
frozen_checks_ml.py

RF/LightGBM B robustness와 Final retrain workload의 frozen 검증값.

B-center walk-forward fold 수와 Final retrain A/B row 수가
사전에 확정한 audit 결과와 일치하는지 검증하며, 불일치하면 fail-fast한다.
"""

import pandas as pd

from src.forecasting.common import folds as day3_folds

FROZEN_B_ROBUSTNESS_ROW_FOLD_COUNT = {1: 25, 2: 24, 4: 22}

FROZEN_FINAL_RETRAIN_ROW_WORKLOAD = {
    1: {"a_rows": 1_496_291, "b_rows": 166_218, "total_rows": 1_662_509},
    2: {"a_rows": 1_483_594, "b_rows": 159_044, "total_rows": 1_642_638},
    4: {"a_rows": 1_458_262, "b_rows": 144_776, "total_rows": 1_603_038},
}


def verify_b_robustness_row_fold_count(dev: pd.DataFrame, horizon: int) -> None:
    """B-center row-based walk-forward fold 수가 frozen 값과 일치하는지 검증한다."""
    folds = day3_folds.generate_b_walkforward_folds(dev, horizon)
    expected = FROZEN_B_ROBUSTNESS_ROW_FOLD_COUNT[horizon]
    actual = len(folds)
    if actual != expected:
        raise RuntimeError(
            f"h{horizon}: B robustness row-based fold 개수={actual}가 frozen audit 값 "
            f"{expected}과 다름 - production fold generator/데이터가 audit 시점과 달라졌을 "
            f"가능성이 있으므로 B robustness를 진행하지 않는다"
        )


def verify_final_retrain_row_workload(dev: pd.DataFrame, horizon: int) -> None:
    """Final retrain의 A/B/전체 eligible row 수가 frozen 값과 일치하는지 검증한다."""
    a_mask = dev["center_id"] == day3_folds.CENTER_A
    b_post_mask = (dev["center_id"] == day3_folds.CENTER_B) & (dev["week_st"] >= day3_folds.B_HISTORY_START)
    target_date = dev["week_st"] + pd.Timedelta(weeks=horizon)
    from src.forecasting.common.config import FINAL_TRAIN_CUTOFF

    a_rows = int((a_mask & (target_date < FINAL_TRAIN_CUTOFF)).sum())
    b_rows = int((b_post_mask & (target_date < FINAL_TRAIN_CUTOFF)).sum())
    total_rows = int(day3_folds.final_retrain_mask(dev, horizon).sum())

    expected = FROZEN_FINAL_RETRAIN_ROW_WORKLOAD[horizon]
    actual = {"a_rows": a_rows, "b_rows": b_rows, "total_rows": total_rows}
    if actual != expected:
        raise RuntimeError(
            f"h{horizon}: Final retrain row workload={actual}가 frozen audit 값 {expected}과 "
            f"다름 - production 데이터/mask가 audit 시점과 달라졌을 가능성이 있으므로 "
            f"Final retrain을 진행하지 않는다"
        )
