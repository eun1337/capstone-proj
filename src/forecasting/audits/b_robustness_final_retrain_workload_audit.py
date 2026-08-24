"""
b_robustness_final_retrain_workload_audit.py
B robustness(post-regime, pre-2024 expanding walk-forward)와 Final retrain(A 전체 +
B post-regime, target_date<2024-01-01)의 row/DL sequence workload를 학습 없이 측정하는
read-only audit. production fold generator(src.forecasting.common.folds)와 DL sequence
builder(src.forecasting.deep_learning.common.sequence_builder)를 그대로 사용한다.
structural NaN(adi/cv2 expanding)은
개발 데이터 전체로 1회만 fit한다 - fit_residual_nan_medians는 어떤 train subset으로 fit해도
NaN을 전부 해소하도록 설계되어 있어(전역 median fallback) fold별로 다시 fit해도 시퀀스
개수(count)에는 영향이 없고, workload 측정만이 목적인 이 audit에서는 불필요한 반복이다.
모델 fit/predict/HPO, predictive metric 계산, artifact 저장은 하지 않는다.
"""

import numpy as np
import pandas as pd

from src.forecasting.common import folds as day3_folds
from src.forecasting.common.config import (
    B_HISTORY_START,
    DEVELOPMENT_PATH,
    FINAL_TRAIN_CUTOFF,
    HORIZONS,
)
from src.forecasting.common.data_loader import load_development
from src.forecasting.deep_learning.common.config import LOOKBACK_CHOICES
from src.forecasting.deep_learning.common.sequence_builder import (
    build_sequences,
    fold_origin_key_set,
    split_batch_by_origin_keys,
)
from src.forecasting.deep_learning.common.structural_nan import (
    apply_residual_nan_medians,
    fit_residual_nan_medians,
)

CENTER_COL = "center_id"
WEEK_COL = "week_st"


def _target_date(df: pd.DataFrame, horizon: int) -> pd.Series:
    return df[WEEK_COL] + pd.Timedelta(weeks=horizon)


def _stats(counts: list[int]) -> dict:
    arr = np.asarray(counts, dtype=float)
    return {"min": int(arr.min()), "mean": float(arr.mean()), "median": float(np.median(arr)),
            "max": int(arr.max()), "sum": int(arr.sum())}


def audit_row_workload(dev: pd.DataFrame, horizon: int) -> dict:
    folds = day3_folds.generate_b_walkforward_folds(dev, horizon)
    target_date = _target_date(dev, horizon)
    train_rows = [int(f["train_mask"].sum()) for f in folds]
    val_rows = [int(f["val_mask"].sum()) for f in folds]
    max_fold = folds[int(np.argmax(train_rows))]
    return {
        "horizon": horizon,
        "folds": folds,
        "number_of_folds": len(folds),
        "first_validation_origin": folds[0]["val_week"],
        "last_validation_origin": folds[-1]["val_week"],
        "train_rows": _stats(train_rows),
        "val_rows": _stats(val_rows),
        "max_train_fold": {
            "fold_id": max_fold["fold"],
            "train_end": target_date[max_fold["train_mask"]].max(),
            "validation_origin": max_fold["val_week"],
        },
    }


def audit_dl_sequence_workload(dev: pd.DataFrame, horizon: int, lookback: int,
                                folds: list[dict], batch) -> dict:
    target_date = _target_date(dev, horizon)
    train_seq, val_seq = [], []
    max_train_n, max_fold_info = -1, None
    for f in folds:
        n_train = len(split_batch_by_origin_keys(batch, fold_origin_key_set(dev, f["train_mask"])).target)
        n_val = len(split_batch_by_origin_keys(batch, fold_origin_key_set(dev, f["val_mask"])).target)
        train_seq.append(n_train)
        val_seq.append(n_val)
        if n_train > max_train_n:
            max_train_n = n_train
            max_fold_info = {
                "fold_id": f["fold"],
                "train_end": target_date[f["train_mask"]].max(),
                "validation_origin": f["val_week"],
            }
    return {
        "horizon": horizon, "lookback": lookback, "folds": len(folds),
        "train_seq": _stats(train_seq), "val_seq": _stats(val_seq),
        "max_train_fold": max_fold_info,
    }


def audit_final_retrain(dev: pd.DataFrame, horizon: int, batches: dict) -> dict:
    target_date = _target_date(dev, horizon)
    a_mask = dev[CENTER_COL] == "A"
    b_post_mask = (dev[CENTER_COL] == "B") & (dev[WEEK_COL] >= B_HISTORY_START)
    a_rows_mask = a_mask & (target_date < FINAL_TRAIN_CUTOFF)
    b_rows_mask = b_post_mask & (target_date < FINAL_TRAIN_CUTOFF)
    combined_mask = day3_folds.final_retrain_mask(dev, horizon)
    assert int(combined_mask.sum()) == int(a_rows_mask.sum()) + int(b_rows_mask.sum())

    combined_keys = fold_origin_key_set(dev, combined_mask)
    sequences = {lb: len(split_batch_by_origin_keys(batches[lb], combined_keys).target)
                 for lb in LOOKBACK_CHOICES}
    return {
        "horizon": horizon,
        "a_rows": int(a_rows_mask.sum()),
        "b_rows": int(b_rows_mask.sum()),
        "total_rows": int(combined_mask.sum()),
        "sequences": sequences,
    }


def _fit_impute_full(dev: pd.DataFrame) -> pd.DataFrame:
    residual_maps = fit_residual_nan_medians(dev)
    imputed, _ = apply_residual_nan_medians(dev, residual_maps)
    return imputed


def main() -> None:
    dev = load_development()
    imputed = _fit_impute_full(dev)

    row_results, seq_results, retrain_results = {}, {}, {}
    for horizon in HORIZONS:
        row_results[horizon] = audit_row_workload(dev, horizon)
        folds = row_results[horizon]["folds"]
        batches = {}
        for lookback in LOOKBACK_CHOICES:
            batch = build_sequences(imputed, horizon, lookback)
            batches[lookback] = batch
            seq_results[(horizon, lookback)] = audit_dl_sequence_workload(dev, horizon, lookback, folds, batch)
        retrain_results[horizon] = audit_final_retrain(dev, horizon, batches)

    print("=" * 100)
    print("TABLE 1 - B robustness row workload")
    print("=" * 100)
    print(f"{'h':>3} {'folds':>6} {'first_origin':>12} {'last_origin':>12} "
          f"{'train_min':>10} {'train_mean':>11} {'train_med':>10} {'train_max':>10} {'train_sum':>10} "
          f"{'val_min':>8} {'val_mean':>8} {'val_med':>8} {'val_max':>8} {'val_sum':>8}")
    for horizon in HORIZONS:
        r = row_results[horizon]
        tr, vr = r["train_rows"], r["val_rows"]
        print(f"{horizon:>3} {r['number_of_folds']:>6} "
              f"{r['first_validation_origin'].date()!s:>12} {r['last_validation_origin'].date()!s:>12} "
              f"{tr['min']:>10} {tr['mean']:>11.1f} {tr['median']:>10.1f} {tr['max']:>10} {tr['sum']:>10} "
              f"{vr['min']:>8} {vr['mean']:>8.1f} {vr['median']:>8.1f} {vr['max']:>8} {vr['sum']:>8}")
    print("\nmax train_rows fold detail:")
    for horizon in HORIZONS:
        m = row_results[horizon]["max_train_fold"]
        print(f"  h{horizon}: fold_id={m['fold_id']} train_end={m['train_end'].date()} "
              f"validation_origin={m['validation_origin'].date()}")

    print()
    print("=" * 100)
    print("TABLE 2 - B robustness DL sequence workload")
    print("=" * 100)
    print(f"{'h':>3} {'lb':>4} {'folds':>6} "
          f"{'tseq_min':>9} {'tseq_mean':>10} {'tseq_med':>9} {'tseq_max':>9} {'tseq_sum':>9} "
          f"{'vseq_min':>9} {'vseq_mean':>10} {'vseq_med':>9} {'vseq_max':>9} {'vseq_sum':>9}")
    for horizon in HORIZONS:
        for lookback in LOOKBACK_CHOICES:
            r = seq_results[(horizon, lookback)]
            ts, vs = r["train_seq"], r["val_seq"]
            print(f"{horizon:>3} {lookback:>4} {r['folds']:>6} "
                  f"{ts['min']:>9} {ts['mean']:>10.1f} {ts['median']:>9.1f} {ts['max']:>9} {ts['sum']:>9} "
                  f"{vs['min']:>9} {vs['mean']:>10.1f} {vs['median']:>9.1f} {vs['max']:>9} {vs['sum']:>9}")
    print("\nmax train_sequences fold detail:")
    for horizon in HORIZONS:
        for lookback in LOOKBACK_CHOICES:
            m = seq_results[(horizon, lookback)]["max_train_fold"]
            print(f"  h{horizon} lb{lookback}: fold_id={m['fold_id']} train_end={m['train_end'].date()} "
                  f"validation_origin={m['validation_origin'].date()}")

    print()
    print("=" * 100)
    print("TABLE 3 - Final retrain workload")
    print("=" * 100)
    print(f"{'h':>3} {'A_rows':>10} {'B_rows':>10} {'total_rows':>10} "
          f"{'lb13_seq':>10} {'lb26_seq':>10}")
    for horizon in HORIZONS:
        r = retrain_results[horizon]
        print(f"{horizon:>3} {r['a_rows']:>10} {r['b_rows']:>10} {r['total_rows']:>10} "
              f"{r['sequences'][13]:>10} {r['sequences'][26]:>10}")

    print()
    print("=" * 100)
    print("audit components (production, read-only)")
    print("=" * 100)
    print(f"source dataset path      : {DEVELOPMENT_PATH}")
    print("fold generator           : src.forecasting.common.folds.generate_b_walkforward_folds")
    print("target boundary/filter   : src.forecasting.common.folds._target_date, "
          "src.forecasting.common.folds.final_retrain_mask")
    print("sequence builder         : src.forecasting.deep_learning.common.sequence_builder.build_sequences")


if __name__ == "__main__":
    main()
