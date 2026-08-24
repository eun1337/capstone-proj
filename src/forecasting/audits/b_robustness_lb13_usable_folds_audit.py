"""
b_robustness_lb13_usable_folds_audit.py
직전 b_robustness_final_retrain_workload_audit.py 결과 중 lookback=13의 fold별 usable
여부(validation_sequences>0)만 추가로 확인하는 read-only audit. lookback=26은 직전
audit에서 전 horizon 전 fold validation_sequences=0이 이미 확인됐으므로 재계산하지 않는다.
fold generator/target boundary/sequence builder는 이전 audit과 동일한 production 함수를
그대로 재사용한다. 모델 fit/predict/HPO, predictive metric 계산은 하지 않는다.
"""

import numpy as np
import pandas as pd

from src.forecasting.common import folds
from src.forecasting.common.config import HORIZONS
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

LOOKBACK = 13


def _stats(values: list[int]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {"min": int(arr.min()), "mean": float(arr.mean()), "median": float(np.median(arr)),
            "max": int(arr.max()), "sum": int(arr.sum())}


def audit_usable_folds(dev: pd.DataFrame, horizon: int, imputed: pd.DataFrame) -> dict:
    fold_list = folds.generate_b_walkforward_folds(dev, horizon)
    batch = build_sequences(imputed, horizon, LOOKBACK)

    per_fold = []
    for f in fold_list:
        n_train_rows = int(f["train_mask"].sum())
        n_train_seq = len(split_batch_by_origin_keys(batch, fold_origin_key_set(dev, f["train_mask"])).target)
        n_val_seq = len(split_batch_by_origin_keys(batch, fold_origin_key_set(dev, f["val_mask"])).target)
        per_fold.append({"fold": f["fold"], "val_week": f["val_week"],
                          "train_rows": n_train_rows, "train_seq": n_train_seq, "val_seq": n_val_seq})

    usable = [f for f in per_fold if f["val_seq"] > 0]
    unusable = [f for f in per_fold if f["val_seq"] == 0]
    usable_origins = sorted(f["val_week"] for f in usable)

    return {
        "horizon": horizon,
        "total_folds": len(per_fold),
        "usable_folds": len(usable),
        "unusable_folds": len(unusable),
        "usable_origins": usable_origins,
        "usable_train_rows": _stats([f["train_rows"] for f in usable]) if usable else None,
        "usable_train_seq": _stats([f["train_seq"] for f in usable]) if usable else None,
        "usable_val_seq": _stats([f["val_seq"] for f in usable]) if usable else None,
    }


def main() -> None:
    dev = load_development()
    residual_maps = fit_residual_nan_medians(dev)
    imputed, _ = apply_residual_nan_medians(dev, residual_maps)

    results = {h: audit_usable_folds(dev, h, imputed) for h in HORIZONS}

    for h in HORIZONS:
        r = results[h]
        print("=" * 100)
        print(f"h{h} - lookback=13 usable fold detail")
        print("=" * 100)
        print(f"total_folds={r['total_folds']} usable_folds={r['usable_folds']} "
              f"unusable_folds={r['unusable_folds']}")
        if r["usable_origins"]:
            print(f"first_usable_validation_origin={r['usable_origins'][0].date()}")
            print(f"last_usable_validation_origin={r['usable_origins'][-1].date()}")
        print("usable_validation_origins=", [d.date().isoformat() for d in r["usable_origins"]])
        for label, s in (("train_rows", r["usable_train_rows"]), ("train_seq", r["usable_train_seq"]),
                          ("val_seq", r["usable_val_seq"])):
            if s is None:
                print(f"{label}: usable fold 없음")
            else:
                print(f"{label}: min={s['min']} mean={s['mean']:.1f} median={s['median']:.1f} "
                      f"max={s['max']} sum={s['sum']}")
        print()

    print("=" * 100)
    print("SUMMARY TABLE - lookback=13 usable folds")
    print("=" * 100)
    print(f"{'h':>3} {'total_folds':>11} {'usable':>7} {'unusable':>9} "
          f"{'first_origin':>12} {'last_origin':>12} "
          f"{'train_seq_sum':>13} {'train_seq_mean':>14} {'train_seq_max':>13} {'val_seq_sum':>11}")
    for h in HORIZONS:
        r = results[h]
        ts = r["usable_train_seq"]
        vs = r["usable_val_seq"]
        first_o = r["usable_origins"][0].date() if r["usable_origins"] else None
        last_o = r["usable_origins"][-1].date() if r["usable_origins"] else None
        ts_sum = ts["sum"] if ts else "N/A"
        ts_mean = f"{ts['mean']:.1f}" if ts else "N/A"
        ts_max = ts["max"] if ts else "N/A"
        vs_sum = vs["sum"] if vs else "N/A"
        print(f"{h:>3} {r['total_folds']:>11} {r['usable_folds']:>7} {r['unusable_folds']:>9} "
              f"{str(first_o):>12} {str(last_o):>12} "
              f"{ts_sum:>13} {ts_mean:>14} {ts_max:>13} {vs_sum:>11}")

    print()
    print("lookback=26: 이전 audit(b_robustness_final_retrain_workload_audit)에서 h1/h2/h4 전 fold "
          "validation_sequences=0으로 확인됨 - 재계산하지 않음, usable_folds=0.")


if __name__ == "__main__":
    main()
