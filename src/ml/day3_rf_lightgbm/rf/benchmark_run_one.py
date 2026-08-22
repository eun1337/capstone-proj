"""
benchmark_run_one.py
RF Search Space 후보 1개 config에 대해 P10 Fold1 전체 데이터로 계산 feasibility(시간/트리 구조/
메모리)를 측정한다. 독립 프로세스로 실행해 config별로 안전하게 중단할 수 있게 한다.
성능 비교/모델 선택 목적이 아니다.
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.ml.day3_rf_lightgbm.common import config as cfg
from src.ml.day3_rf_lightgbm.common import folds as f
from src.ml.day3_rf_lightgbm.common.data_loader import load_development
from src.ml.day3_rf_lightgbm.rf.config import FIXED_PARAMS
from src.ml.day3_rf_lightgbm.rf.preprocessing import RFPreprocessor


def log(tag: str, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}][{tag}] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--validation-year", type=int, default=2022)
    parser.add_argument("--n-estimators", type=int, required=True)
    parser.add_argument("--max-features", type=float, required=True)
    parser.add_argument("--min-samples-leaf", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tag = args.config_name

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    t_total_start = time.perf_counter()

    log(tag, "데이터 로드 시작")
    t0 = time.perf_counter()
    dev = load_development()
    fold = f.generate_expanding_folds(dev, args.validation_year, args.horizon)[0]
    train_df = dev.loc[fold["train_mask"]]
    val_df = dev.loc[fold["val_mask"]]
    data_prepare_time_sec = time.perf_counter() - t0
    log(tag, f"train={len(train_df):,} val={len(val_df):,} (데이터 준비 {data_prepare_time_sec:.1f}s)")

    t0 = time.perf_counter()
    pp = RFPreprocessor()
    X_train = pp.fit_transform(train_df, args.horizon)
    X_val = pp.transform(val_df)
    preprocessing_time_sec = time.perf_counter() - t0
    log(tag, f"preprocessing 완료 {preprocessing_time_sec:.1f}s, X_train shape={X_train.shape}")

    target_col = cfg.TARGET_COLS[args.horizon]
    y_train_log = np.log1p(train_df[target_col].to_numpy(dtype=float))

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        **FIXED_PARAMS,
    )

    log(tag, f"fit 시작 (n_estimators={args.n_estimators}, max_features={args.max_features}, "
             f"min_samples_leaf={args.min_samples_leaf})")
    t0 = time.perf_counter()
    model.fit(X_train, y_train_log)
    fit_time_sec = time.perf_counter() - t0
    log(tag, f"fit 완료 {fit_time_sec:.1f}s")

    t0 = time.perf_counter()
    model.predict(X_val)
    predict_time_sec = time.perf_counter() - t0
    log(tag, f"predict 완료 {predict_time_sec:.1f}s")

    total_time_sec = time.perf_counter() - t_total_start

    depths = [est.get_depth() for est in model.estimators_]
    node_counts = [est.tree_.node_count for est in model.estimators_]
    n_leaves = [est.get_n_leaves() for est in model.estimators_]

    peak_rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mb = peak_rss_raw / (1024 * 1024) if sys.platform == "darwin" else peak_rss_raw / 1024

    result = {
        "config_name": tag,
        "n_estimators": args.n_estimators,
        "max_features": args.max_features,
        "min_samples_leaf": args.min_samples_leaf,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_transformed_features": int(X_train.shape[1]),
        "data_prepare_time_sec": data_prepare_time_sec,
        "preprocessing_time_sec": preprocessing_time_sec,
        "fit_time_sec": fit_time_sec,
        "predict_time_sec": predict_time_sec,
        "total_time_sec": total_time_sec,
        "tree_depth_mean": float(np.mean(depths)),
        "tree_depth_max": int(np.max(depths)),
        "tree_node_count_mean": float(np.mean(node_counts)),
        "tree_node_count_max": int(np.max(node_counts)),
        "tree_n_leaves_mean": float(np.mean(n_leaves)),
        "tree_n_leaves_max": int(np.max(n_leaves)),
        "peak_rss_mb": peak_rss_mb,
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    log(tag, f"결과 저장 완료 -> {args.out}")


if __name__ == "__main__":
    main()
