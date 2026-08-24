"""
p10_lgbm.py

LightGBM P10 technical calibration.

A센터 2022 expanding 4-fold에서 사전 고정한 15개 configuration의
학습 가능성, runtime, best_iteration을 확인한다.
P10 결과는 P13 hyperparameter 선택에 사용하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.forecasting.common import evaluator as ev
from src.forecasting.common import folds as day3_folds
from src.forecasting.common.config import HORIZONS
from src.forecasting.common.data_loader import load_development
from src.forecasting.machine_learning.lightgbm import trainer as lgbm_trainer

FAMILY = "lightgbm"

ANCHOR_HP = {
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 1,
    "lambda_l1": 1e-8,
    "lambda_l2": 1e-8,
}

# bagging_freq 비교가 활성화되도록 C10/C11의 bagging_fraction은 0.7로 둔다.
P10_CONFIGS = {
    "C01_anchor": {},
    "C02_num_leaves_lo": {"num_leaves": 8},
    "C03_num_leaves_hi": {"num_leaves": 1024},
    "C04_min_child_samples_lo": {"min_child_samples": 5},
    "C05_min_child_samples_hi": {"min_child_samples": 5000},
    "C06_feature_fraction_lo": {"feature_fraction": 0.4},
    "C07_feature_fraction_hi": {"feature_fraction": 0.7},
    "C08_bagging_fraction_lo": {"bagging_fraction": 0.4, "bagging_freq": 1},
    "C09_bagging_fraction_hi": {"bagging_fraction": 0.7, "bagging_freq": 1},
    "C10_bagging_freq_hi": {"bagging_fraction": 0.7, "bagging_freq": 7},
    "C11_bagging_freq_mid": {"bagging_fraction": 0.7, "bagging_freq": 3},
    "C12_lambda_l1_hi": {"lambda_l1": 10.0},
    "C13_lambda_l1_lo": {"lambda_l1": 1e-3},
    "C14_lambda_l2_hi": {"lambda_l2": 10.0},
    "C15_lambda_l2_lo": {"lambda_l2": 1e-3},
}


def _resolve_hp(overrides: dict) -> dict:
    hp = dict(ANCHOR_HP)
    hp.update(overrides)
    return hp


def _pooled_metrics(oof_frames: list[pd.DataFrame]) -> dict:
    pooled = pd.concat(oof_frames, ignore_index=True)
    return ev.compute_metrics(
        pooled["y_true"].to_numpy(), pooled["y_pred"].to_numpy(),
        pooled["mase_scale"].to_numpy(),
    )


def run_p10_calibration(sub_a: pd.DataFrame, horizon: int, seed: int) -> dict:
    """15개 고정 configuration을 2022 4-fold에서 실행해 technical calibration 결과를 기록한다."""
    folds = day3_folds.generate_expanding_folds(sub_a, 2022, horizon)
    fold_data = [
        (fold["fold"], sub_a.loc[fold["train_mask"]], sub_a.loc[fold["val_mask"]])
        for fold in folds
    ]
    for fold_id, train_df, val_df in fold_data:
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError(f"h{horizon} fold {fold_id}: train/val row 0개")

    config_results = []
    for config_name, overrides in P10_CONFIGS.items():
        hp = _resolve_hp(overrides)
        fold_records, oof_frames, fold_metrics = [], [], []
        config_status = "ok"

        for fold_id, train_df, val_df in fold_data:
            t0 = time.perf_counter()
            try:
                result = lgbm_trainer.train_and_evaluate_fold(
                    train_df, val_df, horizon, **hp,
                    stage="p10_calibration", model_family=FAMILY,
                    config_id=config_name, seed=seed, fold_id=fold_id,
                )
                elapsed = time.perf_counter() - t0
                fold_records.append({
                    "fold_id": fold_id, "status": "ok", "runtime_sec": elapsed,
                    "best_iteration": result["best_iteration"],
                    "best_validation_loss": result["best_validation_loss"],
                    "wape_min_iteration": result["wape_min_iteration"],
                    "n_train": result["n_train"], "n_val": result["n_val"],
                    "fold_metrics": result["metrics"],
                })
                oof_frames.append(result["oof"])
                fold_metrics.append(result["metrics"])
            except MemoryError as exc:
                elapsed = time.perf_counter() - t0
                config_status = "oom"
                fold_records.append({
                    "fold_id": fold_id, "status": "oom", "runtime_sec": elapsed,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            except Exception as exc:  # noqa: BLE001
                elapsed = time.perf_counter() - t0
                config_status = "failed"
                fold_records.append({
                    "fold_id": fold_id, "status": "failed", "runtime_sec": elapsed,
                    "error": f"{type(exc).__name__}: {exc}",
                })

        entry = {
            "config_name": config_name, "params": hp, "status": config_status,
            "n_folds_ok": sum(1 for r in fold_records if r["status"] == "ok"),
            "n_folds_total": len(fold_data),
            "total_runtime_sec": sum(r["runtime_sec"] for r in fold_records),
            "fold_records": fold_records,
        }
        if config_status == "ok" and len(oof_frames) == len(fold_data):
            pooled = _pooled_metrics(oof_frames)
            entry["pooled_metrics"] = pooled
            entry["worst_fold_wape"] = max(m["wape"] for m in fold_metrics)
        config_results.append(entry)

    return {
        "horizon": horizon, "seed": seed, "n_configs": len(P10_CONFIGS),
        "n_folds_per_config": len(fold_data), "configs": config_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM P10 15-config calibration (2022, A center)")
    parser.add_argument("--output-dir", default="outputs/hpo/lgbm")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for h in HORIZONS:
        print(f"[P10][lightgbm] horizon={h} 15-config calibration 시작 ({len(P10_CONFIGS)} configs x 4 folds)")
        result = run_p10_calibration(sub_a, h, args.seed)
        n_ok = sum(1 for c in result["configs"] if c["status"] == "ok")
        print(f"[P10][lightgbm] h{h}: {n_ok}/{len(P10_CONFIGS)} config 전부 4-fold 성공")
        out_path = out_dir / f"p10_{FAMILY}_h{h}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"[P10][lightgbm] 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
