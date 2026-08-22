# -*- coding: utf-8 -*-
"""
p10_lgbm.py
P10: A센터 2021 train → 2022 Q1~Q4 Initial Search Space 탐색(4-fold). lgbm/config.py의
Initial Search Space 안에서 Optuna(TPE)로 폭넓게 trial을 돌려 fold별/pooled 지표를 기록한다.
여기서는 최종 HP를 선택하지 않는다 - 최종 선택(P16 규칙)은 P13 Final Search Space에서
p13_lgbm.py가 수행하며, 이 결과는 P12에서 Search Space를 좁힐지 검토하는 근거
자료.

suggest_lgbm_hp()는 p13_lgbm.py가 그대로 import해 재사용한다 - Initial/Final Search Space가
같은 한(P12 조정 전) 축 정의가 두 스크립트에서 어긋나지 않도록 이 함수 하나만 둔다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna
import pandas as pd

from src.ml.day3_rf_lightgbm.common import evaluator as ev
from src.ml.day3_rf_lightgbm.common import folds as day3_folds
from src.ml.day3_rf_lightgbm.common.config import HORIZONS
from src.ml.day3_rf_lightgbm.common.data_loader import load_development
from src.ml.day3_rf_lightgbm.lgbm import trainer as lgbm_trainer
from src.ml.day3_rf_lightgbm.lgbm.config import (
    COLSAMPLE_BYTREE_CHOICES,
    LEARNING_RATE_CHOICES,
    MIN_CHILD_SAMPLES_CHOICES,
    NUM_LEAVES_CHOICES,
    N_ESTIMATORS_CHOICES,
    OPTUNA_N_STARTUP_TRIALS,
)

FAMILY = "lightgbm"


def suggest_lgbm_hp(trial: optuna.Trial) -> dict:
    """lgbm/config.py의 Initial/Final Search Space(둘 다 discrete choice) 안에서 Optuna
    TPE가 trial 하나의 하이퍼파라미터 조합을 고르게 한다. P10/P13가 동일 함수를 재사용하므로
    두 stage의 축 정의가 어긋날 수 없다."""
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", N_ESTIMATORS_CHOICES),
        "learning_rate": trial.suggest_categorical("learning_rate", LEARNING_RATE_CHOICES),
        "num_leaves": trial.suggest_categorical("num_leaves", NUM_LEAVES_CHOICES),
        "min_child_samples": trial.suggest_categorical("min_child_samples", MIN_CHILD_SAMPLES_CHOICES),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", COLSAMPLE_BYTREE_CHOICES),
    }


def _pooled_metrics(oof_frames: list[pd.DataFrame]) -> dict:
    pooled = pd.concat(oof_frames, ignore_index=True)
    return ev.compute_metrics(
        pooled["y_true"].to_numpy(), pooled["y_pred"].to_numpy(),
        pooled["mase_scale"].to_numpy(),
    )


def run_p10_search(sub_a: pd.DataFrame, horizon: int, n_trials: int, seed: int) -> dict:
    """P12 검토용 원시 자료만 만든다 - 최종 선택(P16 규칙)은 p13_lgbm.py의 책임."""
    folds = day3_folds.generate_expanding_folds(sub_a, 2022, horizon)
    fold_data = [
        (fold["fold"], sub_a.loc[fold["train_mask"]], sub_a.loc[fold["val_mask"]])
        for fold in folds
    ]
    for fold_id, train_df, val_df in fold_data:
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError(f"h{horizon} fold {fold_id}: train/val row 0개")

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=OPTUNA_N_STARTUP_TRIALS)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    trial_summaries = []

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_lgbm_hp(trial)
        oof_frames, fold_metrics = [], []

        for fold_id, train_df, val_df in fold_data:
            try:
                result = lgbm_trainer.train_and_evaluate_fold(
                    train_df, val_df, horizon, **hp,
                    stage="p10_initial_search", model_family=FAMILY,
                    config_id=f"trial{trial.number}", seed=seed, fold_id=fold_id,
                )
            except Exception as exc:  # noqa: BLE001
                trial_summaries.append({
                    "trial_id": trial.number, "params": hp, "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}", "fold_id": fold_id,
                    "pooled_wape": float("inf"), "pooled_bias": float("nan"),
                })
                return float("inf")
            oof_frames.append(result["oof"])
            fold_metrics.append(result["metrics"])

        pooled = _pooled_metrics(oof_frames)
        trial_summaries.append({
            "trial_id": trial.number, "params": hp, "status": "ok",
            "pooled_wape": pooled["wape"], "pooled_bias": pooled["bias"],
            "pooled_rmse": pooled["rmse"], "pooled_mae": pooled["mae"],
            "pooled_mase": pooled["mase"],
            "worst_fold_wape": max(m["wape"] for m in fold_metrics),
            "per_fold_metrics": fold_metrics,
        })
        return pooled["wape"]

    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    ok_trials = sorted(
        (t for t in trial_summaries if t["status"] == "ok"),
        key=lambda t: t["pooled_wape"],
    )
    failed_trials = [t for t in trial_summaries if t["status"] != "ok"]

    return {
        "horizon": horizon, "n_trials": n_trials, "seed": seed,
        "n_failed_trials": len(failed_trials),
        "all_trials": ok_trials + failed_trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM P10 Initial Search Space 탐색 (2022, A center)")
    parser.add_argument("--output-dir", default="outputs/hpo/lgbm")
    parser.add_argument("--n-trials", type=int, required=True, help="P10 탐색 trial 수(P12 검토용, P13 n_trials와 동일할 필요 없음)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for h in HORIZONS:
        print(f"[P10][lightgbm] horizon={h} Initial Search 탐색 시작 (n_trials={args.n_trials})")
        result = run_p10_search(sub_a, h, args.n_trials, args.seed)
        top = result["all_trials"][0] if result["all_trials"] else None
        if top is not None and top.get("status") == "ok":
            print(
                f"[P10][lightgbm] h{h}: 참고용 best trial={top['trial_id']} "
                f"WAPE={top['pooled_wape']:.4f} Bias={top['pooled_bias']:.4f} (P12에서 사람이 최종 검토)"
            )
        out_path = out_dir / f"p10_{FAMILY}_h{h}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"[P10][lightgbm] 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
