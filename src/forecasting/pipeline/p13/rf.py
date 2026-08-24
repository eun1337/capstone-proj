"""
rf.py

Random Forest P13 HPO runner.

A센터 2023 expanding 4-fold에서 max_features와 min_samples_leaf의 4개 조합을
GridSampler로 모두 평가한다. n_estimators는 100으로 고정한다.
5-family common evaluation key로 filtering한 OOF를 사용해 horizon별 hyperparameter를 선택한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna
import pandas as pd

from src.forecasting.common import folds as day3_folds
from src.forecasting.pipeline.p13 import hpo_common as hc
from src.forecasting.common.config import (
    BIAS_GUARDRAIL_ABS_PCT,
    HORIZONS,
    P13_MODEL_SEED,
    P13_SAMPLER_SEED,
)
from src.forecasting.common.data_loader import load_development
from src.forecasting.machine_learning.rf import trainer as rf_trainer
from src.forecasting.machine_learning.rf.config import (
    FIXED_PARAMS,
    MAX_FEATURES_CHOICES,
    MIN_SAMPLES_LEAF_CHOICES,
    N_ESTIMATORS,
    P13_GRID_SEARCH_SPACE,
)

FAMILY = "rf"
EVALUATION_POPULATION = "five_family_common_keys"

# Frozen P13 Search Space 검증
if tuple(sorted(MAX_FEATURES_CHOICES)) != tuple(sorted((0.1, 1 / 3))):
    raise RuntimeError(f"rf.config.MAX_FEATURES_CHOICES={MAX_FEATURES_CHOICES}가 frozen spec {{0.1, 1/3}}과 다름")
if tuple(sorted(MIN_SAMPLES_LEAF_CHOICES)) != (100, 500):
    raise RuntimeError(f"rf.config.MIN_SAMPLES_LEAF_CHOICES={MIN_SAMPLES_LEAF_CHOICES}가 frozen spec (100,500)과 다름")
if N_ESTIMATORS != 100:
    raise RuntimeError(f"rf.config.N_ESTIMATORS={N_ESTIMATORS}가 frozen spec 100과 다름")
_GRID_SIZE = len(MAX_FEATURES_CHOICES) * len(MIN_SAMPLES_LEAF_CHOICES)
if _GRID_SIZE != 4:
    raise RuntimeError(f"RF grid size={_GRID_SIZE}가 frozen spec 4와 다름")


def suggest_rf_hp(trial: optuna.Trial) -> dict:
    return {
        "max_features": trial.suggest_categorical("max_features", list(MAX_FEATURES_CHOICES)),
        "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", list(MIN_SAMPLES_LEAF_CHOICES)),
        "n_estimators": N_ESTIMATORS,
    }


def _run_folds(hp: dict, fold_data: list, horizon: int, seed: int, stage: str, config_id: str) -> dict:
    """하나의 RF configuration을 4-fold에서 학습·평가하고 native OOF를 반환한다."""
    oof_frames, fold_metrics = [], []
    for fold_id, train_df, val_df in fold_data:
        try:
            result = rf_trainer.train_and_evaluate_fold(
                train_df, val_df, horizon,
                n_estimators=hp["n_estimators"], max_features=hp["max_features"],
                min_samples_leaf=hp["min_samples_leaf"],
                stage=stage, model_family=FAMILY.upper(), config_id=config_id,
                seed=seed, fold_id=fold_id,
            )
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "fold_id": fold_id}
        oof_frames.append(result["oof"])
        fold_metrics.append(result["metrics"])
    return {"status": "ok", "oof_frames": oof_frames, "native_fold_metrics": fold_metrics}


def run_p13_hpo(
    sub_a: pd.DataFrame, horizon: int, seed: int, sampler_seed: int,
    common_eval_keys: pd.DataFrame, common_eval_keys_path: Path,
) -> dict:
    folds = day3_folds.generate_expanding_folds(sub_a, 2023, horizon)
    fold_data = [
        (fold["fold"], sub_a.loc[fold["train_mask"]], sub_a.loc[fold["val_mask"]])
        for fold in folds
    ]
    for fold_id, train_df, val_df in fold_data:
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError(f"h{horizon} fold {fold_id}: train/val row 0개")

    hc.check_common_keys_match_p13_period(common_eval_keys, folds, horizon, common_eval_keys_path)
    horizon_common_keys = common_eval_keys[common_eval_keys["horizon"] == horizon]

    study = hc.make_grid_sampler_study(P13_GRID_SEARCH_SPACE, sampler_seed)
    trial_summaries: list[dict] = []
    filtered_oof_by_trial: dict[int, pd.DataFrame] = {}

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_rf_hp(trial)
        run = _run_folds(hp, fold_data, horizon, seed, "p13_final_hpo", f"trial{trial.number}")

        if run["status"] != "ok":
            trial_summaries.append({
                "trial_id": trial.number, "params": hp, "status": "failed",
                "error": run["error"], "fold_id": run["fold_id"],
                "pooled_wape": float("inf"), "pooled_bias": float("nan"),
                "worst_fold_wape": float("inf"),
            })
            return float("inf")

        pooled_native_oof = pd.concat(run["oof_frames"], ignore_index=True)
        filtered_oof = hc.filter_pooled_oof_by_common_keys(pooled_native_oof, horizon_common_keys)
        pooled_common = hc.ev.compute_metrics(
            filtered_oof["y_true"].to_numpy(), filtered_oof["y_pred"].to_numpy(),
            filtered_oof["mase_scale"].to_numpy(),
        )
        per_fold_common = hc.per_fold_metrics_common(filtered_oof)
        worst_fold_wape = max(m["wape"] for m in per_fold_common.values())

        trial_summaries.append({
            "trial_id": trial.number, "params": hp, "status": "ok",
            "pooled_wape": pooled_common["wape"], "pooled_bias": pooled_common["bias"],
            "pooled_rmse": pooled_common["rmse"], "pooled_mae": pooled_common["mae"],
            "pooled_mase": pooled_common["mase"], "worst_fold_wape": worst_fold_wape,
            "per_fold_metrics_common": per_fold_common,
            "native_fold_metrics_diagnostic": run["native_fold_metrics"],
            "common_eval_key_count": len(horizon_common_keys),
        })
        filtered_oof_by_trial[trial.number] = filtered_oof
        return pooled_common["wape"]

    study.optimize(objective, n_trials=_GRID_SIZE, n_jobs=1)
    hc.check_grid_fully_and_uniquely_evaluated(trial_summaries, P13_GRID_SEARCH_SPACE, horizon)

    ok_trials = [t for t in trial_summaries if t["status"] == "ok"]
    selection = hc.select_best_trial(ok_trials)

    result = {
        "horizon": horizon, "n_trials": _GRID_SIZE, "seed": seed, "sampler_seed": sampler_seed,
        "n_failed_trials": sum(1 for t in trial_summaries if t["status"] != "ok"),
        "evaluation_population": EVALUATION_POPULATION,
        "common_eval_key_path": str(common_eval_keys_path),
        "common_eval_key_count": len(horizon_common_keys),
        "n_estimators_fixed": N_ESTIMATORS,
        "selection": {k: v for k, v in selection.items()},
        "all_trials": trial_summaries,
    }

    sel = selection["selected"]
    if sel is not None:
        result["pooled_metrics_common"] = {
            "wape": sel["pooled_wape"], "bias": sel["pooled_bias"], "rmse": sel["pooled_rmse"],
            "mae": sel["pooled_mae"], "mase": sel["pooled_mase"],
        }
        result["per_fold_metrics_common"] = sel["per_fold_metrics_common"]
        result["native_fold_metrics_diagnostic"] = sel["native_fold_metrics_diagnostic"]
        result["selected_oof_common_filtered"] = filtered_oof_by_trial[sel["trial_id"]]

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="RF P13 Final HPO (2023, A center)")
    parser.add_argument("--output-dir", default="outputs/hpo/rf")
    parser.add_argument("--seed", type=int, default=P13_MODEL_SEED)
    parser.add_argument("--sampler-seed", type=int, default=P13_SAMPLER_SEED)
    parser.add_argument("--common-eval-keys-path", required=True)
    args = parser.parse_args()

    common_eval_keys_path = Path(args.common_eval_keys_path)
    common_eval_keys = hc.load_common_eval_keys(common_eval_keys_path)

    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()

    out_dir = Path(args.output_dir)
    oof_dir = out_dir / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)

    for h in HORIZONS:
        print(f"[P13][{FAMILY}] horizon={h} Final HPO 시작 (n_trials={_GRID_SIZE}, common_eval_keys={common_eval_keys_path})")
        result = run_p13_hpo(sub_a, h, args.seed, args.sampler_seed, common_eval_keys, common_eval_keys_path)

        sel = result["selection"]["selected"]
        if sel is None:
            print(f"[P13][{FAMILY}] h{h}: guardrail(|bias|<={BIAS_GUARDRAIL_ABS_PCT}%p) 통과 trial 없음 - 확인 필요")
        else:
            print(
                f"[P13][{FAMILY}] h{h}: best trial={sel['trial_id']} params={sel['params']} "
                f"WAPE={sel['pooled_wape']:.4f} Bias={sel['pooled_bias']:.4f} "
                f"WorstFoldWAPE={sel['worst_fold_wape']:.4f} reason={result['selection']['reason']}"
            )
            result["selected_oof_common_filtered"].to_parquet(
                oof_dir / f"oof_{FAMILY}_h{h}_trial{sel['trial_id']}_seed{args.seed}_common.parquet",
                index=False,
            )

        out_path = out_dir / f"p13_{FAMILY}_h{h}.json"
        serializable = {k: v for k, v in result.items() if k != "selected_oof_common_filtered"}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
        print(f"[P13][{FAMILY}] 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
