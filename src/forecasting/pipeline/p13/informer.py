"""
informer.py

Informer P13 HPO runner.

A센터 2023 expanding 4-fold에서 n_heads의 2개 후보를 GridSampler로 모두 평가한다.
lookback, e_layers와 나머지 모델·학습 파라미터는 고정한다.
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
from src.forecasting.deep_learning.informer import trainer as informer_trainer
from src.forecasting.deep_learning.informer.config import (
    BATCH_SIZE,
    D_MODEL,
    E_LAYERS,
    LEARNING_RATE,
    LOOKBACK,
    MAX_EPOCHS,
    N_HEADS_CHOICES,
    P13_GRID_SEARCH_SPACE,
    label_len_for,
)

FAMILY = "informer"
EVALUATION_POPULATION = "five_family_common_keys"
EXPECTED_LABEL_LEN = LOOKBACK // 2

# Frozen P13 Search Space 검증
if LOOKBACK != 13:
    raise RuntimeError(f"informer.config.LOOKBACK={LOOKBACK}가 frozen spec 13과 다름")
if EXPECTED_LABEL_LEN != 6 or label_len_for(LOOKBACK) != 6:
    raise RuntimeError(f"label_len_for(LOOKBACK)={label_len_for(LOOKBACK)}가 frozen spec 6과 다름")
if tuple(sorted(N_HEADS_CHOICES)) != (8, 16):
    raise RuntimeError(f"informer.config.N_HEADS_CHOICES={N_HEADS_CHOICES}가 frozen spec (8,16)과 다름")
if MAX_EPOCHS != 2:
    raise RuntimeError(f"informer.config.MAX_EPOCHS={MAX_EPOCHS}가 frozen spec 2와 다름")
if E_LAYERS != 2:
    raise RuntimeError(f"informer.config.E_LAYERS={E_LAYERS}가 frozen spec 2와 다름")
for _n_heads in N_HEADS_CHOICES:
    if D_MODEL % _n_heads != 0:
        raise RuntimeError(f"d_model={D_MODEL}이 n_heads={_n_heads}로 나누어떨어지지 않음")
_GRID_SIZE = len(N_HEADS_CHOICES)
if _GRID_SIZE != 2:
    raise RuntimeError(f"Informer grid size={_GRID_SIZE}가 frozen spec 2와 다름")


def suggest_informer_hp(trial: optuna.Trial) -> dict:
    return {
        "n_heads": trial.suggest_categorical("n_heads", list(N_HEADS_CHOICES)),
        "lookback": LOOKBACK, "e_layers": E_LAYERS, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "max_epochs": MAX_EPOCHS,
    }


def _run_folds(hp: dict, sub_a: pd.DataFrame, folds: list, horizon: int, seed: int,
               stage: str, trial_label: str, checkpoint_root: Path) -> dict:
    """하나의 Informer configuration을 4-fold에서 학습·평가하고 native OOF를 반환한다."""
    oof_frames, fold_metrics = [], []
    for fold in folds:
        fold_id = fold["fold"]
        checkpoint_path = checkpoint_root / f"{trial_label}_fold{fold_id}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = informer_trainer.train_and_evaluate_fold(
                sub_a, fold, horizon, hp["lookback"],
                e_layers=hp["e_layers"], n_heads=hp["n_heads"], batch_size=hp["batch_size"],
                max_epochs=hp["max_epochs"], learning_rate=hp["learning_rate"], seed=seed,
                stage=stage, model_family=FAMILY.upper(), config_id=f"{trial_label}_fold{fold_id}",
                checkpoint_path=str(checkpoint_path),
            )
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "fold_id": fold_id}
        if result["epochs_completed"] != hp["max_epochs"]:
            return {
                "status": "failed", "fold_id": fold_id,
                "error": f"epochs_completed={result['epochs_completed']}가 max_epochs={hp['max_epochs']}과 "
                         "다름 - EarlyStopping이 재도입됐을 가능성",
            }
        if result["n_heads"] != hp["n_heads"]:
            return {
                "status": "failed", "fold_id": fold_id,
                "error": f"trainer가 실제로 받은 n_heads={result['n_heads']}가 요청값 {hp['n_heads']}과 다름",
            }
        if result["label_len"] != EXPECTED_LABEL_LEN:
            return {
                "status": "failed", "fold_id": fold_id,
                "error": f"label_len={result['label_len']}이 frozen spec {EXPECTED_LABEL_LEN}과 다름",
            }
        oof_frames.append(result["oof"])
        fold_metrics.append(result["metrics"])
    return {"status": "ok", "oof_frames": oof_frames, "native_fold_metrics": fold_metrics}


def run_p13_hpo(
    sub_a: pd.DataFrame, horizon: int, seed: int, sampler_seed: int,
    common_eval_keys: pd.DataFrame, common_eval_keys_path: Path, checkpoint_root: Path,
) -> dict:
    folds = day3_folds.generate_expanding_folds(sub_a, 2023, horizon)
    for fold in folds:
        if int(fold["train_mask"].sum()) == 0 or int(fold["val_mask"].sum()) == 0:
            raise ValueError(f"h{horizon} fold {fold['fold']}: train/val row 0개")

    hc.check_common_keys_match_p13_period(common_eval_keys, folds, horizon, common_eval_keys_path)
    horizon_common_keys = common_eval_keys[common_eval_keys["horizon"] == horizon]

    study = hc.make_grid_sampler_study(P13_GRID_SEARCH_SPACE, sampler_seed)
    trial_summaries: list[dict] = []
    filtered_oof_by_trial: dict[int, pd.DataFrame] = {}

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_informer_hp(trial)
        run = _run_folds(hp, sub_a, folds, horizon, seed, "p13_final_hpo",
                          f"h{horizon}_trial{trial.number}", checkpoint_root)

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
        "lookback_fixed": LOOKBACK, "label_len_fixed": EXPECTED_LABEL_LEN, "max_epochs_fixed": MAX_EPOCHS,
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
    parser = argparse.ArgumentParser(description="Informer P13 Final HPO (2023, A center)")
    parser.add_argument("--output-dir", default="outputs/hpo/informer")
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
    checkpoint_root = out_dir / "_checkpoints"
    oof_dir.mkdir(parents=True, exist_ok=True)

    for h in HORIZONS:
        print(f"[P13][{FAMILY}] horizon={h} Final HPO 시작 (n_trials={_GRID_SIZE}, common_eval_keys={common_eval_keys_path})")
        result = run_p13_hpo(sub_a, h, args.seed, args.sampler_seed, common_eval_keys, common_eval_keys_path, checkpoint_root)

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
