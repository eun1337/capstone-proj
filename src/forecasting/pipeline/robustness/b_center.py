"""
b_center.py

P21 winner-only B-center robustness.

horizon별 P21 winner 하나만 seed=42로 평가하며 B robustness 결과로 model selection을
다시 수행하지 않는다. RF/LightGBM은 B post-regime walk-forward fold를 사용하고,
LSTM/TFT/Informer는 lookback=13에서 validation sequence가 존재하는 usable fold만 사용한다.
실행 전 frozen workload 검증으로 production 데이터와 fold 구조의 일관성을 확인한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.forecasting.common import evaluator as ev
from src.forecasting.common import folds as day3_folds
from src.forecasting.pipeline.robustness import frozen_checks_ml as row_checks
from src.forecasting.pipeline.common import family_registry as registry
from src.forecasting.common.config import HORIZONS
from src.forecasting.common.data_loader import load_development
from src.forecasting.pipeline.robustness import frozen_checks_dl as dl_checks

SEED = 42


def _usable_dl_folds(dev: pd.DataFrame, horizon: int) -> list:
    frozen = dl_checks.FROZEN_B_ROBUSTNESS_USABLE_FOLDS[horizon]
    first_origin = pd.Timestamp(frozen["first_origin"])
    last_origin = pd.Timestamp(frozen["last_origin"])
    all_folds = day3_folds.generate_b_walkforward_folds(dev, horizon)
    usable = [f for f in all_folds if first_origin <= f["val_week"] <= last_origin]
    if len(usable) != frozen["usable_folds"]:
        raise RuntimeError(
            f"h{horizon}: 재구성한 usable fold 개수={len(usable)}가 frozen 값 "
            f"{frozen['usable_folds']}과 다름 - production 데이터가 audit 시점과 달라졌을 수 있음"
        )
    return usable


def run_b_robustness_for_winner(dev: pd.DataFrame, horizon: int, winner: dict, checkpoint_root: Path) -> dict:
    """P21 winner 하나를 B-center에서 robustness 평가한다."""
    family = winner["family"]
    hp = winner["params"]

    if family in registry.ROW_BASED_FAMILIES:
        row_checks.verify_b_robustness_row_fold_count(dev, horizon)
        folds = day3_folds.generate_b_walkforward_folds(dev, horizon)
        fold_data = [(f["fold"], dev.loc[f["train_mask"]], dev.loc[f["val_mask"]]) for f in folds]
        mod = registry.module_for(family)
        run = mod._run_folds(hp, fold_data, horizon, SEED, "b_robustness", f"b_robustness_h{horizon}_{family}")
    else:
        # DL은 B-center usable weekly fold를 직접 순회한다.
        dl_checks.verify_b_robustness_usable_folds(dev, horizon)
        folds = _usable_dl_folds(dev, horizon)
        run = _run_dl_b_folds(family, dev, folds, horizon, hp, checkpoint_root)

    if run["status"] != "ok":
        raise RuntimeError(f"{family} h{horizon} B robustness 실패: {run['error']}")

    pooled_oof = pd.concat(run["oof_frames"], ignore_index=True)
    metrics = ev.compute_metrics(pooled_oof["y_true"].to_numpy(), pooled_oof["y_pred"].to_numpy(),
                                  pooled_oof["mase_scale"].to_numpy())
    return {
        "family": family, "horizon": horizon, "params": hp, "seed": SEED,
        "n_folds": len(run["oof_frames"]),
        "pooled_metrics": {"wape": metrics["wape"], "bias": metrics["bias"], "rmse": metrics["rmse"],
                           "mae": metrics["mae"], "mase": metrics["mase"]},
    }


def _run_dl_b_folds(family: str, dev: pd.DataFrame, folds: list, horizon: int, hp: dict,
                     checkpoint_root: Path) -> dict:
    mod = registry.module_for(family)
    oof_frames, fold_metrics = [], []
    for fold in folds:
        fold_id = fold["fold"]
        trial_label = f"b_robustness_h{horizon}_{family}_fold{fold_id}"
        if family == "lstm":
            checkpoint_path = checkpoint_root / f"{trial_label}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            result = mod.lstm_trainer.train_and_evaluate_fold(
                dev, fold, horizon, hp["lookback"], hidden_size=hp["hidden_size"],
                batch_size=hp["batch_size"], max_epochs=hp["max_epochs"], learning_rate=hp["learning_rate"],
                weight_decay=hp["weight_decay"], seed=SEED, stage="b_robustness", model_family="LSTM",
                config_id=trial_label, checkpoint_path=str(checkpoint_path),
            )
        elif family == "tft":
            checkpoint_dir = checkpoint_root / trial_label
            result = mod.tft_trainer.train_and_evaluate_fold(
                dev, fold, horizon, hp["lookback"], hidden_size=hp["hidden_size"],
                hidden_continuous_size=hp["hidden_continuous_size"], attention_head_size=hp["attention_head_size"],
                dropout=hp["dropout"], learning_rate=hp["learning_rate"], gradient_clip_val=hp["gradient_clip_val"],
                batch_size=hp["batch_size"], max_epochs=hp["max_epochs"], seed=SEED, stage="b_robustness",
                model_family="TFT", config_id=trial_label, checkpoint_dir=str(checkpoint_dir),
            )
        else:
            checkpoint_path = checkpoint_root / f"{trial_label}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            result = mod.informer_trainer.train_and_evaluate_fold(
                dev, fold, horizon, hp["lookback"], e_layers=hp["e_layers"], n_heads=hp["n_heads"],
                batch_size=hp["batch_size"], max_epochs=hp["max_epochs"], learning_rate=hp["learning_rate"],
                seed=SEED, stage="b_robustness", model_family="Informer", config_id=trial_label,
                checkpoint_path=str(checkpoint_path),
            )
        if result["epochs_completed"] != hp["max_epochs"]:
            return {"status": "failed", "error": f"{family} fold{fold_id}: epochs_completed != max_epochs"}
        oof_frames.append(result["oof"])
        fold_metrics.append(result["metrics"])
    return {"status": "ok", "oof_frames": oof_frames, "native_fold_metrics": fold_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="P21 winner-only B robustness (seed42 only)")
    parser.add_argument("--p21-dir", default="outputs/p21")
    parser.add_argument("--output-dir", default="outputs/robustness/b_center")
    args = parser.parse_args()

    p21_dir = Path(args.p21_dir)
    dev = load_development()
    out_dir = Path(args.output_dir)
    checkpoint_root = out_dir / "_checkpoints"

    for horizon in HORIZONS:
        h_key = f"h{horizon}"
        p21_path = p21_dir / f"p21_h{horizon}_selection.json"
        with open(p21_path, encoding="utf-8") as f:
            h_result = json.load(f)
        winner = h_result.get("winner")
        if winner is None:
            print(f"[B-ROBUSTNESS] {h_key}: P21 winner 미확정 - 건너뜀")
            continue
        print(f"[B-ROBUSTNESS] {h_key} winner={winner['family']} 시작...")
        result = run_b_robustness_for_winner(dev, horizon, winner, checkpoint_root)
        out_path = out_dir / f"b_robustness_h{horizon}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"[B-ROBUSTNESS] {h_key}: {result['pooled_metrics']} -> {out_path}")


if __name__ == "__main__":
    main()
