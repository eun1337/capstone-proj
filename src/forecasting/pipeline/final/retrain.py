"""
retrain.py

P21 winner-only Final retrain.

horizon별 P21 winner family/configuration 하나를 seed=42로 전체 pre-2024 학습 population에
재학습한다. validation, EarlyStopping, best-epoch/iteration selection은 사용하지 않는다.
학습 완료 후 2024 Holdout에서 사용할 Final model artifact와 metadata를 저장한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.forecasting.pipeline.final import model_artifact as final_model_artifact
from src.forecasting.common import folds as day3_folds
from src.forecasting.pipeline.robustness import frozen_checks_ml as row_checks
from src.forecasting.pipeline.common import family_registry as registry
from src.forecasting.common.config import HORIZONS, P13_MODEL_SEED
from src.forecasting.common.data_loader import load_development
from src.forecasting.pipeline.robustness import frozen_checks_dl as dl_checks

SEED = P13_MODEL_SEED
if SEED != 42:
    raise RuntimeError(f"Final retrain seed는 42여야 함(common.config.P13_MODEL_SEED={SEED})")


def verify_final_retrain_population(dev: pd.DataFrame, horizon: int, family: str) -> pd.Series:
    """Final retrain population이 frozen row/sequence workload와 일치하는지 검증한다."""
    row_checks.verify_final_retrain_row_workload(dev, horizon)
    if family in registry.DL_FAMILIES:
        dl_checks.verify_final_retrain_lb13_sequences(dev, horizon)
    return day3_folds.final_retrain_mask(dev, horizon)


def run_final_retrain_for_winner(dev: pd.DataFrame, horizon: int, winner: dict, out_dir: Path) -> dict:
    """P21 winner 하나를 전체 Final retrain population으로 학습하고 artifact metadata를 저장한다."""
    family, hp = winner["family"], winner["params"]
    mask = verify_final_retrain_population(dev, horizon, family)
    full_df = dev.loc[mask].copy()
    population_rows = len(full_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    if family == "tft":
        model_artifact_target = out_dir / f"{family}_h{horizon}_final"
    elif family in ("rf", "lightgbm"):
        model_artifact_target = out_dir / f"{family}_h{horizon}_final.joblib"
    else:
        model_artifact_target = out_dir / f"{family}_h{horizon}_final.pt"

    fit_result = registry.fit_final_model_for_family(family, full_df, horizon, hp, SEED, model_artifact_target)

    population_unit = "sequences" if family in registry.DL_FAMILIES else "rows"
    population_size = fit_result.get("n_train_sequences", population_rows) if population_unit == "sequences" \
        else fit_result.get("n_train_rows", population_rows)

    metadata_path = out_dir / f"{family}_h{horizon}_final_metadata.json"
    final_model_artifact.save_metadata(
        metadata_path, family=family, horizon=horizon, config=hp, seed=SEED,
        model_artifact_path=fit_result["model_artifact_path"],
        training_population_size=population_size, population_unit=population_unit,
        fit_time_sec=fit_result["fit_time_sec"], epochs_completed=fit_result.get("epochs_completed"),
    )
    return {"metadata_path": str(metadata_path), "fit_result": fit_result, "population_rows": population_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="P21 winner-only Final retrain (seed42 only)")
    parser.add_argument("--p21-dir", default="outputs/p21")
    parser.add_argument("--output-dir", default="outputs/final_retrain")
    args = parser.parse_args()

    p21_dir = Path(args.p21_dir)
    dev = load_development()
    out_dir = Path(args.output_dir)

    for horizon in HORIZONS:
        h_key = f"h{horizon}"
        p21_path = p21_dir / f"p21_h{horizon}_selection.json"

        with open(p21_path, encoding="utf-8") as f:
            h_result = json.load(f)

        winner = h_result.get("winner")
        if winner is None:
            print(f"[FINAL-RETRAIN] {h_key}: P21 winner 미확정 - 건너뜀")
            continue

        print(f"[FINAL-RETRAIN] {h_key} winner={winner['family']} 시작...")
        result = run_final_retrain_for_winner(dev, horizon, winner, out_dir)
        print(f"[FINAL-RETRAIN] {h_key}: {result['metadata_path']}")


if __name__ == "__main__":
    main()
