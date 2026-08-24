"""
holdout_2024.py

Final retrain artifact의 2024 Holdout 평가.

horizon별로 저장된 Final model artifact만 불러와 prediction과 raw-scale metric을 계산한다.
model 재학습, HPO, family/configuration 재선택은 수행하지 않는다.
target_date가 2025로 넘어가는 origin은 평가에서 제외하고 분기별 결과는 target_date 기준으로 계산한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.forecasting.common import evaluator as ev
from src.forecasting.pipeline.final import model_artifact as final_model_artifact
from src.forecasting.common import folds as day3_folds
from src.forecasting.pipeline.common import family_registry as registry
from src.forecasting.common.data_loader import load_development, load_holdout_2024


def run_holdout_for_artifact(holdout_df: pd.DataFrame, horizon: int, final_retrain_metadata_path: Path) -> dict:
    """저장된 Final retrain artifact로 해당 horizon의 2024 Holdout을 평가한다."""
    metadata = final_model_artifact.load_metadata(final_retrain_metadata_path)
    if metadata["horizon"] != horizon:
        raise ValueError(f"artifact horizon={metadata['horizon']}이 요청 horizon={horizon}과 다름")

    expected_n = int(day3_folds.holdout_2024_mask(holdout_df, horizon).sum())
    pred = registry.predict_with_final_model_for_family(metadata["family"], metadata, holdout_df, horizon)

    y_true, y_pred = pred["y_true"], pred["y_pred"]
    keys = pred["keys"]
    dev_history = load_development()
    mase_scale = ev.build_mase_scale(dev_history[["center_id", "sku_id", "week_st", "qty"]], keys)
    metrics = ev.compute_metrics(y_true, y_pred, mase_scale)

    quarter = keys["target_date"].dt.quarter
    per_quarter = {}
    for q in sorted(quarter.unique()):
        m = quarter == q
        per_quarter[int(q)] = ev.compute_metrics(y_true[m.to_numpy()], y_pred[m.to_numpy()], mase_scale[m.to_numpy()])

    missing_info = pred["missing_info"]
    n_missing = missing_info["n_missing"] if missing_info else max(0, expected_n - len(y_true))

    return {
        "family": metadata["family"], "horizon": horizon, "config": metadata["config"], "seed": metadata["seed"],
        "n_expected_predictions": expected_n,
        "n_predictions": len(y_true),
        "n_missing_predictions": n_missing,
        "missing_info": missing_info,
        "overall_metrics": {k: metrics[k] for k in ("wape", "bias", "rmse", "mae", "mase")},
        "per_quarter_metrics": {q: {k: m[k] for k in ("wape", "bias", "rmse", "mae", "mase")} for q, m in per_quarter.items()},
        "final_retrain_metadata_path": str(final_retrain_metadata_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Final retrain artifact 전용 2024 Holdout (재선택 경로 없음)")
    parser.add_argument("--horizon", type=int, required=True, choices=(1, 2, 4))
    parser.add_argument("--final-retrain-metadata-path", required=True)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    holdout_df = load_holdout_2024()
    result = run_holdout_for_artifact(holdout_df, args.horizon, Path(args.final_retrain_metadata_path))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
