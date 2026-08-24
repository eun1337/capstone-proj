"""
runner.py
P12 profiling 공통 실행.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable

from src.forecasting.benchmarks.p12 import common
from src.forecasting.benchmarks.p12.schema import new_experiment_record
from src.forecasting.common import folds
from src.forecasting.common.data_loader import load_development

CENTER = "A"
STAGE = "P10"
VALIDATION_YEAR = 2022
HORIZON = 1
FOLD_INDEX = 0
SEED = 42


def load_a_center_fold(validation_year: int = VALIDATION_YEAR, fold_index: int = FOLD_INDEX):
    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()
    if set(sub_a["center_id"].unique()) != {"A"}:
        raise RuntimeError(f"center_id 필터링 실패: {sorted(sub_a['center_id'].unique())}")
    fold = folds.generate_expanding_folds(sub_a, validation_year, HORIZON)[fold_index]
    train_df = sub_a.loc[fold["train_mask"]]
    val_df = sub_a.loc[fold["val_mask"]]
    return sub_a, fold, train_df, val_df


def run_corner(*, family: str, timing_scope: str, hpo_space: dict[str, Any],
               cheap_config: dict[str, Any], high_cost_config: dict[str, Any],
               fixed_config: dict[str, Any], configuration_basis: dict[str, Any],
               code_path: str, run_model_fn: Callable[..., dict], argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=f"{family} P12 cheap/high-cost profiling.")
    parser.add_argument("corner", choices=["cheap", "high_cost"])
    parser.add_argument("--stage", default=STAGE)
    parser.add_argument("--validation-year", type=int, default=VALIDATION_YEAR)
    parser.add_argument("--fold-index", type=int, default=FOLD_INDEX)
    parser.add_argument("--experiment-id-prefix", default=None)
    args = parser.parse_args(argv)

    corner = args.corner
    prefix = args.experiment_id_prefix or family.lower()
    experiment_id = f"{prefix}_{corner}"
    cheap_experiment_id = f"{prefix}_cheap"

    sub_a, fold, train_df, val_df = load_a_center_fold(args.validation_year, args.fold_index)
    r = run_model_fn(corner, args.stage, sub_a, fold, train_df, val_df)

    data_section = {"center": CENTER, "stage": args.stage, "validation_year": args.validation_year,
                    "horizon": HORIZON, "fold": r["fold_id"]}
    execution_section = {
        "seed": SEED, "device": r["device"], "backend": r["backend"],
        "git_commit": common.get_git_commit(), "git_dirty": common.get_git_dirty(),
        "software_versions": r["software_versions"], **common.get_hardware_info(),
    }

    # high_cost는 같은 experiment_id_prefix의 cheap만 reference로 쓴다(다른 fold/year와
    # 절대 섞이지 않도록).
    cheap_record = common.load_cheap_result(family.lower(), cheap_experiment_id) if corner == "high_cost" else None
    comparable, mismatches = common.check_timing_comparable_to_cheap(
        family=family, data=data_section, execution=execution_section,
        timing_scope=timing_scope if r["total_runtime_sec"] is not None else None,
        tested_config=r["tested_config"], hpo_space=hpo_space, cheap_record=cheap_record,
    )
    cheap_runtime_sec = cheap_record["timing"]["total_runtime_sec"] if cheap_record else None
    runtime_ratio = common.compute_runtime_ratio(r["total_runtime_sec"], cheap_runtime_sec, comparable)

    record = new_experiment_record(
        experiment_id=experiment_id, family=family, corner=corner,
        purpose="COMPUTE_CORNER_MEASUREMENT",
        evidence_status="CANONICAL" if r["status_completion"] == "completed" else "INVALID_INCOMPLETE",
        data={**data_section, "train_period": r["train_period"], "validation_period": r["validation_period"],
              "train_rows": r["train_rows"], "validation_rows": r["validation_rows"],
              "train_sequences": r["train_sequences"], "validation_sequences": r["validation_sequences"]},
        configuration={
            "hpo_space": hpo_space, "cheap_config": cheap_config, "high_cost_config": high_cost_config,
            "requested_config": r["requested_config"], "tested_config": r["tested_config"],
            "fixed_config": fixed_config, "configuration_basis": configuration_basis,
        },
        execution=execution_section,
        timing={
            "timing_scope": timing_scope if r["total_runtime_sec"] is not None else None,
            "data_preparation_sec": r["data_preparation_sec"], "fit_train_sec": r["fit_train_sec"],
            "validation_or_prediction_sec": r["validation_or_prediction_sec"],
            "total_runtime_sec": r["total_runtime_sec"], "cheap_runtime_sec": cheap_runtime_sec,
            "runtime_ratio": runtime_ratio, "timing_comparable_to_cheap": comparable,
        },
        resource={"peak_ram_mb": r["peak_ram_mb"]},
        status={"completion_status": r["status_completion"], "technical_failure": r["technical_failure"],
                "error_message": r["error_message"], "fallback_occurred": r["fallback_occurred"],
                "fallback_reason": r["fallback_reason"]},
        observed_compute_result=r["observed_compute_result"],
        artifact={"code_path": code_path, "canonical_artifact_path": None},
    )

    out_path = common.save_experiment(record, family.lower())
    print(f"저장: {out_path}")
    print(f"status={r['status_completion']} total_runtime_sec={r['total_runtime_sec']} "
          f"fallback_occurred={r['fallback_occurred']} comparable={comparable} "
          f"mismatches={mismatches} runtime_ratio={runtime_ratio}")
    if r["technical_failure"]:
        sys.exit(1)
