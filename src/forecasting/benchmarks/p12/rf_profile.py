"""
rf_profile.py
RF P12 cheap/high-cost profiling.
"""

from src.forecasting.benchmarks.p12 import runner
from src.forecasting.benchmarks.p12.common import get_software_versions, peak_ram_mb
from src.forecasting.benchmarks.p12.runner import HORIZON, SEED
from src.forecasting.benchmarks.p12.schema import NA
from src.forecasting.machine_learning.rf.config import (
    FIXED_PARAMS, MAX_FEATURES_CHOICES, MIN_SAMPLES_LEAF_CHOICES, N_ESTIMATORS_CHOICES,
)
from src.forecasting.machine_learning.rf.trainer import train_and_evaluate_fold

FAMILY = "RF"
TIMING_SCOPE = "trainer_preprocess_fit_predict"

HPO_SPACE = {"max_features": [0.1, 1 / 3], "min_samples_leaf": [100, 500]}
CHEAP_CONFIG = {"max_features": 0.1, "min_samples_leaf": 500}
HIGH_COST_CONFIG = {"max_features": 1 / 3, "min_samples_leaf": 100}
FIXED_CONFIG = {"n_estimators": 100, "max_depth": None, "min_samples_split": 2,
                "bootstrap": True, "criterion": "squared_error", "n_jobs": -1}
CONFIGURATION_BASIS = {
    "n_estimators": ["official_reference", "prior_study"],
    "max_features": ["prior_study"],
    "min_samples_leaf": ["prior_study"],
    "max_depth": ["official_reference"],
    "min_samples_split": ["official_reference"],
    "bootstrap": ["official_reference"],
    "criterion": ["official_reference"],
}

for _cfg in (CHEAP_CONFIG, HIGH_COST_CONFIG):
    if _cfg["max_features"] not in MAX_FEATURES_CHOICES:
        raise ValueError(f"max_features={_cfg['max_features']}가 현재 Search Space에 없음")
    if _cfg["min_samples_leaf"] not in MIN_SAMPLES_LEAF_CHOICES:
        raise ValueError(f"min_samples_leaf={_cfg['min_samples_leaf']}가 현재 Search Space에 없음")
if FIXED_CONFIG["n_estimators"] not in N_ESTIMATORS_CHOICES:
    raise ValueError(f"n_estimators={FIXED_CONFIG['n_estimators']}가 현재 Search Space에 없음")

_production_fixed = (FIXED_PARAMS["max_depth"], FIXED_PARAMS["min_samples_split"],
                     FIXED_PARAMS["bootstrap"], FIXED_PARAMS["criterion"], FIXED_PARAMS["n_jobs"])
_expected_fixed = (FIXED_CONFIG["max_depth"], FIXED_CONFIG["min_samples_split"],
                   FIXED_CONFIG["bootstrap"], FIXED_CONFIG["criterion"], FIXED_CONFIG["n_jobs"])
if _production_fixed != _expected_fixed:
    raise RuntimeError(f"rf.config.FIXED_PARAMS={_production_fixed}가 예상값 {_expected_fixed}과 다름")


def run_model(corner: str, stage: str, sub_a, fold, train_df, val_df) -> dict:
    corner_config = CHEAP_CONFIG if corner == "cheap" else HIGH_COST_CONFIG

    status_completion = "completed"
    technical_failure = False
    error_message = None
    result = None
    try:
        result = train_and_evaluate_fold(
            train_df, val_df, HORIZON,
            n_estimators=FIXED_CONFIG["n_estimators"], max_features=corner_config["max_features"],
            min_samples_leaf=corner_config["min_samples_leaf"],
            stage=stage, model_family=FAMILY, config_id=f"P12_{FAMILY.lower()}_{corner}",
            seed=SEED, fold_id=fold["fold"],
        )
    except Exception as exc:  # noqa: BLE001
        status_completion = "technical_failure"
        technical_failure = True
        error_message = f"{type(exc).__name__}: {exc}"

    tested_config = {**corner_config, **FIXED_CONFIG}
    val_period = f"{fold['val_start'].date()} ~ {fold['val_end'].date()}"
    train_period = (
        f"{train_df['week_st'].min().date()} ~ {train_df['week_st'].max().date()}"
        if len(train_df) > 0 else NA
    )

    return {
        "status_completion": status_completion, "technical_failure": technical_failure,
        "error_message": error_message, "fallback_occurred": False, "fallback_reason": NA,
        "requested_config": tested_config, "tested_config": tested_config,
        "device": "cpu", "backend": "sklearn.ensemble.RandomForestRegressor",
        "software_versions": get_software_versions("scikit-learn", "pandas", "numpy"),
        "total_runtime_sec": result["total_time_sec"] if result else None,
        "data_preparation_sec": result["preprocessing_time_sec"] if result else None,
        "fit_train_sec": result["fit_time_sec"] if result else None,
        "validation_or_prediction_sec": result["predict_time_sec"] if result else None,
        "peak_ram_mb": peak_ram_mb(),
        "train_rows": len(train_df), "validation_rows": len(val_df),
        "train_sequences": NA, "validation_sequences": NA,
        "train_period": train_period, "validation_period": val_period, "fold_id": fold["fold"],
        "observed_compute_result": {"note": "RF: timing/peak_ram 외 추가 진단 없음"},
    }


if __name__ == "__main__":
    runner.run_corner(
        family=FAMILY, timing_scope=TIMING_SCOPE, hpo_space=HPO_SPACE,
        cheap_config=CHEAP_CONFIG, high_cost_config=HIGH_COST_CONFIG, fixed_config=FIXED_CONFIG,
        configuration_basis=CONFIGURATION_BASIS, code_path="src/forecasting/benchmarks/p12/rf_profile.py",
        run_model_fn=run_model,
    )
