"""
lightgbm_profile.py
LightGBM P12 cheap/high-cost profiling.
"""

from src.forecasting.benchmarks.p12 import runner
from src.forecasting.benchmarks.p12.common import get_software_versions, peak_ram_mb
from src.forecasting.benchmarks.p12.runner import HORIZON, SEED
from src.forecasting.benchmarks.p12.schema import NA
import src.forecasting.machine_learning.lightgbm.trainer as lgbm_trainer

FAMILY = "LightGBM"
TIMING_SCOPE = "trainer_preprocess_fit_predict"

HPO_SPACE = {"num_leaves": [8, 31], "min_child_samples": [100, 1000]}
CHEAP_CONFIG = {"num_leaves": 8, "min_child_samples": 1000}
HIGH_COST_CONFIG = {"num_leaves": 31, "min_child_samples": 100}
# feature_fraction/bagging_fraction/bagging_freq/lambda_l1/lambda_l2는 trainer 호출
# kwarg로 직접 넘어가고, 나머지는 lgbm.trainer.FIXED_PARAMS로 재바인딩된다.
_TRAINER_FIXED_KWARGS = {"feature_fraction": 0.4, "bagging_fraction": 0.4, "bagging_freq": 1,
                         "lambda_l1": 0.0, "lambda_l2": 0.0}
_LGBM_FIXED_PARAMS = {"learning_rate": 0.1, "n_estimators": 100, "max_depth": -1,
                      "boosting_type": "gbdt", "objective": "regression",
                      "n_jobs": -1, "verbosity": -1}
FIXED_CONFIG = {**_TRAINER_FIXED_KWARGS, **_LGBM_FIXED_PARAMS}
CONFIGURATION_BASIS = {
    "n_estimators": ["official_reference"], "learning_rate": ["official_reference"],
    "num_leaves": ["prior_study"], "min_child_samples": ["prior_study"],
    "feature_fraction": ["prior_study", "official_compute_guidance"],
    "bagging_fraction": ["prior_study", "official_compute_guidance"],
    "bagging_freq": ["prior_study"],
    "lambda_l1": ["official_reference"], "lambda_l2": ["official_reference"],
    "max_depth": ["official_reference"], "boosting_type": ["official_reference"],
}


def run_model(corner: str, stage: str, sub_a, fold, train_df, val_df) -> dict:
    corner_config = CHEAP_CONFIG if corner == "cheap" else HIGH_COST_CONFIG
    lgbm_trainer.FIXED_PARAMS = _LGBM_FIXED_PARAMS  # 프로세스 내에서만 유효

    status_completion = "completed"
    technical_failure = False
    error_message = None
    result = None
    try:
        result = lgbm_trainer.train_and_evaluate_fold(
            train_df, val_df, HORIZON, **corner_config, **_TRAINER_FIXED_KWARGS,
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
        "device": "cpu", "backend": "lightgbm.LGBMRegressor",
        "software_versions": get_software_versions("lightgbm", "pandas", "numpy"),
        "total_runtime_sec": result["total_time_sec"] if result else None,
        "data_preparation_sec": result["preprocessing_time_sec"] if result else None,
        "fit_train_sec": result["fit_time_sec"] if result else None,
        "validation_or_prediction_sec": result["predict_time_sec"] if result else None,
        "peak_ram_mb": peak_ram_mb(),
        "train_rows": len(train_df), "validation_rows": len(val_df),
        "train_sequences": NA, "validation_sequences": NA,
        "train_period": train_period, "validation_period": val_period, "fold_id": fold["fold"],
        "observed_compute_result": {"note": "LightGBM: timing/peak_ram 외 추가 진단 없음"},
    }


if __name__ == "__main__":
    runner.run_corner(
        family=FAMILY, timing_scope=TIMING_SCOPE, hpo_space=HPO_SPACE,
        cheap_config=CHEAP_CONFIG, high_cost_config=HIGH_COST_CONFIG, fixed_config=FIXED_CONFIG,
        configuration_basis=CONFIGURATION_BASIS, code_path="src/forecasting/benchmarks/p12/lightgbm_profile.py",
        run_model_fn=run_model,
    )
