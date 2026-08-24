"""
tft_profile.py
TFT P12 cheap/high-cost profiling (fixed 1 epoch).
"""

from src.forecasting.benchmarks.p12 import runner
from src.forecasting.benchmarks.p12.common import get_checkpoint_dir, get_software_versions
from src.forecasting.benchmarks.p12.runner import HORIZON, SEED
from src.forecasting.benchmarks.p12.schema import NA
from src.forecasting.deep_learning.tft.config import OPTIMIZER as _PRODUCTION_OPTIMIZER
import src.forecasting.deep_learning.tft.trainer as tft_trainer

FAMILY = "TFT"
TIMING_SCOPE = "single_epoch_forced"
MAX_EPOCHS = 1
LSTM_LAYERS_FIXED = 1

HPO_SPACE = {"hidden_size": [8, 16], "lookback": [13, 26]}
CHEAP_CONFIG = {"hidden_size": 8, "lookback": 13}
HIGH_COST_CONFIG = {"hidden_size": 16, "lookback": 26}
FIXED_CONFIG = {"hidden_continuous_size": 8, "attention_head_size": 1, "dropout": 0.1,
                "learning_rate": 1e-3, "gradient_clip_val": 0.1, "batch_size": 256,
                "lstm_layers": LSTM_LAYERS_FIXED, "optimizer": "adam", "max_epochs": MAX_EPOCHS}
CONFIGURATION_BASIS = {
    "hidden_size": ["official_tutorial"],
    "attention_head_size": ["official_tutorial"],
    "hidden_continuous_size": ["official_reference", "official_tutorial"],
    "dropout": ["official_reference", "official_tutorial"],
    "gradient_clip_val": ["official_tutorial"],
    "lookback": ["task_specific"],
    "batch_size": ["prior_study"],
    "learning_rate": ["official_reference", "official_tutorial"],
    "lstm_layers": ["official_reference"],
    "optimizer": ["official_reference"],
}

if _PRODUCTION_OPTIMIZER != FIXED_CONFIG["optimizer"]:
    raise RuntimeError(f"tft.config.OPTIMIZER={_PRODUCTION_OPTIMIZER!r}가 예상값 "
                       f"{FIXED_CONFIG['optimizer']!r}과 다름")


def run_model(corner: str, stage: str, sub_a, fold, train_df, val_df) -> dict:
    corner_config = CHEAP_CONFIG if corner == "cheap" else HIGH_COST_CONFIG
    tft_trainer.LSTM_LAYERS = LSTM_LAYERS_FIXED  # 프로세스 내에서만 유효(함수 파라미터 아님)

    checkpoint_dir = get_checkpoint_dir(FAMILY.lower(), f"{FAMILY.lower()}_{corner}")

    status_completion = "completed"
    technical_failure = False
    error_message = None
    result = None
    try:
        result = tft_trainer.train_and_evaluate_fold(
            sub_a, fold, HORIZON, corner_config["lookback"],
            hidden_size=corner_config["hidden_size"],
            hidden_continuous_size=FIXED_CONFIG["hidden_continuous_size"],
            attention_head_size=FIXED_CONFIG["attention_head_size"], dropout=FIXED_CONFIG["dropout"],
            learning_rate=FIXED_CONFIG["learning_rate"], gradient_clip_val=FIXED_CONFIG["gradient_clip_val"],
            batch_size=FIXED_CONFIG["batch_size"], max_epochs=MAX_EPOCHS, seed=SEED,
            stage=stage, model_family=FAMILY, config_id=f"P12_{FAMILY.lower()}_{corner}",
            checkpoint_dir=str(checkpoint_dir),
        )
    except Exception as exc:  # noqa: BLE001
        status_completion = "technical_failure"
        technical_failure = True
        error_message = f"{type(exc).__name__}: {exc}"

    tested_config = {**corner_config, **FIXED_CONFIG}
    val_period = f"{fold['val_start'].date()} ~ {fold['val_end'].date()}"

    return {
        "status_completion": status_completion, "technical_failure": technical_failure,
        "error_message": error_message, "fallback_occurred": False, "fallback_reason": NA,
        "requested_config": tested_config, "tested_config": tested_config,
        "device": result["device"] if result else None,
        "backend": "pytorch_forecasting.TemporalFusionTransformer",
        "software_versions": get_software_versions("pytorch-forecasting", "lightning", "torch"),
        "total_runtime_sec": result["total_time_sec"] if result else None,
        "data_preparation_sec": result["preprocessing_time_sec"] if result else None,
        "fit_train_sec": result["train_time_sec"] if result else None,
        "validation_or_prediction_sec": result["predict_time_sec"] if result else None,
        "peak_ram_mb": result["peak_ram_mb"] if result else None,
        "train_rows": NA, "validation_rows": NA,
        "train_sequences": result["n_train_sequences"] if result else None,
        "validation_sequences": result["n_val_sequences"] if result else None,
        "train_period": NA, "validation_period": val_period, "fold_id": fold["fold"],
        "observed_compute_result": {
            "note": "TFT: timing/peak_ram/epochs_completed 외 추가 진단 없음",
            "epochs_completed": result["epochs_completed"] if result else None,
        },
    }


if __name__ == "__main__":
    runner.run_corner(
        family=FAMILY, timing_scope=TIMING_SCOPE, hpo_space=HPO_SPACE,
        cheap_config=CHEAP_CONFIG, high_cost_config=HIGH_COST_CONFIG, fixed_config=FIXED_CONFIG,
        configuration_basis=CONFIGURATION_BASIS, code_path="src/forecasting/benchmarks/p12/tft_profile.py",
        run_model_fn=run_model,
    )
