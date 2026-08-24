"""
lstm_profile.py
LSTM P12 cheap/high-cost profiling (fixed 1 epoch).
"""

from src.forecasting.benchmarks.p12 import runner
from src.forecasting.benchmarks.p12.common import get_checkpoint_dir, get_software_versions
from src.forecasting.benchmarks.p12.runner import HORIZON, SEED
from src.forecasting.benchmarks.p12.schema import NA
from src.forecasting.deep_learning.lstm.config import DROPOUT, NUM_LAYERS
from src.forecasting.deep_learning.lstm.trainer import train_and_evaluate_fold

FAMILY = "LSTM"
TIMING_SCOPE = "full_pipeline_fixed_epochs"
MAX_EPOCHS = 1
FALLBACK_BATCH_SIZE = 512

HPO_SPACE = {"hidden_size": [32, 64], "lookback": [13, 26]}
CHEAP_CONFIG = {"hidden_size": 32, "lookback": 13}
HIGH_COST_CONFIG = {"hidden_size": 64, "lookback": 26}
FIXED_CONFIG = {"learning_rate": 1e-3, "batch_size": 1024, "weight_decay": 0.0,
                "num_layers": NUM_LAYERS, "dropout": DROPOUT, "max_epochs": MAX_EPOCHS}
CONFIGURATION_BASIS = {
    "hidden_size": ["prior_study"], "lookback": ["task_specific"],
    "learning_rate": ["official_reference"], "weight_decay": ["official_reference"],
    "batch_size": ["prior_study"], "num_layers": ["official_reference"],
    "dropout": ["official_reference"],
}

if NUM_LAYERS != 1 or DROPOUT != 0.0:
    raise RuntimeError(f"lstm.config NUM_LAYERS/DROPOUT이 예상과 다름: {NUM_LAYERS}, {DROPOUT}")

# batch_size=1024가 OOM/memory allocation 계열로만 실패하면 512로 1회 fallback한다
# (그 외 예외는 즉시 technical_failure). 1024/512는 서로 다른 checkpoint를 쓴다.
_OOM_SIGNATURES = ("out of memory", "cannot allocate memory", "mps backend out of memory",
                   "unable to allocate", "not enough memory")


def _is_oom_error(exc: Exception) -> bool:
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, RuntimeError):
        return any(sig in str(exc).lower() for sig in _OOM_SIGNATURES)
    return False


def _attempt(corner: str, stage: str, hidden_size: int, lookback: int, batch_size: int,
             checkpoint_suffix: str, sub_a, fold):
    checkpoint_path = get_checkpoint_dir(
        FAMILY.lower(), f"{FAMILY.lower()}_{corner}__{checkpoint_suffix}") / "model.pt"
    return train_and_evaluate_fold(
        sub_a, fold, HORIZON, lookback,
        hidden_size=hidden_size, batch_size=batch_size, max_epochs=MAX_EPOCHS,
        learning_rate=FIXED_CONFIG["learning_rate"], weight_decay=FIXED_CONFIG["weight_decay"],
        seed=SEED, stage=stage, model_family=FAMILY, config_id=f"P12_{FAMILY.lower()}_{corner}",
        checkpoint_path=str(checkpoint_path),
    )


def run_model(corner: str, stage: str, sub_a, fold, train_df, val_df) -> dict:
    corner_config = CHEAP_CONFIG if corner == "cheap" else HIGH_COST_CONFIG

    status_completion = "completed"
    technical_failure = False
    error_message = None
    result = None
    fallback_occurred = False
    fallback_reason = NA
    batch_size_used = FIXED_CONFIG["batch_size"]

    try:
        result = _attempt(corner, stage, corner_config["hidden_size"], corner_config["lookback"],
                          batch_size_used, "batch1024", sub_a, fold)
    except Exception as exc:  # noqa: BLE001
        first_error = f"{type(exc).__name__}: {exc}"
        if _is_oom_error(exc):
            fallback_occurred = True
            fallback_reason = (f"batch_size=1024 OOM 계열 실패({first_error}) -> "
                               f"batch_size={FALLBACK_BATCH_SIZE}로 1회만 fallback")
            batch_size_used = FALLBACK_BATCH_SIZE
            try:
                result = _attempt(corner, stage, corner_config["hidden_size"], corner_config["lookback"],
                                  batch_size_used, "batch512", sub_a, fold)
            except Exception as exc2:  # noqa: BLE001
                status_completion = "technical_failure"
                technical_failure = True
                error_message = (f"1차(batch=1024) OOM: {first_error} | "
                                 f"fallback(batch=512) 실패: {type(exc2).__name__}: {exc2}")
        else:
            status_completion = "technical_failure"
            technical_failure = True
            error_message = f"batch_size=1024 실패(OOM 아님, fallback 미시도): {first_error}"

    requested_config = {**corner_config, **FIXED_CONFIG}
    tested_config = {**corner_config, **FIXED_CONFIG, "batch_size": batch_size_used}
    val_period = f"{fold['val_start'].date()} ~ {fold['val_end'].date()}"
    train_mask_df = sub_a.loc[fold["train_mask"]]
    train_period = (
        f"{train_mask_df['week_st'].min().date()} ~ {train_mask_df['week_st'].max().date()}"
        if len(train_mask_df) > 0 else NA
    )

    return {
        "status_completion": status_completion, "technical_failure": technical_failure,
        "error_message": error_message, "fallback_occurred": fallback_occurred,
        "fallback_reason": fallback_reason,
        "requested_config": requested_config, "tested_config": tested_config,
        "device": result["device"] if result else None, "backend": "torch (LSTM, lstm.trainer)",
        "software_versions": get_software_versions("torch", "pandas", "numpy"),
        "total_runtime_sec": result["total_time_sec"] if result else None,
        "data_preparation_sec": result["preprocessing_time_sec"] if result else None,
        "fit_train_sec": result["train_time_sec"] if result else None,
        "validation_or_prediction_sec": result["predict_time_sec"] if result else None,
        "peak_ram_mb": result["peak_ram_mb"] if result else None,
        "train_rows": NA, "validation_rows": NA,
        "train_sequences": result["n_train_sequences"] if result else None,
        "validation_sequences": result["n_val_sequences"] if result else None,
        "train_period": train_period, "validation_period": val_period, "fold_id": fold["fold"],
        "observed_compute_result": {
            "note": "LSTM: timing/peak_ram/epochs_completed 외 추가 진단 없음",
            "epochs_completed": result["epochs_completed"] if result else None,
        },
    }


if __name__ == "__main__":
    runner.run_corner(
        family=FAMILY, timing_scope=TIMING_SCOPE, hpo_space=HPO_SPACE,
        cheap_config=CHEAP_CONFIG, high_cost_config=HIGH_COST_CONFIG, fixed_config=FIXED_CONFIG,
        configuration_basis=CONFIGURATION_BASIS, code_path="src/forecasting/benchmarks/p12/lstm_profile.py",
        run_model_fn=run_model,
    )
