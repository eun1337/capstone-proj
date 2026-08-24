"""
informer_profile.py
Informer P12 cheap/high-cost profiling (fixed 1 epoch).
"""

from src.forecasting.benchmarks.p12 import runner
from src.forecasting.benchmarks.p12.common import get_checkpoint_dir, get_software_versions
from src.forecasting.benchmarks.p12.runner import HORIZON, SEED
from src.forecasting.benchmarks.p12.schema import NA
from src.forecasting.deep_learning.informer.config import label_len_for
import src.forecasting.deep_learning.informer.trainer as informer_trainer

FAMILY = "Informer"
TIMING_SCOPE = "single_epoch_forced"
MAX_EPOCHS = 1

HPO_SPACE = {"lookback": [13, 26]}
CHEAP_CONFIG = {"lookback": 13}
HIGH_COST_CONFIG = {"lookback": 26}
# d_model/d_ff/factor는 production constant를 그대로 가져오지 않고 P12 의도값을 명시한다 -
# production이 바뀌어도 이 값은 자동으로 따라가지 않으며, 아래에서 별도로 drift를 검증한다.
FIXED_CONFIG = {"e_layers": 2, "n_heads": 8, "d_layers": 1, "d_model": 512, "d_ff": 2048,
                "factor": 5, "dropout": 0.05, "learning_rate": 1e-4, "batch_size": 32,
                "max_epochs": MAX_EPOCHS}
CONFIGURATION_BASIS = {
    "e_layers": ["official_reference"], "n_heads": ["official_reference"],
    "d_layers": ["official_reference"], "d_model": ["official_reference"],
    "d_ff": ["official_reference"], "factor": ["official_reference"],
    "dropout": ["official_reference"], "learning_rate": ["official_reference"],
    "batch_size": ["official_reference"], "lookback": ["task_specific"],
}

_production = (informer_trainer.D_MODEL, informer_trainer.D_FF, informer_trainer.FACTOR)
_expected = (FIXED_CONFIG["d_model"], FIXED_CONFIG["d_ff"], FIXED_CONFIG["factor"])
if _production != _expected:
    raise RuntimeError(f"production informer.trainer의 D_MODEL/D_FF/FACTOR={_production}가 "
                       f"예상값 {_expected}과 다름")


def run_model(corner: str, stage: str, sub_a, fold, train_df, val_df) -> dict:
    corner_config = CHEAP_CONFIG if corner == "cheap" else HIGH_COST_CONFIG
    lookback = corner_config["lookback"]

    expected_label_len = lookback // 2
    if label_len_for(lookback) != expected_label_len:
        raise RuntimeError(f"label_len_for({lookback})={label_len_for(lookback)}가 "
                           f"floor(lookback/2)={expected_label_len}과 다름")

    # d_layers/dropout은 함수 파라미터가 아니라 informer.trainer의 모듈 전역이라, 이
    # 프로세스 안에서만 그 전역을 재바인딩해야 FIXED_CONFIG 값이 실제로 적용된다.
    informer_trainer.D_LAYERS = FIXED_CONFIG["d_layers"]
    informer_trainer.DROPOUT = FIXED_CONFIG["dropout"]

    checkpoint_path = get_checkpoint_dir(FAMILY.lower(), f"{FAMILY.lower()}_{corner}") / "model.pt"

    status_completion = "completed"
    technical_failure = False
    error_message = None
    result = None
    try:
        result = informer_trainer.train_and_evaluate_fold(
            sub_a, fold, HORIZON, lookback,
            e_layers=FIXED_CONFIG["e_layers"], n_heads=FIXED_CONFIG["n_heads"],
            batch_size=FIXED_CONFIG["batch_size"], max_epochs=MAX_EPOCHS,
            learning_rate=FIXED_CONFIG["learning_rate"], seed=SEED,
            stage=stage, model_family=FAMILY, config_id=f"P12_{FAMILY.lower()}_{corner}",
            checkpoint_path=str(checkpoint_path),
        )
        if result["label_len"] != expected_label_len:
            raise RuntimeError(f"production이 실제로 만든 label_len={result['label_len']}이 "
                               f"floor(lookback/2)={expected_label_len}과 다름")
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
        "backend": "torch (Informer, informer.trainer)",
        "software_versions": get_software_versions("torch", "pandas", "numpy"),
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
            "note": "Informer: timing/peak_ram/epochs_completed 외 추가 진단 없음",
            "epochs_completed": result["epochs_completed"] if result else None,
            "label_len": result["label_len"] if result else expected_label_len,
        },
    }


if __name__ == "__main__":
    runner.run_corner(
        family=FAMILY, timing_scope=TIMING_SCOPE, hpo_space=HPO_SPACE,
        cheap_config=CHEAP_CONFIG, high_cost_config=HIGH_COST_CONFIG, fixed_config=FIXED_CONFIG,
        configuration_basis=CONFIGURATION_BASIS, code_path="src/forecasting/benchmarks/p12/informer_profile.py",
        run_model_fn=run_model,
    )
