"""
dl_high_lb13_supplementary_profile.py
B robustness feasibility audit(lookback=26은 전 horizon/fold validation_sequences=0)에 따라
DL Search Space에서 lookback=26을 제외하고 lookback=13으로 고정하면서, 기존 P12 High
configuration(lookback=26 기준)이 최종 Search Space와 달라졌다. 이 스크립트는 새 High
configuration(lookback=13) 3개(LSTM/TFT/Informer)만 P13-scale(2023 Fold4, h1)로 추가
측정하는 supplementary compute profiling이다. 기존 Cheap은 재실행하지 않고 기존 cheap
artifact를 읽기 전용으로만 참조한다. runner.load_a_center_fold/schema/common과 각 family
production trainer는 그대로 재사용하며, 각 profile 모듈의 FIXED_CONFIG/CONFIGURATION_BASIS도
값 재정의 없이 그대로 가져다 쓴다. predictive metric(WAPE/Bias/RMSE/MAE/MASE)은 trainer가
내부적으로 계산하더라도 이 스크립트의 artifact/출력에는 포함하지 않는다.
"""

import hashlib
import json
from pathlib import Path

from src.forecasting.benchmarks.p12.common import (
    DEFAULT_OUTPUT_ROOT,
    check_timing_comparable_to_cheap,
    compute_runtime_ratio,
    get_checkpoint_dir,
    get_git_commit,
    get_git_dirty,
    get_hardware_info,
    get_software_versions,
    load_cheap_result,
    save_experiment,
)
from src.forecasting.benchmarks.p12.informer_profile import CONFIGURATION_BASIS as INF_BASIS
from src.forecasting.benchmarks.p12.informer_profile import FAMILY as INF_FAMILY
from src.forecasting.benchmarks.p12.informer_profile import FIXED_CONFIG as INF_FIXED_ORIG
from src.forecasting.benchmarks.p12.lstm_profile import FALLBACK_BATCH_SIZE
from src.forecasting.benchmarks.p12.lstm_profile import CONFIGURATION_BASIS as LSTM_BASIS
from src.forecasting.benchmarks.p12.lstm_profile import FAMILY as LSTM_FAMILY
from src.forecasting.benchmarks.p12.lstm_profile import FIXED_CONFIG as LSTM_FIXED
from src.forecasting.benchmarks.p12.lstm_profile import _attempt as lstm_attempt
from src.forecasting.benchmarks.p12.lstm_profile import _is_oom_error
from src.forecasting.benchmarks.p12.runner import HORIZON, SEED, load_a_center_fold
from src.forecasting.benchmarks.p12.schema import NA, new_experiment_record
from src.forecasting.benchmarks.p12.tft_profile import CONFIGURATION_BASIS as TFT_BASIS
from src.forecasting.benchmarks.p12.tft_profile import FAMILY as TFT_FAMILY
from src.forecasting.benchmarks.p12.tft_profile import FIXED_CONFIG as TFT_FIXED
from src.forecasting.benchmarks.p12.tft_profile import LSTM_LAYERS_FIXED
from src.forecasting.deep_learning.informer.config import label_len_for
import src.forecasting.deep_learning.informer.trainer as informer_trainer
import src.forecasting.deep_learning.tft.trainer as tft_trainer

CODE_PATH = "src/forecasting/benchmarks/p12/dl_high_lb13_supplementary_profile.py"
STAGE = "P13"
VALIDATION_YEAR = 2023
FOLD_INDEX = 3
PURPOSE = "COMPUTE_CORNER_MEASUREMENT_SUPPLEMENTARY_HIGH_LB13"

INF_FIXED = {k: v for k, v in INF_FIXED_ORIG.items() if k != "n_heads"}

RUNS = {
    "LSTM": {
        "family": LSTM_FAMILY, "timing_scope": "full_pipeline_fixed_epochs",
        "hpo_space": {"hidden_size": [32, 64]},
        "cheap_config": {"hidden_size": 32, "lookback": 13},
        "high_cost_config": {"hidden_size": 64, "lookback": 13},
        "fixed_config": LSTM_FIXED, "configuration_basis": LSTM_BASIS,
        "cheap_experiment_id": "lstm_p13_2023_fold4_cheap",
        "experiment_id": "lstm_p13_2023_fold4_supp_high_lb13_high_cost",
    },
    "TFT": {
        "family": TFT_FAMILY, "timing_scope": "single_epoch_forced",
        "hpo_space": {"hidden_size": [8, 16]},
        "cheap_config": {"hidden_size": 8, "lookback": 13},
        "high_cost_config": {"hidden_size": 16, "lookback": 13},
        "fixed_config": TFT_FIXED, "configuration_basis": TFT_BASIS,
        "cheap_experiment_id": "tft_p13_2023_fold4_cheap",
        "experiment_id": "tft_p13_2023_fold4_supp_high_lb13_high_cost",
    },
    "Informer": {
        "family": INF_FAMILY, "timing_scope": "single_epoch_forced",
        "hpo_space": {"n_heads": [8, 16]},
        "cheap_config": {"n_heads": 8, "lookback": 13},
        "high_cost_config": {"n_heads": 16, "lookback": 13},
        "fixed_config": INF_FIXED, "configuration_basis": INF_BASIS,
        "cheap_experiment_id": "informer_p13_2023_fold4_cheap",
        "experiment_id": "informer_p13_2023_fold4_supp_high_lb13_high_cost",
    },
}


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_existing_artifacts() -> dict:
    snapshot = {}
    for family in RUNS:
        for suffix in ("cheap", "high_cost"):
            path = DEFAULT_OUTPUT_ROOT / family.lower() / f"{family.lower()}_p13_2023_fold4_{suffix}.json"
            snapshot[str(path)] = _file_sha256(path)
    return snapshot


def _validate_configs() -> None:
    lstm_cfg = {**RUNS["LSTM"]["high_cost_config"], **LSTM_FIXED}
    assert lstm_cfg["hidden_size"] == 64 and lstm_cfg["lookback"] == 13
    print(f"[VALIDATE] LSTM requested_config = {lstm_cfg}")

    tft_cfg = {**RUNS["TFT"]["high_cost_config"], **TFT_FIXED}
    assert tft_cfg["hidden_size"] == 16 and tft_cfg["lookback"] == 13
    print(f"[VALIDATE] TFT requested_config = {tft_cfg}")

    inf_cfg = {**RUNS["Informer"]["high_cost_config"], **INF_FIXED, "n_heads": 16}
    expected_label_len = 13 // 2
    assert inf_cfg["n_heads"] == 16 and inf_cfg["lookback"] == 13
    assert label_len_for(13) == expected_label_len == 6
    assert informer_trainer.D_MODEL % inf_cfg["n_heads"] == 0, (
        f"d_model={informer_trainer.D_MODEL}이 n_heads={inf_cfg['n_heads']}로 나누어떨어지지 않음"
    )
    print(f"[VALIDATE] Informer requested_config = {inf_cfg} (label_len={expected_label_len})")


def _run_lstm(sub_a, fold) -> dict:
    spec = RUNS["LSTM"]
    hidden_size, lookback = spec["high_cost_config"]["hidden_size"], spec["high_cost_config"]["lookback"]
    batch_size_used = LSTM_FIXED["batch_size"]
    status_completion, technical_failure, error_message = "completed", False, None
    fallback_occurred, fallback_reason = False, NA
    result = None
    try:
        result = lstm_attempt("supp_high_lb13", STAGE, hidden_size, lookback, batch_size_used,
                              "batch1024", sub_a, fold)
    except Exception as exc:  # noqa: BLE001
        first_error = f"{type(exc).__name__}: {exc}"
        if _is_oom_error(exc):
            fallback_occurred = True
            fallback_reason = (f"batch_size=1024 OOM 계열 실패({first_error}) -> "
                               f"batch_size={FALLBACK_BATCH_SIZE}로 1회만 fallback")
            batch_size_used = FALLBACK_BATCH_SIZE
            try:
                result = lstm_attempt("supp_high_lb13", STAGE, hidden_size, lookback, batch_size_used,
                                      "batch512", sub_a, fold)
            except Exception as exc2:  # noqa: BLE001
                status_completion, technical_failure = "technical_failure", True
                error_message = (f"1차(batch=1024) OOM: {first_error} | "
                                 f"fallback(batch=512) 실패: {type(exc2).__name__}: {exc2}")
        else:
            status_completion, technical_failure = "technical_failure", True
            error_message = f"batch_size=1024 실패(OOM 아님, fallback 미시도): {first_error}"

    requested_config = {**spec["high_cost_config"], **LSTM_FIXED}
    tested_config = {**requested_config, "batch_size": batch_size_used}
    val_period = f"{fold['val_start'].date()} ~ {fold['val_end'].date()}"
    train_mask_df = sub_a.loc[fold["train_mask"]]
    train_period = (f"{train_mask_df['week_st'].min().date()} ~ {train_mask_df['week_st'].max().date()}"
                    if len(train_mask_df) > 0 else NA)

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
            "note": "LSTM supplementary(lb13 revised High): timing/peak_ram/epochs_completed 외 추가 진단 없음",
            "epochs_completed": result["epochs_completed"] if result else None,
        },
    }


def _run_tft(sub_a, fold) -> dict:
    spec = RUNS["TFT"]
    hidden_size, lookback = spec["high_cost_config"]["hidden_size"], spec["high_cost_config"]["lookback"]
    tft_trainer.LSTM_LAYERS = LSTM_LAYERS_FIXED

    checkpoint_dir = get_checkpoint_dir("tft", "tft_supp_high_lb13")
    status_completion, technical_failure, error_message, result = "completed", False, None, None
    try:
        result = tft_trainer.train_and_evaluate_fold(
            sub_a, fold, HORIZON, lookback,
            hidden_size=hidden_size, hidden_continuous_size=TFT_FIXED["hidden_continuous_size"],
            attention_head_size=TFT_FIXED["attention_head_size"], dropout=TFT_FIXED["dropout"],
            learning_rate=TFT_FIXED["learning_rate"], gradient_clip_val=TFT_FIXED["gradient_clip_val"],
            batch_size=TFT_FIXED["batch_size"], max_epochs=TFT_FIXED["max_epochs"], seed=SEED,
            stage=STAGE, model_family="TFT", config_id="P12_tft_supp_high_lb13",
            checkpoint_dir=str(checkpoint_dir),
        )
    except Exception as exc:  # noqa: BLE001
        status_completion, technical_failure = "technical_failure", True
        error_message = f"{type(exc).__name__}: {exc}"

    tested_config = {**spec["high_cost_config"], **TFT_FIXED}
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
            "note": "TFT supplementary(lb13 revised High): timing/peak_ram/epochs_completed 외 추가 진단 없음",
            "epochs_completed": result["epochs_completed"] if result else None,
        },
    }


def _run_informer(sub_a, fold) -> dict:
    spec = RUNS["Informer"]
    lookback, n_heads = spec["high_cost_config"]["lookback"], spec["high_cost_config"]["n_heads"]
    expected_label_len = lookback // 2
    informer_trainer.D_LAYERS = INF_FIXED["d_layers"]
    informer_trainer.DROPOUT = INF_FIXED["dropout"]

    checkpoint_path = get_checkpoint_dir("informer", "informer_supp_high_lb13") / "model.pt"
    status_completion, technical_failure, error_message, result = "completed", False, None, None
    n_heads_confirmed = None
    try:
        result = informer_trainer.train_and_evaluate_fold(
            sub_a, fold, HORIZON, lookback,
            e_layers=INF_FIXED["e_layers"], n_heads=n_heads,
            batch_size=INF_FIXED["batch_size"], max_epochs=INF_FIXED["max_epochs"],
            learning_rate=INF_FIXED["learning_rate"], seed=SEED,
            stage=STAGE, model_family="Informer", config_id="P12_informer_supp_high_lb13",
            checkpoint_path=str(checkpoint_path),
        )
        n_heads_confirmed = result["n_heads"]
        if n_heads_confirmed != n_heads:
            raise RuntimeError(f"trainer가 실제로 받은 n_heads={n_heads_confirmed}가 요청값 {n_heads}과 다름")
        if result["label_len"] != expected_label_len:
            raise RuntimeError(f"production이 만든 label_len={result['label_len']}이 "
                               f"floor(lookback/2)={expected_label_len}과 다름")
    except Exception as exc:  # noqa: BLE001
        status_completion, technical_failure = "technical_failure", True
        error_message = f"{type(exc).__name__}: {exc}"

    tested_config = {**spec["high_cost_config"], **INF_FIXED}
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
            "note": "Informer supplementary(lb13 revised High, n_heads=16 복원): "
                    "timing/peak_ram/epochs_completed 외 추가 진단 없음",
            "epochs_completed": result["epochs_completed"] if result else None,
            "label_len": result["label_len"] if result else expected_label_len,
            "n_heads_confirmed_by_trainer": n_heads_confirmed,
        },
    }


def _assemble_and_save(family: str, spec: dict, r: dict) -> Path:
    data_section = {"center": "A", "stage": STAGE, "validation_year": VALIDATION_YEAR,
                    "horizon": HORIZON, "fold": r["fold_id"]}
    execution_section = {
        "seed": SEED, "device": r["device"], "backend": r["backend"],
        "git_commit": get_git_commit(), "git_dirty": get_git_dirty(),
        "software_versions": r["software_versions"], **get_hardware_info(),
    }

    cheap_record = load_cheap_result(family.lower(), spec["cheap_experiment_id"])
    comparable, mismatches = check_timing_comparable_to_cheap(
        family=family, data=data_section, execution=execution_section,
        timing_scope=spec["timing_scope"] if r["total_runtime_sec"] is not None else None,
        tested_config=r["tested_config"], hpo_space=spec["hpo_space"], cheap_record=cheap_record,
    )
    cheap_runtime_sec = cheap_record["timing"]["total_runtime_sec"] if cheap_record else None
    runtime_ratio = compute_runtime_ratio(r["total_runtime_sec"], cheap_runtime_sec, comparable)
    if not comparable:
        print(f"  [INFO] {family}: 참조 cheap 기록과 비교 불가(mismatches={mismatches}) - "
              f"artifact 내 runtime_ratio는 None, 별도 비교는 report 단계에서 직접 계산")

    record = new_experiment_record(
        experiment_id=spec["experiment_id"], family=family, corner="high_cost", purpose=PURPOSE,
        evidence_status="CANONICAL" if r["status_completion"] == "completed" else "INVALID_INCOMPLETE",
        data={**data_section, "train_period": r["train_period"], "validation_period": r["validation_period"],
              "train_rows": r["train_rows"], "validation_rows": r["validation_rows"],
              "train_sequences": r["train_sequences"], "validation_sequences": r["validation_sequences"]},
        configuration={
            "hpo_space": spec["hpo_space"], "cheap_config": spec["cheap_config"],
            "high_cost_config": spec["high_cost_config"],
            "requested_config": r["requested_config"], "tested_config": r["tested_config"],
            "fixed_config": spec["fixed_config"], "configuration_basis": spec["configuration_basis"],
        },
        execution=execution_section,
        timing={
            "timing_scope": spec["timing_scope"] if r["total_runtime_sec"] is not None else None,
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
        artifact={"code_path": CODE_PATH, "canonical_artifact_path": None},
    )
    out_path = save_experiment(record, family.lower())
    print(f"[SAVED] {family}: {out_path}")
    print(f"  status={r['status_completion']} total_runtime_sec={r['total_runtime_sec']} "
          f"fallback_occurred={r['fallback_occurred']}")
    return out_path


def main() -> None:
    before = _snapshot_existing_artifacts()
    _validate_configs()

    sub_a, fold, _, _ = load_a_center_fold(VALIDATION_YEAR, FOLD_INDEX)

    saved_paths = {}
    print("\n[RUN] LSTM supplementary High(lb13)...")
    saved_paths["LSTM"] = _assemble_and_save("LSTM", RUNS["LSTM"], _run_lstm(sub_a, fold))

    print("\n[RUN] TFT supplementary High(lb13)...")
    saved_paths["TFT"] = _assemble_and_save("TFT", RUNS["TFT"], _run_tft(sub_a, fold))

    print("\n[RUN] Informer supplementary High(lb13)...")
    saved_paths["Informer"] = _assemble_and_save("Informer", RUNS["Informer"], _run_informer(sub_a, fold))

    after = _snapshot_existing_artifacts()
    artifacts_unchanged = (before == after)

    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    header = (f"{'model':>9} {'cheap_cfg':>26} {'new_high_cfg':>22} {'cheap_sec':>10} {'high_sec':>10} "
              f"{'ratio':>7} {'cheap_fit':>10} {'high_fit':>10} {'fit_ratio':>9} "
              f"{'train_seq':>10} {'peak_rss_mb':>12} {'fallback':>9} {'status':>18}")
    print(header)

    all_feasible = True
    n_heads_confirmed = None
    for family in ("LSTM", "TFT", "Informer"):
        spec = RUNS[family]
        new_rec = json.loads(saved_paths[family].read_text(encoding="utf-8"))
        cheap_rec = json.loads((DEFAULT_OUTPUT_ROOT / family.lower() /
                                f"{spec['cheap_experiment_id']}.json").read_text(encoding="utf-8"))

        cheap_total = cheap_rec["timing"]["total_runtime_sec"]
        high_total = new_rec["timing"]["total_runtime_sec"]
        cheap_fit = cheap_rec["timing"]["fit_train_sec"]
        high_fit = new_rec["timing"]["fit_train_sec"]
        total_ratio = (high_total / cheap_total) if (cheap_total and high_total) else None
        fit_ratio = (high_fit / cheap_fit) if (cheap_fit and high_fit) else None
        status = new_rec["status"]["completion_status"]
        if status != "completed":
            all_feasible = False

        print(f"{family:>9} {str(spec['cheap_config']):>26} {str(spec['high_cost_config']):>22} "
              f"{cheap_total:>10.2f} {str(high_total):>10} "
              f"{(f'{total_ratio:.2f}x' if total_ratio else 'N/A'):>7} "
              f"{cheap_fit:>10.2f} {str(high_fit):>10} "
              f"{(f'{fit_ratio:.2f}x' if fit_ratio else 'N/A'):>9} "
              f"{str(new_rec['data']['train_sequences']):>10} "
              f"{str(new_rec['resource']['peak_ram_mb']):>12} "
              f"{str(new_rec['status']['fallback_occurred']):>9} {status:>18}")

        print(f"  (참고) cheap train_sequences={cheap_rec['data']['train_sequences']}, "
              f"cheap peak_ram_mb={cheap_rec['resource']['peak_ram_mb']} "
              f"(단일 run RAM 차이는 방향성 있는 효과로 해석하지 않음)")

        if family == "Informer":
            n_heads_confirmed = new_rec["observed_compute_result"].get("n_heads_confirmed_by_trainer")

    print("\n" + "=" * 100)
    print("최종 확인")
    print("=" * 100)
    print(f"1. 세 New High 모두 technical feasible: {all_feasible}")
    print(f"2. Informer n_heads=16이 실제 trainer에 적용됨(trainer 반환값): {n_heads_confirmed == 16}")
    print(f"3. 기존 P13 Fold4 cheap/high_cost artifact(6개) 변경 없음: {artifacts_unchanged}")
    print("4. production training code/data 수정: 없음(이 스크립트는 신규 파일이며 기존 파일을 "
          "편집하지 않음. tft_trainer.LSTM_LAYERS/informer_trainer.D_LAYERS·DROPOUT 재바인딩은 "
          "기존 tft_profile.py/informer_profile.py와 동일하게 프로세스 내 전역 변수 재설정일 "
          "뿐 파일을 변경하지 않음)")
    print("5. 새로 생성/수정한 파일:")
    print(f"   - {CODE_PATH} (신규)")
    for family in ("LSTM", "TFT", "Informer"):
        print(f"   - {saved_paths[family]} (신규)")


if __name__ == "__main__":
    main()
