"""
dl_high_lb13_supplementary_profile_scale_sweep.py
2023 Fold4에서 이미 완료한 dl_high_lb13_supplementary_profile.py의 동일 New High
configuration(lookback=13 고정, LSTM hidden_size=64 / TFT hidden_size=16 /
Informer n_heads=16)을 2022 Fold1(P10)과 2023 Fold1(P13)에도 추가로 측정해
2022F1 -> 2023F1 -> 2023F4 규모별 compute evidence를 완성하는 supplementary profiling.
2023 Fold4는 기존 완료 artifact를 읽기 전용으로만 재사용하고 재실행하지 않으며, 기존
Cheap도 재실행하지 않는다. Fold4 스크립트의 run 함수(_run_lstm/_run_tft/_run_informer/
_assemble_and_save)는 STAGE/VALIDATION_YEAR 전역만 스코프별로 재바인딩해 그대로
재사용한다(함수 본문은 전혀 수정하지 않음).
"""

import hashlib
import json
from pathlib import Path

import src.forecasting.archive.benchmarks_p12.dl_high_lb13_supplementary_profile as supp
from src.forecasting.benchmarks.p12.common import DEFAULT_OUTPUT_ROOT
from src.forecasting.benchmarks.p12.runner import load_a_center_fold

CODE_PATH = "src/forecasting/benchmarks/p12/dl_high_lb13_supplementary_profile_scale_sweep.py"
FAMILIES = ("LSTM", "TFT", "Informer")

SCALES = (
    {"label": "2022_F1", "stage": "P10", "validation_year": 2022, "fold_index": 0},
    {"label": "2023_F1", "stage": "P13", "validation_year": 2023, "fold_index": 0},
)

RUN_FN = {"LSTM": supp._run_lstm, "TFT": supp._run_tft, "Informer": supp._run_informer}

_EXISTING_FILENAMES_TO_WATCH = (
    "{f}_cheap.json", "{f}_high_cost.json",
    "{f}_p13_2023_fold1_cheap.json", "{f}_p13_2023_fold1_high_cost.json",
    "{f}_p13_2023_fold4_cheap.json", "{f}_p13_2023_fold4_high_cost.json",
    "{f}_p13_2023_fold4_supp_high_lb13_high_cost.json",
)


def _file_sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _snapshot_existing_artifacts() -> dict:
    snapshot = {}
    for family in FAMILIES:
        fam_lower = family.lower()
        for pattern in _EXISTING_FILENAMES_TO_WATCH:
            path = DEFAULT_OUTPUT_ROOT / fam_lower / pattern.format(f=fam_lower)
            snapshot[str(path)] = _file_sha256(path)
    return snapshot


def _spec_for_scale(family: str, scale_label: str) -> dict:
    base = supp.RUNS[family]
    fam_lower = family.lower()
    if scale_label == "2022_F1":
        cheap_experiment_id = f"{fam_lower}_cheap"
        experiment_id = f"{fam_lower}_supp_high_lb13_high_cost"
    else:
        cheap_experiment_id = f"{fam_lower}_p13_2023_fold1_cheap"
        experiment_id = f"{fam_lower}_p13_2023_fold1_supp_high_lb13_high_cost"
    return {**base, "cheap_experiment_id": cheap_experiment_id, "experiment_id": experiment_id}


def _run_one(family: str, scale: dict) -> Path:
    saved_stage, saved_year = supp.STAGE, supp.VALIDATION_YEAR
    supp.STAGE, supp.VALIDATION_YEAR = scale["stage"], scale["validation_year"]
    try:
        sub_a, fold, _, _ = load_a_center_fold(scale["validation_year"], scale["fold_index"])
        r = RUN_FN[family](sub_a, fold)
        spec = _spec_for_scale(family, scale["label"])
        return supp._assemble_and_save(family, spec, r)
    finally:
        supp.STAGE, supp.VALIDATION_YEAR = saved_stage, saved_year


def main() -> None:
    before = _snapshot_existing_artifacts()
    supp._validate_configs()

    saved_paths = {}
    for scale in SCALES:
        for family in FAMILIES:
            print(f"\n[RUN] {family} {scale['label']} supplementary High(lb13)...")
            saved_paths[(family, scale["label"])] = _run_one(family, scale)

    after = _snapshot_existing_artifacts()
    artifacts_unchanged = (before == after)

    fold4_paths = {
        family: DEFAULT_OUTPUT_ROOT / family.lower() /
        f"{family.lower()}_p13_2023_fold4_supp_high_lb13_high_cost.json"
        for family in FAMILIES
    }

    print("\n" + "=" * 120)
    print("SUMMARY TABLE - LSTM/TFT/Informer New High(lb13) compute scale sweep")
    print("=" * 120)
    header = (f"{'model':>9} {'2022F1_total':>12} {'2022F1_fit':>10} {'2023F1_total':>12} "
              f"{'2023F1_fit':>10} {'2023F4_total':>12} {'2023F4_fit':>10} "
              f"{'22->23F1':>9} {'23F1->23F4':>10} {'22->23F4':>9} {'status':>30}")
    print(header)

    all_feasible = True
    for family in FAMILIES:
        rec_2022 = json.loads(saved_paths[(family, "2022_F1")].read_text(encoding="utf-8"))
        rec_2023f1 = json.loads(saved_paths[(family, "2023_F1")].read_text(encoding="utf-8"))
        rec_2023f4 = json.loads(fold4_paths[family].read_text(encoding="utf-8"))

        t22 = rec_2022["timing"]["total_runtime_sec"]
        f22 = rec_2022["timing"]["fit_train_sec"]
        t23f1 = rec_2023f1["timing"]["total_runtime_sec"]
        f23f1 = rec_2023f1["timing"]["fit_train_sec"]
        t23f4 = rec_2023f4["timing"]["total_runtime_sec"]
        f23f4 = rec_2023f4["timing"]["fit_train_sec"]

        r1 = (t23f1 / t22) if (t22 and t23f1) else None
        r2 = (t23f4 / t23f1) if (t23f1 and t23f4) else None
        r3 = (t23f4 / t22) if (t22 and t23f4) else None

        statuses = {
            "2022_F1": rec_2022["status"]["completion_status"],
            "2023_F1": rec_2023f1["status"]["completion_status"],
            "2023_F4": rec_2023f4["status"]["completion_status"],
        }
        if any(s != "completed" for s in statuses.values()):
            all_feasible = False
        status_str = f"22F1={statuses['2022_F1']},23F1={statuses['2023_F1']},23F4={statuses['2023_F4']}"

        print(f"{family:>9} {t22:>12.2f} {f22:>10.2f} {t23f1:>12.2f} {f23f1:>10.2f} "
              f"{t23f4:>12.2f} {f23f4:>10.2f} "
              f"{(f'{r1:.2f}x' if r1 else 'N/A'):>9} {(f'{r2:.2f}x' if r2 else 'N/A'):>10} "
              f"{(f'{r3:.2f}x' if r3 else 'N/A'):>9} {status_str:>30}")

    print("\n" + "=" * 120)
    print("최종 확인")
    print("=" * 120)
    print(f"1. 총 6개 run 모두 technical feasible: {all_feasible}")
    print(f"2. 기존 artifact(cheap/high_cost/fold4-supplementary) 변경 없음: {artifacts_unchanged}")
    print("3. production training code/data 수정: 없음(신규 파일만 추가, tft_trainer.LSTM_LAYERS/"
          "informer_trainer.D_LAYERS·DROPOUT 재바인딩은 기존 profile 스크립트와 동일한 프로세스 내 "
          "전역 변수 재설정일 뿐 파일을 변경하지 않음)")
    print("4. 새로 생성된 파일:")
    print(f"   - {CODE_PATH} (신규)")
    for (family, label), path in saved_paths.items():
        print(f"   - {path} (신규)")


if __name__ == "__main__":
    main()
