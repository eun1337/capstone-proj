"""
dl_high_lb13_supplementary_informer_2023f1_resume.py
직전 scale-sweep 실행이 외부 요인으로 중단되어 6개 run 중 Informer 2023 P13 Fold1 하나만
누락됐다. 이미 저장된 5개 결과(LSTM/TFT/Informer 2022_F1, LSTM/TFT 2023_F1)는 절대
재실행하지 않고, 누락된 Informer 2023_F1 하나만 dl_high_lb13_supplementary_profile_
scale_sweep.py의 기존 run 함수(_run_one)를 그대로 재사용해 추가 실행한다. 실행 후에는
6개를 다시 돌리지 않고 이미 저장된 artifact 6개 + 기존 2023 Fold4 supplementary artifact를
읽기 전용으로 로드해 scale-sweep SUMMARY TABLE만 다시 출력한다.
"""

import hashlib
import json
from pathlib import Path

import src.forecasting.archive.benchmarks_p12.dl_high_lb13_supplementary_profile as supp
import src.forecasting.archive.benchmarks_p12.dl_high_lb13_supplementary_profile_scale_sweep as sweep
from src.forecasting.benchmarks.p12.common import DEFAULT_OUTPUT_ROOT

FAMILIES = ("LSTM", "TFT", "Informer")
TARGET_FAMILY, TARGET_SCALE_LABEL = "Informer", "2023_F1"

ARTIFACT_PATHS = {
    ("LSTM", "2022_F1"): DEFAULT_OUTPUT_ROOT / "lstm" / "lstm_supp_high_lb13_high_cost.json",
    ("TFT", "2022_F1"): DEFAULT_OUTPUT_ROOT / "tft" / "tft_supp_high_lb13_high_cost.json",
    ("Informer", "2022_F1"): DEFAULT_OUTPUT_ROOT / "informer" / "informer_supp_high_lb13_high_cost.json",
    ("LSTM", "2023_F1"): DEFAULT_OUTPUT_ROOT / "lstm" / "lstm_p13_2023_fold1_supp_high_lb13_high_cost.json",
    ("TFT", "2023_F1"): DEFAULT_OUTPUT_ROOT / "tft" / "tft_p13_2023_fold1_supp_high_lb13_high_cost.json",
    ("Informer", "2023_F1"): DEFAULT_OUTPUT_ROOT / "informer" /
        "informer_p13_2023_fold1_supp_high_lb13_high_cost.json",
    ("LSTM", "2023_F4"): DEFAULT_OUTPUT_ROOT / "lstm" / "lstm_p13_2023_fold4_supp_high_lb13_high_cost.json",
    ("TFT", "2023_F4"): DEFAULT_OUTPUT_ROOT / "tft" / "tft_p13_2023_fold4_supp_high_lb13_high_cost.json",
    ("Informer", "2023_F4"): DEFAULT_OUTPUT_ROOT / "informer" /
        "informer_p13_2023_fold4_supp_high_lb13_high_cost.json",
}

_ALREADY_DONE = [k for k in ARTIFACT_PATHS if k != (TARGET_FAMILY, TARGET_SCALE_LABEL)]


def _file_sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _snapshot_already_done() -> dict:
    return {str(ARTIFACT_PATHS[k]): _file_sha256(ARTIFACT_PATHS[k]) for k in _ALREADY_DONE}


def main() -> None:
    target_path = ARTIFACT_PATHS[(TARGET_FAMILY, TARGET_SCALE_LABEL)]
    if target_path.exists():
        raise FileExistsError(f"{target_path}가 이미 존재함 - 재실행하지 않고 종료")

    before = _snapshot_already_done()
    supp._validate_configs()

    scale = next(s for s in sweep.SCALES if s["label"] == TARGET_SCALE_LABEL)
    print(f"[RUN] {TARGET_FAMILY} {TARGET_SCALE_LABEL} supplementary High(lb13)...")
    saved_path = sweep._run_one(TARGET_FAMILY, scale)
    assert saved_path == target_path, f"저장 경로 불일치: {saved_path} != {target_path}"

    after = _snapshot_already_done()
    already_done_unchanged = (before == after)

    print("\n" + "=" * 120)
    print("SUMMARY TABLE - LSTM/TFT/Informer New High(lb13) compute scale sweep")
    print("=" * 120)
    header = (f"{'model':>9} {'2022F1_total':>12} {'2022F1_fit':>10} {'2023F1_total':>12} "
              f"{'2023F1_fit':>10} {'2023F4_total':>12} {'2023F4_fit':>10} "
              f"{'22->23F1':>9} {'23F1->23F4':>10} {'22->23F4':>9} {'status':>30}")
    print(header)

    all_feasible = True
    for family in FAMILIES:
        rec_2022 = json.loads(ARTIFACT_PATHS[(family, "2022_F1")].read_text(encoding="utf-8"))
        rec_2023f1 = json.loads(ARTIFACT_PATHS[(family, "2023_F1")].read_text(encoding="utf-8"))
        rec_2023f4 = json.loads(ARTIFACT_PATHS[(family, "2023_F4")].read_text(encoding="utf-8"))

        t22, f22 = rec_2022["timing"]["total_runtime_sec"], rec_2022["timing"]["fit_train_sec"]
        t23f1, f23f1 = rec_2023f1["timing"]["total_runtime_sec"], rec_2023f1["timing"]["fit_train_sec"]
        t23f4, f23f4 = rec_2023f4["timing"]["total_runtime_sec"], rec_2023f4["timing"]["fit_train_sec"]

        r1 = (t23f1 / t22) if (t22 and t23f1) else None
        r2 = (t23f4 / t23f1) if (t23f1 and t23f4) else None
        r3 = (t23f4 / t22) if (t22 and t23f4) else None

        statuses = {"2022_F1": rec_2022["status"]["completion_status"],
                    "2023_F1": rec_2023f1["status"]["completion_status"],
                    "2023_F4": rec_2023f4["status"]["completion_status"]}
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
    print(f"2. 기존 5개 완료 artifact 변경 없음: {already_done_unchanged}")
    print("3. production training code/data 수정: 없음")
    print(f"4. 새로 생성된 파일: {target_path} (신규), "
          f"{__file__.rsplit('capstone-proj/', 1)[-1]} (신규)")


if __name__ == "__main__":
    main()
