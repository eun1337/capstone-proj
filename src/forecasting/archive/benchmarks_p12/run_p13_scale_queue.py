"""P12 2023 P13-scale(Fold1/Fold4) cheap/high-cost 실행 큐. 순차 실행, 병렬 금지."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from src.forecasting.benchmarks.p12.common import DEFAULT_OUTPUT_ROOT

REPO_ROOT = Path(__file__).resolve().parents[4]
LOG_DIR = DEFAULT_OUTPUT_ROOT / "queue_logs"
PY = sys.executable

_FAMILIES = ("rf", "lightgbm", "lstm", "tft", "informer")
_MODULES = {"rf": "rf_profile", "lightgbm": "lightgbm_profile", "lstm": "lstm_profile",
           "tft": "tft_profile", "informer": "informer_profile"}
_FOLDS = (("fold1", 0), ("fold4", 3))


def _build_steps() -> list[dict]:
    steps = []
    for fold_label, fold_index in _FOLDS:
        for family in _FAMILIES:
            prefix = f"{family}_p13_2023_{fold_label}"
            for corner in ("cheap", "high_cost"):
                steps.append({
                    "id": f"{prefix}_{corner}", "family": family,
                    "cmd": [PY, "-m", f"src.forecasting.benchmarks.p12.{_MODULES[family]}", corner,
                            "--stage", "P13", "--validation-year", "2023",
                            "--fold-index", str(fold_index), "--experiment-id-prefix", prefix],
                })
    return steps


STEPS = _build_steps()


def _existing_result_status(family: str, experiment_id: str) -> str | None:
    path = DEFAULT_OUTPUT_ROOT / family / f"{experiment_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        record = json.load(fh)
    if record.get("evidence_status") == "CANONICAL" and record.get("status", {}).get("completion_status") == "completed":
        return "already_completed"
    return "invalid_incomplete_exists"


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"p13_scale_queue_{time.strftime('%Y%m%dT%H%M%S')}.jsonl"

    with open(log_path, "w", encoding="utf-8") as log_fh:
        for step in STEPS:
            entry = {"experiment_id": step["id"], "cmd": step["cmd"]}

            existing = _existing_result_status(step["family"], step["id"])
            if existing == "already_completed":
                entry["status"] = "skipped"
                entry["reason"] = "이미 CANONICAL+completed 결과가 존재함"
                print(f"[SKIPPED] {step['id']}")
                log_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                log_fh.flush()
                continue
            if existing == "invalid_incomplete_exists":
                entry["status"] = "failed"
                entry["reason"] = "INVALID_INCOMPLETE 결과가 이미 존재함 - 자동 삭제/overwrite하지 않음"
                print(f"[FAILED] {step['id']} (기존 INVALID_INCOMPLETE artifact 존재)")
                log_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                log_fh.flush()
                continue

            start = time.time()
            entry["start_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start))
            print(f"[START] {step['id']}")

            proc = subprocess.run(step["cmd"], cwd=str(REPO_ROOT), capture_output=True, text=True)
            end = time.time()

            entry.update(
                end_iso=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(end)),
                wall_time_sec=end - start,
                returncode=proc.returncode,
                status="completed" if proc.returncode == 0 else "failed",
                stdout_tail=proc.stdout[-2000:],
                stderr_tail=proc.stderr[-2000:],
            )
            print(f"[{entry['status'].upper()}] {step['id']} wall_time_sec={entry['wall_time_sec']:.1f} "
                  f"returncode={proc.returncode}")
            log_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log_fh.flush()

    print(f"큐 로그: {log_path}")


if __name__ == "__main__":
    main()
