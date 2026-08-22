"""
compute_benchmark.py
RF/LSTM/TFT/Informer 1차 Compute Benchmark 오케스트레이터. 각 family의 실제
production train_and_evaluate_fold를 그대로 호출하는 독립 worker 프로세스
(_rf_compute_bench_worker.py 등)를 순차 실행하고, family당 60분 timeout을 강제한다.
timeout 시 해당 family만 안전하게 중단하고 status를 기록한 뒤 다음 family로 진행한다.
성능 metric은 worker가 참고용으로 함께 기록하지만 이 스크립트는 그 값을 어떤 판단에도
쓰지 않는다. Search Space/architecture/feature/HPO는 변경하지 않는다. 결과 CSV
파일명(four_family_compute_benchmark.csv)은 기존 실측 artifact이므로 스크립트/모듈
이름이 바뀌어도 그대로 유지한다.
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

TIMEOUT_SEC = 3600
OUTPUT_DIR = Path("outputs/benchmarks")
RESULT_CSV = OUTPUT_DIR / "four_family_compute_benchmark.csv"

REPO_ROOT = Path(__file__).resolve().parents[3]


def run_family(family: str, cmd: list, out_path: Path) -> dict:
    print(f"[{family}] 시작: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, timeout=TIMEOUT_SEC,
            capture_output=True, text=True,
        )
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            print(f"[{family}] 비정상 종료(returncode={proc.returncode}), stderr 마지막 부분:", flush=True)
            print(proc.stderr[-3000:], flush=True)
            return {
                "family": family, "configuration": None, "n_train": None, "n_val": None,
                "data_prep_time_sec": None, "train_time_sec": None, "predict_time_sec": None,
                "total_time_sec": elapsed, "epochs_completed": None, "best_epoch": None,
                "peak_ram_mb": None, "device": None, "device_memory_mb": None,
                "device_memory_metric": None, "status": "error",
                "notes": f"worker 프로세스 오류(returncode={proc.returncode}): {proc.stderr[-500:]}",
            }
        if not out_path.exists():
            return {
                "family": family, "configuration": None, "n_train": None, "n_val": None,
                "data_prep_time_sec": None, "train_time_sec": None, "predict_time_sec": None,
                "total_time_sec": elapsed, "epochs_completed": None, "best_epoch": None,
                "peak_ram_mb": None, "device": None, "device_memory_mb": None,
                "device_memory_metric": None, "status": "error",
                "notes": "worker가 정상 종료했으나 결과 파일이 생성되지 않음",
            }
        with open(out_path, encoding="utf-8") as fh:
            result = json.load(fh)
        print(f"[{family}] 완료: {elapsed:.1f}s", flush=True)
        return result

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        print(f"[{family}] {TIMEOUT_SEC/60:.0f}분 초과 - 중단", flush=True)
        return {
            "family": family, "configuration": None, "n_train": None, "n_val": None,
            "data_prep_time_sec": None, "train_time_sec": None, "predict_time_sec": None,
            "total_time_sec": elapsed, "epochs_completed": None, "best_epoch": None,
            "peak_ram_mb": None, "device": None, "device_memory_mb": None,
            "device_memory_metric": None, "status": ">60min / compute-heavy",
            "notes": f"{TIMEOUT_SEC/60:.0f}분 초과로 프로세스 중단, 어느 단계였는지는 알 수 없음(결과 파일 미생성)",
        }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        rf_out = tmpdir / "rf_result.json"
        results.append(run_family(
            "RF",
            [sys.executable, "-m", "src.ml.benchmarks._rf_compute_bench_worker", "--out", str(rf_out)],
            rf_out,
        ))

        lstm_out = tmpdir / "lstm_result.json"
        lstm_ckpt = tmpdir / "lstm_ckpt.pt"
        results.append(run_family(
            "LSTM",
            [sys.executable, "-m", "src.ml.benchmarks._lstm_compute_bench_worker",
             "--out", str(lstm_out), "--checkpoint", str(lstm_ckpt)],
            lstm_out,
        ))

        tft_out = tmpdir / "tft_result.json"
        tft_ckpt_dir = tmpdir / "tft_ckpt"
        results.append(run_family(
            "TFT",
            [sys.executable, "-m", "src.ml.benchmarks._tft_compute_bench_worker",
             "--out", str(tft_out), "--checkpoint-dir", str(tft_ckpt_dir)],
            tft_out,
        ))

        informer_out = tmpdir / "informer_result.json"
        informer_ckpt = tmpdir / "informer_ckpt.pt"
        results.append(run_family(
            "Informer",
            [sys.executable, "-m", "src.ml.benchmarks._informer_compute_bench_worker",
             "--out", str(informer_out), "--checkpoint", str(informer_ckpt)],
            informer_out,
        ))

    df = pd.DataFrame([{
        "family": r["family"], "horizon": 1, "fold": 1,
        "configuration": r.get("configuration"),
        "n_train": r.get("n_train"), "n_val": r.get("n_val"),
        "data_prep_time_sec": r.get("data_prep_time_sec"),
        "train_time_sec": r.get("train_time_sec"),
        "predict_time_sec": r.get("predict_time_sec"),
        "total_time_sec": r.get("total_time_sec"),
        "epochs_completed": r.get("epochs_completed"),
        "best_epoch": r.get("best_epoch"),
        "peak_ram_mb": r.get("peak_ram_mb"),
        "device": r.get("device"),
        "device_memory_mb": r.get("device_memory_mb"),
        "device_memory_metric": r.get("device_memory_metric"),
        "status": r.get("status"),
        "notes": r.get("notes"),
    } for r in results])

    df.to_csv(RESULT_CSV, index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 20)
    print()
    print(df.to_string(index=False))
    print()
    print(f"저장: {RESULT_CSV}")


if __name__ == "__main__":
    main()
