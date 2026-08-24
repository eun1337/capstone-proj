"""
progress.py
Forecasting 공통 진행률/소요시간 기록. config x horizon x fold 단위 실행이 끝날 때마다
호출부가 측정한 시간값을 받아 진행률/평균 실행시간/ETA를 계산해 record(dict)로 반환한다.
시간 측정(perf_counter) 자체는 호출부 책임이며, 여기서는 상태를 들고 있지 않는다.
"""

import math
from datetime import datetime, timedelta


def _format_hms(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_progress_record(
    *,
    model_family,
    stage,
    config_id,
    horizon,
    fold_id,
    completed: int,
    total: int,
    train_n: int,
    valid_n: int,
    fit_time_sec: float,
    predict_time_sec: float,
    run_time_sec: float,
    elapsed_sec: float,
) -> dict:
    """실행 1건의 timing을 받아 진행률/avg_run_sec/ETA/expected_finish를 계산한 record를 반환한다."""
    if total <= 0:
        raise ValueError(f"total은 1 이상이어야 함: {total!r}")
    if completed < 1:
        raise ValueError(f"completed는 1 이상이어야 함: {completed!r}")
    if completed > total:
        raise ValueError(f"completed가 total을 초과함: {completed!r} > {total!r}")
    if train_n < 0:
        raise ValueError(f"train_n은 음수일 수 없음: {train_n!r}")
    if valid_n < 0:
        raise ValueError(f"valid_n은 음수일 수 없음: {valid_n!r}")
    for name, value in (("fit_time_sec", fit_time_sec), ("predict_time_sec", predict_time_sec),
                        ("run_time_sec", run_time_sec), ("elapsed_sec", elapsed_sec)):
        if not math.isfinite(value):
            raise ValueError(f"{name}이 finite가 아님: {value!r}")
        if value < 0:
            raise ValueError(f"{name}은 음수일 수 없음: {value!r}")

    remaining = total - completed
    avg_run_sec = elapsed_sec / completed
    eta_sec = avg_run_sec * remaining
    expected_finish = datetime.now() + timedelta(seconds=eta_sec)

    return {
        "model_family": model_family, "stage": stage, "config_id": config_id,
        "horizon": horizon, "fold_id": fold_id,
        "completed": completed, "total": total,
        "train_n": train_n, "valid_n": valid_n,
        "fit_time_sec": fit_time_sec, "predict_time_sec": predict_time_sec, "run_time_sec": run_time_sec,
        "elapsed_sec": elapsed_sec, "avg_run_sec": avg_run_sec, "eta_sec": eta_sec,
        "expected_finish": expected_finish,
    }


def format_progress_line(record: dict) -> str:
    return (
        f"[{record['model_family']}][{record['stage']}] {record['completed']}/{record['total']} | "
        f"{record['config_id']} | h{record['horizon']} | fold {record['fold_id']} | "
        f"train={record['train_n']} valid={record['valid_n']} | "
        f"fit={record['fit_time_sec']:.1f}s pred={record['predict_time_sec']:.1f}s | "
        f"elapsed={_format_hms(record['elapsed_sec'])} ETA={_format_hms(record['eta_sec'])} | "
        f"finish={record['expected_finish']:%Y-%m-%d %H:%M}"
    )


def print_progress(record: dict) -> None:
    print(format_progress_line(record))
