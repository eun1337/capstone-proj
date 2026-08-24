"""
run_queue.py

P13 family x horizon 단위 순차 실행 queue.

기존 family runner(informer/tft/rf/lstm/lightgbm)의 HPO 로직을 그대로 subprocess로 호출한다.
family/horizon마다 공식 산출물(JSON + selected OOF) 존재 여부와 실행 조건(seed/sampler_seed/
evaluation_population/common_eval_key_path) 일치 여부를 확인해 SKIP / PARTIAL / RUN을 판정하고,
PARTIAL이나 실행 실패(또는 subprocess 성공 후 산출물 재검증 실패)가 발생하면 즉시 queue를 중단한다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.forecasting.common.config import HORIZONS, P13_MODEL_SEED, P13_SAMPLER_SEED

# 각 family runner의 argparse --output-dir 기본값과 동일하다.
# lightgbm만 output-dir 기본값이 "outputs/hpo/lgbm"이고, JSON/OOF 파일명 자체는 FAMILY="lightgbm"을 사용한다.
FAMILY_OUTPUT_DIR = {
    "lightgbm": Path("outputs/hpo/lgbm"),
    "rf": Path("outputs/hpo/rf"),
    "lstm": Path("outputs/hpo/lstm"),
    "tft": Path("outputs/hpo/tft"),
    "informer": Path("outputs/hpo/informer"),
}
FAMILIES = tuple(FAMILY_OUTPUT_DIR)

QUEUE_LOG_DIR = Path("outputs/hpo/_queue_logs")

# 5개 runner 모두 result 딕셔너리에 동일한 값을 저장한다(FAMILY 값과 무관하게 고정 문자열).
EVALUATION_POPULATION = "five_family_common_keys"


def _official_json_path(family: str, horizon: int) -> Path:
    return FAMILY_OUTPUT_DIR[family] / f"p13_{family}_h{horizon}.json"


def _official_oof_dir(family: str) -> Path:
    return FAMILY_OUTPUT_DIR[family] / "oof"


def _oof_glob_pattern(family: str, horizon: int) -> str:
    return f"oof_{family}_h{horizon}_trial*_seed*_common.parquet"


def check_artifact_status(
    family: str, horizon: int, common_eval_keys_path: Path,
) -> tuple[str, list[str]]:
    """family/horizon의 공식 산출물 상태를 판정한다. 반환값: (SKIP|PARTIAL|RUN, 상세 메시지 목록)."""
    json_path = _official_json_path(family, horizon)
    oof_dir = _official_oof_dir(family)
    matched_oof_files = sorted(oof_dir.glob(_oof_glob_pattern(family, horizon))) if oof_dir.exists() else []

    if not json_path.exists():
        if matched_oof_files:
            return "PARTIAL", [
                f"공식 JSON 없음: {json_path}",
                f"OOF만 존재: {[str(p) for p in matched_oof_files]}",
            ]
        return "RUN", [f"공식 JSON 없음: {json_path}", "OOF 없음"]

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return "PARTIAL", [f"JSON 파싱 실패: {json_path} ({type(exc).__name__}: {exc})"]

    if data.get("horizon") != horizon or "selection" not in data:
        return "PARTIAL", [f"JSON 내용이 손상되었거나 horizon 필드 불일치: {json_path}"]

    # 기존 JSON이 현재 queue 실행 조건(runner default seed/sampler_seed 및 요청된 common-eval-keys-path)과
    # 같은지 확인한다. queue는 --seed/--sampler-seed를 별도로 받지 않으므로 각 runner의 default인
    # P13_MODEL_SEED/P13_SAMPLER_SEED를 기준으로 비교한다.
    mismatch: list[str] = []
    if data.get("seed") != P13_MODEL_SEED:
        mismatch.append(f"seed 불일치: json={data.get('seed')}, 기대값(P13_MODEL_SEED)={P13_MODEL_SEED}")
    if data.get("sampler_seed") != P13_SAMPLER_SEED:
        mismatch.append(
            f"sampler_seed 불일치: json={data.get('sampler_seed')}, "
            f"기대값(P13_SAMPLER_SEED)={P13_SAMPLER_SEED}"
        )
    if data.get("evaluation_population") != EVALUATION_POPULATION:
        mismatch.append(
            f"evaluation_population 불일치: json={data.get('evaluation_population')}, "
            f"기대값={EVALUATION_POPULATION}"
        )
    json_key_path = data.get("common_eval_key_path")
    if not json_key_path:
        mismatch.append(f"common_eval_key_path 없음(JSON 손상 가능): json={json_key_path}")
    else:
        try:
            if Path(json_key_path).resolve() != Path(common_eval_keys_path).resolve():
                mismatch.append(
                    f"common_eval_key_path 불일치: json={json_key_path}, "
                    f"요청값={common_eval_keys_path} (Path.resolve() 기준 비교)"
                )
        except Exception as exc:  # noqa: BLE001
            mismatch.append(f"common_eval_key_path 비교 실패: json={json_key_path} ({exc})")
    if mismatch:
        return "PARTIAL", [f"기존 JSON이 현재 queue 실행 조건과 다름: {json_path}"] + mismatch

    all_trials = data.get("all_trials", [])
    n_ok_trials = sum(1 for t in all_trials if isinstance(t, dict) and t.get("status") == "ok")

    selected = data["selection"].get("selected")

    if selected is None:
        # selected=None 안전성 검증: status=="ok" trial이 하나도 없으면 "정상 guardrail 실패"가 아니라
        # 전체 trial이 실패한 비정상 상태이므로 SKIP 처리하지 않는다.
        if n_ok_trials == 0:
            return "PARTIAL", [
                f"selection.selected=None이고 status=='ok'인 trial도 0개임 - 정상적인 bias guardrail "
                f"실패가 아니라 전체 trial 실패 가능성: {json_path}",
            ]
        # guardrail(|bias|<=BIAS_GUARDRAIL_ABS_PCT) 통과 trial이 없으면 runner는 OOF를 저장하지 않는다.
        # 성공 trial이 있는 상태에서의 이 경우는 OOF 부재가 정상 완료 상태이다.
        if matched_oof_files:
            return "PARTIAL", [
                f"selection.selected=None(guardrail 통과 trial 없음)인데 예상치 못한 OOF 파일 존재: "
                f"{[str(p) for p in matched_oof_files]}",
            ]
        return "SKIP", [
            f"JSON 완료(성공 trial {n_ok_trials}개 중 guardrail 통과 trial 없음 - OOF 없음이 정상): {json_path}",
        ]

    trial_id = selected.get("trial_id")
    seed = data.get("seed")
    expected_oof = oof_dir / f"oof_{family}_h{horizon}_trial{trial_id}_seed{seed}_common.parquet"

    # stale/중복 OOF 검증: selected trial의 OOF가 정확히 1개만 존재해야 한다.
    if expected_oof not in matched_oof_files:
        detail = [f"selected trial의 OOF 없음: {expected_oof}"]
        if matched_oof_files:
            detail.append(f"대신 다른 OOF 파일 존재(stale 가능성): {[str(p) for p in matched_oof_files]}")
        return "PARTIAL", detail

    if len(matched_oof_files) > 1:
        stale = [str(p) for p in matched_oof_files if p != expected_oof]
        return "PARTIAL", [
            f"selected trial의 OOF({expected_oof})는 존재하지만 stale/중복 OOF도 함께 존재함: {stale}",
        ]

    try:
        oof_df = pd.read_parquet(expected_oof, columns=["y_true", "y_pred"])
        if len(oof_df) == 0:
            return "PARTIAL", [f"OOF parquet이 비어 있음: {expected_oof}"]
    except Exception as exc:  # noqa: BLE001
        return "PARTIAL", [f"OOF parquet 로드 실패: {expected_oof} ({type(exc).__name__}: {exc})"]

    return "SKIP", [f"JSON+OOF 정상 존재(중복 없음): {json_path}, {expected_oof}"]


def build_command(family: str, horizon: int, common_eval_keys_path: Path) -> list[str]:
    return [
        sys.executable, "-m", f"src.forecasting.pipeline.p13.{family}",
        "--horizon", str(horizon),
        "--common-eval-keys-path", str(common_eval_keys_path),
    ]


def run_one(family: str, horizon: int, common_eval_keys_path: Path, log_path: Path) -> tuple[int, float]:
    """기존 family runner를 subprocess로 호출하고 stdout/stderr를 log_path에 기록한다."""
    cmd = build_command(family, horizon, common_eval_keys_path)
    start = datetime.now()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"[queue] family={family} horizon={horizon}\n")
        logf.write(f"[queue] command={' '.join(cmd)}\n")
        logf.write(f"[queue] start={start.isoformat()}\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        status = "SUCCESS" if proc.returncode == 0 else "FAILED"
        logf.write(
            f"[queue] end={end.isoformat()} elapsed={elapsed:.1f}s "
            f"status={status} returncode={proc.returncode}\n"
        )
    return proc.returncode, elapsed


def _append_log(log_path: Path, msg: str) -> None:
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(msg + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="P13 family x horizon 순차 실행 queue")
    parser.add_argument("--families", nargs="+", required=True, choices=list(FAMILIES),
                        help=f"실행할 family 목록(순서 유지). 허용값: {FAMILIES}")
    parser.add_argument("--common-eval-keys-path", required=True,
                        help="5-family P13(2023) common evaluation key parquet/csv 경로")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 subprocess/HPO 없이 RUN/SKIP/PARTIAL 판정과 호출 예정 command만 출력")
    args = parser.parse_args()

    common_eval_keys_path = Path(args.common_eval_keys_path)
    if not common_eval_keys_path.exists():
        print(f"[queue] common evaluation key 파일이 존재하지 않음(fail-fast): {common_eval_keys_path}")
        return 2

    jobs = [(family, h) for family in args.families for h in HORIZONS]

    plan: list[tuple[str, int, str, list[str]]] = []
    partial_found = False
    for family, h in jobs:
        status, detail = check_artifact_status(family, h, common_eval_keys_path)
        plan.append((family, h, status, detail))
        if status == "PARTIAL":
            partial_found = True

    if args.dry_run:
        print(f"[dry-run] 총 {len(jobs)}개 작업(family x horizon)")
        for family, h, status, detail in plan:
            print(f"[dry-run] {family} h{h}: {status}")
            for d in detail:
                print(f"    - {d}")
            if status == "RUN":
                cmd = build_command(family, h, common_eval_keys_path)
                print(f"    command: {' '.join(cmd)}")
        if partial_found:
            print("[dry-run] PARTIAL 상태가 감지됨 - 실제 실행 시 이 지점에서 queue 전체가 STOP됨")
        return 0

    queue_start = datetime.now()
    QUEUE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary_log_path = QUEUE_LOG_DIR / f"queue_{queue_start.strftime('%Y%m%d_%H%M%S')}.log"
    summary_log = open(summary_log_path, "w", encoding="utf-8")

    def emit(msg: str) -> None:
        print(msg)
        summary_log.write(msg + "\n")
        summary_log.flush()

    emit(f"[queue] 시작 {queue_start.isoformat()} - 총 {len(jobs)}개 작업 - summary_log={summary_log_path}")

    if partial_found:
        emit("[queue] PARTIAL 상태 발견 - 자동 overwrite/restart 불가, queue 전체 STOP")
        for family, h, status, detail in plan:
            if status == "PARTIAL":
                emit(f"  PARTIAL: {family} h{h}")
                for d in detail:
                    emit(f"    - {d}")
        summary_log.close()
        return 2

    total = len(jobs)
    completed = 0
    elapsed_list: list[float] = []
    # ETA는 SKIP을 제외하고 실제 RUN이 필요한 job 수만 기준으로 계산한다.
    n_run_remaining = sum(1 for _, _, status, _ in plan if status == "RUN")

    for family, h, status, _detail in plan:
        if status == "SKIP":
            emit(f"[queue] SKIP {family} h{h} - 공식 산출물 이미 존재")
            completed += 1
            continue

        job_start = datetime.now()
        emit(f"[queue] RUN {family} h{h} start={job_start.isoformat()}")
        log_path = QUEUE_LOG_DIR / f"{family}_h{h}.log"
        returncode, elapsed = run_one(family, h, common_eval_keys_path, log_path)
        job_end = datetime.now()

        if returncode != 0:
            emit(
                f"[queue] FAILED {family} h{h} end={job_end.isoformat()} elapsed={elapsed:.1f}s "
                f"returncode={returncode} log={log_path}"
            )
            emit(f"[queue] STOP - 이후 작업 진행하지 않음. 완료={completed}/{total}")
            summary_log.close()
            return 1

        # subprocess returncode=0만으로는 부족하므로 공식 산출물(JSON+selected OOF)이 실제로
        # 완료 상태인지 재검증한다. SKIP이 아니면(PARTIAL/RUN 모두) 실패로 처리하고 즉시 STOP한다.
        post_status, post_detail = check_artifact_status(family, h, common_eval_keys_path)
        _append_log(log_path, f"[queue] post-check status={post_status}")
        for d in post_detail:
            _append_log(log_path, f"[queue]   - {d}")

        if post_status != "SKIP":
            emit(
                f"[queue] FAILED {family} h{h} - subprocess returncode=0이지만 산출물 재검증 결과="
                f"{post_status} (비정상 종료로 처리)"
            )
            for d in post_detail:
                emit(f"    - {d}")
            emit(f"[queue] STOP - 이후 작업 진행하지 않음. 완료={completed}/{total} log={log_path}")
            summary_log.close()
            return 1

        completed += 1
        n_run_remaining -= 1
        elapsed_list.append(elapsed)
        emit(
            f"[queue] SUCCESS {family} h{h} end={job_end.isoformat()} elapsed={elapsed:.1f}s "
            f"completed={completed}/{total} log={log_path}"
        )
        if n_run_remaining > 0:
            avg_elapsed = sum(elapsed_list) / len(elapsed_list)
            emit(
                f"[queue] ETA(완료된 RUN 작업 평균 elapsed 기준, 남은 RUN {n_run_remaining}개) "
                f"~= {avg_elapsed * n_run_remaining:.1f}s"
            )

    queue_end = datetime.now()
    total_elapsed = (queue_end - queue_start).total_seconds()
    emit(
        f"[queue] 종료 {queue_end.isoformat()} elapsed={total_elapsed:.1f}s "
        f"완료={completed}/{total}"
    )
    summary_log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
