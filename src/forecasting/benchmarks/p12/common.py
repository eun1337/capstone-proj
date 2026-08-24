"""
common.py
P12 profiling 공통 유틸리티.
"""

from __future__ import annotations

import json
import platform
import resource
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from src.forecasting.benchmarks.p12.schema import validate_schema

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "benchmarks" / "p12"
DEFAULT_CHECKPOINT_ROOT = DEFAULT_OUTPUT_ROOT / "_checkpoints"


def get_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                 capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def get_git_dirty() -> Optional[bool]:
    try:
        result = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                 capture_output=True, text=True, timeout=10)
        return bool(result.stdout.strip()) if result.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def get_software_versions(*package_names: str) -> dict[str, Optional[str]]:
    import importlib.metadata

    versions: dict[str, Optional[str]] = {"python": platform.python_version()}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def get_hardware_info() -> dict[str, Optional[str]]:
    cpu_model = platform.processor() or None
    if sys.platform == "darwin":
        try:
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                     capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                cpu_model = result.stdout.strip()
        except Exception:  # noqa: BLE001
            pass
    return {"machine": platform.node() or None, "cpu_model": cpu_model}


def peak_ram_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024  # macOS=byte, Linux=KB


def get_checkpoint_dir(family: str, name: str, *,
                        checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT) -> Path:
    ckpt_dir = checkpoint_root / family / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def load_cheap_result(family: str, experiment_id: Optional[str] = None, *,
                       output_root: Path = DEFAULT_OUTPUT_ROOT) -> Optional[dict]:
    """cheap JSON이 없거나 CANONICAL+completed+corner=='cheap'가 아니면 None(참조로 쓰지 않음).
    experiment_id 생략 시 기존 2022 기본값(f"{family}_cheap")을 그대로 찾는다."""
    experiment_id = experiment_id or f"{family}_cheap"
    path = output_root / family / f"{experiment_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        record = json.load(fh)
    if (record.get("evidence_status") == "CANONICAL"
            and record.get("status", {}).get("completion_status") == "completed"
            and record.get("corner") == "cheap"):
        return record
    return None


def check_timing_comparable_to_cheap(*, family: str, data: dict[str, Any], execution: dict[str, Any],
                                       timing_scope: Optional[str], tested_config: dict[str, Any],
                                       hpo_space: dict[str, Any],
                                       cheap_record: Optional[dict]) -> tuple[bool, list[str]]:
    """cheap과 동일 환경/동일 non-HPO 설정일 때만 비교 가능. hpo_space의 축은 cheap/high_cost
    사이에서 원래 다르므로 비교 대상에서 제외하고, 그 외 tested_config(실제 사용값, LSTM
    fallback처럼 요청값과 달라질 수 있음)는 전부 비교한다."""
    if cheap_record is None:
        return False, ["cheap_record 없음"]

    mismatches: list[str] = []
    if cheap_record.get("family") != family:
        mismatches.append(f"family: {cheap_record.get('family')!r} != {family!r}")

    ref_data = cheap_record.get("data", {})
    for key in ("center", "stage", "validation_year", "fold", "horizon"):
        if ref_data.get(key) != data.get(key):
            mismatches.append(f"data.{key}: {ref_data.get(key)!r} != {data.get(key)!r}")

    ref_timing_scope = cheap_record.get("timing", {}).get("timing_scope")
    if ref_timing_scope != timing_scope:
        mismatches.append(f"timing_scope: {ref_timing_scope!r} != {timing_scope!r}")

    ref_execution = cheap_record.get("execution", {})
    for key in ("seed", "device", "backend", "git_commit", "git_dirty", "software_versions",
                "machine", "cpu_model"):
        if ref_execution.get(key) != execution.get(key):
            mismatches.append(f"{key}: {ref_execution.get(key)!r} != {execution.get(key)!r}")

    ref_tested = cheap_record.get("configuration", {}).get("tested_config", {})
    non_hpo_keys = (set(ref_tested) | set(tested_config)) - set(hpo_space)
    for key in non_hpo_keys:
        if ref_tested.get(key) != tested_config.get(key):
            mismatches.append(f"tested_config.{key}: {ref_tested.get(key)!r} != {tested_config.get(key)!r}")

    return (len(mismatches) == 0), mismatches


def compute_runtime_ratio(tested_total_sec: Optional[float], cheap_total_sec: Optional[float],
                           comparable: bool) -> Optional[float]:
    if not comparable or tested_total_sec is None or not cheap_total_sec:
        return None
    return tested_total_sec / cheap_total_sec


def save_experiment(record: dict, family: str, *, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    out_dir = output_root / family
    out_path = out_dir / f"{record['experiment_id']}.json"
    record["artifact"]["canonical_artifact_path"] = str(out_path)

    errors = validate_schema(record)
    if errors:
        raise ValueError("experiment record가 canonical schema를 위반함:\n" + "\n".join(errors))

    if out_path.exists():
        raise FileExistsError(f"{out_path}가 이미 존재함 - 다른 experiment_id를 써라")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2, default=str)
    return out_path
