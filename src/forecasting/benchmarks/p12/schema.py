"""
schema.py
P12 profiling 결과 스키마.
"""

from __future__ import annotations

from typing import Any

NA = "N/A"

EVIDENCE_STATUS_VALUES = ("CANONICAL", "INVALID_INCOMPLETE")
CORNER_VALUES = ("cheap", "high_cost")
COMPLETION_STATUS_VALUES = ("completed", "technical_failure", "incomplete")
TIMING_SCOPE_VALUES = (
    "trainer_preprocess_fit_predict",
    "full_pipeline_fixed_epochs",
    "single_epoch_forced",
)

_TOP_LEVEL_SECTIONS = (
    "experiment_id", "family", "corner", "purpose", "evidence_status",
    "data", "configuration", "execution", "timing", "resource",
    "status", "observed_compute_result", "artifact",
)

_FORBIDDEN_METRIC_KEYS = {"wape", "bias", "rmse", "mae", "mase", "validation_score",
                           "predictive_score", "metrics"}


def _find_forbidden_metric_keys(obj: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in _FORBIDDEN_METRIC_KEYS:
                found.add(k)
            found |= _find_forbidden_metric_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            found |= _find_forbidden_metric_keys(item)
    return found


def new_experiment_record(
    *,
    experiment_id: str,
    family: str,
    corner: str,
    purpose: str,
    evidence_status: str,
    data: dict[str, Any],
    configuration: dict[str, Any],
    execution: dict[str, Any],
    timing: dict[str, Any],
    resource: dict[str, Any],
    status: dict[str, Any],
    observed_compute_result: dict[str, Any],
    artifact: dict[str, Any],
) -> dict:
    if evidence_status not in EVIDENCE_STATUS_VALUES:
        raise ValueError(f"evidence_status는 {EVIDENCE_STATUS_VALUES} 중 하나여야 함: {evidence_status!r}")
    if corner not in CORNER_VALUES:
        raise ValueError(f"corner는 {CORNER_VALUES} 중 하나여야 함: {corner!r}")
    return {
        "experiment_id": experiment_id, "family": family, "corner": corner, "purpose": purpose,
        "evidence_status": evidence_status, "data": data, "configuration": configuration,
        "execution": execution, "timing": timing, "resource": resource,
        "status": status, "observed_compute_result": observed_compute_result, "artifact": artifact,
    }


def validate_schema(record: dict) -> list[str]:
    errors: list[str] = []

    for key in _TOP_LEVEL_SECTIONS:
        if key not in record:
            errors.append(f"필수 top-level 키 누락: {key!r}")
    if errors:
        return errors

    if record["evidence_status"] not in EVIDENCE_STATUS_VALUES:
        errors.append(f"evidence_status 허용값 위반: {record['evidence_status']!r}")
    if record["corner"] not in CORNER_VALUES:
        errors.append(f"corner 허용값 위반: {record['corner']!r}")

    data = record["data"]
    for k in ("center", "stage", "validation_year", "horizon", "fold",
              "train_period", "validation_period", "train_rows", "validation_rows",
              "train_sequences", "validation_sequences"):
        if k not in data:
            errors.append(f"data.{k} 누락")

    config = record["configuration"]
    for k in ("hpo_space", "cheap_config", "high_cost_config", "requested_config",
              "tested_config", "fixed_config", "configuration_basis"):
        if k not in config:
            errors.append(f"configuration.{k} 누락")

    execution = record["execution"]
    for k in ("seed", "device", "backend", "git_commit", "git_dirty", "software_versions",
              "machine", "cpu_model"):
        if k not in execution:
            errors.append(f"execution.{k} 누락")

    timing = record["timing"]
    for k in ("timing_scope", "data_preparation_sec", "fit_train_sec",
              "validation_or_prediction_sec", "total_runtime_sec", "cheap_runtime_sec",
              "runtime_ratio", "timing_comparable_to_cheap"):
        if k not in timing:
            errors.append(f"timing.{k} 누락")
    if timing.get("timing_scope") not in (*TIMING_SCOPE_VALUES, None, NA):
        errors.append(f"timing.timing_scope 허용값 위반: {timing.get('timing_scope')!r}")
    if timing.get("timing_comparable_to_cheap") is True and timing.get("runtime_ratio") is None \
            and timing.get("cheap_runtime_sec") is not None:
        errors.append("timing_comparable_to_cheap=True인데 runtime_ratio가 None임")

    status = record["status"]
    for k in ("completion_status", "technical_failure", "error_message",
              "fallback_occurred", "fallback_reason"):
        if k not in status:
            errors.append(f"status.{k} 누락")
    if status.get("completion_status") not in (*COMPLETION_STATUS_VALUES, None):
        errors.append(f"status.completion_status 허용값 위반: {status.get('completion_status')!r}")
    if record["evidence_status"] == "CANONICAL" and status.get("completion_status") != "completed":
        errors.append("evidence_status=CANONICAL이려면 status.completion_status==completed여야 함")

    if "peak_ram_mb" not in record["resource"]:
        errors.append("resource.peak_ram_mb 누락")

    leaked = _find_forbidden_metric_keys(record.get("observed_compute_result"))
    if leaked:
        errors.append(f"observed_compute_result에 predictive metric 혼입됨: {leaked}")

    artifact = record["artifact"]
    for k in ("code_path", "canonical_artifact_path"):
        if k not in artifact:
            errors.append(f"artifact.{k} 누락")

    return errors
