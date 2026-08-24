"""
model_artifact.py

Final retrain model artifact의 공통 metadata 저장/검증.

family별 trainer가 저장한 model artifact 경로와 학습 configuration, seed,
training population 정보를 기록한다. Final retrain artifact는 validation,
EarlyStopping, best-epoch selection을 사용하지 않은 경우에만 2024 Holdout에 사용할 수 있다.
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = 1


def save_metadata(
    metadata_path,
    *,
    family: str,
    horizon: int,
    config: dict,
    seed: int,
    model_artifact_path,
    training_population_size: int,
    population_unit: str,
    fit_time_sec: float,
    epochs_completed: int | None = None,
) -> Path:
    if population_unit not in ("rows", "sequences"):
        raise ValueError(f"population_unit은 'rows'|'sequences'여야 함: {population_unit!r}")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "family": family, "horizon": horizon, "config": config, "seed": seed,
        "model_artifact_path": str(model_artifact_path),
        "training_population_size": training_population_size,
        "population_unit": population_unit,
        "fit_time_sec": fit_time_sec,
        "epochs_completed": epochs_completed,
        "validation_used": False,
        "early_stopping_used": False,
        "best_epoch_selection_used": False,
    }
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
    return metadata_path


def load_metadata(metadata_path) -> dict:
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Final retrain artifact metadata가 없음: {metadata_path}")
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    for key in ("validation_used", "early_stopping_used", "best_epoch_selection_used"):
        if metadata.get(key) is not False:
            raise ValueError(
                f"{metadata_path}: {key}={metadata.get(key)!r}가 False가 아님 - Final retrain "
                "artifact가 아닐 가능성(2024 holdout에 쓸 수 없음)"
            )
    return metadata
