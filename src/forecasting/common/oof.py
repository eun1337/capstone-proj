"""
oof.py
Forecasting 공통 OOF row 생성/검증/저장. metric 계산은 evaluator.compute_metrics()의
책임이며, 이 파일은 fold별 validation row 단위 실제값/예측값만 다룬다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.common.config import HORIZONS

REQUIRED_KEY_COLS = ("center_id", "sku_id", "week_st", "target_date")
METADATA_COLS = ("stage", "model_family", "config_id", "seed", "horizon", "fold_id")
VALUE_COLS = ("y_true", "y_pred_log", "y_pred", "mase_scale")
BASE_OOF_COLUMNS = METADATA_COLS + REQUIRED_KEY_COLS + VALUE_COLS
DUPLICATE_KEY_COLS = ("stage", "model_family", "config_id", "seed", "horizon") + REQUIRED_KEY_COLS


def _require_identifier(name: str, value) -> None:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise ValueError(f"{name}에 유효한 식별 값이 필요함: {value!r}")


def build_oof_frame(
    keys: pd.DataFrame,
    y_true,
    y_pred_log,
    y_pred,
    mase_scale,
    *,
    stage,
    model_family,
    config_id,
    seed,
    horizon,
    fold_id,
) -> pd.DataFrame:
    """validation row의 key/실제값/예측값/MASE scale을 동일 schema의 OOF DataFrame으로 만든다."""
    missing_keys = [c for c in REQUIRED_KEY_COLS if c not in keys.columns]
    if missing_keys:
        raise KeyError(f"keys에 필수 컬럼 누락: {missing_keys}")

    if horizon not in HORIZONS:
        raise ValueError(f"지원하지 않는 horizon: {horizon!r} (허용값: {HORIZONS})")
    _require_identifier("stage", stage)
    _require_identifier("model_family", model_family)
    _require_identifier("config_id", config_id)

    y_true = np.asarray(y_true, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mase_scale = np.asarray(mase_scale, dtype=float)

    n = len(keys)
    for name, arr in (("y_true", y_true), ("y_pred_log", y_pred_log), ("y_pred", y_pred), ("mase_scale", mase_scale)):
        if len(arr) != n:
            raise ValueError(f"{name} 길이 불일치: {len(arr)} vs keys {n}")

    if not np.isfinite(y_true).all():
        raise ValueError("y_true에 NaN/Inf가 존재함")
    if not np.isfinite(y_pred_log).all():
        raise ValueError("y_pred_log에 NaN/Inf가 존재함")
    if not np.isfinite(y_pred).all():
        raise ValueError("y_pred에 NaN/Inf가 존재함")
    if (y_pred < 0).any():
        raise ValueError("y_pred에 음수가 존재함")

    oof = pd.DataFrame(index=range(n))
    oof["stage"] = stage
    oof["model_family"] = model_family
    oof["config_id"] = config_id
    oof["seed"] = seed
    oof["horizon"] = horizon
    oof["fold_id"] = fold_id
    oof["center_id"] = keys["center_id"].to_numpy()
    oof["sku_id"] = keys["sku_id"].to_numpy()
    oof["week_st"] = keys["week_st"].to_numpy()
    oof["target_date"] = keys["target_date"].to_numpy()
    if "row_id" in keys.columns:
        oof["row_id"] = keys["row_id"].to_numpy()
    oof["y_true"] = y_true
    oof["y_pred_log"] = y_pred_log
    oof["y_pred"] = y_pred
    oof["mase_scale"] = mase_scale

    validate_oof_frame(oof, expected_n=n)
    return oof


def validate_oof_frame(oof: pd.DataFrame, expected_n: int | None = None) -> None:
    """schema/길이/중복/finite/horizon/target_date 정합성을 검증한다."""
    missing = [c for c in BASE_OOF_COLUMNS if c not in oof.columns]
    if missing:
        raise KeyError(f"OOF에 필수 컬럼 누락: {missing}")

    if len(oof) == 0:
        raise ValueError("OOF가 비어 있음")
    if expected_n is not None and len(oof) != expected_n:
        raise ValueError(f"OOF row 수 불일치: {len(oof)} vs expected {expected_n}")

    dup = oof.duplicated(subset=list(DUPLICATE_KEY_COLS)).sum()
    if dup:
        raise ValueError(f"OOF 중복 row {dup}건 (stage/model_family/config_id/seed/horizon/center_id/sku_id/week_st/target_date 기준, fold_id 무관)")

    y_true = oof["y_true"].to_numpy(dtype=float)
    y_pred_log = oof["y_pred_log"].to_numpy(dtype=float)
    y_pred = oof["y_pred"].to_numpy(dtype=float)
    if not np.isfinite(y_true).all():
        raise ValueError("y_true에 NaN/Inf가 존재함")
    if not np.isfinite(y_pred_log).all():
        raise ValueError("y_pred_log에 NaN/Inf가 존재함")
    if not np.isfinite(y_pred).all():
        raise ValueError("y_pred에 NaN/Inf가 존재함")
    if (y_pred < 0).any():
        raise ValueError("y_pred에 음수가 존재함")

    bad_horizon = ~oof["horizon"].isin(HORIZONS)
    if bad_horizon.any():
        raise ValueError(f"지원하지 않는 horizon 존재: {sorted(oof.loc[bad_horizon, 'horizon'].unique())}")

    expected_target_date = oof["week_st"] + pd.to_timedelta(oof["horizon"] * 7, unit="D")
    if not (oof["target_date"] == expected_target_date).all():
        raise ValueError("target_date != week_st + horizon주 인 row가 존재함")


def save_oof(oof: pd.DataFrame, path) -> None:
    """validate_oof_frame 통과 후 parquet으로 저장한다. 경로는 호출부가 정한다."""
    validate_oof_frame(oof)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(path, index=False)
