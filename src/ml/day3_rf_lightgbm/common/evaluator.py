"""
evaluator.py
Day3 RF/LightGBM 공통 평가 - WAPE/Bias/RMSE/MAE/MASE. 원 수량(raw) scale에서 계산하며,
MASE는 통계모델 트랙(06_evaluate_statistical_models.py)과 동일하게 (center_id, sku_id)별
lag-1 naive scale을 사용한 row-level scaled error 평균이다.
"""

import numpy as np
import pandas as pd


def inverse_transform_prediction(pred_log) -> np.ndarray:
    """log1p 예측값을 raw quantity로 역변환하고 음수는 0으로 clip한다."""
    pred_log = np.asarray(pred_log, dtype=float)
    if not np.isfinite(pred_log).all():
        raise ValueError("pred_log에 NaN/Inf가 존재함")
    raw = np.clip(np.expm1(pred_log), 0.0, None)
    if not np.isfinite(raw).all():
        raise ValueError("역변환 결과에 NaN/Inf가 존재함(expm1 overflow)")
    return raw


def build_mase_scale(history_df: pd.DataFrame, keys_df: pd.DataFrame) -> np.ndarray:
    """history_df(과거 실측)로 (center_id, sku_id)별 lag-1 naive scale
    (=mean(|qty_t-qty_(t-1)|), week_st 순서)을 만들고 keys_df의 행 순서(center_id, sku_id)에
    매핑해 반환한다. n_obs<2, scale<=0, NaN/Inf, history 자체에 없는 조합은 NaN이 된다
    (compute_metrics가 해당 row를 MASE에서만 제외한다)."""
    dup = history_df.duplicated(subset=["center_id", "sku_id", "week_st"]).sum()
    if dup:
        raise ValueError(f"history_df의 (center_id, sku_id, week_st) 중복 {dup}건 - 임의 aggregate 금지")

    ordered = history_df.sort_values(["center_id", "sku_id", "week_st"])
    grp = ordered.groupby(["center_id", "sku_id"])["qty"]
    scale = grp.apply(lambda s: s.diff().abs().mean())
    n_obs = grp.size()
    scale = scale.where((n_obs >= 2) & np.isfinite(scale) & (scale > 0))

    key_index = pd.MultiIndex.from_frame(keys_df[["center_id", "sku_id"]])
    return scale.reindex(key_index).to_numpy()


def compute_metrics(y_true, y_pred, mase_scale=None) -> dict:
    """WAPE/Bias/RMSE/MAE/MASE를 계산한다. mase_scale이 주어지지 않으면 MASE는 NaN.
    입력은 위치 기준(numpy 변환)으로만 다뤄 pandas index 정렬에 의한 재정렬을 막는다."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        raise ValueError("y_true/y_pred가 비어 있음")
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true/y_pred 길이 불일치: {len(y_true)} vs {len(y_pred)}")
    if not np.isfinite(y_true).all():
        raise ValueError("y_true에 NaN/Inf가 존재함")
    if not np.isfinite(y_pred).all():
        raise ValueError("y_pred에 NaN/Inf가 존재함")
    if (y_pred < 0).any():
        raise ValueError("y_pred에 음수가 존재함(inverse_transform_prediction 적용 여부 확인)")

    n = len(y_true)
    err = y_pred - y_true
    denom = float(np.sum(y_true))

    wape = float(np.sum(np.abs(err)) / denom * 100) if denom != 0 else np.nan
    bias = float(np.sum(err) / denom * 100) if denom != 0 else np.nan
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    mase, mase_n = np.nan, 0
    if mase_scale is not None:
        mase_scale = np.asarray(mase_scale, dtype=float)
        if len(mase_scale) != n:
            raise ValueError(f"mase_scale 길이 불일치: {len(mase_scale)} vs {n}")
        valid = np.isfinite(mase_scale) & (mase_scale > 0)
        mase_n = int(valid.sum())
        if mase_n > 0:
            mase = float(np.mean(np.abs(err[valid]) / mase_scale[valid]))

    return {"wape": wape, "bias": bias, "rmse": rmse, "mae": mae, "mase": mase, "n": n, "mase_n": mase_n}
