"""outputs/model_comparison/weekly_error_2024.csv, coverage_summary_2024.csv,
stat_model_comparison_2024.csv, cross_track_predictions_2024.parquet 공용 로더.
05 상세 분석/Q&A 전용 — read-only. parquet은 27M행/100MB+라 전체 로드 없이
pyarrow.dataset filter pushdown으로 요청된 SKU·Center·Horizon 행만 읽는다."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from services.dashboard_data import load_product_master

OUTPUTS_DIR = Path(__file__).resolve().parents[3] / "outputs" / "model_comparison"

@lru_cache(maxsize=None)
def load_sku_product_lookup() -> pd.DataFrame:
    """product_master.parquet에서 (center, sku_id)별 상품명/옵션코드 조회용 테이블만 추출."""
    pm = load_product_master()
    return pm[["center_id", "sku_id", "상품명", "option_code"]].rename(columns={"center_id": "center"})

STAT_MODEL = "SARIMA"
STAT_VARIANT = "none"
ML_MODEL = "Hurdle-LightGBM"
ML_VARIANT = "operational_final"

STAT_KEY_TO_MODEL_VARIANT = {
    "ARIMA_S0": ("ARIMA", "S0"),
    "ARIMAX_S1": ("ARIMAX", "S1"),
    "ARIMAX_S2": ("ARIMAX", "S2"),
    "ARIMAX_S3": ("ARIMAX", "S3"),
    "ARIMAX_S4": ("ARIMAX", "S4"),
    "SARIMA": ("SARIMA", "none"),
    "SARIMAX_S4": ("SARIMAX", "S4"),
}

COVERAGE_SCOPE_LABELS = {"model_fit": "Model-fit", "fallback": "Fallback", "cold_start": "Cold-start"}

def weighted_weekly_combine(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """actual_sum/prediction_sum은 그대로 합산, WAPE는 sum(|err|)=WAPE/100*actual_sum로
    복원한 뒤 합산 그룹 기준으로 재계산 — 여러 center/comparison_scope 행을 하나의
    주차별 WAPE로 합칠 때 쓰는 표준 가중평균(04 WAPE Profile과 동일 원리)."""
    tmp = df.copy()
    tmp["_abs_err"] = tmp["WAPE"] / 100.0 * tmp["actual_sum"]
    g = tmp.groupby(group_cols, as_index=False).agg(
        actual_sum=("actual_sum", "sum"),
        prediction_sum=("prediction_sum", "sum"),
        _abs_err=("_abs_err", "sum"),
    )
    g["WAPE"] = g["_abs_err"] / g["actual_sum"] * 100
    g["Bias"] = (g["prediction_sum"] - g["actual_sum"]) / g["actual_sum"] * 100
    return g.drop(columns=["_abs_err"])

@lru_cache(maxsize=None)
def load_weekly_error() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "weekly_error_2024.csv")

@lru_cache(maxsize=None)
def load_stat_model_comparison() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "stat_model_comparison_2024.csv")

@lru_cache(maxsize=None)
def load_coverage_summary() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "coverage_summary_2024.csv")

@lru_cache(maxsize=None)
def _predictions_dataset():
    return ds.dataset(OUTPUTS_DIR / "cross_track_predictions_2024.parquet", format="parquet")

def query_sku_predictions(sku_id: str, center: str, horizon: int) -> pd.DataFrame:
    """선택 SKU의 SARIMA/Hurdle-LightGBM 주간 actual·prediction·comparison_scope —
    filter pushdown으로 해당 SKU/Center/Horizon 행만 읽어 27M행 전체 로드를 피한다."""
    dataset = _predictions_dataset()
    filt = (
        (ds.field("sku_id") == sku_id)
        & (ds.field("center_id") == center)
        & (ds.field("horizon") == horizon)
        & (ds.field("model").isin([STAT_MODEL, ML_MODEL]))
    )
    tbl = dataset.to_table(
        columns=["sku_id", "center_id", "horizon", "model", "variant", "target_date", "actual", "prediction", "comparison_scope"],
        filter=filt,
    )
    return tbl.to_pandas()

TS_MODELS = [
    ("SARIMA", "none", "SARIMA", True),
    ("Hurdle-LightGBM", "operational_final", "H-LGBM", True),
    ("ARIMA", "S0", "ARIMA", False),
    ("ARIMAX", "S4", "ARIMAX", False),
    ("SARIMAX", "S4", "SARIMAX", False),
]
TS_MODEL_LABELS = {m: label for m, _v, label, _d in TS_MODELS}

TS_PREDICTION_COLUMNS = [
    "sku_id", "center_id", "horizon", "model", "variant", "target_date", "actual", "prediction",
    "forecast_source", "development_n_obs", "model_applied_flag", "fallback_flag", "coldstart_flag",
    "comparison_scope",
]

def search_products(query: str, limit: int = 8) -> pd.DataFrame:
    """상품명/바코드/SKU 부분일치 검색. (center_id, sku_id) 행 단위로 반환하되
    같은 sku_id가 어느 센터들에 있는지(centers_available)는 product_master 전체 기준으로 계산한다
    (검색 결과가 한쪽 센터만 걸려도 다른 센터 존재 여부를 알 수 있어야 하므로)."""
    pm = load_product_master()
    q = (query or "").strip()
    if not q:
        return pm.iloc[0:0]
    mask = (
        pm["상품명"].str.contains(q, case=False, na=False, regex=False)
        | pm["barcode"].astype(str).str.contains(q, na=False, regex=False)
        | pm["sku_id"].str.contains(q, case=False, na=False, regex=False)
    )
    hit = pm[mask]
    if hit.empty:
        return hit
    centers_map = pm.groupby("sku_id")["center_id"].apply(lambda s: sorted(s.unique().tolist()))
    out = hit.head(limit).copy()
    out["centers_available"] = out["sku_id"].map(centers_map)
    return out[["sku_id", "center_id", "상품명", "barcode", "option_code", "centers_available"]]

def get_product_info(sku_id: str, center: str) -> dict | None:
    pm = load_product_master()
    hit = pm[(pm["sku_id"] == sku_id) & (pm["center_id"] == center)]
    if hit.empty:
        return None
    row = hit.iloc[0]
    centers_available = sorted(pm[pm["sku_id"] == sku_id]["center_id"].unique().tolist())
    return {
        "sku_id": sku_id, "center": center,
        "product_name": row["상품명"], "barcode": str(row["barcode"]), "option_code": row["option_code"],
        "category_large": row["KAN_대분류"], "category_middle": row["KAN_중분류"], "category_small": row["KAN_소분류"],
        "centers_available": centers_available,
    }

def query_sku_predictions_multi(sku_id: str, center: str, horizon: int, models: list[tuple[str, str]]) -> pd.DataFrame:
    """query_sku_predictions와 동일한 filter-pushdown 패턴 — cold-start/fallback/development_n_obs 등
    05 상세분석 전용 컬럼까지 함께 읽는다(불필요한 전체 컬럼 로드는 피함)."""
    dataset = _predictions_dataset()
    model_names = [m for m, _v in models]
    filt = (
        (ds.field("sku_id") == sku_id)
        & (ds.field("center_id") == center)
        & (ds.field("horizon") == horizon)
        & (ds.field("model").isin(model_names))
    )
    tbl = dataset.to_table(columns=TS_PREDICTION_COLUMNS, filter=filt)
    df = tbl.to_pandas()
    if df.empty:
        return df
    wanted = set(models)
    return df[df.apply(lambda r: (r["model"], r["variant"]) in wanted, axis=1)].reset_index(drop=True)

def _wape_bias(pred: pd.Series, act: pd.Series) -> tuple[float | None, float | None]:
    actual_sum = float(act.sum())
    if actual_sum == 0:
        return None, None
    wape = float((pred - act).abs().sum() / actual_sum * 100)
    bias = float((pred.sum() - actual_sum) / actual_sum * 100)
    return wape, bias

def compute_sku_summary(df: pd.DataFrame, weeks_window: str) -> dict:
    """df: query_sku_predictions_multi() 결과(여러 모델 행 혼재).
    actual은 (sku,center,horizon,target_date)마다 모델 수만큼 반복 저장돼 있으므로,
    target_date 기준으로 중복 제거한 실제 수요 시계열을 먼저 만든 뒤에만 actual 기반 지표를 계산한다."""
    result: dict = {
        "weeks_available": 0, "weeks_used": 0, "actual_consistent": True, "inconsistent_dates": [],
        "avg_actual": None, "zero_demand_ratio": None, "zero_demand_count": None, "actual_total": None,
        "models": [], "winner_model": None,
        "diagnostics": {}, "window_dates": [],
    }
    if df.empty:
        return result

    by_date = df.groupby("target_date")["actual"]
    nunique = by_date.nunique()
    inconsistent = nunique[nunique > 1].index
    if len(inconsistent):
        result["actual_consistent"] = False
        result["inconsistent_dates"] = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in sorted(inconsistent)[:5]]

    actual_series = df.drop_duplicates(subset="target_date").set_index("target_date")["actual"].sort_index()
    all_dates = list(actual_series.index)
    result["weeks_available"] = len(all_dates)
    if weeks_window != "all":
        n = int(weeks_window)
        window_dates = all_dates[-n:] if len(all_dates) > n else all_dates
    else:
        window_dates = all_dates
    result["weeks_used"] = len(window_dates)
    result["window_dates"] = list(window_dates)
    if not window_dates:
        return result

    actual_window = actual_series.loc[window_dates]
    result["avg_actual"] = float(actual_window.mean())
    result["zero_demand_count"] = int((actual_window == 0).sum())
    result["zero_demand_ratio"] = result["zero_demand_count"] / len(window_dates) * 100
    result["actual_total"] = float(actual_window.sum())

    window_set = set(window_dates)
    model_rows = []
    error_series_by_model: dict[str, pd.Series] = {}
    for model_name, variant, label, is_default in TS_MODELS:
        mdf = df[(df["model"] == model_name) & (df["variant"] == variant) & (df["target_date"].isin(window_set))]
        if mdf.empty:
            continue
        merged = mdf.merge(actual_window.rename("actual_dedup"), left_on="target_date", right_index=True)
        pred, act = merged["prediction"], merged["actual_dedup"]
        wape, bias = _wape_bias(pred, act)
        err = (pred - act).abs()
        mae = float(err.mean())
        rmse = float(np.sqrt(((pred - act) ** 2).mean()))
        dev_obs = merged["development_n_obs"].dropna()
        scope_counts = merged["comparison_scope"].value_counts()
        applied_counts = merged["model_applied_flag"].value_counts()
        source_counts = merged["forecast_source"].fillna("unknown").value_counts()
        n = len(merged)
        row = {
            "model": model_name, "variant": variant, "label": label, "is_default": is_default,
            "n_weeks_used": n,
            "wape": wape, "bias": bias, "mae": mae, "rmse": rmse,
            "actual_total": float(act.sum()), "prediction_total": float(pred.sum()),
            "fallback_ratio": float(scope_counts.get("fallback", 0)) / n * 100,
            "cold_start_ratio": float((merged["coldstart_flag"] == 1).sum()) / n * 100,
            "model_fit_ratio": float(scope_counts.get("model_fit", 0)) / n * 100,
            "development_n_obs": float(dev_obs.iloc[0]) if len(dev_obs) else None,
            "model_applied_rows": int(applied_counts.get(1, 0)),
            "comparison_scope_counts": {str(k): int(v) for k, v in scope_counts.items()},
            "forecast_source_counts": {str(k): int(v) for k, v in source_counts.items()},
        }
        model_rows.append(row)
        error_series_by_model[model_name] = pd.Series(err.values, index=merged["target_date"].values)

    result["models"] = model_rows
    ranked = sorted((r for r in model_rows if r["wape"] is not None), key=lambda r: r["wape"])
    result["winner_model"] = ranked[0]["model"] if ranked else None

    diag: dict = {}
    diag["zero_demand"] = {"count": result["zero_demand_count"], "total": len(window_dates), "ratio": result["zero_demand_ratio"]}
    any_coldstart = next((r for r in model_rows if r["cold_start_ratio"] > 0), None)
    diag["cold_start"] = {"present": any_coldstart is not None, "detail": [{"model": r["model"], "ratio": r["cold_start_ratio"]} for r in model_rows if r["cold_start_ratio"] > 0]}
    diag["fallback"] = {"present": any(r["fallback_ratio"] > 0 for r in model_rows), "detail": [{"model": r["model"], "ratio": r["fallback_ratio"]} for r in model_rows if r["fallback_ratio"] > 0]}
    dev_obs_vals = [r["development_n_obs"] for r in model_rows if r["development_n_obs"] is not None]
    diag["short_history"] = {"development_n_obs": min(dev_obs_vals) if dev_obs_vals else None, "is_short": (min(dev_obs_vals) < 52) if dev_obs_vals else None}
    diag["bias_direction"] = [
        {"model": r["model"], "bias": r["bias"], "tendency": ("과소예측" if r["bias"] is not None and r["bias"] <= -20 else "과대예측" if r["bias"] is not None and r["bias"] >= 20 else "허용범위(±20%)" if r["bias"] is not None else None)}
        for r in model_rows
    ]

    if result["winner_model"] and len(window_dates) >= 4:
        std = float(actual_window.std())
        if std > 0:
            z = (actual_window - actual_window.mean()) / std
            k = min(5, max(1, len(window_dates) // 4))
            demand_shift_dates = set(z.abs().sort_values(ascending=False).head(k).index)
            err_s = error_series_by_model.get(result["winner_model"])
            if err_s is not None and len(err_s) >= k:
                error_dates = set(err_s.sort_values(ascending=False).head(k).index)
                overlap = demand_shift_dates & error_dates
                if overlap:
                    diag["demand_shift_error_overlap"] = {
                        "k": k, "overlap_count": len(overlap), "model": result["winner_model"],
                        "overlap_dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in sorted(overlap)],
                    }

    result["diagnostics"] = diag
    return result

@lru_cache(maxsize=3)
def representative_skus(horizon: int) -> list[dict]:
    dataset = _predictions_dataset()
    filt = (
        (ds.field("horizon") == horizon)
        & (((ds.field("model") == STAT_MODEL) & (ds.field("variant") == STAT_VARIANT))
           | ((ds.field("model") == ML_MODEL) & (ds.field("variant") == ML_VARIANT)))
    )
    columns = [
        "center_id", "sku_id", "target_date", "model", "actual", "prediction",
        "fallback_flag", "coldstart_flag", "development_n_obs", "comparison_scope",
    ]
    df = dataset.to_table(columns=columns, filter=filt).to_pandas()
    if df.empty:
        return []
    keys = ["center_id", "sku_id", "target_date"]
    stat = df[df.model == STAT_MODEL].drop_duplicates(keys)
    ml = df[df.model == ML_MODEL].drop_duplicates(keys)
    common = stat.merge(ml, on=keys, suffixes=("_stat", "_ml"), validate="one_to_one")
    common = common[np.isclose(common.actual_stat, common.actual_ml)].copy()
    common["actual"] = common.actual_stat
    common["stat_abs"] = (common.prediction_stat - common.actual).abs()
    common["ml_abs"] = (common.prediction_ml - common.actual).abs()
    common["stat_error"] = common.prediction_stat - common.actual
    common["ml_error"] = common.prediction_ml - common.actual
    grouped = common.groupby(["center_id", "sku_id"], as_index=False).agg(
        n_rows=("target_date", "size"), actual_sum=("actual", "sum"),
        zero_ratio=("actual", lambda s: float((s == 0).mean() * 100)),
        stat_abs=("stat_abs", "sum"), ml_abs=("ml_abs", "sum"),
        stat_error=("stat_error", "sum"), ml_error=("ml_error", "sum"),
        fallback_rows=("fallback_flag_stat", lambda s: int((s == 1).sum())),
        cold_start_rows=("coldstart_flag_stat", lambda s: int((s == 1).sum())),
        development_n_obs=("development_n_obs_stat", "first"),
    )
    valid = grouped[(grouped.actual_sum > 0) & (grouped.n_rows >= 20)].copy()
    if valid.empty:
        return []
    for prefix in ["stat", "ml"]:
        valid[f"{prefix}_wape"] = valid[f"{prefix}_abs"] / valid.actual_sum * 100
        valid[f"{prefix}_bias"] = valid[f"{prefix}_error"] / valid.actual_sum * 100
    valid["fit_score"] = valid[["stat_wape", "ml_wape"]].mean(axis=1) + valid[["stat_bias", "ml_bias"]].abs().mean(axis=1)
    valid["error_score"] = valid[["stat_wape", "ml_wape"]].mean(axis=1)
    used: set[tuple[str, str]] = set()

    def choose(label: str, frame: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> dict | None:
        candidates = frame.sort_values(sort_cols, ascending=ascending)
        row = next((r for r in candidates.itertuples() if (r.center_id, r.sku_id) not in used), None)
        if row is None:
            return None
        used.add((row.center_id, row.sku_id))
        return {
            "key": label, "sku_id": row.sku_id, "center": row.center_id,
            "actual_sum": float(row.actual_sum), "n_rows": int(row.n_rows),
            "stat_wape": float(row.stat_wape), "ml_wape": float(row.ml_wape),
            "stat_bias": float(row.stat_bias), "ml_bias": float(row.ml_bias),
            "zero_ratio": float(row.zero_ratio), "fallback_rows": int(row.fallback_rows),
        }

    cases = [
        choose("similar", valid[valid.zero_ratio < 80], ["fit_score", "actual_sum"], [True, False]),
        choose("high_error", valid, ["error_score", "actual_sum"], [False, False]),
        choose("intermittent", valid[valid.zero_ratio >= 50], ["zero_ratio", "actual_sum"], [False, False]),
        choose("fallback", valid[valid.fallback_rows > 0], ["fallback_rows", "actual_sum"], [False, False]),
    ]
    return [case for case in cases if case is not None]
