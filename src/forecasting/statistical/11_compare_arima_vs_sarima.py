"""
11_compare_arima_vs_sarima.py
ARIMA(S0, data/ml/day2_statistical_models/arima/arima_holdout_2024.parquet)와 SARIMA/
SARIMAX(S4)(data/ml/day2_statistical_models/sarima_sarimax_sku/sarima_eval_common_panel.
parquet, model 컬럼으로 SARIMA/SARIMAX_S4 구분)을 (center_id, sku_id, forecast_origin,
horizon) 공통 key에서 3-way 비교하는 완전히 독립적인 스크립트. 01_build_arima_orders.py/
02_run_arima_holdout.py/06_evaluate_statistical_models.py/statistical_utils.py 등 기존
ARIMA 파이프라인 코드는 import하지 않고, pandas만으로 두 산출물 parquet과
development_2021_2023.parquet을 직접 읽어 계산한다(08_sku_sarima_sarimax.py도
import하지 않음 — MASE scale까지 이 스크립트 안에서 독립적으로 재계산).

지표 정의는 10_evaluate_sarima.py와 완전히 동일:
    error := prediction - actual
    WAPE = sum(|error|) / sum(|actual|)   (Primary)
    Bias = sum(error)    / sum(|actual|)
    RMSE = sqrt(mean(error^2))
    MAE  = mean(|error|)
    MASE = mean(|error| / scale_sku)  (scale_sku = SKU별 development 구간 lag-1 naive
           MAE: A 2021-01-04~2023-12-25 전체, B 2023-07-03~2023-12-25 Post-regime만.
           scale이 0/NaN/inf이거나 development n_obs<2인 SKU는 MASE 집계에서만 제외)

원칙:
    - prediction을 삭제·재계산하지 않는다. outlier 제거, clipping 추가, metric 정의
      임의 변경도 하지 않는다.
    - SARIMA/SARIMAX의 m/order/override 규칙을 재검토하거나 바꾸지 않는다 — sarima_eval_
      common_panel.parquet에 이미 override가 반영된 prediction_final을 있는 그대로
      비교 대상으로 쓴다.
    - actual 값이 두 파일 간 다른 key가 있어도 임의로 하나를 선택하지 않는다: 애초에
      단일 actual로 병합하지 않고 각 모델은 각자 원본 파일의 actual로 자신의 지표를
      계산한다(ARIMA 지표는 arima_actual, SARIMA/SARIMAX 지표는 각자의 actual). 두
      actual이 다른 key의 개수만 진단 정보로 audit에 남긴다.
    - B센터는 post-regime 관측 이력이 짧아 SARIMA/SARIMAX 계절(m) 구조가 탐색적으로
      결정됐다는 점을, 해석이나 개선책 제안 없이 결과표 옆에 원칙 문구로만 병기한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"

ARIMA_HOLDOUT_PATH = DATA_DIR / "ml" / "day2_statistical_models" / "arima" / "arima_holdout_2024.parquet"
SARIMA_PANEL_PATH = DATA_DIR / "ml" / "day2_statistical_models" / "sarima_sarimax_sku" / "sarima_eval_common_panel.parquet"
DEV_PATH = DATA_DIR / "development_2021_2023.parquet"

OUT_DIR = DATA_DIR / "ml" / "day2_statistical_models"
OUT_AUDIT = OUT_DIR / "arima_vs_sarima_compare_audit.csv"
OUT_METRICS = OUT_DIR / "arima_vs_sarima_compare_metrics.csv"

CENTER_COL, SKU_COL, WEEK_COL, QTY_COL = "center_id", "sku_id", "week_st", "qty"
KEY_COLS = [CENTER_COL, SKU_COL, "forecast_origin", "horizon"]
HORIZONS = [1, 2, 4]
CENTERS = ["A", "B"]
MODELS_SX = ["SARIMA", "SARIMAX_S4"]

A_HISTORY_START = pd.Timestamp("2021-01-04")
B_HISTORY_START = pd.Timestamp("2023-07-03")
DEV_MAX_WEEK = pd.Timestamp("2023-12-25")

B_SEASONAL_STRUCTURE_NOTE = (
    "B센터는 post-regime 관측 이력이 2023-07-03~2023-12-25(최대 26주)로 짧아, SARIMA/"
    "SARIMAX(S4) seasonal 구조(m 후보 {13,26}, m=52는 104주 미만이라 원천 배제)가 이 짧은 "
    "이력 내에서 탐색적으로 결정되었다(08_sku_sarima_sarimax.py m_candidates_for_history "
    "규칙). 이 사실은 결과 해석이 아니라 원칙 기록으로만 병기한다."
)


def _require_inputs() -> None:
    missing = []
    if not ARIMA_HOLDOUT_PATH.exists():
        missing.append(str(ARIMA_HOLDOUT_PATH))
    if not SARIMA_PANEL_PATH.exists():
        missing.append(str(SARIMA_PANEL_PATH))
    if missing:
        raise SystemExit(
            "[ERROR] 필요한 입력 파일이 없습니다:\n  " + "\n  ".join(missing) +
            "\nARIMA holdout은 01_build_arima_orders.py -> 02_run_arima_holdout.py, "
            "SARIMA/SARIMAX panel은 08_sku_sarima_sarimax.py -> 09_diagnose_sarima_"
            "instability.py -> 10_evaluate_sarima.py를 순서대로 먼저 실행하세요."
        )


def _load_dev_mase_scale() -> pd.DataFrame:
    """SKU별 development 구간(A 전체, B Post-regime만) lag-1 naive MAE.
    01/02/statistical_utils를 참조하지 않고 development_2021_2023.parquet을 직접 읽어
    10_evaluate_sarima.py의 _dev_stats()와 동일한 정의를 이 스크립트 안에서 재현한다."""
    dev_df = pd.read_parquet(DEV_PATH, columns=[CENTER_COL, SKU_COL, WEEK_COL, QTY_COL])
    is_a = (dev_df[CENTER_COL] == "A") & (dev_df[WEEK_COL] >= A_HISTORY_START) & (dev_df[WEEK_COL] <= DEV_MAX_WEEK)
    is_b = (dev_df[CENTER_COL] == "B") & (dev_df[WEEK_COL] >= B_HISTORY_START) & (dev_df[WEEK_COL] <= DEV_MAX_WEEK)
    dev_df = dev_df.loc[is_a | is_b].sort_values([CENTER_COL, SKU_COL, WEEK_COL])
    dev_df["abs_diff"] = dev_df.groupby([CENTER_COL, SKU_COL], observed=True)[QTY_COL].diff().abs()

    stats = dev_df.groupby([CENTER_COL, SKU_COL], observed=True).agg(
        development_n_obs=(QTY_COL, "size"),
        mase_scale=("abs_diff", "mean"),
    ).reset_index()
    valid = (
        stats["mase_scale"].notna()
        & np.isfinite(stats["mase_scale"])
        & (stats["mase_scale"] > 0)
        & (stats["development_n_obs"] >= 2)
    )
    stats["mase_scale_valid"] = valid
    return stats


def _metrics_block(pred: pd.Series, actual: pd.Series, mase_scale: pd.Series, mase_scale_valid: pd.Series) -> dict:
    """10_evaluate_sarima.py의 _metrics_block()과 완전히 동일한 계산식."""
    err = pred - actual
    abs_actual_sum = actual.abs().sum()
    n = len(pred)
    wape = float(err.abs().sum() / abs_actual_sum) if abs_actual_sum > 0 else np.nan
    bias = float(err.sum() / abs_actual_sum) if abs_actual_sum > 0 else np.nan
    rmse = float(np.sqrt((err ** 2).mean())) if n > 0 else np.nan
    mae = float(err.abs().mean()) if n > 0 else np.nan

    valid_mask = mase_scale_valid.fillna(False)
    n_mase = int(valid_mask.sum())
    if n_mase > 0:
        mase = float((pred[valid_mask] - actual[valid_mask]).abs().div(mase_scale[valid_mask]).mean())
    else:
        mase = np.nan

    return {"n": n, "WAPE": wape, "Bias": bias, "RMSE": rmse, "MAE": mae, "MASE": mase, "n_mase": n_mase}


def main() -> None:
    _require_inputs()

    arima = pd.read_parquet(
        ARIMA_HOLDOUT_PATH,
        columns=[*KEY_COLS, "prediction", "actual"],
    ).rename(columns={"prediction": "arima_prediction", "actual": "arima_actual"})
    # ARIMA holdout의 horizon은 "h1"/"h2"/"h4" 문자열(SARIMA/SARIMAX는 1/2/4 정수) — key
    # merge를 위해 형식만 정수로 맞춘다(값 자체는 그대로, 매핑 오류 방지용 명시적 검증 포함).
    _horizon_map = {"h1": 1, "h2": 2, "h4": 4}
    unknown_horizons = set(arima["horizon"].unique()) - set(_horizon_map)
    assert not unknown_horizons, f"예상 밖 ARIMA horizon 값: {unknown_horizons}"
    arima["horizon"] = arima["horizon"].map(_horizon_map).astype(int)
    n_arima_total = len(arima)
    n_key_dup_arima = int(arima.duplicated(subset=KEY_COLS, keep=False).sum())

    sx_panel = pd.read_parquet(
        SARIMA_PANEL_PATH,
        columns=["model", *KEY_COLS, "prediction_final", "actual", "has_observed_history"],
    ).rename(columns={"prediction_final": "sx_prediction", "actual": "sx_actual"})

    dev_stats = _load_dev_mase_scale()

    audit_rows_all = []
    metric_rows_all = []
    per_model_merged = {}

    for model_label in MODELS_SX:
        sx = sx_panel[sx_panel["model"] == model_label].drop(columns=["model"])
        n_sx_total = len(sx)
        n_key_dup_sx = int(sx.duplicated(subset=KEY_COLS, keep=False).sum())

        merged = arima.merge(sx, on=KEY_COLS, how="inner")
        n_common = len(merged)
        n_arima_only = n_arima_total - n_common
        n_sx_only = n_sx_total - n_common
        n_key_dup_merged = int(merged.duplicated(subset=KEY_COLS, keep=False).sum())

        both_actual_present = merged["arima_actual"].notna() & merged["sx_actual"].notna()
        actual_mismatch = both_actual_present & ~np.isclose(
            merged["arima_actual"].fillna(np.nan), merged["sx_actual"].fillna(np.nan),
            rtol=1e-9, atol=1e-9, equal_nan=True,
        )
        n_actual_mismatch = int(actual_mismatch.sum())
        n_actual_compared = int(both_actual_present.sum())

        merged = merged.merge(dev_stats, on=[CENTER_COL, SKU_COL], how="left")
        merged["mase_scale_valid"] = merged["mase_scale_valid"].infer_objects(copy=False).fillna(False)
        merged["has_observed_history"] = merged["has_observed_history"].infer_objects(copy=False).fillna(False)
        per_model_merged[model_label] = merged

        audit_rows_all.append({
            "model": model_label,
            "n_arima_total": n_arima_total,
            "n_sx_total": n_sx_total,
            "n_common_key_inner_join": n_common,
            "excluded_n_arima_only": n_arima_only,
            "excluded_n_sx_only": n_sx_only,
            "key_dup_count_arima": n_key_dup_arima,
            "key_dup_count_sx": n_key_dup_sx,
            "key_dup_count_merged": n_key_dup_merged,
            "actual_compared_both_present": n_actual_compared,
            "actual_mismatch_count": n_actual_mismatch,
            "actual_mismatch_flag_needs_investigation": n_actual_mismatch != 0,
            "main_n_has_observed_history_true": int(merged["has_observed_history"].sum()),
            "auxiliary_n_has_observed_history_false": int((~merged["has_observed_history"]).sum()),
        })

        main_panel = merged[merged["has_observed_history"]]
        for h in HORIZONS:
            h_sub = main_panel[main_panel["horizon"] == h]
            for center in [*CENTERS, "ALL"]:
                sub = h_sub if center == "ALL" else h_sub[h_sub[CENTER_COL] == center]
                arima_m = _metrics_block(sub["arima_prediction"], sub["arima_actual"],
                                          sub["mase_scale"], sub["mase_scale_valid"])
                sx_m = _metrics_block(sub["sx_prediction"], sub["sx_actual"],
                                       sub["mase_scale"], sub["mase_scale_valid"])
                wape_improve_pp = (
                    (arima_m["WAPE"] - sx_m["WAPE"]) * 100
                    if pd.notna(arima_m["WAPE"]) and pd.notna(sx_m["WAPE"]) else np.nan
                )
                metric_rows_all.append({
                    "horizon": h, "center_id": center, "model": "ARIMA", "compared_against": model_label,
                    "n": arima_m["n"], "WAPE": arima_m["WAPE"], "Bias": arima_m["Bias"], "RMSE": arima_m["RMSE"],
                    "MAE": arima_m["MAE"], "MASE": arima_m["MASE"], "n_mase": arima_m["n_mase"],
                    "wape_improvement_pp_arima_to_model": wape_improve_pp,
                })
                metric_rows_all.append({
                    "horizon": h, "center_id": center, "model": model_label, "compared_against": model_label,
                    "n": sx_m["n"], "WAPE": sx_m["WAPE"], "Bias": sx_m["Bias"], "RMSE": sx_m["RMSE"],
                    "MAE": sx_m["MAE"], "MASE": sx_m["MASE"], "n_mase": sx_m["n_mase"],
                    "wape_improvement_pp_arima_to_model": wape_improve_pp,
                })

    audit_df = pd.DataFrame(audit_rows_all)
    audit_long = audit_df.melt(id_vars="model", var_name="metric", value_name="value")
    note_row = pd.DataFrame([{"model": "ALL", "metric": "note_b_seasonal_structure", "value": B_SEASONAL_STRUCTURE_NOTE}])
    pd.concat([audit_long, note_row], ignore_index=True).to_csv(OUT_AUDIT, index=False, encoding="utf-8-sig")

    metrics_df = pd.DataFrame(metric_rows_all)[
        ["horizon", "center_id", "model", "compared_against", "n", "WAPE", "Bias", "RMSE", "MAE", "MASE", "n_mase",
         "wape_improvement_pp_arima_to_model"]
    ]
    metrics_df.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")

    print("=" * 90)
    for a in audit_rows_all:
        print(f"[감사][{a['model']}] ARIMA={a['n_arima_total']:,}, {a['model']}={a['n_sx_total']:,}, "
              f"공통key={a['n_common_key_inner_join']:,} (ARIMA-only 제외={a['excluded_n_arima_only']:,}, "
              f"{a['model']}-only 제외={a['excluded_n_sx_only']:,}), "
              f"actual비교={a['actual_compared_both_present']:,}, 불일치={a['actual_mismatch_count']:,}")
    print(f"  note_b_seasonal_structure: {B_SEASONAL_STRUCTURE_NOTE}")
    print()
    print("[결론] (숫자만)")
    for model_label in MODELS_SX:
        print(f"  --- ARIMA vs {model_label} ---")
        rows = [r for r in metric_rows_all if r["compared_against"] == model_label and r["center_id"] == "ALL"]
        for h in HORIZONS:
            arima_r = next(r for r in rows if r["horizon"] == h and r["model"] == "ARIMA")
            sx_r = next(r for r in rows if r["horizon"] == h and r["model"] == model_label)
            print(f"  h{h} ALL: ARIMA WAPE={arima_r['WAPE']:.4f} MASE={arima_r['MASE']:.4f} | "
                  f"{model_label} WAPE={sx_r['WAPE']:.4f} MASE={sx_r['MASE']:.4f} | "
                  f"개선폭={arima_r['wape_improvement_pp_arima_to_model']:.2f}%p")
    print()
    print("[저장]")
    print(f"  {OUT_AUDIT}")
    print(f"  {OUT_METRICS}")


if __name__ == "__main__":
    main()
