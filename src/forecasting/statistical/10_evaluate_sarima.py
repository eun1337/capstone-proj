"""
10_evaluate_sarima.py
SARIMA/SARIMAX(S4) SKU 트랙(08_sku_sarima_sarimax.py) 최종 성능 평가. 08이 SKU마다 SARIMA와
SARIMAX(S4)를 함께 fit해 forecast_results.parquet에 둘 다 저장한다는 점이 확인되어, 이
스크립트는 두 model을 각각 독립적으로(override도 model별로 분리 적용) 평가한다.

완전히 독립적인 스크립트다 — 02_run_arima_holdout.py/06_evaluate_statistical_models.py/
07_diagnose_arimax_economic_instability.py/statistical_utils.py(기존 ARIMA/ARIMAX
트랙 파일)는 로컬에 존재하지 않아 참조·import하지 않는다. 이 트랙 자체의 소스인
08_sku_sarima_sarimax.py(CENTER_COL/SKU_COL/QTY_COL/load_dev_frame)와, 같은 폴더의
09_diagnose_sarima_instability.py 산출물(override 규칙, model 컬럼 포함)만 의존한다.

이 스크립트가 하지 않는 것(원칙):
    - prediction/outlier를 삭제·재계산하지 않는다.
    - winsorizing, clipping을 추가하지 않는다.
    - metric 정의를 임의로 바꾸지 않는다.
    - forecast_results.parquet을 재작성하지 않는다(읽기 전용).

지표 정의(WAPE/Bias/RMSE/MAE/MASE, error := prediction - actual):
    WAPE = sum(|error|) / sum(|actual|)                (Primary)
    Bias = sum(error)    / sum(|actual|)                (부호 있는 상대 편향, WAPE와 같은 분모)
    RMSE = sqrt(mean(error^2))
    MAE  = mean(|error|)
    MASE = mean(|error| / scale_sku)  (scale_sku = 해당 SKU의 development 기간 lag-1 naive
           MAE. scale이 0/NaN/inf이거나 development n_obs<2인 SKU는 MASE 집계에서만 제외
           — 다른 지표에는 영향 없음)

model별 base panel: SARIMA는 forecast_source in {sarima, constant, naive_mean}, SARIMAX_S4는
{sarimax_s4, constant, naive_mean}. constant/naive_mean은 구조 탐색 자체가 실패한 SKU의
공용 fallback이라(process_sku에서 search가 None이면 SARIMA/SARIMAX 어느 쪽도 fit되지
않고 fallback_forecast_sku 하나만 호출됨) 두 model 평가에 동일하게 재사용된다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]

# sku_sarima_sarimax 모듈이 08_sku_sarima_sarimax.py로 숫자 접두사가 붙어 `import
# sku_sarima_sarimax`(식별자가 숫자로 시작 불가)로는 로드할 수 없으므로, 경로 기반
# importlib으로 명시적으로 로드한다(CENTER_COL/SKU_COL/QTY_COL/load_dev_frame 재사용 목적은 동일).
import importlib.util as _importlib_util

_skm_spec = _importlib_util.spec_from_file_location(
    "sku_sarima_sarimax", BASE_DIR / "src" / "ml" / "day2_statistical_models" / "08_sku_sarima_sarimax.py"
)
skm = _importlib_util.module_from_spec(_skm_spec)
_skm_spec.loader.exec_module(skm)

RESULT_DIR = BASE_DIR / "data" / "ml" / "day2_statistical_models" / "sarima_sarimax_sku"
FORECAST_RESULTS_PATH = RESULT_DIR / "forecast_results.parquet"
DIAGNOSIS_PATH = RESULT_DIR / "sarima_instability_diagnosis.csv"
OVERRIDE_PATH = RESULT_DIR / "sarima_status_override.csv"

OUT_AUDIT = RESULT_DIR / "sarima_eval_audit.csv"
OUT_COVERAGE = RESULT_DIR / "sarima_eval_coverage.csv"
OUT_COMMON_PANEL = RESULT_DIR / "sarima_eval_common_panel.parquet"
OUT_METRICS = RESULT_DIR / "sarima_eval_metrics.csv"
OUT_SOURCE_BREAKDOWN = RESULT_DIR / "sarima_eval_source_breakdown.csv"
OUT_COLDSTART_AUX = RESULT_DIR / "sarima_eval_coldstart_aux.csv"

KEY_COLS = [skm.CENTER_COL, skm.SKU_COL, "forecast_origin", "horizon"]
HORIZONS = [1, 2, 4]
CENTERS = ["A", "B"]

MODEL_SOURCES = {"SARIMA": "sarima", "SARIMAX_S4": "sarimax_s4"}


def _require_upstream_outputs() -> None:
    missing = [p for p in (DIAGNOSIS_PATH, OVERRIDE_PATH) if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise SystemExit(
            f"[ERROR] 필요한 09 산출물이 없습니다: {names}. "
            "09_diagnose_sarima_instability.py를 먼저 실행하세요."
        )


def _load_base_panel(df: pd.DataFrame, model_source: str) -> pd.DataFrame:
    return df[df["forecast_source"].isin([model_source, "constant", "naive_mean"])].copy()


def _apply_override(panel: pd.DataFrame, override_all: pd.DataFrame, model_label: str) -> tuple[pd.DataFrame, dict]:
    override_model = override_all[override_all["model"] == model_label]
    override_key = override_model[[*KEY_COLS, "override_scope", "override_prediction"]].copy()

    merged = panel.merge(override_key, on=KEY_COLS, how="left", indicator=True)
    is_override = merged["_merge"] == "both"
    n_override_applied = int(is_override.sum())
    n_override_rows_in_file = len(override_key)
    n_override_unmatched = n_override_rows_in_file - n_override_applied

    merged["forecast_source_final"] = merged["forecast_source"]
    merged.loc[is_override & (merged["override_scope"] == "sku_level"), "forecast_source_final"] = "naive_mean_override_sku"
    merged.loc[is_override & (merged["override_scope"] == "row_level"), "forecast_source_final"] = "naive_mean_override_row"
    merged["prediction_final"] = np.where(is_override, merged["override_prediction"], merged["prediction"])

    merged = merged.drop(columns=["_merge", "override_scope", "override_prediction"])
    merged["override_applied"] = is_override

    audit_extra = {
        "override_rows_in_file": n_override_rows_in_file,
        "override_applied_matched": n_override_applied,
        "override_unmatched_in_file": n_override_unmatched,
        "override_sku_level_applied": int((merged["forecast_source_final"] == "naive_mean_override_sku").sum()),
        "override_row_level_applied": int((merged["forecast_source_final"] == "naive_mean_override_row").sum()),
    }
    return merged, audit_extra


def _dev_stats() -> pd.DataFrame:
    """SKU별 development n_obs, has_observed_history, MASE scale(lag-1 naive MAE, QTY_COL 기준).
    model과 무관 — SKU 단위로 1회만 계산."""
    dev_df = skm.load_dev_frame()
    dev_df = dev_df.sort_values([skm.CENTER_COL, skm.SKU_COL, skm.WEEK_COL])
    dev_df["abs_diff"] = dev_df.groupby([skm.CENTER_COL, skm.SKU_COL], observed=True)[skm.QTY_COL].diff().abs()

    stats = dev_df.groupby([skm.CENTER_COL, skm.SKU_COL], observed=True).agg(
        development_n_obs=(skm.QTY_COL, "size"),
        mase_scale=("abs_diff", "mean"),
    ).reset_index()
    stats["has_observed_history"] = stats["development_n_obs"] > 0
    valid = (
        stats["mase_scale"].notna()
        & np.isfinite(stats["mase_scale"])
        & (stats["mase_scale"] > 0)
        & (stats["development_n_obs"] >= 2)
    )
    stats["mase_scale_valid"] = valid
    return stats


def _metrics_block(sub: pd.DataFrame) -> dict:
    """sub: prediction_final/actual/mase_scale/mase_scale_valid 컬럼을 가진 행 집합."""
    err = sub["prediction_final"] - sub["actual"]
    abs_actual_sum = sub["actual"].abs().sum()
    n = len(sub)
    wape = float(err.abs().sum() / abs_actual_sum) if abs_actual_sum > 0 else np.nan
    bias = float(err.sum() / abs_actual_sum) if abs_actual_sum > 0 else np.nan
    rmse = float(np.sqrt((err ** 2).mean())) if n > 0 else np.nan
    mae = float(err.abs().mean()) if n > 0 else np.nan

    mase_rows = sub[sub["mase_scale_valid"]]
    n_mase = len(mase_rows)
    if n_mase > 0:
        mase = float((mase_rows["prediction_final"] - mase_rows["actual"]).abs().div(mase_rows["mase_scale"]).mean())
    else:
        mase = np.nan

    return {"n": n, "WAPE": wape, "Bias": bias, "RMSE": rmse, "MAE": mae, "MASE": mase, "n_mase": n_mase}


def process_model(model_label: str, model_source: str, df: pd.DataFrame, override_all: pd.DataFrame,
                   dev_stats: pd.DataFrame) -> dict:
    panel = _load_base_panel(df, model_source)
    n_before_override_join = len(panel)

    panel, override_audit = _apply_override(panel, override_all, model_label)

    panel = panel.merge(dev_stats, on=[skm.CENTER_COL, skm.SKU_COL], how="left")
    panel["has_observed_history"] = panel["has_observed_history"].fillna(False)
    panel["mase_scale_valid"] = panel["mase_scale_valid"].fillna(False)

    n_key_dup = int(panel.duplicated(subset=KEY_COLS, keep=False).sum())
    n_horizon_bad = int((~panel["horizon"].isin(HORIZONS)).sum())

    actual_isna = panel["actual"].isna()
    n_actual_censored_excluded = int(actual_isna.sum())
    eval_panel = panel.loc[~actual_isna].copy()

    pred_isfinite = np.isfinite(eval_panel["prediction_final"])
    pred_nonneg = eval_panel["prediction_final"] >= 0
    n_pred_nonfinite = int((~pred_isfinite).sum())
    n_pred_negative = int((pred_isfinite & ~pred_nonneg).sum())

    audit = {
        "model": model_label,
        "n_base_panel_rows": n_before_override_join,
        "key_dup_count": n_key_dup,
        "horizon_invalid_count": n_horizon_bad,
        "actual_censored_excluded": n_actual_censored_excluded,
        "n_eval_panel_rows": len(eval_panel),
        "prediction_nonfinite_count": n_pred_nonfinite,
        "prediction_negative_count": n_pred_negative,
        **override_audit,
    }

    n_main = int(eval_panel["has_observed_history"].sum())
    n_cold = int((~eval_panel["has_observed_history"]).sum())
    coverage_rows = [
        {"model": model_label, "stage": f"base_panel({model_source}+constant+naive_mean)",
         "n": n_before_override_join, "excluded_n": 0},
        {"model": model_label, "stage": "actual_censored_excluded", "n": len(eval_panel),
         "excluded_n": n_actual_censored_excluded},
    ]
    for center in CENTERS:
        n_c = int((eval_panel[skm.CENTER_COL] == center).sum())
        coverage_rows.append({"model": model_label, "stage": f"eval_panel_center_{center}", "n": n_c, "excluded_n": np.nan})
    for h in HORIZONS:
        n_h = int((eval_panel["horizon"] == h).sum())
        coverage_rows.append({"model": model_label, "stage": f"eval_panel_horizon_h{h}", "n": n_h, "excluded_n": np.nan})
    coverage_rows.append({"model": model_label, "stage": "main(has_observed_history=True)", "n": n_main, "excluded_n": np.nan})
    coverage_rows.append({"model": model_label, "stage": "cold_start(has_observed_history=False)", "n": n_cold, "excluded_n": np.nan})

    panel_out_cols = [*KEY_COLS, "forecast_source", "forecast_source_final", "override_applied",
                       "prediction", "prediction_final", "actual", "has_observed_history",
                       "mase_scale", "mase_scale_valid"]
    common_panel_out = eval_panel[panel_out_cols].copy()
    common_panel_out.insert(0, "model", model_label)

    main_panel = eval_panel[eval_panel["has_observed_history"]]
    metric_rows = []
    for h in HORIZONS:
        h_sub = main_panel[main_panel["horizon"] == h]
        for center in [*CENTERS, "ALL"]:
            sub = h_sub if center == "ALL" else h_sub[h_sub[skm.CENTER_COL] == center]
            row = {"model": model_label, "horizon": h, "center_id": center}
            row.update(_metrics_block(sub))
            metric_rows.append(row)
    metrics_df = pd.DataFrame(metric_rows)[
        ["model", "horizon", "center_id", "n", "WAPE", "Bias", "RMSE", "MAE", "MASE", "n_mase"]
    ]

    src_rows = []
    total_n = len(main_panel)
    for src, g in main_panel.groupby("forecast_source_final"):
        row = {"model": model_label, "forecast_source_final": src,
               "share_pct": round(len(g) / total_n * 100, 3) if total_n > 0 else np.nan}
        row.update(_metrics_block(g))
        src_rows.append(row)
    src_df = pd.DataFrame(src_rows)[
        ["model", "forecast_source_final", "n", "share_pct", "WAPE", "Bias", "RMSE", "MAE", "MASE", "n_mase"]
    ].sort_values("n", ascending=False) if src_rows else pd.DataFrame(
        columns=["model", "forecast_source_final", "n", "share_pct", "WAPE", "Bias", "RMSE", "MAE", "MASE", "n_mase"])

    non_override_metrics = _metrics_block(main_panel[~main_panel["override_applied"]])
    override_only_metrics = _metrics_block(main_panel[main_panel["override_applied"]])
    overall_metrics = _metrics_block(main_panel)

    cold_panel = eval_panel[~eval_panel["has_observed_history"]]
    cold_rows = []
    if len(cold_panel) > 0:
        for h in HORIZONS:
            h_sub = cold_panel[cold_panel["horizon"] == h]
            for center in [*CENTERS, "ALL"]:
                sub = h_sub if center == "ALL" else h_sub[h_sub[skm.CENTER_COL] == center]
                row = {"model": model_label, "horizon": h, "center_id": center}
                row.update(_metrics_block(sub))
                cold_rows.append(row)
    cold_df = pd.DataFrame(cold_rows, columns=["model", "horizon", "center_id", "n", "WAPE", "Bias", "RMSE", "MAE", "MASE", "n_mase"])

    return {
        "audit": audit, "coverage_rows": coverage_rows, "common_panel": common_panel_out,
        "metrics_df": metrics_df, "src_df": src_df, "cold_df": cold_df,
        "n_main": n_main, "n_cold": n_cold, "eval_panel_len": len(eval_panel),
        "non_override_metrics": non_override_metrics, "override_only_metrics": override_only_metrics,
        "overall_metrics": overall_metrics,
    }


def _reason_note(all_audits: list[dict]) -> str:
    parts = ", ".join(
        f"{a['model']} {a['override_applied_matched']}행(sku_level {a['override_sku_level_applied']}/"
        f"row_level {a['override_row_level_applied']})"
        for a in all_audits
    )
    return (
        f"override 적용 현황 — {parts} (09_diagnose_sarima_instability.py 자동탐지 결과): "
        "재현 검증된 수치 불안정성(fit_sarimax의 enforce_stationarity=False/enforce_invertibility="
        "False 하 near-unit-root 계수가 refit=False 재귀 append 중 발산) 때문이며, 2024 홀드아웃 "
        "결과를 보고 구조(m/order)를 재탐색한 것이 아니라 기존 실패-fallback 체계(naive_mean)를 "
        "그대로 적용한 것이다."
    )


def main() -> None:
    _require_upstream_outputs()

    df = pd.read_parquet(FORECAST_RESULTS_PATH)  # 읽기 전용
    override_all = pd.read_csv(OVERRIDE_PATH, encoding="utf-8-sig", parse_dates=["forecast_origin"])
    dev_stats = _dev_stats()

    results = {}
    for model_label, model_source in MODEL_SOURCES.items():
        results[model_label] = process_model(model_label, model_source, df, override_all, dev_stats)

    audit_df = pd.DataFrame([r["audit"] for r in results.values()])
    audit_long = audit_df.melt(id_vars="model", var_name="metric", value_name="value")
    note_row = pd.DataFrame([{"model": "ALL", "metric": "note", "value": _reason_note([r["audit"] for r in results.values()])}])
    pd.concat([audit_long, note_row], ignore_index=True).to_csv(OUT_AUDIT, index=False, encoding="utf-8-sig")

    coverage_all = pd.DataFrame([row for r in results.values() for row in r["coverage_rows"]])
    coverage_all.to_csv(OUT_COVERAGE, index=False, encoding="utf-8-sig")

    common_panel_all = pd.concat([r["common_panel"] for r in results.values()], ignore_index=True)
    common_panel_all.to_parquet(OUT_COMMON_PANEL, index=False)

    metrics_all = pd.concat([r["metrics_df"] for r in results.values()], ignore_index=True)
    metrics_all.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")

    src_all = pd.concat([r["src_df"] for r in results.values()], ignore_index=True)
    src_all.to_csv(OUT_SOURCE_BREAKDOWN, index=False, encoding="utf-8-sig")

    cold_all = pd.concat([r["cold_df"] for r in results.values()], ignore_index=True)
    cold_all.to_csv(OUT_COLDSTART_AUX, index=False, encoding="utf-8-sig")

    print("=" * 90)
    for model_label, r in results.items():
        a = r["audit"]
        print(f"[감사][{model_label}] key_dup={a['key_dup_count']}, horizon_invalid={a['horizon_invalid_count']}, "
              f"actual_censored_excluded={a['actual_censored_excluded']}, "
              f"prediction_nonfinite={a['prediction_nonfinite_count']}, prediction_negative={a['prediction_negative_count']}")
        print(f"[override][{model_label}] 파일 내 행={a['override_rows_in_file']}, 적용(매칭)={a['override_applied_matched']}, "
              f"불일치={a['override_unmatched_in_file']} (sku_level={a['override_sku_level_applied']}, "
              f"row_level={a['override_row_level_applied']})")
    print()
    print("[결론] (숫자만)")
    for model_label, r in results.items():
        print(f"  --- {model_label} ---")
        print(f"  평가 대상 행수(actual censored 제외 후): {r['eval_panel_len']:,} "
              f"(Main {r['n_main']:,} / cold-start {r['n_cold']:,})")
        pooled_by_h = {int(row.horizon): row for row in r["metrics_df"][r["metrics_df"].center_id == "ALL"].itertuples()}
        for h in HORIZONS:
            row = pooled_by_h.get(h)
            if row is not None:
                print(f"  h{h} ALL센터 Main: n={row.n:,}, WAPE={row.WAPE:.4f}, Bias={row.Bias:.4f}, "
                      f"RMSE={row.RMSE:.4f}, MAE={row.MAE:.4f}, MASE={row.MASE:.4f}(n_mase={row.n_mase:,})")
        nom, oom, om = r["non_override_metrics"], r["override_only_metrics"], r["overall_metrics"]
        print(f"  override 적용 {r['audit']['override_applied_matched']}건 — ALL-Main 기준 WAPE: "
              f"미적용행={nom['WAPE']:.4f}, override행={oom['WAPE']:.4f}, 전체(최종)={om['WAPE']:.4f}")
    print()
    print("[저장] (6개 산출물, 전부 model 컬럼으로 SARIMA/SARIMAX_S4 구분)")
    for p in [OUT_AUDIT, OUT_COVERAGE, OUT_COMMON_PANEL, OUT_METRICS, OUT_SOURCE_BREAKDOWN, OUT_COLDSTART_AUX]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
