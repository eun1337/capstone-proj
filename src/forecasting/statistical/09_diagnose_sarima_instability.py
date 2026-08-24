"""
09_diagnose_sarima_instability.py
SARIMA/SARIMAX(S4) SKU 트랙(08_sku_sarima_sarimax.py) 사후진단 단계 — 07_diagnose_arimax_
economic_instability.py와 동일 역할(수치 불안정 SKU를 사후에 자동 탐지해 별도 status로
분리)을 이 SKU 단위 트랙에 대해 수행한다.

배경 및 범위 변경 이력(2026-08-22):
    1차: forecast_results.parquet(forecast_source=='sarima')을 수동 조사해 literal Inf가
    있는 SKU 2건을 확인, 재현 검증까지 완료함. 원인: fit_sarimax()의 enforce_stationarity=
    False/enforce_invertibility=False 하에서 near-unit-root 계수가 refit=False 재귀
    append 중 발산.
    2차: abs(prediction_unclipped)>=1e15(overflow 직전) 조건까지 전체 스캔해 SKU 7건 추가.
    3차: 절대값(1e15) 기준을 SKU별 development 실측 스케일 대비 상대 기준(dev_max_qty*100)
    으로 교체(04/07 ARIMAX economic exog 진단과 동일 원칙).
    4차(이 버전): forecast_source=='sarima'만 탐지 대상이었으나, 08이 SARIMA와 SARIMAX(S4)
    를 SKU마다 함께 fit하고 둘 다 forecast_results.parquet에 저장한다는 점이 확인되어,
    SARIMAX(S4, forecast_source=='sarimax_s4')에도 동일 탐지 규칙을 독립적으로 적용한다.
    두 모델은 서로 다른 fit 결과이므로 어느 한쪽만 불안정할 수 있어 SKU 단위로 따로 판정한다.

이 스크립트가 하지 않는 것(원칙):
    - forecast_results.parquet을 재작성하지 않는다(읽기 전용).
    - prediction 값을 삭제·재계산하지 않는다.
    - outlier 제거, winsorizing, clipping을 추가하지 않는다.
    - m/order/enforce_stationarity/enforce_invertibility 등 fit 옵션을 건드리지 않는다.
    - 02_run_arima_holdout.py/06_evaluate_statistical_models.py/07_diagnose_arimax_
      economic_instability.py/statistical_utils.py(기존 ARIMA/ARIMAX 트랙 파일)를 참조·
      import하지 않는다.

탐지 규칙(SKU ID/threshold 하드코딩 없음, model in {sarima, sarimax_s4} 각각 독립 적용):
    각 (center_id, sku_id)마다 development 기간(A: 2021-01-04~2023-12-25 전체,
    B: 2023-07-03~2023-12-25 Post-regime만) 실측 qty의 최댓값 dev_max_qty를 구한다
    (0이면 1로 floor). threshold_used = max(dev_max_qty * 100, 10) — "development 관측
    최댓값의 100배를 넘는 예측은 정상적 수요 변동으로 설명할 수 없다"는 독립적 도메인 기준
    (하한 10은 dev_max_qty가 매우 작은/0인 SKU가 사소한 변동까지 걸리는 것을 막기 위한 여유값).
    해당 model의 행 중 하나라도 (a) prediction_unclipped==inf, 또는 (b) abs(prediction_
    unclipped) > threshold_used 를 만족하면 그 (model, SKU)를 numerical_instability로
    분류한다. SARIMA와 SARIMAX(S4)는 서로 다른 fit이므로 한쪽만 flag될 수 있다(동일 SKU가
    두 model 모두에서 독립적으로 flag되는 것도 가능).

override granularity 규칙 (model마다 독립 적용, 이전과 동일 원칙):
    - (a)에 해당하는 (model, SKU) -> override_scope='sku_level': 그 model의 해당 SKU
      155행 전체를 naive_mean으로 override 대상 지정.
    - (a) 없이 (b)만 해당 -> override_scope='row_level': threshold_used를 넘는 행만
      override 대상 지정.
    naive_mean override 값(development 기간 qty 평균)은 model과 무관하게 SKU 단위로
    동일하게 계산된다(fallback_forecast_sku()와 동일 정의).

sarima_status_override.csv에는 어느 model의 행을 override하는지 구분하기 위해 model
컬럼을 포함한다 — SARIMA와 SARIMAX(S4)는 forecast_results.parquet에서 동일한
(center_id, sku_id, forecast_origin, horizon) key를 공유하므로, 10_evaluate_sarima.py가
override를 적용할 때 key만으로 join하면 안 되고 model까지 함께 매칭해야 한다.
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
DIAGNOSIS_OUT_PATH = RESULT_DIR / "sarima_instability_diagnosis.csv"
OVERRIDE_OUT_PATH = RESULT_DIR / "sarima_status_override.csv"

THRESHOLD_MULTIPLIER = 100  # dev_max_qty의 몇 배까지를 "정상적 수요 변동"으로 볼지 (독립적 도메인 기준)
THRESHOLD_FLOOR = 10        # dev_max_qty가 매우 작거나 0(->1로 floor)인 SKU를 위한 최소 여유값
DEV_MAX_QTY_FLOOR = 1       # development 전 구간 무판매(dev_max_qty==0)인 SKU의 floor 값

MODEL_SOURCES = {"sarima": "SARIMA", "sarimax_s4": "SARIMAX_S4"}

# 이번 세션에서 수동으로 확인한 2개 SKU(sarima 기준, 검증용 diff 대조 기준일 뿐)
MANUALLY_CONFIRMED_SARIMA_SKUS = {
    ("A", "18801007235582*BX*1"): 34.337078651685395,
    ("A", "8803322002439*EA*1"): 7.051282051282051,
}

ORDER_COLS = ["selected_p", "selected_d", "selected_q",
              "selected_P", "selected_D", "selected_Q", "selected_m"]

CAUSE_NOTE = (
    "enforce_stationarity=False/enforce_invertibility=False 하에서 near-unit-root 계수가 "
    "refit=False 재귀 append를 통해 발산한 것으로 확인됨(2026-08-22 세션 sarima에서 재현 "
    "검증 완료, 결정론적 재현). 탐지 기준: prediction_unclipped가 Inf이거나 절댓값이 "
    f"threshold_used(=max(dev_max_qty*{THRESHOLD_MULTIPLIER}, {THRESHOLD_FLOOR})) 초과인 "
    "행이 있는 (model, SKU)를 numerical_instability로 판정. SARIMA와 SARIMAX(S4)에 각각 "
    "독립적으로 적용됨(model 컬럼 참조). override 단위: Inf가 하나라도 있으면 해당 model의 "
    "SKU 전체(sku_level), Inf 없이 threshold 초과 값만 있으면 해당 행만(row_level)."
)


def _dev_max_qty() -> pd.DataFrame:
    """SKU별 development 기간(A 전체, B Post-regime만) 실측 qty 최댓값. 0이면 1로 floor.
    model과 무관 — SKU 단위로 1회만 계산."""
    dev_df = skm.load_dev_frame()
    g = dev_df.groupby([skm.CENTER_COL, skm.SKU_COL], observed=True)[skm.QTY_COL].max().reset_index()
    g = g.rename(columns={skm.QTY_COL: "dev_max_qty"})
    g["dev_max_qty"] = g["dev_max_qty"].clip(lower=DEV_MAX_QTY_FLOOR)
    g["threshold_used"] = np.maximum(g["dev_max_qty"] * THRESHOLD_MULTIPLIER, THRESHOLD_FLOOR)
    return g


def _sku_stats(model_df: pd.DataFrame) -> pd.DataFrame:
    """단일 model(source)의 SKU 단위 통계: has_inf, n_total, n_pathological,
    pct_pathological, override_scope, dev_max_qty, threshold_used, n_exceed_threshold."""
    g = model_df.copy()
    g["is_inf"] = np.isinf(g["prediction_unclipped"])
    g["is_over_threshold_not_inf"] = (~g["is_inf"]) & (g["prediction_unclipped"].abs() > g["threshold_used"])
    g["is_pathological"] = g["is_inf"] | g["is_over_threshold_not_inf"]

    stats = g.groupby([skm.CENTER_COL, skm.SKU_COL]).agg(
        n_total=("is_pathological", "size"),
        n_inf=("is_inf", "sum"),
        n_exceed_threshold=("is_over_threshold_not_inf", "sum"),
        n_pathological=("is_pathological", "sum"),
        dev_max_qty=("dev_max_qty", "first"),
        threshold_used=("threshold_used", "first"),
    ).reset_index()
    stats["has_inf"] = stats["n_inf"] > 0
    stats["pct_pathological"] = (stats["n_pathological"] / stats["n_total"] * 100).round(1)
    stats["override_scope"] = np.where(stats["has_inf"], "sku_level", "row_level")
    return stats[stats["n_pathological"] > 0].reset_index(drop=True)


def compute_naive_mean_level(dev_sku_df: pd.DataFrame) -> float:
    """fallback_forecast_sku()와 동일 정의: development 이력이 아예 없으면 0.0, 전부 동일값이면
    그 값(constant), 아니면 평균(naive_mean). model과 무관 — SKU 단위 값."""
    if len(dev_sku_df) == 0:
        return 0.0
    y_raw = dev_sku_df[skm.QTY_COL].to_numpy(dtype=float)
    if np.all(y_raw == y_raw[0]):
        return float(y_raw[0])
    return float(y_raw.mean())


def process_model(source: str, model_label: str, df: pd.DataFrame, dev_max_df: pd.DataFrame,
                   dev_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """단일 model(source)에 대해 탐지+override 생성. (diag_rows, override_rows, report_info) 반환."""
    model_df = df[df["forecast_source"] == source].copy()
    model_df = model_df.merge(dev_max_df, on=[skm.CENTER_COL, skm.SKU_COL], how="left")
    n_no_dev_max = int(model_df["dev_max_qty"].isna().sum())
    if n_no_dev_max:
        print(f"  [경고][{model_label}] dev_max_qty를 못 찾은 행 {n_no_dev_max}건")

    stats = _sku_stats(model_df)
    flagged_keys = set(zip(stats[skm.CENTER_COL], stats[skm.SKU_COL]))
    scope_map = {(r[skm.CENTER_COL], r[skm.SKU_COL]): r["override_scope"] for _, r in stats.iterrows()}
    pct_map = {(r[skm.CENTER_COL], r[skm.SKU_COL]): r["pct_pathological"] for _, r in stats.iterrows()}
    has_inf_map = {(r[skm.CENTER_COL], r[skm.SKU_COL]): bool(r["has_inf"]) for _, r in stats.iterrows()}
    dev_max_map = {(r[skm.CENTER_COL], r[skm.SKU_COL]): r["dev_max_qty"] for _, r in stats.iterrows()}
    thresh_map = {(r[skm.CENTER_COL], r[skm.SKU_COL]): r["threshold_used"] for _, r in stats.iterrows()}
    n_exceed_map = {(r[skm.CENTER_COL], r[skm.SKU_COL]): int(r["n_exceed_threshold"]) for _, r in stats.iterrows()}

    print(f"[탐지][{model_label}] numerical_instability SKU 수: {len(flagged_keys)}")
    for c, s in sorted(flagged_keys):
        print(f"  - {c}/{s}  scope={scope_map[(c, s)]}  dev_max_qty={dev_max_map[(c, s)]:.1f}  "
              f"threshold_used={thresh_map[(c, s)]:.1f}  초과행={n_exceed_map[(c, s)]}  "
              f"병리적비율={pct_map[(c, s)]}%")

    if not flagged_keys:
        empty_diag = pd.DataFrame(columns=[skm.CENTER_COL, skm.SKU_COL, "forecast_origin", "horizon",
                                            *ORDER_COLS, "with_intercept", "development_n_obs",
                                            "dev_max_qty", "threshold_used", "n_exceed_threshold",
                                            "prediction_unclipped", "prediction", "is_inf",
                                            "is_neg_one_clipped", "override_scope",
                                            "sku_pct_pathological", "sku_has_inf", "model"])
        empty_override = pd.DataFrame(columns=[skm.CENTER_COL, skm.SKU_COL, "forecast_origin", "horizon",
                                                "model", "original_forecast_source", "sarima_status",
                                                "override_scope", "override_forecast_source",
                                                "override_reason", "override_prediction_unclipped",
                                                "override_prediction"])
        return empty_diag, empty_override, {"flagged_keys": flagged_keys, "scope_map": scope_map,
                                             "naive_mean_map": {}}

    key_mask = model_df.apply(lambda r: (r[skm.CENTER_COL], r[skm.SKU_COL]) in flagged_keys, axis=1)
    flagged_rows = model_df.loc[key_mask].copy()
    flagged_rows["is_inf"] = np.isinf(flagged_rows["prediction_unclipped"])
    flagged_rows["is_over_threshold_not_inf"] = (~flagged_rows["is_inf"]) & (
        flagged_rows["prediction_unclipped"].abs() > flagged_rows["threshold_used"]
    )

    dev_n_obs_map: dict[tuple[str, str], int] = {}
    naive_mean_map: dict[tuple[str, str], float] = {}
    for c, s in flagged_keys:
        dev_sku = dev_df[(dev_df[skm.CENTER_COL] == c) & (dev_df[skm.SKU_COL] == s)]
        dev_n_obs_map[(c, s)] = len(dev_sku)
        naive_mean_map[(c, s)] = compute_naive_mean_level(dev_sku)

    diag = flagged_rows[[skm.CENTER_COL, skm.SKU_COL, "forecast_origin", "horizon", *ORDER_COLS,
                          "with_intercept", "prediction_unclipped", "prediction", "is_inf"]].copy()
    diag["is_neg_one_clipped"] = np.isclose(diag["prediction_unclipped"], -1.0)
    diag["development_n_obs"] = diag.apply(lambda r: dev_n_obs_map[(r[skm.CENTER_COL], r[skm.SKU_COL])], axis=1)
    diag["dev_max_qty"] = diag.apply(lambda r: dev_max_map[(r[skm.CENTER_COL], r[skm.SKU_COL])], axis=1)
    diag["threshold_used"] = diag.apply(lambda r: thresh_map[(r[skm.CENTER_COL], r[skm.SKU_COL])], axis=1)
    diag["n_exceed_threshold"] = diag.apply(lambda r: n_exceed_map[(r[skm.CENTER_COL], r[skm.SKU_COL])], axis=1)
    diag["override_scope"] = diag.apply(lambda r: scope_map[(r[skm.CENTER_COL], r[skm.SKU_COL])], axis=1)
    diag["sku_pct_pathological"] = diag.apply(lambda r: pct_map[(r[skm.CENTER_COL], r[skm.SKU_COL])], axis=1)
    diag["sku_has_inf"] = diag.apply(lambda r: has_inf_map[(r[skm.CENTER_COL], r[skm.SKU_COL])], axis=1)
    diag["model"] = model_label
    diag = diag[[skm.CENTER_COL, skm.SKU_COL, "forecast_origin", "horizon", *ORDER_COLS,
                 "with_intercept", "development_n_obs", "dev_max_qty", "threshold_used",
                 "n_exceed_threshold", "prediction_unclipped", "prediction", "is_inf",
                 "is_neg_one_clipped", "override_scope", "sku_pct_pathological", "sku_has_inf", "model"]]

    sku_level_keys = {k for k, v in scope_map.items() if v == "sku_level"}
    row_level_keys = {k for k, v in scope_map.items() if v == "row_level"}
    is_sku_level_row = flagged_rows.apply(lambda r: (r[skm.CENTER_COL], r[skm.SKU_COL]) in sku_level_keys, axis=1)
    is_row_level_hit = flagged_rows.apply(
        lambda r: (r[skm.CENTER_COL], r[skm.SKU_COL]) in row_level_keys, axis=1
    ) & flagged_rows["is_over_threshold_not_inf"]

    override_rows = flagged_rows.loc[is_sku_level_row | is_row_level_hit].copy()
    override = override_rows[[skm.CENTER_COL, skm.SKU_COL, "forecast_origin", "horizon"]].copy()
    override["model"] = model_label
    override["original_forecast_source"] = source
    override["sarima_status"] = "numerical_instability"
    override["override_forecast_source"] = "naive_mean"
    override["override_scope"] = override.apply(lambda r: scope_map[(r[skm.CENTER_COL], r[skm.SKU_COL])], axis=1)
    override["override_reason"] = override.apply(
        lambda r: (
            f"development fit 자체는 수렴했으나 2024 rolling append 중 near-unit-root 계수가 "
            f"재귀적으로 발산해 {model_label} SKU 전체(155행)에서 literal Inf가 관측됨(자동탐지 "
            "조건 a) — 구조 재탐색이 아니라 기존 실패-fallback 체계(naive_mean) 적용"
            if r["override_scope"] == "sku_level" else
            f"특정 origin/horizon에서만 {model_label} append 상태가 일시적으로 발산해 development "
            f"실측 최댓값({dev_max_map[(r[skm.CENTER_COL], r[skm.SKU_COL])]:.1f})의 "
            f"{THRESHOLD_MULTIPLIER}배(threshold_used="
            f"{thresh_map[(r[skm.CENTER_COL], r[skm.SKU_COL])]:.1f})를 초과함(자동탐지 조건 b) — "
            "해당 행만 naive_mean으로 override, 나머지 행은 원래 예측 유지"
        ), axis=1
    )
    override["override_prediction_unclipped"] = override.apply(
        lambda r: naive_mean_map[(r[skm.CENTER_COL], r[skm.SKU_COL])], axis=1
    )
    override["override_prediction"] = override["override_prediction_unclipped"].clip(lower=0.0)
    override = override[[skm.CENTER_COL, skm.SKU_COL, "forecast_origin", "horizon", "model",
                          "original_forecast_source", "sarima_status", "override_scope",
                          "override_forecast_source", "override_reason",
                          "override_prediction_unclipped", "override_prediction"]]

    report_info = {"flagged_keys": flagged_keys, "scope_map": scope_map, "naive_mean_map": naive_mean_map}
    return diag, override, report_info


def main() -> None:
    df = pd.read_parquet(FORECAST_RESULTS_PATH)  # 읽기 전용, 이 스크립트는 이 파일을 재작성하지 않음
    dev_max_df = _dev_max_qty()
    dev_df = skm.load_dev_frame()

    all_diag = []
    all_override = []
    reports = {}
    for source, label in MODEL_SOURCES.items():
        diag, override, info = process_model(source, label, df, dev_max_df, dev_df)
        all_diag.append(diag)
        all_override.append(override)
        reports[label] = info

    diag_all = pd.concat(all_diag, ignore_index=True).sort_values(["model", skm.SKU_COL, "forecast_origin", "horizon"]).reset_index(drop=True)
    override_all = pd.concat(all_override, ignore_index=True).sort_values(["model", skm.SKU_COL, "forecast_origin", "horizon"]).reset_index(drop=True)

    note_row = {c: "" for c in diag_all.columns}
    note_row[skm.CENTER_COL] = "NOTE"
    note_row[skm.SKU_COL] = CAUSE_NOTE
    diag_out = pd.concat([diag_all, pd.DataFrame([note_row])], ignore_index=True)
    diag_out.to_csv(DIAGNOSIS_OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[저장] {DIAGNOSIS_OUT_PATH}  (데이터 {len(diag_all)}행 + NOTE 1행 = 총 {len(diag_out)}행)")

    override_all.to_csv(OVERRIDE_OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[저장] {OVERRIDE_OUT_PATH}  ({len(override_all)}행)")

    print("=" * 90)
    for label in MODEL_SOURCES.values():
        sub = override_all[override_all["model"] == label]
        sku_level = sub[sub["override_scope"] == "sku_level"]
        row_level = sub[sub["override_scope"] == "row_level"]
        n_sku_level_skus = sku_level[[skm.CENTER_COL, skm.SKU_COL]].drop_duplicates().shape[0]
        n_row_level_skus = row_level[[skm.CENTER_COL, skm.SKU_COL]].drop_duplicates().shape[0]
        print(f"[{label}] sku_level: {n_sku_level_skus}개 SKU / {len(sku_level)}행, "
              f"row_level: {n_row_level_skus}개 SKU / {len(row_level)}행, "
              f"합계 {len(sub)}행")

    print()
    print(f"[총합] override 파일 총 행수: {len(override_all)}행")

    print("=" * 90)
    print("[검증] 수동 확인 SKU(sarima 기준) vs 자동 탐지 diff")
    sarima_flagged = reports["SARIMA"]["flagged_keys"]
    sarima_scope = reports["SARIMA"]["scope_map"]
    sarima_naive = reports["SARIMA"]["naive_mean_map"]
    manual_keys = set(MANUALLY_CONFIRMED_SARIMA_SKUS.keys())
    missing = manual_keys - sarima_flagged
    if missing:
        print(f"  -> [불일치] 자동탐지가 놓친 SKU: {sorted(missing)}")
    else:
        print("  -> 수동 확인 2개 SKU 전부 SARIMA 자동탐지에 포함됨:")
        for k in sorted(manual_keys):
            print(f"     {k}: scope={sarima_scope.get(k)}")
    print()
    print("[검증] override_prediction(naive_mean) 값 대조 (수동 vs 자동, sarima 기준)")
    for key, manual_val in MANUALLY_CONFIRMED_SARIMA_SKUS.items():
        auto_val = sarima_naive.get(key)
        match = np.isclose(manual_val, auto_val, rtol=1e-9, atol=1e-9) if auto_val is not None else False
        print(f"  {key}: 수동={manual_val}  자동={auto_val}  일치={match}")


if __name__ == "__main__":
    main()
