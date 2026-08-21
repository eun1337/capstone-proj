"""
06_evaluate_statistical_models.py

2024 Holdout에서 S0~S4 통계모델의 최종 성능을 평가한다.

(center_id, sku_id, forecast_origin, horizon) 공통 key를 기준으로
동일한 평가 모집단을 구성하고, 자기 관측 history가 있는 행을 Main으로 평가한다.
WAPE, Bias, RMSE, MAE, MASE와 forecast source별 성능,
coverage 및 cold-start 보조 결과를 함께 산출한다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
DAY2_DIR = BASE_DIR / "data" / "ml" / "day2_statistical_models"
ARIMA_HOLDOUT_PATH = DAY2_DIR / "arima_holdout_2024.parquet"
ARIMAX_HOLDOUT_PATH = DAY2_DIR / "arimax_holdout_2024.parquet"
DEV_PATH = BASE_DIR / "data" / "development_2021_2023.parquet"
OUT_DIR = DAY2_DIR

CENTER_COL = "center_id"
SKU_COL = "sku_id"
WEEK_COL = "week_st"
QTY_COL = "qty"

KEY_COLS = [CENTER_COL, SKU_COL, "forecast_origin", "horizon"]
PANEL_COLS = KEY_COLS + ["target_week", "forecast_source", "has_observed_history", "prediction", "actual", "model"]
MODELS = ["S0", "S1", "S2", "S3", "S4"]
HORIZONS = ["h1", "h2", "h4"]
HORIZON_WEEKS = {"h1": 1, "h2": 2, "h4": 4}
SOURCE_CATEGORY = {"arima": "normal", "arimax": "normal", "constant": "constant", "naive_mean": "naive_mean"}
COLDSTART_SOURCES = [
    "coldstart_subcategory_mean", "coldstart_midcategory_mean",
    "coldstart_category_mean", "coldstart_center_mean",
]

A_START, A_END = pd.Timestamp("2021-01-04"), pd.Timestamp("2023-12-25")
B_START, B_END = pd.Timestamp("2023-07-03"), pd.Timestamp("2023-12-25")

AUDIT_PATH = OUT_DIR / "statistical_eval_audit.csv"
COVERAGE_PATH = OUT_DIR / "statistical_eval_coverage.csv"
COMMON_PANEL_PATH = OUT_DIR / "statistical_eval_common_panel.parquet"
METRICS_PATH = OUT_DIR / "statistical_eval_metrics_s0_s4.csv"
SOURCE_BREAKDOWN_PATH = OUT_DIR / "statistical_eval_source_breakdown.csv"
COLDSTART_AUX_PATH = OUT_DIR / "statistical_eval_coldstart_aux.csv"
MASE_SCALES_PATH = OUT_DIR / "statistical_eval_mase_scales.parquet"


def _load_model_frame(path: Path, model_col: str = None, model_value: str = None) -> pd.DataFrame:
    """단일 모델 파일(model_value)이든 block 컬럼으로 여러 모델을 겸하는 파일(model_col)이든
    같은 형태로 불러온다 — 향후 SARIMA/SARIMAX 파일을 추가할 때 이 함수만 재사용하면 된다.
    두 원본 parquet은 컬럼이 훨씬 많으므로 평가에 쓰는 컬럼만 읽는다."""
    read_cols = [c for c in PANEL_COLS if c != "model"]
    if model_col:
        read_cols = read_cols + [model_col]
    df = pd.read_parquet(path, columns=read_cols)
    df["model"] = df[model_col] if model_col else model_value
    return df[PANEL_COLS]


def load_panel() -> pd.DataFrame:
    s0 = _load_model_frame(ARIMA_HOLDOUT_PATH, model_value="S0")
    sx = _load_model_frame(ARIMAX_HOLDOUT_PATH, model_col="exog_block")
    return pd.concat([s0, sx], ignore_index=True)


def build_common_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """key당 model이 5개(MODELS 전부) 존재하면 공통 key다 — Python set 5개 대신
    groupby.transform으로 판정해 수백만 key에서도 메모리를 O(panel) 수준으로 유지한다."""
    valid = panel[panel["actual"].notna()].copy()
    model_count = valid.groupby(KEY_COLS)["model"].transform("nunique")
    return valid[model_count == len(MODELS)]


def _scopes(df: pd.DataFrame) -> list:
    return [("A", df[df[CENTER_COL] == "A"]), ("B", df[df[CENTER_COL] == "B"]), ("ALL", df)]


def run_audit(panel: pd.DataFrame, common: pd.DataFrame, main_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def _row(check, passed, detail=""):
        rows.append({"check": check, "passed": bool(passed), "detail": detail})

    bad_models = set(panel["model"].unique()) - set(MODELS)
    missing_models = set(MODELS) - set(panel["model"].unique())
    _row("model_presence", not bad_models and not missing_models, f"미확인 {bad_models}, 누락 {missing_models}")

    for m in MODELS:
        dup = int(panel.loc[panel["model"] == m, KEY_COLS].duplicated().sum())
        _row(f"key_uniqueness_{m}", dup == 0, f"중복 {dup}건")

    horizon_set = set(panel["horizon"].unique())
    _row("horizon_domain", horizon_set == set(HORIZONS), f"실제 horizon 집합: {horizon_set}")

    step = panel["horizon"].map(HORIZON_WEEKS).apply(lambda w: pd.Timedelta(weeks=w))
    bad_target = int((panel["target_week"] != panel["forecast_origin"] + step).sum())
    _row("target_week_formula", bad_target == 0, f"불일치 {bad_target}건")

    pred = panel["prediction"].to_numpy(dtype=float)
    _row("prediction_finite_nonneg", bool(np.isfinite(pred).all()) and bool((pred >= 0).all()), "")

    _row("actual_nonnegative", bool((common["actual"] >= 0).all()), "")  # WAPE/Bias가 sum(actual)을 그대로 분모로 쓰므로 필수 전제

    # model별 key uniqueness/domain은 위에서 이미 확인했으므로 key당 distinct model 수만 보면 된다
    sx = panel[panel["model"] != "S0"]
    block_count = sx.groupby(KEY_COLS, observed=True)["model"].nunique()
    n_incomplete = int((block_count != 4).sum())
    _row("s1_s4_block_completeness", n_incomplete == 0, f"S1~S4 누락 key {n_incomplete}건")

    for col in ["target_week", "actual", "has_observed_history"]:
        n_bad = int((common.groupby(KEY_COLS)[col].nunique(dropna=False) > 1).sum())
        _row(f"common_key_{col}_consistent", n_bad == 0, f"모델 간 값 불일치 {n_bad}건")

    n_not5 = int((common.groupby(KEY_COLS).size() != len(MODELS)).sum())
    _row("common_panel_exactly_5_models", n_not5 == 0, f"5개가 아닌 key {n_not5}건")

    _row("common_panel_nonempty", len(common) > 0, f"행 수 {len(common)}")

    for scope, scope_df in _scopes(main_panel):
        for horizon in HORIZONS:
            counts = scope_df.loc[scope_df["horizon"] == horizon].groupby("model").size().reindex(MODELS, fill_value=0)
            ok = counts.nunique() <= 1 and counts.min() > 0  # 5개 모두 0건도 "동일"이라 별도로 막는다
            _row(f"main_panel_n_equal_{scope}_{horizon}", ok, f"{counts.to_dict()}")

    return pd.DataFrame(rows)


def build_coverage(panel: pd.DataFrame, common: pd.DataFrame) -> pd.DataFrame:
    """common-key intersection에서 S0~S4 중 어느 모델이 얼마나 빠졌는지 scope×horizon×model
    단위로 남긴다 — audit은 PASS/FAIL만 주므로 어디서 얼마나 빠졌는지는 이 파일로 확인한다."""
    candidate = panel[panel["actual"].notna()]
    rows = []
    for (scope, cand_df), (_, comm_df) in zip(_scopes(candidate), _scopes(common)):
        for horizon in HORIZONS:
            for model in MODELS:
                candidate_n = int(((cand_df["horizon"] == horizon) & (cand_df["model"] == model)).sum())
                common_n = int(((comm_df["horizon"] == horizon) & (comm_df["model"] == model)).sum())
                excluded_n = candidate_n - common_n
                rows.append({
                    "scope": scope, "horizon": horizon, "model": model,
                    "candidate_n": candidate_n, "common_n": common_n, "excluded_n": excluded_n,
                    "excluded_rate": excluded_n / candidate_n if candidate_n else np.nan,
                })
    return pd.DataFrame(rows)


def build_mase_scales(dev: pd.DataFrame) -> pd.DataFrame:
    a = dev[(dev[CENTER_COL] == "A") & dev[WEEK_COL].between(A_START, A_END)]
    b = dev[(dev[CENTER_COL] == "B") & dev[WEEK_COL].between(B_START, B_END)]
    sub = pd.concat([a, b], ignore_index=True).sort_values([CENTER_COL, SKU_COL, WEEK_COL])

    grp = sub.groupby([CENTER_COL, SKU_COL])[QTY_COL]
    scale_df = pd.DataFrame({"scale": grp.apply(lambda s: s.diff().abs().mean()), "n_obs": grp.size()}).reset_index()
    scale_df["mase_valid"] = (scale_df["n_obs"] >= 2) & np.isfinite(scale_df["scale"]) & (scale_df["scale"] > 0)
    scale_df["mase_scale"] = np.where(scale_df["mase_valid"], scale_df["scale"], np.nan)
    return scale_df


def compute_metrics(sub: pd.DataFrame) -> dict:
    """sub는 mase_scale이 이미 merge된 panel/common의 부분집합이어야 한다(호출마다 merge하면
    build_metrics_table/source_breakdown/coldstart_aux 합쳐 수백 번 반복 merge가 발생한다)."""
    actual = sub["actual"].to_numpy(dtype=float)
    pred = sub["prediction"].to_numpy(dtype=float)
    err = pred - actual
    n = len(sub)
    denom = np.sum(actual)

    rmse = float(np.sqrt(np.mean(err ** 2))) if n else np.nan
    mae = float(np.mean(np.abs(err))) if n else np.nan
    wape = float(np.sum(np.abs(err)) / denom * 100) if denom > 0 else np.nan
    bias = float(np.sum(err) / denom * 100) if denom > 0 else np.nan

    scale = sub["mase_scale"].to_numpy(dtype=float)
    valid = np.isfinite(scale) & (scale > 0)
    mase = float(np.mean(np.abs(err[valid]) / scale[valid])) if valid.any() else np.nan

    return {
        "n": n, "RMSE": rmse, "MAE": mae, "WAPE": wape, "Bias(%)": bias, "MASE": mase,
        "mase_valid_n": int(valid.sum()), "mase_excluded_n": int(n - valid.sum()),
        "mase_excluded_rate": float((n - valid.sum()) / n) if n else np.nan,
    }


def build_metrics_table(main_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, scope_df in _scopes(main_panel):
        for horizon in HORIZONS:
            for model in MODELS:
                sub = scope_df[(scope_df["horizon"] == horizon) & (scope_df["model"] == model)]
                rows.append({"scope": scope, "horizon": horizon, "model": model, **compute_metrics(sub)})
    return pd.DataFrame(rows)


def build_source_breakdown(main_panel: pd.DataFrame) -> pd.DataFrame:
    main_panel = main_panel.assign(source_category=main_panel["forecast_source"].map(SOURCE_CATEGORY))
    rows = []
    for scope, scope_df in _scopes(main_panel):
        for horizon in HORIZONS:
            for model in MODELS:
                model_scope = scope_df[(scope_df["horizon"] == horizon) & (scope_df["model"] == model)]
                total_n = len(model_scope)
                for cat in ["normal", "constant", "naive_mean"]:
                    sub = model_scope[model_scope["source_category"] == cat]
                    m = compute_metrics(sub)
                    rows.append({
                        "scope": scope, "horizon": horizon, "model": model, "source_category": cat,
                        "ratio": m["n"] / total_n if total_n else np.nan, **m,
                    })
    return pd.DataFrame(rows)


def build_coldstart_aux(common: pd.DataFrame) -> pd.DataFrame:
    cold = common[~common["has_observed_history"]]
    rows = []
    for scope, scope_df in _scopes(cold):
        for horizon in HORIZONS:
            for model in MODELS:
                model_scope = scope_df[(scope_df["horizon"] == horizon) & (scope_df["model"] == model)]
                total_n = len(model_scope)
                for src in COLDSTART_SOURCES:
                    sub = model_scope[model_scope["forecast_source"] == src]
                    m = compute_metrics(sub)
                    rows.append({
                        "scope": scope, "horizon": horizon, "model": model, "forecast_source": src,
                        "ratio": m["n"] / total_n if total_n else np.nan, **m,
                    })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    common = build_common_panel(panel)
    main_panel = common[common["has_observed_history"]]

    audit = run_audit(panel, common, main_panel)
    audit.to_csv(AUDIT_PATH, index=False)
    build_coverage(panel, common).to_csv(COVERAGE_PATH, index=False)
    del panel  # 이후로는 common/main_panel만 필요
    n_fail = int((~audit["passed"]).sum())
    if n_fail:
        raise AssertionError(f"audit 실패 {n_fail}건:\n{audit[~audit['passed']].to_string(index=False)}")

    dev = pd.read_parquet(DEV_PATH, columns=[CENTER_COL, SKU_COL, WEEK_COL, QTY_COL])
    scale_df = build_mase_scales(dev)
    scale_df.to_parquet(MASE_SCALES_PATH, index=False)

    # metric 계산마다 merge하는 대신 여기서 한 번만 merge한다(호출 수백 회분 절약). common_panel
    # 산출물 스키마는 그대로 유지해야 하므로 mase_scale은 별도 프레임에만 붙인다.
    common_scaled = common.merge(scale_df[[CENTER_COL, SKU_COL, "mase_scale"]], on=[CENTER_COL, SKU_COL], how="left")
    main_panel = common_scaled[common_scaled["has_observed_history"]]

    metrics = build_metrics_table(main_panel)
    metrics.to_csv(METRICS_PATH, index=False)

    source_breakdown = build_source_breakdown(main_panel)
    source_breakdown.to_csv(SOURCE_BREAKDOWN_PATH, index=False)

    coldstart_aux = build_coldstart_aux(common_scaled)
    coldstart_aux.to_csv(COLDSTART_AUX_PATH, index=False)

    common.to_parquet(COMMON_PANEL_PATH, index=False)

    n_valid = int(scale_df["mase_valid"].sum())
    n_excluded = int((~scale_df["mase_valid"]).sum())
    print(f"[Audit] 총 {len(audit)}건 중 실패 {n_fail}건")
    print(f"[MASE scale] valid={n_valid:,} excluded={n_excluded:,} rate={n_excluded / len(scale_df):.2%}")
    print(f"[common panel] {len(common):,}행")
    print(f"저장 완료: {AUDIT_PATH}")
    print(f"저장 완료: {COVERAGE_PATH}")
    print(f"저장 완료: {COMMON_PANEL_PATH}")
    print(f"저장 완료: {METRICS_PATH}")
    print(f"저장 완료: {SOURCE_BREAKDOWN_PATH}")
    print(f"저장 완료: {COLDSTART_AUX_PATH}")
    print(f"저장 완료: {MASE_SCALES_PATH}")


if __name__ == "__main__":
    main()
