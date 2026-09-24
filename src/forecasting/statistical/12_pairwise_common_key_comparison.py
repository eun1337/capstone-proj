"""
12_pairwise_common_key_comparison.py

통계모델 7종(ARIMA, ARIMAX-S1~S4, SARIMA, SARIMAX-S4) 중 외생변수 pair(ARIMA vs
ARIMAX-S1/S2/S3/S4, SARIMA vs SARIMAX-S4)를 각각 "두 모델 모두 예측이 존재하는 공통
key(center_id, sku_id, forecast_origin, horizon)"로 inner join해 재계산한다.
ARIMA vs SARIMA/SARIMAX-S4 페어는 11_compare_arima_vs_sarima.py가 이미 담당하므로
이 스크립트는 재계산하지 않고, 그 결과 파일(arima_vs_sarima_compare_metrics.csv)을
그대로 읽어 최종 통합 파일에만 옮겨 적는다(중복 계산·값 불일치 방지).

전체 7개 모델을 한꺼번에 교집합하지 않는다 — pair마다 그 pair에 필요한 두 모델의
공통 key만 사용해 모집단이 불필요하게 줄어드는 것을 막는다(요청사항).

Main 모집단 정의(06_evaluate_statistical_models.py/10_evaluate_sarima.py/
11_compare_arima_vs_sarima.py와 동일한 기존 관례를 그대로 따름):
  - 두 모델 모두 해당 key에 예측이 존재(inner join)
  - 두 모델 모두 actual이 not null (target_week가 2024 범위를 벗어나는 origin 등 제외)
  - 두 모델 모두 has_observed_history=True (cold-start 보조 결과는 여기서 제외하고
    build_family_coverage()가 별도 파일로 집계 — 카드4에서 fallback/coverage로 노출)
  - actual 값이 두 모델 파일 간 다르면(같은 key인데 ground truth가 다르면) 비교하지
    않고 audit에만 기록(11의 원칙과 동일)

metric 정의는 src.forecasting.common.evaluator(compute_metrics/build_mase_scale)를
그대로 사용한다 — WAPE/Bias는 %(기존 outputs/model_comparison/stat_model_comparison_
2024.csv와 동일 스케일), RMSE/MAE는 판매수량 단위, MASE는 SKU별 development
lag-1 naive scale(A 전체 + B post-regime).

이 스크립트는 새 파일만 만든다 — 기존 outputs/model_comparison/stat_model_comparison_
2024.csv, stat_common_metrics_2024.csv 등은 전혀 읽거나 수정하지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.common import evaluator as ev
from src.forecasting.common.config import B_HISTORY_START

BASE_DIR = Path(__file__).resolve().parents[3]
DAY2_DIR = BASE_DIR / "data" / "ml" / "day2_statistical_models"
DEV_PATH = BASE_DIR / "data" / "development_2021_2023.parquet"

ARIMA_PATH = DAY2_DIR / "arima_holdout_2024.parquet"
ARIMAX_PATH = DAY2_DIR / "arimax_holdout_2024.parquet"
SARIMA_PANEL_PATH = DAY2_DIR / "sarima_sarimax_sku" / "sarima_eval_common_panel.parquet"
ARIMA_VS_SARIMA_METRICS_PATH = DAY2_DIR / "arima_vs_sarima_compare_metrics.csv"

OUT_DIR = BASE_DIR / "outputs" / "model_comparison"
METRICS_OUT = OUT_DIR / "stat_pairwise_common_key_metrics_2024.csv"
AUDIT_OUT = OUT_DIR / "stat_pairwise_common_key_audit_2024.csv"
COVERAGE_OUT = OUT_DIR / "stat_family_coverage_2024.csv"

KEY = ["center_id", "sku_id", "forecast_origin", "horizon"]
UNIT = ["center_id", "sku_id"]
HORIZONS = [1, 2, 4]
CENTERS = ["A", "B", "ALL"]
A_HISTORY_START = pd.Timestamp("2021-01-04")
DEV_MAX_WEEK = pd.Timestamp("2023-12-25")
HORIZON_STR_MAP = {"h1": 1, "h2": 2, "h4": 4}

COLDSTART_SOURCES = {
    "coldstart_subcategory_mean", "coldstart_midcategory_mean",
    "coldstart_category_mean", "coldstart_center_mean",
}
NORMAL_SOURCES = {"arima", "arimax", "sarima", "sarimax_s4"}


def _dev_mase_history() -> pd.DataFrame:
    dev = pd.read_parquet(DEV_PATH, columns=["center_id", "sku_id", "week_st", "qty"])
    is_a = (dev.center_id == "A") & (dev.week_st >= A_HISTORY_START) & (dev.week_st <= DEV_MAX_WEEK)
    is_b = (dev.center_id == "B") & (dev.week_st >= B_HISTORY_START) & (dev.week_st <= DEV_MAX_WEEK)
    return dev.loc[is_a | is_b]


def _load_arima() -> pd.DataFrame:
    df = pd.read_parquet(ARIMA_PATH, columns=KEY + ["prediction", "actual", "has_observed_history"])
    df = df.rename(columns={"prediction": "pred", "has_observed_history": "hist"})
    df["horizon"] = df["horizon"].map(HORIZON_STR_MAP).astype(int)
    return df


def _load_arimax(block: str) -> pd.DataFrame:
    df = pd.read_parquet(
        ARIMAX_PATH, columns=KEY + ["exog_block", "prediction", "actual", "has_observed_history"],
    )
    df = df[df["exog_block"] == block].drop(columns=["exog_block"])
    df = df.rename(columns={"prediction": "pred", "has_observed_history": "hist"})
    df["horizon"] = df["horizon"].map(HORIZON_STR_MAP).astype(int)
    return df


def _load_sarima_family(model_label: str) -> pd.DataFrame:
    df = pd.read_parquet(
        SARIMA_PANEL_PATH,
        columns=["model", *KEY, "prediction_final", "actual", "has_observed_history"],
    )
    df = df[df["model"] == model_label].drop(columns=["model"])
    return df.rename(columns={"prediction_final": "pred", "has_observed_history": "hist"})


LOADERS = {
    "ARIMA": _load_arima,
    "ARIMAX_S1": lambda: _load_arimax("S1"),
    "ARIMAX_S2": lambda: _load_arimax("S2"),
    "ARIMAX_S3": lambda: _load_arimax("S3"),
    "ARIMAX_S4": lambda: _load_arimax("S4"),
    "SARIMA": lambda: _load_sarima_family("SARIMA"),
    "SARIMAX_S4": lambda: _load_sarima_family("SARIMAX_S4"),
}

PAIRS = [
    ("ARIMA", "ARIMAX_S1"),
    ("ARIMA", "ARIMAX_S2"),
    ("ARIMA", "ARIMAX_S3"),
    ("ARIMA", "ARIMAX_S4"),
    ("SARIMA", "SARIMAX_S4"),
]


def compute_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dev_hist = _dev_mase_history()
    needed = sorted({name for pair in PAIRS for name in pair})
    loaded = {name: LOADERS[name]() for name in needed}

    metric_rows: list[dict] = []
    audit_rows: list[dict] = []

    for left, right in PAIRS:
        pair_id = f"{left}_vs_{right}"
        L = loaded[left].rename(columns={"pred": "pred_l", "actual": "actual_l", "hist": "hist_l"})
        R = loaded[right].rename(columns={"pred": "pred_r", "actual": "actual_r", "hist": "hist_r"})

        n_l_total, n_r_total = len(L), len(R)
        merged = L.merge(R, on=KEY, how="inner", validate="one_to_one")
        n_common_key = len(merged)

        both_actual = merged["actual_l"].notna() & merged["actual_r"].notna()
        mismatch = both_actual & ~np.isclose(
            merged["actual_l"].where(both_actual, 0.0), merged["actual_r"].where(both_actual, 0.0),
            rtol=1e-9, atol=1e-9,
        )
        n_mismatch = int(mismatch.sum())

        main_mask = both_actual & ~mismatch & merged["hist_l"] & merged["hist_r"]
        main_df = merged.loc[main_mask].copy()
        main_df["actual"] = main_df["actual_l"]

        audit_rows.append({
            "pair": pair_id, "left": left, "right": right,
            "n_left_total": n_l_total, "n_right_total": n_r_total,
            "n_common_key_inner_join": n_common_key,
            "n_left_only_excluded": n_l_total - n_common_key,
            "n_right_only_excluded": n_r_total - n_common_key,
            "n_actual_both_present": int(both_actual.sum()),
            "n_actual_mismatch": n_mismatch,
            "n_main_both_has_history": int(main_mask.sum()),
            "n_excluded_coldstart_either_side": int((both_actual & ~mismatch & ~(merged["hist_l"] & merged["hist_r"])).sum()),
        })

        for horizon in HORIZONS:
            h_df = main_df[main_df["horizon"] == horizon]
            for center in CENTERS:
                sub = h_df if center == "ALL" else h_df[h_df["center_id"] == center]
                if sub.empty:
                    continue
                mase_scale = ev.build_mase_scale(dev_hist, sub[UNIT])
                n_sku = int(sub[UNIT].drop_duplicates().shape[0])
                for model_name, pred_col in [(left, "pred_l"), (right, "pred_r")]:
                    m = ev.compute_metrics(sub["actual"].to_numpy(), sub[pred_col].to_numpy(), mase_scale)
                    metric_rows.append({
                        "pair": pair_id, "model": model_name,
                        "compared_against": right if model_name == left else left,
                        "center_id": center, "horizon": horizon,
                        "n_rows": m["n"], "n_sku": n_sku,
                        "WAPE": m["wape"], "Bias": m["bias"], "RMSE": m["rmse"], "MAE": m["mae"],
                        "MASE": m["mase"], "mase_n": m["mase_n"],
                    })

    return pd.DataFrame(metric_rows), pd.DataFrame(audit_rows)


def _merge_arima_vs_sarima_results(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """11_compare_arima_vs_sarima.py가 이미 계산한 ARIMA vs SARIMA/SARIMAX_S4 결과를
    동일 스키마(WAPE/Bias %, n_rows, n_sku)로 변환해 합친다. 11의 WAPE/Bias는 fraction
    (예: 1.0118)이라 이 파일의 %(예: 101.18) 관례에 맞춰 100을 곱한다 — 값 자체를
    재계산하지 않고 표시 스케일만 맞춘다."""
    if not ARIMA_VS_SARIMA_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"{ARIMA_VS_SARIMA_METRICS_PATH} 없음 — 11_compare_arima_vs_sarima.py를 먼저 실행하세요."
        )
    df = pd.read_csv(ARIMA_VS_SARIMA_METRICS_PATH)
    df = df.rename(columns={"n": "n_rows"})
    df["pair"] = df["compared_against"].map(lambda m: f"ARIMA_vs_{m}")
    df["WAPE"] = df["WAPE"] * 100
    df["Bias"] = df["Bias"] * 100
    return df[["pair", "model", "compared_against", "center_id", "horizon", "n_rows", "n_sku",
               "WAPE", "Bias", "RMSE", "MAE", "MASE", "n_mase"]]


def _source_category(source: pd.Series) -> pd.Series:
    return np.select(
        [source.isin(NORMAL_SOURCES), source.eq("constant"), source.eq("naive_mean"), source.isin(COLDSTART_SOURCES)],
        ["normal", "constant", "naive_mean", "coldstart"],
        default="other",
    )


def build_family_coverage() -> pd.DataFrame:
    """각 모델의 own 전체 모집단(pair로 좁히지 않은 population) 기준 정상/fallback/
    cold-start/override 건수. 06/10 스크립트가 이미 계산한 coverage CSV를 재사용하지
    않고, 이 스크립트가 로드한 원본 holdout parquet에서 직접 집계해 pairwise 결과와
    완전히 동일한 소스·동일한 has_observed_history 정의를 쓴다(다른 스크립트의 별도
    필터와 섞이지 않도록)."""
    rows = []

    arima = pd.read_parquet(ARIMA_PATH, columns=["forecast_source", "arima_status", "actual", "has_observed_history"])
    arima = arima.rename(columns={"arima_status": "status"})
    arimax = pd.read_parquet(ARIMAX_PATH, columns=["exog_block", "forecast_source", "arimax_status", "actual", "has_observed_history"])
    sarima_panel = pd.read_parquet(
        SARIMA_PANEL_PATH,
        columns=["model", "forecast_source", "forecast_source_final", "override_applied", "actual", "has_observed_history"],
    )

    def _summarize(name: str, df: pd.DataFrame, source_col: str, status: pd.Series | None, override: pd.Series | None) -> dict:
        cat = pd.Series(_source_category(df[source_col]), index=df.index)
        n_total = len(df)
        row = {
            "model": name,
            "n_total": n_total,
            "n_actual_present": int(df["actual"].notna().sum()),
            "n_has_observed_history_true": int(df["has_observed_history"].sum()),
            "n_has_observed_history_false": int((~df["has_observed_history"]).sum()),
            "n_normal": int((cat == "normal").sum()),
            "n_fallback_constant": int((cat == "constant").sum()),
            "n_fallback_naive_mean": int((cat == "naive_mean").sum()),
            "n_fallback_coldstart": int((cat == "coldstart").sum()),
            "normal_rate": (cat == "normal").sum() / n_total if n_total else None,
            "fallback_rate": (cat != "normal").sum() / n_total if n_total else None,
        }
        if status is not None:
            for value, count in status.value_counts().items():
                row[f"status_{value}"] = int(count)
        if override is not None:
            row["n_override_instability"] = int(override.sum())
        return row

    rows.append(_summarize("ARIMA", arima, "forecast_source", arima["status"], None))
    for blk in ["S1", "S2", "S3", "S4"]:
        sub = arimax[arimax["exog_block"] == blk]
        rows.append(_summarize(f"ARIMAX_{blk}", sub, "forecast_source", sub["arimax_status"], None))
    for model_label, out_name in [("SARIMA", "SARIMA"), ("SARIMAX_S4", "SARIMAX_S4")]:
        sub = sarima_panel[sarima_panel["model"] == model_label]
        rows.append(_summarize(out_name, sub, "forecast_source_final", None, sub["override_applied"]))

    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_new, audit_new = compute_pairs()
    metrics_arima_sarima = _merge_arima_vs_sarima_results(metrics_new)
    metrics_all = pd.concat([metrics_arima_sarima, metrics_new], ignore_index=True)
    metrics_all.to_csv(METRICS_OUT, index=False, encoding="utf-8-sig")
    audit_new.to_csv(AUDIT_OUT, index=False, encoding="utf-8-sig")

    coverage = build_family_coverage()
    coverage.to_csv(COVERAGE_OUT, index=False, encoding="utf-8-sig")

    print(f"[저장] {METRICS_OUT} ({len(metrics_all)}행)")
    print(f"[저장] {AUDIT_OUT} ({len(audit_new)}행)")
    print(f"[저장] {COVERAGE_OUT} ({len(coverage)}행)")

    print()
    print("[요약] ALL center, WAPE(%)")
    for pair in metrics_all["pair"].unique():
        sub = metrics_all[(metrics_all["pair"] == pair) & (metrics_all["center_id"] == "ALL")]
        for h in HORIZONS:
            hs = sub[sub["horizon"] == h]
            if hs.empty:
                continue
            vals = " | ".join(f"{r.model}={r.WAPE:.2f}" for r in hs.itertuples())
            print(f"  {pair} h{h}: {vals}")


if __name__ == "__main__":
    main()
