"""
13_all_family_common_panel.py

통계모델 7종(ARIMA, ARIMAX-S1~S4, SARIMA, SARIMAX-S4) 전체가 동시에 예측을 갖는
"단일 공통 평가 panel"을 만든다. 11_compare_arima_vs_sarima.py /
12_pairwise_common_key_comparison.py의 pair별 common-key 비교는 그대로 두고
건드리지 않는다 — 이 스크립트는 "7개 모델 전부를 한 row 집합에서 동시에 비교"하는
용도로 완전히 별도 산출물만 만든다.

common key: (center_id, sku_id, forecast_origin, horizon) — 7개 모델 전부에 해당
key의 예측이 존재해야 한다(전부 inner join). 그 위에 다음 조건을 동일 적용한다
(06_evaluate_statistical_models.py/10_evaluate_sarima.py/11/12와 동일한 기존 관례):
  - 7개 모델 전부 actual not null
  - 7개 모델 전부 has_observed_history=True (cold-start 제외 — 카드4가 별도로 다룸)
  - 7개 모델의 actual이 서로 다르면(같은 key인데 ground truth 불일치) 비교하지 않고
    audit에만 기록

metric은 src.forecasting.common.evaluator(compute_metrics/build_mase_scale)를 그대로
사용 — WAPE/Bias는 %, RMSE/MAE는 판매수량, MASE는 SKU별 development lag-1 naive scale
(A 전체 + B post-regime), 11/12 산출물과 동일 스케일.
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

OUT_DIR = BASE_DIR / "outputs" / "model_comparison"
METRICS_OUT = OUT_DIR / "stat_all_family_common_panel_metrics_2024.csv"
AUDIT_OUT = OUT_DIR / "stat_all_family_common_panel_audit_2024.csv"

KEY = ["center_id", "sku_id", "forecast_origin", "horizon"]
UNIT = ["center_id", "sku_id"]
HORIZONS = [1, 2, 4]
CENTERS = ["A", "B", "ALL"]
A_HISTORY_START = pd.Timestamp("2021-01-04")
DEV_MAX_WEEK = pd.Timestamp("2023-12-25")
HORIZON_STR_MAP = {"h1": 1, "h2": 2, "h4": 4}

MODEL_ORDER = ["ARIMA", "ARIMAX_S1", "ARIMAX_S2", "ARIMAX_S3", "ARIMAX_S4", "SARIMA", "SARIMAX_S4"]


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
    df = pd.read_parquet(ARIMAX_PATH, columns=KEY + ["exog_block", "prediction", "actual", "has_observed_history"])
    df = df[df["exog_block"] == block].drop(columns=["exog_block"])
    df = df.rename(columns={"prediction": "pred", "has_observed_history": "hist"})
    df["horizon"] = df["horizon"].map(HORIZON_STR_MAP).astype(int)
    return df


def _load_sarima_family(model_label: str) -> pd.DataFrame:
    df = pd.read_parquet(
        SARIMA_PANEL_PATH, columns=["model", *KEY, "prediction_final", "actual", "has_observed_history"],
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


def build_common_panel() -> tuple[pd.DataFrame, dict]:
    frames = {name: LOADERS[name]() for name in MODEL_ORDER}
    audit: dict = {}

    for name, df in frames.items():
        audit[f"n_total_{name}"] = len(df)
        dup = df.duplicated(subset=KEY).sum()
        if dup:
            raise ValueError(f"{name}: key 중복 {dup}건 — inner join 전 정합성 위반")

    key_sets = {name: pd.MultiIndex.from_frame(df[KEY]) for name, df in frames.items()}
    common_index = key_sets[MODEL_ORDER[0]]
    for name in MODEL_ORDER[1:]:
        common_index = common_index.intersection(key_sets[name])
    audit["n_key_intersection_all_7"] = len(common_index)

    common_keys = common_index.to_frame(index=False)[KEY]

    panel = common_keys.copy()
    for name in MODEL_ORDER:
        merged = common_keys.merge(
            frames[name][KEY + ["pred", "actual", "hist"]], on=KEY, how="left", validate="one_to_one",
        )
        assert merged[["pred", "actual", "hist"]].notna().all().all(), f"{name}: 공통 key인데 값이 없는 행 존재(로직 오류)"
        panel[f"pred_{name}"] = merged["pred"].to_numpy()
        panel[f"actual_{name}"] = merged["actual"].to_numpy()
        panel[f"hist_{name}"] = merged["hist"].to_numpy()

    actual_cols = [f"actual_{name}" for name in MODEL_ORDER]
    all_actual_present = panel[actual_cols].notna().all(axis=1)
    audit["n_actual_all_present"] = int(all_actual_present.sum())
    audit["n_actual_missing_any"] = int((~all_actual_present).sum())

    ref = panel[f"actual_{MODEL_ORDER[0]}"].to_numpy()
    mismatch = np.zeros(len(panel), dtype=bool)
    for name in MODEL_ORDER[1:]:
        other = panel[f"actual_{name}"].to_numpy()
        mismatch |= ~np.isclose(ref, other, rtol=1e-9, atol=1e-9, equal_nan=True)
    audit["n_actual_mismatch_across_7_models"] = int((mismatch & all_actual_present).sum())

    hist_cols = [f"hist_{name}" for name in MODEL_ORDER]
    all_hist_true = panel[hist_cols].all(axis=1)
    audit["n_has_observed_history_all_true"] = int(all_hist_true.sum())
    audit["n_has_observed_history_excluded"] = int((~all_hist_true).sum())

    main_mask = all_actual_present & ~mismatch & all_hist_true
    audit["n_main_final"] = int(main_mask.sum())
    main = panel.loc[main_mask].copy()
    main["actual"] = main[f"actual_{MODEL_ORDER[0]}"]
    main["n_sku_total"] = common_keys[UNIT].drop_duplicates().shape[0]

    return main, audit


def compute_metrics(main: pd.DataFrame) -> pd.DataFrame:
    dev_hist = _dev_mase_history()
    rows = []
    for horizon in HORIZONS:
        h_df = main[main["horizon"] == horizon]
        for center in CENTERS:
            sub = h_df if center == "ALL" else h_df[h_df["center_id"] == center]
            if sub.empty:
                continue
            mase_scale = ev.build_mase_scale(dev_hist, sub[UNIT])
            n_sku = int(sub[UNIT].drop_duplicates().shape[0])
            for name in MODEL_ORDER:
                m = ev.compute_metrics(sub["actual"].to_numpy(), sub[f"pred_{name}"].to_numpy(), mase_scale)
                rows.append({
                    "model": name, "center_id": center, "horizon": horizon,
                    "n_rows": m["n"], "n_sku": n_sku,
                    "WAPE": m["wape"], "Bias": m["bias"], "RMSE": m["rmse"], "MAE": m["mae"],
                    "MASE": m["mase"], "mase_n": m["mase_n"],
                })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel, audit = build_common_panel()
    metrics = compute_metrics(panel)

    metrics.to_csv(METRICS_OUT, index=False, encoding="utf-8-sig")
    pd.DataFrame([audit]).T.reset_index().rename(columns={"index": "metric", 0: "value"}).to_csv(
        AUDIT_OUT, index=False, encoding="utf-8-sig",
    )

    print(f"[저장] {METRICS_OUT} ({len(metrics)}행)")
    print(f"[저장] {AUDIT_OUT}")
    print()
    for k, v in audit.items():
        print(f"  {k}: {v:,}" if isinstance(v, (int, np.integer)) else f"  {k}: {v}")
    print()
    print("[요약] ALL center, WAPE(%) / Bias(%)")
    for h in HORIZONS:
        sub = metrics[(metrics["center_id"] == "ALL") & (metrics["horizon"] == h)]
        vals = " | ".join(f"{r.model}={r.WAPE:.2f}/{r.Bias:.2f}" for r in sub.itertuples())
        print(f"  h{h}: {vals}")


if __name__ == "__main__":
    main()
