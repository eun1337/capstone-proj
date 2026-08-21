"""
07_diagnose_arimax_economic_instability.py

Economic 변수를 포함한 S1/S4의 극단 예측 원인을 사후진단한다.

저장된 예측 결과를 변경하지 않고 prediction tail,
Economic exog의 Development 범위 이탈·상관관계,
추정 계수 및 동일 모델 재현 여부를 분석한다.

모델 재선택, 이상치 제거 또는 2024 결과를 이용한 사후 수정은 수행하지 않는다.
"""

from pathlib import Path
import importlib
import sys

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
m02 = importlib.import_module("02_run_arima_holdout")
m04 = importlib.import_module("04_prepare_arimax_full_exog")
m05 = importlib.import_module("05_run_arimax_holdout")

CENTER_COL = m02.CENTER_COL
WEEK_COL = m02.WEEK_COL
SKU_COL = "sku_id"
KEY_COLS = [CENTER_COL, SKU_COL, "forecast_origin", "horizon"]
ECONOMIC_COLS = m04.ECONOMIC_COLS
HORIZONS = list(m02.HORIZON_WEEKS)

DAY2_DIR = m02.OUT_DIR
ARIMA_HOLDOUT_PATH = m02.HOLDOUT_OUT_PATH
ARIMAX_HOLDOUT_PATH = m05.HOLDOUT_OUT_PATH

DIST_PATH = DAY2_DIR / "arimax_economic_prediction_distribution.csv"
EXTREME_ROWS_PATH = DAY2_DIR / "arimax_economic_extreme_rows.csv"
EXOG_DIAG_PATH = DAY2_DIR / "arimax_economic_exog_diagnostics.csv"
REFIT_DIAG_PATH = DAY2_DIR / "arimax_economic_refit_diagnostics.csv"

TOP_N_EXTREME = 200  # 진단용 샘플 크기일 뿐 배제 기준 아님
TOP_N_REFIT = 15  # (center,sku,exog_block)당 1회 재fit — 개별 fit 비용 때문에 상한을 둠


def load_panels() -> pd.DataFrame:
    cols = KEY_COLS + [
        "target_week", "forecast_source", "arimax_status", "exog_block",
        "development_n_obs", "selected_p", "selected_d", "selected_q", "with_intercept",
        "current_n_obs", "dev_fit_converged", "state_update_success", "forecast_success",
        "prediction", "actual",
    ]
    sx = pd.read_parquet(ARIMAX_HOLDOUT_PATH, columns=cols)
    s0 = pd.read_parquet(ARIMA_HOLDOUT_PATH, columns=KEY_COLS + ["prediction"])
    s0 = s0.rename(columns={"prediction": "prediction_S0"})
    return sx, s0


def build_distribution(sx: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for center in ["A", "B"]:
        for horizon in HORIZONS:
            for block in ["S1", "S2", "S3", "S4"]:
                pred = sx.loc[
                    (sx[CENTER_COL] == center) & (sx["horizon"] == horizon) & (sx["exog_block"] == block),
                    "prediction",
                ].to_numpy(dtype=float)
                if len(pred) == 0:
                    continue
                rows.append({
                    "center_id": center, "horizon": horizon, "exog_block": block, "n": len(pred),
                    "max": pred.max(), "median": np.median(pred),
                    "p99": np.percentile(pred, 99), "p99.9": np.percentile(pred, 99.9),
                    "p99.99": np.percentile(pred, 99.99),
                })
    return pd.DataFrame(rows)


def build_extreme_rows(sx: pd.DataFrame, s0: pd.DataFrame) -> pd.DataFrame:
    econ = sx[sx["exog_block"].isin(["S1", "S4"])]
    top = econ.sort_values("prediction", ascending=False).head(TOP_N_EXTREME).copy()

    # 같은 key에서 S0/S2/S3 prediction과 비교(폭주가 exog block에 국한되는지 확인)
    others = sx[sx["exog_block"].isin(["S2", "S3"])].pivot_table(
        index=KEY_COLS, columns="exog_block", values="prediction", aggfunc="first"
    ).rename(columns={"S2": "prediction_S2", "S3": "prediction_S3"})
    top = top.merge(others, on=KEY_COLS, how="left").merge(s0, on=KEY_COLS, how="left")
    return top


def _dev_slice(week_arr: np.ndarray, n_obs_dev: int) -> pd.DatetimeIndex:
    return pd.to_datetime(week_arr[:n_obs_dev])


def build_exog_diagnostics(top: pd.DataFrame, df: pd.DataFrame, historical_exog: pd.DataFrame,
                            future_exog_indexed: pd.DataFrame) -> pd.DataFrame:
    """key당: (a) future exog가 그 SKU의 Development 관측 범위 안인지/표준화 거리,
    (b) Development exog 3종의 분산·상관·rank·condition number(저분산/공선성 진단)."""
    rows = []
    cache = {}
    for _, r in top.drop_duplicates(KEY_COLS).iterrows():
        center, sku, origin, horizon = r[CENTER_COL], r[SKU_COL], r["forecast_origin"], r["horizon"]
        if center not in cache:
            sku_arrays = m02._build_sku_arrays(df[df[CENTER_COL] == center])
            hist_lookup = historical_exog[historical_exog[CENTER_COL] == center].set_index(WEEK_COL)[ECONOMIC_COLS]
            cache[center] = (sku_arrays, hist_lookup)
        sku_arrays, hist_lookup = cache[center]
        if sku not in sku_arrays:
            continue
        week_arr, _ = sku_arrays[sku]
        n_obs_dev = int(r["development_n_obs"])
        dev_exog = hist_lookup.reindex(_dev_slice(week_arr, n_obs_dev))

        step = m02.HORIZON_WEEKS[horizon]
        future_row = future_exog_indexed.loc[(center, origin, step)]

        row = {"center_id": center, "sku_id": sku, "forecast_origin": origin, "horizon": horizon,
               "development_n_obs": n_obs_dev}
        for col in ECONOMIC_COLS:
            dmin, dmax = dev_exog[col].min(), dev_exog[col].max()
            dmean, dstd = dev_exog[col].mean(), dev_exog[col].std()
            fval = float(future_row[col])
            row[f"{col}_dev_min"], row[f"{col}_dev_max"] = dmin, dmax
            row[f"{col}_dev_mean"], row[f"{col}_dev_std"] = dmean, dstd
            row[f"{col}_future"] = fval
            row[f"{col}_out_of_dev_range"] = bool(fval < dmin or fval > dmax)
            row[f"{col}_zscore"] = (fval - dmean) / dstd if dstd and np.isfinite(dstd) and dstd > 0 else np.nan

        X = dev_exog[ECONOMIC_COLS].to_numpy(dtype=float)
        valid_mask = np.isfinite(X).all(axis=1)
        Xv = X[valid_mask]
        if len(Xv) >= 2:
            corr = np.corrcoef(Xv, rowvar=False)
            row["corr_ccsi_cpi1"] = corr[0, 1]
            row["corr_ccsi_cpi2"] = corr[0, 2]
            row["corr_cpi1_cpi2"] = corr[1, 2]
            row["matrix_rank"] = int(np.linalg.matrix_rank(Xv))
            row["condition_number"] = float(np.linalg.cond(Xv))
        rows.append(row)
    return pd.DataFrame(rows)


def build_refit_diagnostics(top: pd.DataFrame, df: pd.DataFrame, historical_exog: pd.DataFrame,
                             future_exog_indexed: pd.DataFrame) -> pd.DataFrame:
    """05와 동일한 (p,d,q)+with_intercept, 동일 Development history/exog로 다시 fit하고
    같은 forecast_origin까지 실제 관측치를 이어붙여(append) 같은 극단 예측이 재현되는지 본다."""
    candidates = top.sort_values("prediction", ascending=False).drop_duplicates([CENTER_COL, SKU_COL, "exog_block"])
    candidates = candidates.head(TOP_N_REFIT)

    rows = []
    for _, r in candidates.iterrows():
        center, sku, origin, horizon, block = r[CENTER_COL], r[SKU_COL], r["forecast_origin"], r["horizon"], r["exog_block"]
        block_cols = m05.ARIMAX_BLOCKS[block]
        sku_arrays = m02._build_sku_arrays(df[df[CENTER_COL] == center])
        hist_lookup = historical_exog[historical_exog[CENTER_COL] == center].set_index(WEEK_COL)[
            m04.ECONOMIC_COLS + m04.COVID_COLS + m04.HOLIDAY_COLS
        ]
        week_arr, qty_arr = sku_arrays[sku]

        p, d, q, with_intercept = int(r["selected_p"]), int(r["selected_d"]), int(r["selected_q"]), bool(r["with_intercept"])
        n_obs_dev = int(r["development_n_obs"])
        history_qty_dev = qty_arr[:n_obs_dev]
        history_exog_dev = m05._slice_exog(week_arr, hist_lookup, block_cols, n_obs_dev, sku, block)

        fit = m05._development_final_fit_arimax(history_qty_dev, history_exog_dev, p, d, q, with_intercept)
        row = {
            "center_id": center, "sku_id": sku, "exog_block": block, "forecast_origin": origin, "horizon": horizon,
            "selected_p": p, "selected_d": d, "selected_q": q, "with_intercept": with_intercept,
            "development_n_obs": n_obs_dev, "stored_prediction": float(r["prediction"]),
            "refit_dev_status": fit["status"], "refit_dev_converged": fit["converged"],
        }

        if fit["status"] == "success":
            res = fit["res"]
            param_names = res.model.param_names
            exog_coefs = pd.Series(res.params, index=param_names)[res.model.exog_names]
            exog_pvalues = pd.Series(res.pvalues, index=param_names)[res.model.exog_names]
            for col, coef, pval in zip(block_cols, exog_coefs.to_numpy(), exog_pvalues.to_numpy()):
                if col in ECONOMIC_COLS:
                    row[f"coef_{col}"] = coef
                    row[f"pvalue_{col}"] = pval

            n_obs_target = int(r["current_n_obs"])
            state_res = res
            if n_obs_target > n_obs_dev:
                new_qty = qty_arr[n_obs_dev:n_obs_target]
                new_exog = m05._slice_exog(week_arr, hist_lookup, block_cols, n_obs_target, sku, block, start_idx=n_obs_dev)
                upd = m05._state_append_arimax(state_res, new_qty, new_exog, fit["params"])
                row["reappend_success"] = upd["success"]
                row["reappend_params_exact_equal"] = bool(np.array_equal(np.asarray(upd["res"].params), fit["params"])) if upd["success"] else None
                state_res = upd["res"] if upd["success"] else None

            if state_res is not None:
                n_periods = m02.HORIZON_WEEKS[horizon]
                future_exog = m05._future_exog_matrix(future_exog_indexed, center, origin, block_cols, n_periods)
                fc = m05._state_forecast_arimax(state_res, future_exog, n_periods)
                row["reforecast_success"] = fc["success"]
                if fc["success"]:
                    _, reproduced_pred = m05.expm1_clip(fc["forecast"][n_periods - 1])
                    row["reproduced_prediction"] = reproduced_pred
                    row["reproduced_matches_stored"] = bool(np.isclose(reproduced_pred, row["stored_prediction"], rtol=1e-6))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    sx, s0 = load_panels()
    historical_exog = pd.read_parquet(m04.HISTORICAL_EXOG_PATH)
    future_exog = pd.read_parquet(m04.FUTURE_EXOG_PATH)
    future_exog_indexed = future_exog.set_index([CENTER_COL, "forecast_origin", "step"]).sort_index()

    distribution = build_distribution(sx)
    distribution.to_csv(DIST_PATH, index=False)

    extreme_rows = build_extreme_rows(sx, s0)
    extreme_rows.to_csv(EXTREME_ROWS_PATH, index=False)

    df = m02.load_combined_history()
    exog_diag = build_exog_diagnostics(extreme_rows, df, historical_exog, future_exog_indexed)
    exog_diag.to_csv(EXOG_DIAG_PATH, index=False)

    refit_diag = build_refit_diagnostics(extreme_rows, df, historical_exog, future_exog_indexed)
    refit_diag.to_csv(REFIT_DIAG_PATH, index=False)

    print(f"[분포] {len(distribution)}행 -> {DIST_PATH}")
    print(f"[극단치] {len(extreme_rows)}행 -> {EXTREME_ROWS_PATH}")
    print(f"[exog 진단] {len(exog_diag)}행 -> {EXOG_DIAG_PATH}")
    print(f"[재fit 진단] {len(refit_diag)}행 -> {REFIT_DIAG_PATH}")


if __name__ == "__main__":
    main()
