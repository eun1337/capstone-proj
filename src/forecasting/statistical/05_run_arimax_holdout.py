"""
05_run_arimax_holdout.py

2024 Final Holdout용 Frozen ARIMAX S1~S4 예측을 생성한다.

01에서 확정한 동일한 (p,d,q)+with_intercept를 유지한 채,
각 exog block의 AR/MA 및 외생변수 계수를 Development에서 1회 추정한다.
2024에는 새 y와 해당 시점 historical exog를 append(refit=False)하여 state만 갱신하며,
parameter를 재추정하지 않는다.
"""

from pathlib import Path
import importlib
import os
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd
from pmdarima import ARIMA

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from common import (  # noqa: E402
    CENTER_COL, WEEK_COL, QTY_COL,
    candidate_convergence, expm1_clip, coldstart_fallback, lookup_actual,
)

SKU_COL = "sku_id"

m02 = importlib.import_module("02_run_arima_holdout")
m04 = importlib.import_module("04_prepare_arimax_full_exog")

CENTERS = m02.CENTERS
ORIGINS = m02.ORIGINS
HORIZON_WEEKS = m02.HORIZON_WEEKS
HOLDOUT_MAX_WEEK = m02.HOLDOUT_MAX_WEEK
DEV_FIELDS = m02.DEV_FIELDS
DEV_FIELD_DEFAULTS = m02.DEV_FIELD_DEFAULTS

BASE_DIR = Path(__file__).resolve().parents[3]
ORDERS_PATH = m02.ORDERS_PATH
OUT_DIR = m02.OUT_DIR

ARIMAX_BLOCKS = {k: v for k, v in m04.EXOG_BLOCKS.items() if k != "S0"}  # S1~S4만
BLOCK_ORDER = ["S1", "S2", "S3", "S4"]

HOLDOUT_OUT_PATH = OUT_DIR / "arimax_holdout_2024.parquet"
CHECKPOINT_DIR = OUT_DIR / "arimax_holdout_2024_frozen_checkpoints"  # rolling-refit checkpoint와 별도 경로
CHECKPOINT_EVERY = 20000
PROGRESS_FIRST_MILESTONE = 1000

ALLOWED_SOURCES = {
    "arimax", "constant", "naive_mean",
    "coldstart_subcategory_mean", "coldstart_midcategory_mean",
    "coldstart_category_mean", "coldstart_center_mean",
}
ALLOWED_STATUSES = {
    "success",
    "dev_fit_failure", "dev_fit_nonconverged", "dev_fit_convergence_unknown",
    "state_update_failure", "forecast_failure",
    "no_development_order", "not_applicable",
}
ARIMAX_PATH_STATUSES = {
    "success",
    "dev_fit_failure", "dev_fit_nonconverged", "dev_fit_convergence_unknown",
    "state_update_failure", "forecast_failure",
}

COLUMN_ORDER = [
    "center_id", "sku_id", "forecast_origin", "target_week", "horizon", "exog_block",
    "forecast_source", "arimax_status", "has_observed_history",
    *DEV_FIELDS,
    "current_n_obs", "dev_fit_converged", "state_update_success", "forecast_success",
    "prediction_unclipped", "prediction", "negative_prediction_flag", "actual", "error",
]


def _kept_horizons_for_origin(origin: pd.Timestamp) -> list:
    """target_week<=2024-12-30인 horizon만 평가 대상(연말 right-censoring)."""
    return [h for h, step in HORIZON_WEEKS.items() if origin + pd.Timedelta(weeks=step) <= HOLDOUT_MAX_WEEK]


def _slice_exog(week_arr: np.ndarray, historical_exog_lookup: pd.DataFrame, block_cols: list,
                 end_idx: int, sku_id: str, block: str, start_idx: int = 0) -> np.ndarray:
    """SKU 자체 week와 block exog를 week_st로 정렬한다. 04 재검증(NaN 0건, 센터별 week
    완전 커버)에 따라 leading-NaN 등 예외를 마스킹하지 않고 1:1 정렬만 확인한다."""
    weeks = pd.to_datetime(week_arr[start_idx:end_idx])
    exog_sub = historical_exog_lookup.reindex(weeks)
    assert exog_sub[block_cols].notna().all().all(), \
        f"{sku_id}/{block}: historical exog가 self-history와 week 기준 1:1로 정렬되지 않음(NaN 발생)"
    return exog_sub[block_cols].to_numpy(dtype=float)


def _future_exog_matrix(future_exog_indexed: pd.DataFrame, center_id: str, origin: pd.Timestamp,
                         block_cols: list, n_periods: int) -> np.ndarray:
    rows = []
    for step in range(1, n_periods + 1):
        r = future_exog_indexed.loc[(center_id, origin, step)]
        if not bool(r["exog_available"]):
            raise ValueError(f"n_periods={n_periods}인데 step={step} exog_available=False(right-censoring 계산 불일치)")
        rows.append([float(r[c]) for c in block_cols])
    return np.array(rows, dtype=float)


def _development_final_fit_arimax(history_qty: np.ndarray, history_exog: np.ndarray,
                                   p: int, d: int, q: int, with_intercept: bool) -> dict:
    """Development 전체 구간에 block별 계수를 1회만 추정한다(2024에서는 재추정하지 않음)."""
    out = {"status": "dev_fit_failure", "converged": None, "res": None, "params": None, "error": ""}
    train_log1p = np.log1p(history_qty)
    try:
        model = ARIMA(order=(p, d, q), with_intercept=with_intercept, suppress_warnings=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_log1p, X=history_exog)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    conv = candidate_convergence(model)
    if conv is False:
        out["status"] = "dev_fit_nonconverged"
        out["converged"] = False
        return out
    if conv is not True:  # "Unknown"
        out["status"] = "dev_fit_convergence_unknown"
        out["converged"] = None
        return out

    out["status"] = "success"
    out["converged"] = True
    out["res"] = model.arima_res_
    out["params"] = np.asarray(model.arima_res_.params).copy()
    return out


def _state_append_arimax(state_res, new_qty: np.ndarray, new_exog: np.ndarray, dev_params: np.ndarray) -> dict:
    """새로 확정된 y+historical exog를 state에만 반영한다(exog=, refit=False -> parameter
    재추정 없음). 재추정 없이 그대로 상속되므로 exact equality로 검증한다."""
    out = {"res": None, "success": False, "error": ""}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_res = state_res.append(np.log1p(new_qty), exog=new_exog, refit=False)
        if not np.array_equal(np.asarray(new_res.params), dev_params):
            raise AssertionError("append(refit=False) 후 parameter가 Development fit과 달라짐(재추정 의심)")
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["res"] = new_res
    out["success"] = True
    return out


def _state_forecast_arimax(state_res, future_exog: np.ndarray, n_periods: int) -> dict:
    out = {"forecast": None, "success": False, "error": ""}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = np.asarray(state_res.get_forecast(steps=n_periods, exog=future_exog).predicted_mean)
        if not np.isfinite(fc).all():
            raise ValueError("forecast에 NaN/inf 포함")
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["forecast"] = fc
    out["success"] = True
    return out


def _get_or_init_sku_state_arimax(sku_states: dict, sku_id: str, block: str, dev_row: dict,
                                   week_arr: np.ndarray, qty_arr: np.ndarray, block_cols: list,
                                   historical_exog_lookup: pd.DataFrame) -> dict:
    """SKU를 처음 만났을 때만 Development final fit을 수행한다. sku_states는 block 하나가
    끝나면 통째로 버려지는 dict라 key는 sku_id만으로 충분하다(block마다 별도 dict)."""
    if sku_id in sku_states:
        return sku_states[sku_id]

    p, d, q = int(dev_row["selected_p"]), int(dev_row["selected_d"]), int(dev_row["selected_q"])
    with_intercept = bool(dev_row["with_intercept"])
    n_obs_dev = int(np.searchsorted(week_arr, np.datetime64(m02.DEVELOPMENT_END_WEEK, "D"), side="right"))
    assert n_obs_dev == int(dev_row["development_n_obs"]), (
        f"{sku_id}/{block}: Development fit에 사용된 n_obs({n_obs_dev})가 "
        f"orders 파일의 development_n_obs({dev_row['development_n_obs']})와 불일치"
    )
    history_qty_dev = qty_arr[:n_obs_dev]
    history_exog_dev = _slice_exog(week_arr, historical_exog_lookup, block_cols, n_obs_dev, sku_id, block)

    fit = _development_final_fit_arimax(history_qty_dev, history_exog_dev, p, d, q, with_intercept)
    state = {
        "dev_status": fit["status"], "dev_converged": fit["converged"], "dev_error": fit["error"],
        "res": fit["res"], "dev_params": fit["params"],
        "last_n_obs": n_obs_dev, "broken": fit["status"] != "success",
        "broken_status": None, "broken_reason": fit["error"],
    }
    sku_states[sku_id] = state
    return state


def _build_arimax_rows_for_unit(center_id: str, sku_id: str, origin: pd.Timestamp,
                                 week_arr: np.ndarray, qty_arr: np.ndarray, dev_row: dict | None,
                                 coldstart_means: dict, purchase_cat_map: pd.DataFrame,
                                 first_sale: pd.Timestamp | None, existed_before_regime: bool,
                                 exog_block: str, block_cols: list,
                                 historical_exog_lookup: pd.DataFrame, future_exog_indexed: pd.DataFrame,
                                 kept_horizons: list, sku_states: dict) -> list[dict]:
    n_obs = int(np.searchsorted(week_arr, np.datetime64(origin, "D"), side="right"))
    history_qty = qty_arr[:n_obs]

    has_observed_history = (first_sale is not None and first_sale <= origin) or bool(existed_before_regime)
    has_dev_order = dev_row is not None and dev_row["development_status"] == "selected"

    forecast_array = None
    flat_value = None
    dev_fit_converged = None
    state_update_success = None
    forecast_success = None
    error = ""

    if not has_observed_history:
        sku_cat = m02._resolve_coldstart_category(sku_id, origin, purchase_cat_map)
        if sku_cat is not None:
            value, level = coldstart_fallback(sku_cat, coldstart_means)
        else:
            center_val = coldstart_means["center"]
            value, level = (float(center_val) if pd.notna(center_val) else 0.0), "center"
        flat_value = max(value, 0.0)
        forecast_source = f"coldstart_{level}_mean"
        arimax_status = "not_applicable"

    elif has_dev_order:
        state = _get_or_init_sku_state_arimax(sku_states, sku_id, exog_block, dev_row,
                                               week_arr, qty_arr, block_cols, historical_exog_lookup)
        dev_fit_converged = state["dev_converged"]

        if state["dev_status"] != "success":
            arimax_status = state["dev_status"]
            forecast_source = "naive_mean"
            flat_value = float(np.mean(history_qty)) if len(history_qty) else 0.0
            error = state["dev_error"]
        else:
            if not state["broken"] and n_obs > state["last_n_obs"]:
                new_qty = qty_arr[state["last_n_obs"]:n_obs]
                new_exog = _slice_exog(week_arr, historical_exog_lookup, block_cols, n_obs,
                                        sku_id, exog_block, start_idx=state["last_n_obs"])
                upd = _state_append_arimax(state["res"], new_qty, new_exog, state["dev_params"])
                state_update_success = upd["success"]
                if upd["success"]:
                    state["res"] = upd["res"]
                    state["last_n_obs"] = n_obs
                else:
                    state["broken"] = True
                    state["broken_status"] = "state_update_failure"
                    state["broken_reason"] = upd["error"]
            elif not state["broken"]:
                state_update_success = True  # 새 관측치 없음(no-op) — 실패 아님

            if state["broken"]:
                # broken_status는 최초 실패(state_update 또는 forecast) 원인을 그대로
                # 보존한다 — 이후 origin에서 원인이 뒤바뀌지 않는다.
                arimax_status = state["broken_status"]
                forecast_source = "naive_mean"
                flat_value = float(np.mean(history_qty)) if len(history_qty) else 0.0
                error = state["broken_reason"]
            else:
                n_periods = max(HORIZON_WEEKS[h] for h in kept_horizons)
                future_exog = _future_exog_matrix(future_exog_indexed, center_id, origin, block_cols, n_periods)
                fc = _state_forecast_arimax(state["res"], future_exog, n_periods)
                forecast_success = fc["success"]
                if fc["success"]:
                    forecast_array = fc["forecast"]
                    arimax_status = "success"
                    forecast_source = "arimax"
                else:
                    state["broken"] = True
                    state["broken_status"] = "forecast_failure"
                    state["broken_reason"] = fc["error"]
                    arimax_status = "forecast_failure"
                    forecast_source = "naive_mean"
                    flat_value = float(np.mean(history_qty)) if len(history_qty) else 0.0
                    error = fc["error"]

    else:
        # existed_before_regime=True인 B센터 SKU가 레짐 이후 아직 행이 하나도 없으면
        # n_obs==0인 채로 여기 들어올 수 있다 — np.mean([])/history_qty[0] 금지.
        if n_obs == 0:
            forecast_source = "constant"
            flat_value = 0.0
        else:
            is_constant_now = bool(len(np.unique(history_qty)) == 1)
            if is_constant_now:
                forecast_source = "constant"
                flat_value = float(history_qty[0])
            else:
                forecast_source = "naive_mean"
                flat_value = float(np.mean(history_qty))
        arimax_status = "no_development_order"

    dev_fields = m02._dev_fields(dev_row)

    rows = []
    for h_label in kept_horizons:
        step = HORIZON_WEEKS[h_label]
        target_week = origin + pd.Timedelta(weeks=step)
        actual = lookup_actual(week_arr, qty_arr, target_week)

        if forecast_array is not None:
            pred_unclipped, pred_final = expm1_clip(forecast_array[step - 1])
        else:
            pred_unclipped = flat_value
            pred_final = max(flat_value, 0.0)

        rows.append({
            "center_id": center_id, "sku_id": sku_id,
            "forecast_origin": origin, "target_week": target_week, "horizon": h_label,
            "exog_block": exog_block,
            "forecast_source": forecast_source, "arimax_status": arimax_status,
            "has_observed_history": has_observed_history,
            **dev_fields,
            "current_n_obs": n_obs,
            "dev_fit_converged": dev_fit_converged,
            "state_update_success": state_update_success,
            "forecast_success": forecast_success,
            "prediction_unclipped": pred_unclipped, "prediction": pred_final,
            "negative_prediction_flag": bool(pred_unclipped < 0),
            "actual": actual, "error": error,
        })
    return rows


def _load_done_keys(checkpoint_dir: Path) -> tuple:
    if not checkpoint_dir.exists():
        return set(), 0
    chunk_paths = sorted(checkpoint_dir.glob("chunk_*.parquet"))
    done_keys = set()
    for p in chunk_paths:
        keys_df = pd.read_parquet(p, columns=[CENTER_COL, SKU_COL, "forecast_origin", "exog_block"])
        done_keys.update(
            (c, s, pd.Timestamp(o), b)
            for c, s, o, b in zip(keys_df[CENTER_COL], keys_df[SKU_COL], keys_df["forecast_origin"], keys_df["exog_block"])
        )
    return done_keys, len(chunk_paths)


def _save_chunk(rows: list, checkpoint_dir: Path, chunk_idx: int) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = m02._chunk_path(checkpoint_dir, chunk_idx)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows)[COLUMN_ORDER].to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def _print_arimax_progress(processed_this_run: int, arimax_path_count: int, already_done: int,
                            total_expected: int, elapsed_sec: float) -> None:
    done_total = already_done + processed_this_run
    avg_sec = elapsed_sec / processed_this_run if processed_this_run else float("nan")
    remaining_units = max(total_expected - done_total, 0)
    eta_sec = avg_sec * remaining_units
    total_est_sec = avg_sec * total_expected
    ratio = done_total / total_expected if total_expected else float("nan")

    print(f"[progress] {done_total:,}/{total_expected:,}" + (f" ({ratio:.1%})" if total_expected else ""))
    print(f"  ARIMAX(Frozen) state 처리 건수(이번 실행): {arimax_path_count:,}")
    print(f"  경과시간(이번 실행): {m02._format_timedelta(elapsed_sec)}")
    print(f"  1건당 평균 처리시간: {avg_sec:.4f}초")
    print(f"  예상 전체 소요시간: {m02._format_timedelta(total_est_sec)}")
    print(f"  예상 남은시간: {m02._format_timedelta(eta_sec)}")


def build_holdout(df: pd.DataFrame, orders_df: pd.DataFrame, holdout_raw: pd.DataFrame,
                   historical_exog: pd.DataFrame, future_exog: pd.DataFrame,
                   checkpoint_dir: Path = CHECKPOINT_DIR, checkpoint_every: int = CHECKPOINT_EVERY) -> pd.DataFrame:
    dev_lookup = {(r[CENTER_COL], r[SKU_COL]): r for r in orders_df.to_dict("records")}
    skus_by_target_week = m02._build_skus_by_target_week(holdout_raw)
    future_exog_indexed = future_exog.set_index([CENTER_COL, "forecast_origin", "step"]).sort_index()

    total_expected_units = m02._count_total_units(skus_by_target_week) * len(BLOCK_ORDER)
    done_keys, next_chunk_idx = _load_done_keys(checkpoint_dir)
    already_done = len(done_keys)
    print(f"[사전 계산] 전체 예정 처리 건수(center,sku,forecast_origin,exog_block): {total_expected_units:,}건 "
          f"(기존 checkpoint 완료: {already_done:,}건)")

    current_chunk_rows: list = []
    units_in_chunk = 0
    processed_this_run = 0
    arimax_path_count = 0
    run_start_time = time.perf_counter()

    for center_id in CENTERS:
        center_df = df[df[CENTER_COL] == center_id]
        sku_arrays = m02._build_sku_arrays(center_df)
        first_sale_weeks = m02._first_sale_weeks(center_df)
        existed_before_regime_flags = m02._existed_before_regime_flags(center_df)
        veteran_skus = {sku for sku, flag in existed_before_regime_flags.items() if flag}
        purchase_cat_map = m02._load_purchase_category_map(center_id)
        hist_exog_lookup = (
            historical_exog[historical_exog[CENTER_COL] == center_id]
            .set_index(WEEK_COL)[m04.ECONOMIC_COLS + m04.COVID_COLS + m04.HOLIDAY_COLS]
        )

        # origin에만 의존하고 block에는 의존하지 않는 값은 block loop 밖에서 1회만 계산해 재사용
        kept_horizons_by_origin = {}
        coldstart_means_by_origin = {}
        population_by_origin = {}
        for origin in ORIGINS:
            kept_horizons_by_origin[origin] = _kept_horizons_for_origin(origin)

            sold_by_origin = {sku for sku, fsw in first_sale_weeks.items() if fsw <= origin}
            donor_skus = sold_by_origin | veteran_skus
            donor_df = center_df[center_df[SKU_COL].isin(donor_skus)]
            coldstart_means_by_origin[origin] = m02.compute_coldstart_means(donor_df, origin)

            target_weeks = [origin + pd.Timedelta(weeks=w) for w in HORIZON_WEEKS.values()]
            population = set()
            for tw in target_weeks:
                population |= skus_by_target_week.get((center_id, tw), set())
            population_by_origin[origin] = sorted(population)

        for block in BLOCK_ORDER:
            block_cols = ARIMAX_BLOCKS[block]
            sku_states: dict = {}  # 이 block 안에서만 유지되는 SKU당 frozen state — block이
            # 끝나면 재할당으로 자연스럽게 GC돼 peak 메모리가 block 1개분으로 줄어든다.

            for origin in ORIGINS:  # append state가 이어지도록 반드시 시간순
                kept_horizons = kept_horizons_by_origin[origin]
                coldstart_means = coldstart_means_by_origin[origin]
                population = population_by_origin[origin]

                for sku_id in population:
                    week_arr, qty_arr = sku_arrays[sku_id]
                    dev_row = dev_lookup.get((center_id, sku_id))
                    has_dev_order = dev_row is not None and dev_row["development_status"] == "selected"
                    first_sale = first_sale_weeks.get(sku_id)
                    existed_before_regime = existed_before_regime_flags.get(sku_id, False)

                    key = (center_id, sku_id, origin, block)
                    already = key in done_keys

                    if already and not has_dev_order:
                        continue  # state 없는 branch(cold-start/no_development_order)는 완료됐으면 skip

                    new_rows = _build_arimax_rows_for_unit(
                        center_id, sku_id, origin, week_arr, qty_arr, dev_row,
                        coldstart_means, purchase_cat_map, first_sale, existed_before_regime,
                        block, block_cols, hist_exog_lookup, future_exog_indexed, kept_horizons,
                        sku_states,
                    )

                    if already:
                        # has_dev_order라 state는 방금 갱신했지만(재구성 목적), 출력은 이미
                        # 저장돼 있으므로 이번 결과는 버린다.
                        continue

                    current_chunk_rows.extend(new_rows)
                    if new_rows and new_rows[0]["arimax_status"] in ARIMAX_PATH_STATUSES:
                        arimax_path_count += 1
                    units_in_chunk += 1
                    processed_this_run += 1

                    if processed_this_run == PROGRESS_FIRST_MILESTONE:
                        _print_arimax_progress(processed_this_run, arimax_path_count, already_done,
                                                total_expected_units, time.perf_counter() - run_start_time)

                    if units_in_chunk >= checkpoint_every:
                        _save_chunk(current_chunk_rows, checkpoint_dir, next_chunk_idx)
                        next_chunk_idx += 1
                        current_chunk_rows = []
                        units_in_chunk = 0
                        _print_arimax_progress(processed_this_run, arimax_path_count, already_done,
                                                total_expected_units, time.perf_counter() - run_start_time)

    if units_in_chunk > 0:
        _save_chunk(current_chunk_rows, checkpoint_dir, next_chunk_idx)
        current_chunk_rows = []

    return (
        m02._load_all_chunks(checkpoint_dir)[COLUMN_ORDER]
        .sort_values([CENTER_COL, SKU_COL, "forecast_origin", "exog_block", "horizon"])
        .reset_index(drop=True)
    )


def run_audit(holdout_df: pd.DataFrame, orders_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """02_run_arima_holdout.run_audit()와 동일한 항목은 그대로, block/right-censoring
    관련 항목만 추가한 PASS/FAIL 리포트(assert로 중단하지 않음)."""
    rows = []

    def _row(check, passed, detail):
        rows.append({"check": check, "passed": bool(passed), "detail": detail})

    _row("no_auto_arima_import", "auto_arima" not in globals(), "")

    dup = int(holdout_df.duplicated(subset=[CENTER_COL, SKU_COL, "forecast_origin", "horizon", "exog_block"]).sum())
    _row("no_duplicate_unit_horizon", dup == 0, f"중복 {dup}건")

    bad_horizon = set(holdout_df["horizon"].unique()) - set(HORIZON_WEEKS)
    _row("horizon_domain", not bad_horizon, f"허용되지 않은 horizon: {bad_horizon}")

    bad_block = set(holdout_df["exog_block"].unique()) - set(BLOCK_ORDER)
    _row("exog_block_domain", not bad_block, f"허용되지 않은 exog_block: {bad_block}")

    # domain 검사만으로는 block 전체 누락(예: 특정 unit에서 S3가 통째로 비는 경우)을 못 잡는다
    block_sets = holdout_df.groupby([CENTER_COL, SKU_COL, "forecast_origin"], observed=True)["exog_block"].apply(lambda s: frozenset(s))
    n_incomplete = int((block_sets != frozenset(BLOCK_ORDER)).sum())
    _row("exog_block_set_complete", n_incomplete == 0, f"exog_block 집합이 {{S1,S2,S3,S4}}가 아닌 unit {n_incomplete}건")

    bad_source = set(holdout_df["forecast_source"].unique()) - ALLOWED_SOURCES
    _row("forecast_source_domain", not bad_source, f"허용되지 않은 forecast_source: {bad_source}")
    bad_status = set(holdout_df["arimax_status"].unique()) - ALLOWED_STATUSES
    _row("arimax_status_domain", not bad_status, f"허용되지 않은 arimax_status: {bad_status}")

    expected_target = holdout_df["forecast_origin"] + holdout_df["horizon"].map(HORIZON_WEEKS).apply(lambda w: pd.Timedelta(weeks=w))
    bad_target = int((holdout_df["target_week"] != expected_target).sum())
    _row("target_week_formula", bad_target == 0, f"불일치 {bad_target}건")

    # 연말 right-censoring: 실제 생성된 horizon 집합이 _kept_horizons_for_origin과 일치하는지
    units = holdout_df.drop_duplicates([CENTER_COL, SKU_COL, "forecast_origin", "exog_block"])[
        [CENTER_COL, SKU_COL, "forecast_origin", "exog_block"]
    ]
    actual_horizons = holdout_df.groupby([CENTER_COL, SKU_COL, "forecast_origin", "exog_block"])["horizon"].apply(lambda s: frozenset(s))
    expected_horizons = units["forecast_origin"].drop_duplicates().apply(lambda o: frozenset(_kept_horizons_for_origin(o)))
    origin_to_expected = dict(zip(units["forecast_origin"].drop_duplicates(), expected_horizons))
    mismatch = sum(
        1 for key, hs in actual_horizons.items() if hs != origin_to_expected[key[2]]
    )
    _row("right_censoring_horizon_set", mismatch == 0, f"kept_horizons 불일치 {mismatch}건")

    should_be_target_within_range = int((holdout_df["target_week"] > HOLDOUT_MAX_WEEK).sum())
    _row("no_row_beyond_2024", should_be_target_within_range == 0, f"2024-12-30 초과 target_week 행 {should_be_target_within_range}건")

    # Frozen 고유: dev_fit_converged는 (sku,block)당 Development final fit 1회로만 정해지므로
    # 같은 (center,sku,block)의 모든 행에서 값이 하나여야 한다(중간에 값이 바뀌면 재추정 의심).
    path_rows = holdout_df[holdout_df["arimax_status"].isin(ARIMAX_PATH_STATUSES)]
    conv_nunique = path_rows.groupby([CENTER_COL, SKU_COL, "exog_block"])["dev_fit_converged"].nunique(dropna=False)
    _row("dev_fit_converged_constant_per_sku_block", bool((conv_nunique <= 1).all()),
         f"(center,sku,exog_block) 내에서 dev_fit_converged 값이 바뀐 경우 {int((conv_nunique > 1).sum())}건")

    arimax_rows = holdout_df[holdout_df["forecast_source"] == "arimax"]
    ok_status = (arimax_rows["arimax_status"] == "success").all() if len(arimax_rows) else True
    ok_conv = (arimax_rows["dev_fit_converged"] == True).all() if len(arimax_rows) else True  # noqa: E712
    ok_upd = (arimax_rows["state_update_success"] == True).all() if len(arimax_rows) else True  # noqa: E712
    ok_fc = (arimax_rows["forecast_success"] == True).all() if len(arimax_rows) else True  # noqa: E712
    _row("arimax_success_state_consistency",
     bool(ok_status) and bool(ok_conv) and bool(ok_upd) and bool(ok_fc),
     "forecast_source=arimax인데 arimax_status/dev_fit_converged/state_update_success/forecast_success 중 불일치")

    pred = holdout_df["prediction"].to_numpy(dtype=float)
    pred_unclipped = holdout_df["prediction_unclipped"].to_numpy(dtype=float)
    _row("prediction_finite", bool(np.isfinite(pred).all()) and bool(np.isfinite(pred_unclipped).all()), "")
    _row("prediction_nonnegative", bool((pred >= 0).all()), "")
    _row("negative_flag_consistency", bool((holdout_df["negative_prediction_flag"] == (holdout_df["prediction_unclipped"] < 0)).all()), "")

    cold_mask = holdout_df["forecast_source"].str.startswith("coldstart_")
    _row("coldstart_matches_has_observed_history",
         bool((holdout_df.loc[cold_mask, "has_observed_history"] == False).all())  # noqa: E712
         and bool((holdout_df.loc[~cold_mask, "has_observed_history"] == True).all()),  # noqa: E712
         "")

    b_regime_flags = df.loc[df[CENTER_COL] == "B", [SKU_COL, m02.EXISTED_BEFORE_REGIME_COL]].drop_duplicates(SKU_COL)
    veteran_b_skus = set(b_regime_flags.loc[b_regime_flags[m02.EXISTED_BEFORE_REGIME_COL], SKU_COL])
    b_rows = holdout_df[holdout_df[CENTER_COL] == "B"]
    veteran_violation = b_rows[b_rows[SKU_COL].isin(veteran_b_skus) & (b_rows["has_observed_history"] == False)]  # noqa: E712
    _row("b_existed_before_regime_has_history", veteran_violation.empty, f"위반 {len(veteran_violation)}건")

    holdout_units = holdout_df.drop_duplicates([CENTER_COL, SKU_COL])[[CENTER_COL, SKU_COL] + DEV_FIELDS]
    dev_merged = holdout_units.merge(
        orders_df[[CENTER_COL, SKU_COL] + DEV_FIELDS], on=[CENTER_COL, SKU_COL],
        how="left", suffixes=("", "_orig"), indicator=True,
    )
    present = dev_merged[dev_merged["_merge"] == "both"]
    dev_mismatch = 0
    for col in DEV_FIELDS:
        left, right = present[col], present[f"{col}_orig"]
        both_nan = left.isna() & right.isna()
        dev_mismatch += int((~both_nan & (left != right)).sum())
    _row("development_order_unchanged", dev_mismatch == 0, f"불일치 {dev_mismatch}건")

    absent = dev_merged[dev_merged["_merge"] == "left_only"]
    default_violations = 0
    for col in DEV_FIELDS:
        default = DEV_FIELD_DEFAULTS[col]
        val = absent[col]
        if default is None or (isinstance(default, float) and np.isnan(default)):
            bad = absent[val.notna()]
        else:
            bad = absent[val != default]
        default_violations += len(bad)
    _row("development_default_for_new_sku", default_violations == 0, f"orders에 없는 SKU의 기본값 미유지 {default_violations}건")

    actual_ref = df.rename(columns={WEEK_COL: "target_week", QTY_COL: "actual_ref"})[[CENTER_COL, SKU_COL, "target_week", "actual_ref"]]
    checked = holdout_df.merge(actual_ref, on=[CENTER_COL, SKU_COL, "target_week"], how="left")
    mismatch_actual = checked[checked["actual"].notna() & (checked["actual"] != checked["actual_ref"])]
    should_be_nan = checked[checked["actual"].notna() & checked["actual_ref"].isna()]
    should_be_filled = checked[checked["actual"].isna() & checked["actual_ref"].notna()]
    _row(
        "actual_integrity",
        mismatch_actual.empty and should_be_nan.empty and should_be_filled.empty,
        f"불일치 {len(mismatch_actual)}건, 원본 없는데 채워짐 {len(should_be_nan)}건, "
        f"원본 있는데 비어있음 {len(should_be_filled)}건",
    )

    units_n_obs = holdout_df.drop_duplicates([CENTER_COL, SKU_COL, "forecast_origin"])[
        [CENTER_COL, SKU_COL, "forecast_origin", "current_n_obs"]
    ].sort_values("forecast_origin").reset_index(drop=True)
    df_sorted = df.sort_values(WEEK_COL).reset_index(drop=True)
    df_sorted[CENTER_COL] = df_sorted[CENTER_COL].astype(str)  # merge_asof by= dtype(category)를 holdout_df와 맞춤
    df_sorted["n_obs_asof"] = df_sorted.groupby([CENTER_COL, SKU_COL], observed=True).cumcount() + 1
    checked_n_obs = pd.merge_asof(
        units_n_obs, df_sorted[[CENTER_COL, SKU_COL, WEEK_COL, "n_obs_asof"]],
        left_on="forecast_origin", right_on=WEEK_COL, by=[CENTER_COL, SKU_COL], direction="backward",
    )
    checked_n_obs["n_obs_asof"] = checked_n_obs["n_obs_asof"].fillna(0).astype(int)
    n_obs_mismatch = checked_n_obs[checked_n_obs["n_obs_asof"] != checked_n_obs["current_n_obs"]]
    _row("current_n_obs_integrity", n_obs_mismatch.empty, f"불일치 {len(n_obs_mismatch)}건")

    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = m02.load_combined_history()
    orders_df = pd.read_parquet(ORDERS_PATH)
    holdout_raw = pd.read_parquet(m02.HOLDOUT_PATH, columns=[CENTER_COL, SKU_COL, WEEK_COL])
    historical_exog = pd.read_parquet(m04.HISTORICAL_EXOG_PATH)
    future_exog = pd.read_parquet(m04.FUTURE_EXOG_PATH)

    holdout_df = build_holdout(df, orders_df, holdout_raw, historical_exog, future_exog)

    audit = run_audit(holdout_df, orders_df, df)
    n_fail = int((~audit["passed"]).sum())
    if n_fail:
        raise AssertionError(f"audit 실패 {n_fail}건:\n{audit[~audit['passed']].to_string(index=False)}")

    holdout_df.to_parquet(HOLDOUT_OUT_PATH, index=False)
    if CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)

    print(f"[전체 예측행] {len(holdout_df):,}")
    print(f"[Audit] 총 {len(audit)}건 중 실패 {n_fail}건")
    print(f"저장 완료: {HOLDOUT_OUT_PATH}")


if __name__ == "__main__":
    main()
