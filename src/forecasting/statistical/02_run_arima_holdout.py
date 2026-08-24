"""
02_run_arima_holdout.py

2024 Final Holdout용 Frozen ARIMA 예측을 생성한다.

01에서 확정한 SKU별 ARIMA 구조로 Development 전체에서 계수를 1회 추정하고,
2024에는 새 실제 관측값을 append(refit=False)하여 state만 갱신한다.
모델 parameter는 재추정하지 않으며 h1/h2/h4 예측과
constant, naive_mean, cold-start fallback 결과를 함께 저장한다.
"""

from pathlib import Path
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
    WEEK_COL, QTY_COL, CENTER_COL, SUBCAT_COL, MIDCAT_COL, LARGECAT_COL,
    FORECAST_N_PERIODS, FORECAST_STEP_TO_HORIZON,
    candidate_convergence, expm1_clip,
    compute_coldstart_means, coldstart_fallback, lookup_actual,
)

BASE_DIR = Path(__file__).resolve().parents[3]
DEV_PATH = BASE_DIR / "data" / "development_2021_2023.parquet"
HOLDOUT_PATH = BASE_DIR / "data" / "holdout_2024.parquet"
OUT_DIR = BASE_DIR / "data" / "ml" / "day2_statistical_models"

SKU_COL = "sku_id"
CENTERS = ["A", "B"]

A_HISTORY_START = pd.Timestamp("2021-01-04")
B_HISTORY_START = pd.Timestamp("2023-07-03")
HOLDOUT_MAX_WEEK = pd.Timestamp("2024-12-30")  # 2025 이후 데이터 누출 방지용 상한 안전장치
DEVELOPMENT_END_WEEK = pd.Timestamp("2023-12-25")  # A/B 공통(시작 window만 다름)

EXISTED_BEFORE_REGIME_COL = "existed_before_regime"  # SKU당 상수, A센터는 항상 False

# 구매 원천 데이터와 동일한 sku_id 재구성 규칙
# (바코드+"*"+옵션코드+"*"+상품클러스터).
PURCHASE_MASTER_PATH = BASE_DIR / "data" / "final" / "cleaned_purchase_cleaned_for_pred.parquet"
PURCHASE_RAW_CENTER_COL = "센터"
PURCHASE_SKU_PARTS = ["바코드", "옵션코드", "상품클러스터"]
PURCHASE_FIRST_STOCK_COL = "최초입고일"
SKU_SEP = "*"

FORECAST_ORIGIN_START = pd.Timestamp("2023-12-25")
N_ORIGINS = 53  # h1이 2024년 53개 주차 전체를 덮는 origin 수
ORIGINS = [FORECAST_ORIGIN_START + pd.Timedelta(weeks=i) for i in range(N_ORIGINS)]

ORDERS_PATH = OUT_DIR / "arima_development_orders.parquet"
HOLDOUT_OUT_PATH = OUT_DIR / "arima_holdout_2024.parquet"
CHECKPOINT_DIR = OUT_DIR / "arima_holdout_2024_frozen_checkpoints"  # rolling-refit checkpoint와 별도 경로
CHECKPOINT_EVERY = 20000
PROGRESS_FIRST_MILESTONE = 1000

HORIZON_WEEKS = {"h1": 1, "h2": 2, "h4": 4}

ALLOWED_SOURCES = {
    "arima", "constant", "naive_mean",
    "coldstart_subcategory_mean", "coldstart_midcategory_mean",
    "coldstart_category_mean", "coldstart_center_mean",
}
ALLOWED_STATUSES = {
    "success",
    "dev_fit_failure", "dev_fit_nonconverged", "dev_fit_convergence_unknown",
    "state_update_failure", "forecast_failure",
    "no_development_order", "not_applicable",
}
# Development final fit(성공/실패 무관)이 실제로 시도된 SKU에서만 나오는 arima_status —
# 진행률의 "ARIMA(Frozen) 처리 건수" 집계용.
ARIMA_PATH_STATUSES = {
    "success",
    "dev_fit_failure", "dev_fit_nonconverged", "dev_fit_convergence_unknown",
    "state_update_failure", "forecast_failure",
}

DEV_FIELDS = [
    "development_n_obs", "development_constant", "auto_selected_order", "auto_selected_converged",
    "selected_p", "selected_d", "selected_q", "with_intercept", "selected_aicc", "development_fallback_used",
]
DEV_FIELD_DEFAULTS = {
    "development_n_obs": np.nan, "development_constant": None,
    "auto_selected_order": "", "auto_selected_converged": "",
    "selected_p": np.nan, "selected_d": np.nan, "selected_q": np.nan,
    "with_intercept": None, "selected_aicc": np.nan,
    "development_fallback_used": None,
}

COLUMN_ORDER = [
    "center_id", "sku_id", "forecast_origin", "target_week", "horizon",
    "forecast_source", "arima_status", "has_observed_history",
    *DEV_FIELDS,
    "current_n_obs", "dev_fit_converged", "state_update_success", "forecast_success",
    "prediction_unclipped", "prediction", "negative_prediction_flag", "actual", "error",
]


def load_combined_history() -> pd.DataFrame:
    """development(2021~2023) + holdout(2024)를 결합한 read-only history. A/B window와
    2024 상한을 재검증한다."""
    cols = [CENTER_COL, SKU_COL, WEEK_COL, QTY_COL, SUBCAT_COL, MIDCAT_COL, LARGECAT_COL, EXISTED_BEFORE_REGIME_COL]
    dev = pd.read_parquet(DEV_PATH, columns=cols)
    hold = pd.read_parquet(HOLDOUT_PATH, columns=cols)
    df = pd.concat([dev, hold], ignore_index=True)

    assert df[WEEK_COL].max() <= HOLDOUT_MAX_WEEK, "2024 이후 데이터가 섞여 있음"
    dup = int(df.duplicated(subset=[CENTER_COL, SKU_COL, WEEK_COL]).sum())
    assert dup == 0, f"development/holdout 결합 데이터에 (center,sku,week) 중복 {dup}건"

    a_bad = df[(df[CENTER_COL] == "A") & ~((df[WEEK_COL] >= A_HISTORY_START) & (df[WEEK_COL] <= HOLDOUT_MAX_WEEK))]
    assert a_bad.empty, f"A센터에 지정 window({A_HISTORY_START.date()}~{HOLDOUT_MAX_WEEK.date()}) 밖 데이터 존재"
    b_bad = df[(df[CENTER_COL] == "B") & ~((df[WEEK_COL] >= B_HISTORY_START) & (df[WEEK_COL] <= HOLDOUT_MAX_WEEK))]
    assert b_bad.empty, f"B센터에 지정 window({B_HISTORY_START.date()}~{HOLDOUT_MAX_WEEK.date()}) 밖 데이터 존재"

    regime_nunique = df.groupby([CENTER_COL, SKU_COL], observed=True)[EXISTED_BEFORE_REGIME_COL].nunique()
    assert (regime_nunique <= 1).all(), \
        f"existed_before_regime이 (center_id, sku_id) 내에서 일관되지 않음: {int((regime_nunique > 1).sum())}건"
    a_regime_true = df[(df[CENTER_COL] == "A") & df[EXISTED_BEFORE_REGIME_COL]]
    assert a_regime_true.empty, f"A센터에 existed_before_regime=True인 행 존재: {len(a_regime_true)}건"

    return df


def _build_sku_arrays(center_df: pd.DataFrame) -> dict:
    arrays = {}
    for sku_id, g in center_df.sort_values(WEEK_COL).groupby(SKU_COL, observed=True, sort=False):
        arrays[sku_id] = (
            g[WEEK_COL].to_numpy(dtype="datetime64[D]"),
            g[QTY_COL].to_numpy(dtype=float),
        )
    return arrays


def _first_sale_weeks(center_df: pd.DataFrame) -> dict:
    """coldstart_flag(8주 warmup 포함)가 아니라 qty>0인 첫 주를 직접 계산한다."""
    sold = center_df[center_df[QTY_COL] > 0]
    return sold.groupby(SKU_COL, observed=True)[WEEK_COL].min().to_dict()


def _existed_before_regime_flags(center_df: pd.DataFrame) -> dict:
    return center_df.groupby(SKU_COL, observed=True)[EXISTED_BEFORE_REGIME_COL].first().to_dict()


def _load_purchase_category_map(center_id: str) -> pd.DataFrame:
    cols = [PURCHASE_RAW_CENTER_COL] + PURCHASE_SKU_PARTS + [PURCHASE_FIRST_STOCK_COL, SUBCAT_COL, MIDCAT_COL, LARGECAT_COL]
    purchase = pd.read_parquet(PURCHASE_MASTER_PATH, columns=cols)
    purchase = purchase[purchase[PURCHASE_RAW_CENTER_COL] == center_id].copy()

    purchase[SKU_COL] = (
        purchase[PURCHASE_SKU_PARTS[0]].astype(str) + SKU_SEP +
        purchase[PURCHASE_SKU_PARTS[1]].astype(str) + SKU_SEP +
        purchase[PURCHASE_SKU_PARTS[2]].astype(str)
    )
    purchase[PURCHASE_FIRST_STOCK_COL] = pd.to_datetime(purchase[PURCHASE_FIRST_STOCK_COL])

    return (
        purchase.sort_values(PURCHASE_FIRST_STOCK_COL)
        .groupby(SKU_COL, observed=True)
        .first()[[PURCHASE_FIRST_STOCK_COL, SUBCAT_COL, MIDCAT_COL, LARGECAT_COL]]
    )


def _resolve_coldstart_category(sku_id: str, origin: pd.Timestamp, purchase_cat_map: pd.DataFrame):
    """마스터에 없으면 None(호출부가 coldstart_center_mean으로 fallback)."""
    if sku_id in purchase_cat_map.index:
        row = purchase_cat_map.loc[sku_id]
        if pd.notna(row[PURCHASE_FIRST_STOCK_COL]) and row[PURCHASE_FIRST_STOCK_COL] <= origin:
            return row[[SUBCAT_COL, MIDCAT_COL, LARGECAT_COL]]
    return None


def _build_skus_by_target_week(holdout_raw: pd.DataFrame) -> dict:
    """(center_id, week_st) -> 그 주에 holdout 행이 있는 sku_id 집합."""
    grouped = holdout_raw.groupby([CENTER_COL, WEEK_COL], observed=True)[SKU_COL]
    return {key: set(g) for key, g in grouped}


def _dev_fields(dev_row: dict | None) -> dict:
    if dev_row is None:
        return dict(DEV_FIELD_DEFAULTS)
    return {k: dev_row[k] for k in DEV_FIELDS}


def _development_final_fit(history_qty: np.ndarray, p: int, d: int, q: int, with_intercept: bool) -> dict:
    """Development 전체 구간에 SKU당 계수를 1회만 추정한다(2024에서는 재추정하지 않음)."""
    out = {"status": "dev_fit_failure", "converged": None, "res": None, "params": None, "error": ""}
    train_log1p = np.log1p(history_qty)
    try:
        model = ARIMA(order=(p, d, q), with_intercept=with_intercept, suppress_warnings=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_log1p)
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


def _state_append(state_res, new_qty: np.ndarray, dev_params: np.ndarray) -> dict:
    """새로 확정된 관측치를 state에만 반영한다(refit=False -> parameter 재추정 없음). append는
    파라미터를 재추정 없이 그대로 상속하므로 float 오차조차 없어야 정상이다 — exact equality로
    검증해 근사 오차 뒤에 숨은 재추정을 놓치지 않는다."""
    out = {"res": None, "success": False, "error": ""}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_res = state_res.append(np.log1p(new_qty), refit=False)
        if not np.array_equal(np.asarray(new_res.params), dev_params):
            raise AssertionError("append(refit=False) 후 parameter가 Development fit과 달라짐(재추정 의심)")
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["res"] = new_res
    out["success"] = True
    return out


def _state_forecast(state_res, n_periods: int = FORECAST_N_PERIODS) -> dict:
    out = {"forecast": None, "success": False, "error": ""}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = np.asarray(state_res.get_forecast(steps=n_periods).predicted_mean)
        if not np.isfinite(fc).all():
            raise ValueError("forecast에 NaN/inf 포함")
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["forecast"] = fc
    out["success"] = True
    return out


def _get_or_init_sku_state(sku_states: dict, sku_id: str, dev_row: dict,
                            week_arr: np.ndarray, qty_arr: np.ndarray) -> dict:
    """SKU를 처음 만났을 때만 Development final fit을 수행하고, 이후에는 저장된 state를
    그대로 재사용한다(SKU당 fit 1회, origin마다 재fit 없음)."""
    if sku_id in sku_states:
        return sku_states[sku_id]

    p, d, q = int(dev_row["selected_p"]), int(dev_row["selected_d"]), int(dev_row["selected_q"])
    with_intercept = bool(dev_row["with_intercept"])
    n_obs_dev = int(np.searchsorted(week_arr, np.datetime64(DEVELOPMENT_END_WEEK, "D"), side="right"))
    assert n_obs_dev == int(dev_row["development_n_obs"]), (
        f"{sku_id}: Development fit에 사용된 n_obs({n_obs_dev})가 "
        f"orders 파일의 development_n_obs({dev_row['development_n_obs']})와 불일치"
    )
    history_qty_dev = qty_arr[:n_obs_dev]

    fit = _development_final_fit(history_qty_dev, p, d, q, with_intercept)
    state = {
        "dev_status": fit["status"], "dev_converged": fit["converged"], "dev_error": fit["error"],
        "res": fit["res"], "dev_params": fit["params"],
        "last_n_obs": n_obs_dev, "broken": fit["status"] != "success",
        "broken_status": None, "broken_reason": fit["error"],
    }
    sku_states[sku_id] = state
    return state


def _build_rows_for_unit(center_id: str, sku_id: str, origin: pd.Timestamp,
                          week_arr: np.ndarray, qty_arr: np.ndarray, dev_row: dict | None,
                          coldstart_means: dict, purchase_cat_map: pd.DataFrame,
                          first_sale: pd.Timestamp | None, existed_before_regime: bool,
                          sku_states: dict) -> list[dict]:
    n_obs = int(np.searchsorted(week_arr, np.datetime64(origin, "D"), side="right"))
    history_qty = qty_arr[:n_obs]

    has_observed_history = (first_sale is not None and first_sale <= origin) or bool(existed_before_regime)
    has_dev_order = dev_row is not None and dev_row["development_status"] == "selected"

    forecast_array = None  # log1p 4-step (arima 성공 시에만)
    flat_value = None  # raw qty 기준 flat 예측(그 외 모든 forecast_source)
    dev_fit_converged = None
    state_update_success = None
    forecast_success = None
    error = ""

    if not has_observed_history:
        sku_cat = _resolve_coldstart_category(sku_id, origin, purchase_cat_map)
        if sku_cat is not None:
            value, level = coldstart_fallback(sku_cat, coldstart_means)
        else:
            center_val = coldstart_means["center"]
            value, level = (float(center_val) if pd.notna(center_val) else 0.0), "center"
        flat_value = max(value, 0.0)
        forecast_source = f"coldstart_{level}_mean"
        arima_status = "not_applicable"

    elif has_dev_order:
        state = _get_or_init_sku_state(sku_states, sku_id, dev_row, week_arr, qty_arr)
        dev_fit_converged = state["dev_converged"]

        if state["dev_status"] != "success":
            arima_status = state["dev_status"]  # dev_fit_failure | dev_fit_nonconverged | dev_fit_convergence_unknown
            forecast_source = "naive_mean"
            flat_value = float(np.mean(history_qty)) if len(history_qty) else 0.0
            error = state["dev_error"]
        else:
            if not state["broken"] and n_obs > state["last_n_obs"]:
                new_obs = qty_arr[state["last_n_obs"]:n_obs]
                upd = _state_append(state["res"], new_obs, state["dev_params"])
                state_update_success = upd["success"]
                if upd["success"]:
                    state["res"] = upd["res"]
                    state["last_n_obs"] = n_obs
                else:
                    state["broken"] = True
                    state["broken_status"] = "state_update_failure"
                    state["broken_reason"] = upd["error"]
            elif not state["broken"]:
                state_update_success = True  # 새 관측치 없음(no-op) — append 자체가 불필요했을 뿐 실패 아님

            if state["broken"]:
                # broken_status는 최초로 broken=True가 된 시점(state_update 또는 forecast)의
                # 원인을 그대로 보존한다 — 재검사하지 않으므로 이후 origin에서 원인이
                # state_update_failure로 뒤바뀌지 않는다.
                arima_status = state["broken_status"]
                forecast_source = "naive_mean"
                flat_value = float(np.mean(history_qty)) if len(history_qty) else 0.0
                error = state["broken_reason"]
            else:
                fc = _state_forecast(state["res"])
                forecast_success = fc["success"]
                if fc["success"]:
                    forecast_array = fc["forecast"]
                    arima_status = "success"
                    forecast_source = "arima"
                else:
                    state["broken"] = True
                    state["broken_status"] = "forecast_failure"
                    state["broken_reason"] = fc["error"]
                    arima_status = "forecast_failure"
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
        arima_status = "no_development_order"

    dev_fields = _dev_fields(dev_row)

    rows = []
    for step, h_label in FORECAST_STEP_TO_HORIZON.items():
        target_week = origin + pd.Timedelta(weeks=HORIZON_WEEKS[h_label])
        actual = lookup_actual(week_arr, qty_arr, target_week)

        if forecast_array is not None:
            pred_unclipped, pred_final = expm1_clip(forecast_array[step])
        else:
            pred_unclipped = flat_value
            pred_final = max(flat_value, 0.0)

        rows.append({
            "center_id": center_id, "sku_id": sku_id,
            "forecast_origin": origin, "target_week": target_week, "horizon": h_label,
            "forecast_source": forecast_source, "arima_status": arima_status,
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


def _chunk_path(checkpoint_dir: Path, chunk_idx: int) -> Path:
    return checkpoint_dir / f"chunk_{chunk_idx:06d}.parquet"


def _load_done_keys(checkpoint_dir: Path) -> tuple[set, int]:
    """기존 chunk 파일들에서 key 컬럼만 가볍게 읽어 완료 단위를 파악하고, 다음 신규
    chunk가 이어서 쓸 인덱스를 함께 반환한다."""
    if not checkpoint_dir.exists():
        return set(), 0
    chunk_paths = sorted(checkpoint_dir.glob("chunk_*.parquet"))
    done_keys = set()
    for p in chunk_paths:
        keys_df = pd.read_parquet(p, columns=[CENTER_COL, SKU_COL, "forecast_origin"])
        done_keys.update(
            (c, s, pd.Timestamp(o))
            for c, s, o in zip(keys_df[CENTER_COL], keys_df[SKU_COL], keys_df["forecast_origin"])
        )
    return done_keys, len(chunk_paths)


def _save_chunk(rows: list, checkpoint_dir: Path, chunk_idx: int) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = _chunk_path(checkpoint_dir, chunk_idx)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows)[COLUMN_ORDER].to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def _load_all_chunks(checkpoint_dir: Path) -> pd.DataFrame:
    chunk_paths = sorted(checkpoint_dir.glob("chunk_*.parquet"))
    frames = [pd.read_parquet(p) for p in chunk_paths]
    return pd.concat(frames, ignore_index=True)


def _count_total_units(skus_by_target_week: dict) -> int:
    """fitting 없이 전체 (center,sku,forecast_origin) 처리 예정 건수만 population 규칙
    그대로(target_week에 holdout 행이 있는 SKU의 union) 빠르게 센다."""
    total = 0
    for center_id in CENTERS:
        for origin in ORIGINS:
            target_weeks = [origin + pd.Timedelta(weeks=w) for w in HORIZON_WEEKS.values()]
            population = set()
            for tw in target_weeks:
                population |= skus_by_target_week.get((center_id, tw), set())
            total += len(population)
    return total


def _format_timedelta(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "n/a"
    return str(pd.Timedelta(seconds=seconds)).split(".")[0]


def _print_progress(processed_this_run: int, arima_path_count: int, already_done: int,
                     total_expected: int, elapsed_sec: float) -> None:
    """순수 계측/출력용. 예측값·population·checkpoint 등 파이프라인 결과에는 관여하지 않는다."""
    done_total = already_done + processed_this_run
    avg_sec = elapsed_sec / processed_this_run if processed_this_run else float("nan")
    remaining_units = max(total_expected - done_total, 0)
    eta_sec = avg_sec * remaining_units
    total_est_sec = avg_sec * total_expected
    ratio = done_total / total_expected if total_expected else float("nan")

    print(f"[progress] {done_total:,}/{total_expected:,}" + (f" ({ratio:.1%})" if total_expected else ""))
    print(f"  ARIMA(Frozen) state 처리 건수(이번 실행): {arima_path_count:,}")
    print(f"  경과시간(이번 실행): {_format_timedelta(elapsed_sec)}")
    print(f"  1건당 평균 처리시간: {avg_sec:.4f}초")
    print(f"  예상 전체 소요시간: {_format_timedelta(total_est_sec)}")
    print(f"  예상 남은시간: {_format_timedelta(eta_sec)}")


def _count_b_veteran_zero_history(df: pd.DataFrame, holdout_raw: pd.DataFrame) -> int:
    """B센터 has_observed_history=True(existed_before_regime 포함)인데 forecast_origin
    시점 current_n_obs==0인 case 수를 full run 전에 미리 센다 — 0건이면 현재
    defensive branch(n_obs==0 -> constant/flat_value=0.0) 그대로 유지하면 되고, 존재하면
    임의로 새 정책을 만들지 않고 건수만 보고한다."""
    center_df = df[df[CENTER_COL] == "B"]
    sku_arrays = _build_sku_arrays(center_df)
    first_sale_weeks = _first_sale_weeks(center_df)
    existed_before_regime_flags = _existed_before_regime_flags(center_df)
    skus_by_target_week = _build_skus_by_target_week(holdout_raw[holdout_raw[CENTER_COL] == "B"])

    n_zero = 0
    for origin in ORIGINS:
        target_weeks = [origin + pd.Timedelta(weeks=w) for w in HORIZON_WEEKS.values()]
        population = set()
        for tw in target_weeks:
            population |= skus_by_target_week.get(("B", tw), set())

        for sku_id in population:
            week_arr, _ = sku_arrays[sku_id]
            n_obs = int(np.searchsorted(week_arr, np.datetime64(origin, "D"), side="right"))
            first_sale = first_sale_weeks.get(sku_id)
            existed_before_regime = existed_before_regime_flags.get(sku_id, False)
            has_observed_history = (first_sale is not None and first_sale <= origin) or bool(existed_before_regime)
            if has_observed_history and n_obs == 0:
                n_zero += 1
    return n_zero


def build_holdout(df: pd.DataFrame, orders_df: pd.DataFrame, holdout_raw: pd.DataFrame,
                   checkpoint_dir: Path = CHECKPOINT_DIR, checkpoint_every: int = CHECKPOINT_EVERY) -> pd.DataFrame:
    dev_lookup = {(r[CENTER_COL], r[SKU_COL]): r for r in orders_df.to_dict("records")}
    skus_by_target_week = _build_skus_by_target_week(holdout_raw)

    total_expected_units = _count_total_units(skus_by_target_week)
    done_keys, next_chunk_idx = _load_done_keys(checkpoint_dir)
    already_done = len(done_keys)
    print(f"[사전 계산] 전체 예정 처리 건수(center,sku,forecast_origin): {total_expected_units:,}건 "
          f"(기존 checkpoint 완료: {already_done:,}건)")

    current_chunk_rows: list = []
    units_in_chunk = 0
    processed_this_run = 0
    arima_path_count = 0
    run_start_time = time.perf_counter()

    for center_id in CENTERS:
        center_df = df[df[CENTER_COL] == center_id]
        sku_arrays = _build_sku_arrays(center_df)
        first_sale_weeks = _first_sale_weeks(center_df)
        existed_before_regime_flags = _existed_before_regime_flags(center_df)
        veteran_skus = {sku for sku, flag in existed_before_regime_flags.items() if flag}
        purchase_cat_map = _load_purchase_category_map(center_id)
        sku_states: dict = {}  # SKU당 Development final fit 1회 + append로 이어지는 frozen state

        for origin in ORIGINS:
            sold_by_origin = {sku for sku, fsw in first_sale_weeks.items() if fsw <= origin}
            donor_skus = sold_by_origin | veteran_skus
            donor_df = center_df[center_df[SKU_COL].isin(donor_skus)]
            coldstart_means = compute_coldstart_means(donor_df, origin)

            target_weeks = [origin + pd.Timedelta(weeks=w) for w in HORIZON_WEEKS.values()]
            population = set()
            for tw in target_weeks:
                population |= skus_by_target_week.get((center_id, tw), set())
            population = sorted(population)

            for sku_id in population:
                dev_row = dev_lookup.get((center_id, sku_id))
                has_dev_order = dev_row is not None and dev_row["development_status"] == "selected"
                key = (center_id, sku_id, origin)
                already = key in done_keys

                if already and not has_dev_order:
                    continue  # state 없는 branch(cold-start/no_development_order)는 완료됐으면 그냥 skip

                week_arr, qty_arr = sku_arrays[sku_id]
                first_sale = first_sale_weeks.get(sku_id)
                existed_before_regime = existed_before_regime_flags.get(sku_id, False)
                new_rows = _build_rows_for_unit(
                    center_id, sku_id, origin, week_arr, qty_arr, dev_row,
                    coldstart_means, purchase_cat_map, first_sale, existed_before_regime,
                    sku_states,
                )

                if already:
                    # has_dev_order라 state는 방금 갱신했지만(재구성 목적), 출력은 이미 저장돼
                    # 있으므로 이번 결과는 버린다.
                    continue

                current_chunk_rows.extend(new_rows)
                if new_rows and new_rows[0]["arima_status"] in ARIMA_PATH_STATUSES:
                    arima_path_count += 1
                units_in_chunk += 1
                processed_this_run += 1

                if processed_this_run == PROGRESS_FIRST_MILESTONE:
                    _print_progress(processed_this_run, arima_path_count, already_done,
                                     total_expected_units, time.perf_counter() - run_start_time)

                if units_in_chunk >= checkpoint_every:
                    _save_chunk(current_chunk_rows, checkpoint_dir, next_chunk_idx)
                    next_chunk_idx += 1
                    current_chunk_rows = []
                    units_in_chunk = 0
                    _print_progress(processed_this_run, arima_path_count, already_done,
                                     total_expected_units, time.perf_counter() - run_start_time)

    if units_in_chunk > 0:
        _save_chunk(current_chunk_rows, checkpoint_dir, next_chunk_idx)
        current_chunk_rows = []

    return (
        _load_all_chunks(checkpoint_dir)[COLUMN_ORDER]
        .sort_values([CENTER_COL, SKU_COL, "forecast_origin", "horizon"])
        .reset_index(drop=True)
    )


def run_audit(holdout_df: pd.DataFrame, orders_df: pd.DataFrame, df: pd.DataFrame) -> None:
    assert "auto_arima" not in globals(), "auto_arima가 import되어 있음 — 2024 holdout에서는 사용 금지"

    dup = int(holdout_df.duplicated(subset=[CENTER_COL, SKU_COL, "forecast_origin", "horizon"]).sum())
    assert dup == 0, f"(center_id, sku_id, forecast_origin, horizon) 중복 {dup}건"

    bad_horizon = set(holdout_df["horizon"].unique()) - set(HORIZON_WEEKS)
    assert not bad_horizon, f"허용되지 않은 horizon 존재: {bad_horizon}"

    bad_source = set(holdout_df["forecast_source"].unique()) - ALLOWED_SOURCES
    assert not bad_source, f"허용되지 않은 forecast_source 존재: {bad_source}"
    bad_status = set(holdout_df["arima_status"].unique()) - ALLOWED_STATUSES
    assert not bad_status, f"허용되지 않은 arima_status 존재: {bad_status}"

    expected_target = holdout_df["forecast_origin"] + holdout_df["horizon"].map(HORIZON_WEEKS).apply(lambda w: pd.Timedelta(weeks=w))
    assert (holdout_df["target_week"] == expected_target).all(), "target_week != forecast_origin + horizon 주수"

    actual_ref = df.rename(columns={WEEK_COL: "target_week", QTY_COL: "actual_ref"})[[CENTER_COL, SKU_COL, "target_week", "actual_ref"]]
    checked = holdout_df.merge(actual_ref, on=[CENTER_COL, SKU_COL, "target_week"], how="left")
    mismatch = checked[checked["actual"].notna() & (checked["actual"] != checked["actual_ref"])]
    assert mismatch.empty, f"actual 값이 실측과 불일치: {len(mismatch)}건"
    should_be_nan = checked[checked["actual"].notna() & checked["actual_ref"].isna()]
    assert should_be_nan.empty, f"actual이 채워졌는데 원본에 해당 주 데이터가 없음: {len(should_be_nan)}건"
    should_be_filled = checked[checked["actual"].isna() & checked["actual_ref"].notna()]
    assert should_be_filled.empty, f"원본에 데이터가 있는데 actual이 비어 있음: {len(should_be_filled)}건"

    units = holdout_df.drop_duplicates([CENTER_COL, SKU_COL, "forecast_origin"])[
        [CENTER_COL, SKU_COL, "forecast_origin", "current_n_obs"]
    ].sort_values("forecast_origin").reset_index(drop=True)
    df_sorted = df.sort_values(WEEK_COL).reset_index(drop=True)
    # merge_asof by= key dtype 일치 보정: 원본 df의 center_id는 category, holdout_df는 object라
    # dtype mismatch로 실패할 수 있음(계산 방식은 동일, dtype만 맞춤).
    units[CENTER_COL] = units[CENTER_COL].astype(str)
    df_sorted[CENTER_COL] = df_sorted[CENTER_COL].astype(str)
    df_sorted["n_obs_asof"] = df_sorted.groupby([CENTER_COL, SKU_COL]).cumcount() + 1
    checked_n_obs = pd.merge_asof(
        units, df_sorted[[CENTER_COL, SKU_COL, WEEK_COL, "n_obs_asof"]],
        left_on="forecast_origin", right_on=WEEK_COL, by=[CENTER_COL, SKU_COL], direction="backward",
    )
    checked_n_obs["n_obs_asof"] = checked_n_obs["n_obs_asof"].fillna(0).astype(int)
    n_obs_mismatch = checked_n_obs[checked_n_obs["n_obs_asof"] != checked_n_obs["current_n_obs"]]
    assert n_obs_mismatch.empty, f"current_n_obs가 forecast_origin까지의 실제 history 개수와 불일치(history 누출 가능성): {len(n_obs_mismatch)}건"

    # orders_df에 있는 SKU는 field-by-field 동일성, 없는 SKU(2024 신규)는 DEV_FIELD_DEFAULTS
    # 유지 여부를 각각 검사한다(Development order 불변성).
    holdout_units = holdout_df.drop_duplicates([CENTER_COL, SKU_COL])[[CENTER_COL, SKU_COL] + DEV_FIELDS]
    dev_merged = holdout_units.merge(
        orders_df[[CENTER_COL, SKU_COL] + DEV_FIELDS], on=[CENTER_COL, SKU_COL],
        how="left", suffixes=("", "_orig"), indicator=True,
    )

    present = dev_merged[dev_merged["_merge"] == "both"]
    for col in DEV_FIELDS:
        left, right = present[col], present[f"{col}_orig"]
        both_nan = left.isna() & right.isna()
        mismatch_col = present[~both_nan & (left != right)]
        assert mismatch_col.empty, f"Development {col} 값이 holdout 결과에서 변경됨(주문 파일에 존재하는 SKU): {len(mismatch_col)}건"

    absent = dev_merged[dev_merged["_merge"] == "left_only"]
    for col in DEV_FIELDS:
        default = DEV_FIELD_DEFAULTS[col]
        val = absent[col]
        if default is None or (isinstance(default, float) and np.isnan(default)):
            bad = absent[val.notna()]
        else:
            bad = absent[val != default]
        assert bad.empty, f"Development order 없는 SKU의 {col}이 기본값을 유지하지 않음: {len(bad)}건"

    # Frozen 고유: dev_fit_converged는 SKU당 Development final fit 1회로만 정해지므로
    # 같은 SKU의 모든 행에서 값이 하나여야 한다(중간에 값이 바뀌면 재추정이 있었다는 뜻).
    arima_path_rows = holdout_df[holdout_df["arima_status"].isin(ARIMA_PATH_STATUSES)]
    conv_nunique = arima_path_rows.groupby([CENTER_COL, SKU_COL])["dev_fit_converged"].nunique(dropna=False)
    assert (conv_nunique <= 1).all(), \
        f"SKU 내에서 dev_fit_converged 값이 바뀐 경우 존재(재추정 의심): {int((conv_nunique > 1).sum())}건"

    arima_rows = holdout_df[holdout_df["forecast_source"] == "arima"]
    assert (arima_rows["arima_status"] == "success").all(), "forecast_source=arima인데 arima_status!=success인 행 존재"
    assert (arima_rows["dev_fit_converged"] == True).all(), "forecast_source=arima인데 Development fit이 미수렴인 행 존재"  # noqa: E712
    assert (arima_rows["state_update_success"] == True).all(), "forecast_source=arima인데 state_update_success!=True인 행 존재"  # noqa: E712
    assert (arima_rows["forecast_success"] == True).all(), "forecast_source=arima인데 forecast_success!=True인 행 존재"  # noqa: E712

    pred = holdout_df["prediction"].to_numpy(dtype=float)
    pred_unclipped = holdout_df["prediction_unclipped"].to_numpy(dtype=float)
    assert np.isfinite(pred).all(), "prediction에 NaN/inf 존재"
    assert np.isfinite(pred_unclipped).all(), "prediction_unclipped에 NaN/inf 존재"
    assert (pred >= 0).all(), "prediction에 음수 존재"

    assert (holdout_df["negative_prediction_flag"] == (holdout_df["prediction_unclipped"] < 0)).all(), \
        "negative_prediction_flag가 prediction_unclipped<0과 불일치"

    cold_mask = holdout_df["forecast_source"].str.startswith("coldstart_")
    assert (holdout_df.loc[cold_mask, "has_observed_history"] == False).all(), \
        "cold-start 행인데 has_observed_history=True"  # noqa: E712
    assert (holdout_df.loc[~cold_mask, "has_observed_history"] == True).all(), \
        "Main self-history 행인데 has_observed_history=False"  # noqa: E712

    b_regime_flags = df.loc[df[CENTER_COL] == "B", [SKU_COL, EXISTED_BEFORE_REGIME_COL]].drop_duplicates(SKU_COL)
    veteran_b_skus = set(b_regime_flags.loc[b_regime_flags[EXISTED_BEFORE_REGIME_COL], SKU_COL])
    b_rows = holdout_df[holdout_df[CENTER_COL] == "B"]
    veteran_violation = b_rows[b_rows[SKU_COL].isin(veteran_b_skus) & (b_rows["has_observed_history"] == False)]  # noqa: E712
    assert veteran_violation.empty, \
        f"B센터 existed_before_regime=True SKU인데 has_observed_history=False인 행 존재: {len(veteran_violation)}건"


def build_summary(holdout_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = {"A": holdout_df[holdout_df[CENTER_COL] == "A"], "B": holdout_df[holdout_df[CENTER_COL] == "B"], "ALL": holdout_df}
    for scope, sub in scopes.items():
        src_counts = sub["forecast_source"].value_counts()
        rows.append({
            "scope": scope,
            "n_total": len(sub),
            "n_arima": int(src_counts.get("arima", 0)),
            "n_constant": int(src_counts.get("constant", 0)),
            "n_naive_mean": int(src_counts.get("naive_mean", 0)),
            "n_coldstart_subcategory": int(src_counts.get("coldstart_subcategory_mean", 0)),
            "n_coldstart_midcategory": int(src_counts.get("coldstart_midcategory_mean", 0)),
            "n_coldstart_category": int(src_counts.get("coldstart_category_mean", 0)),
            "n_coldstart_center": int(src_counts.get("coldstart_center_mean", 0)),
            "n_dev_fit_failure": int((sub["arima_status"] == "dev_fit_failure").sum()),
            "n_dev_fit_nonconverged": int((sub["arima_status"] == "dev_fit_nonconverged").sum()),
            "n_dev_fit_convergence_unknown": int((sub["arima_status"] == "dev_fit_convergence_unknown").sum()),
            "n_state_update_failure": int((sub["arima_status"] == "state_update_failure").sum()),
            "n_forecast_failure": int((sub["arima_status"] == "forecast_failure").sum()),
            "n_negative_clipped": int(sub["negative_prediction_flag"].sum()),
        })
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame) -> None:
    print("=" * 80)
    for scope in ["A", "B", "ALL"]:
        r = summary[summary["scope"] == scope].iloc[0]
        print(f"[{scope}]")
        print(f"  전체 예측행: {r['n_total']:,}")
        print(f"  arima: {r['n_arima']:,}")
        print(f"  constant: {r['n_constant']:,}")
        print(f"  naive_mean: {r['n_naive_mean']:,}")
        print(f"  coldstart_subcategory_mean: {r['n_coldstart_subcategory']:,}")
        print(f"  coldstart_midcategory_mean: {r['n_coldstart_midcategory']:,}")
        print(f"  coldstart_category_mean: {r['n_coldstart_category']:,}")
        print(f"  coldstart_center_mean: {r['n_coldstart_center']:,}")
        print(f"  dev_fit failure / nonconverged / convergence_unknown: "
              f"{r['n_dev_fit_failure']:,} / {r['n_dev_fit_nonconverged']:,} / {r['n_dev_fit_convergence_unknown']:,}")
        print(f"  state_update_failure / forecast_failure: {r['n_state_update_failure']:,} / {r['n_forecast_failure']:,}")
        print(f"  negative prediction clipping 수: {r['n_negative_clipped']:,}")
        print()

    print("audit: PASS")
    print("[저장 경로]")
    print(f"  {HOLDOUT_OUT_PATH}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_combined_history()
    orders_df = pd.read_parquet(ORDERS_PATH)
    holdout_raw = pd.read_parquet(HOLDOUT_PATH, columns=[CENTER_COL, SKU_COL, WEEK_COL])

    n_b_veteran_zero = _count_b_veteran_zero_history(df, holdout_raw)
    print(f"[사전 점검] B센터 has_observed_history=True & current_n_obs==0: {n_b_veteran_zero:,}건")

    holdout_df = build_holdout(df, orders_df, holdout_raw)

    run_audit(holdout_df, orders_df, df)
    holdout_df.to_parquet(HOLDOUT_OUT_PATH, index=False)
    if CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)

    summary = build_summary(holdout_df)
    print_summary(summary)


if __name__ == "__main__":
    main()
