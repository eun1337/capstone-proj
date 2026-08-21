"""
01_build_arima_orders.py

Development 데이터만 사용하여 SKU별 ARIMA 구조를 확정한다.

eligible SKU마다 auto_arima 탐색을 1회 수행하고,
AICc 1순위 후보가 수렴하지 않으면 같은 탐색에서 수렴한 후보 중
AICc가 가장 낮은 (p,d,q)+with_intercept를 최종 구조로 선택한다.

2024 데이터는 구조 결정에 사용하지 않는다.
"""

from pathlib import Path
import os
import sys
import warnings

import numpy as np
import pandas as pd
from pmdarima import auto_arima

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from statistical_utils import (  # noqa: E402
    AUTO_ARIMA_KWARGS, MIN_HISTORY, WEEK_COL, QTY_COL, CENTER_COL,
    extract_valid_fit_candidates, candidate_convergence, candidate_aicc,
)

BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_PATH = BASE_DIR / "data" / "development_2021_2023.parquet"
OUT_DIR = BASE_DIR / "data" / "ml" / "day2_statistical_models"

SKU_COL = "sku_id"
CENTERS = ["A", "B"]

A_WINDOW_START = pd.Timestamp("2021-01-04")
A_WINDOW_END = pd.Timestamp("2023-12-25")
B_WINDOW_START = pd.Timestamp("2023-07-03")
B_WINDOW_END = pd.Timestamp("2023-12-25")
DEV_MAX_WEEK = pd.Timestamp("2023-12-25")  # 2024 데이터 누출 방지용 상한 안전장치

ORDERS_PATH = OUT_DIR / "arima_development_orders.parquet"
SUMMARY_PATH = OUT_DIR / "arima_development_orders_summary.csv"
ORDER_DIST_PATH = OUT_DIR / "arima_development_order_distribution.csv"
CHECKPOINT_PATH = OUT_DIR / "arima_development_orders.checkpoint.parquet"
CHECKPOINT_EVERY = 500  # 이 개수만큼 SKU를 처리할 때마다 checkpoint를 갱신

COLUMN_ORDER = [
    "center_id", "sku_id", "development_n_obs", "development_constant",
    "auto_arima_attempted", "auto_selected_order", "auto_selected_with_intercept",
    "auto_selected_aicc", "auto_selected_converged",
    "selected_p", "selected_d", "selected_q", "with_intercept", "selected_aicc",
    "development_fallback_used", "development_status", "search_warning_count", "error",
]

STATUSES = ["selected", "short_history", "constant", "no_converged_candidate", "auto_arima_failure"]


def load_source() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_PATH, columns=[CENTER_COL, SKU_COL, WEEK_COL, QTY_COL])
    assert df[WEEK_COL].max() <= DEV_MAX_WEEK, "development 입력에 2024 이후 데이터가 섞여 있음"

    a_out_of_window = df[(df[CENTER_COL] == "A") &
                          ~((df[WEEK_COL] >= A_WINDOW_START) & (df[WEEK_COL] <= A_WINDOW_END))]
    assert a_out_of_window.empty, f"A센터에 지정 window({A_WINDOW_START.date()}~{A_WINDOW_END.date()}) 밖 데이터 존재"
    a = df[(df[CENTER_COL] == "A") & (df[WEEK_COL] >= A_WINDOW_START) & (df[WEEK_COL] <= A_WINDOW_END)]

    b_out_of_window = df[(df[CENTER_COL] == "B") &
                          ~((df[WEEK_COL] >= B_WINDOW_START) & (df[WEEK_COL] <= B_WINDOW_END))]
    assert b_out_of_window.empty, f"B센터에 지정 window({B_WINDOW_START.date()}~{B_WINDOW_END.date()}) 밖 데이터 존재"
    b = df[(df[CENTER_COL] == "B") & (df[WEEK_COL] >= B_WINDOW_START) & (df[WEEK_COL] <= B_WINDOW_END)]

    return pd.concat([a, b], ignore_index=True)


def _empty_row(center_id: str, sku_id: str, n_obs: int, is_constant: bool) -> dict:
    return {
        "center_id": center_id, "sku_id": sku_id,
        "development_n_obs": n_obs, "development_constant": is_constant,
        "auto_arima_attempted": False,
        "auto_selected_order": "", "auto_selected_with_intercept": None,
        "auto_selected_aicc": np.nan, "auto_selected_converged": "",
        "selected_p": np.nan, "selected_d": np.nan, "selected_q": np.nan,
        "with_intercept": None, "selected_aicc": np.nan,
        "development_fallback_used": False, "development_status": "",
        "search_warning_count": 0, "error": "",
    }


def _select_order(qty: np.ndarray) -> dict:
    """eligible SKU 1개에 auto_arima search를 정확히 1회 수행하고 order를 확정한다."""
    out = {
        "auto_arima_attempted": True,
        "auto_selected_order": "", "auto_selected_with_intercept": None,
        "auto_selected_aicc": np.nan, "auto_selected_converged": "",
        "selected_p": np.nan, "selected_d": np.nan, "selected_q": np.nan,
        "with_intercept": None, "selected_aicc": np.nan,
        "development_fallback_used": False, "development_status": "no_converged_candidate",
        "search_warning_count": 0, "error": "",
    }

    train_log1p = np.log1p(qty)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ret = auto_arima(train_log1p, return_valid_fits=True, **AUTO_ARIMA_KWARGS)
        out["search_warning_count"] = len(caught)
        candidates = extract_valid_fit_candidates(ret)
    except Exception as e:
        out["development_status"] = "auto_arima_failure"
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    best = candidates[0]
    best_converged = candidate_convergence(best)
    out["auto_selected_order"] = str(best.order)
    out["auto_selected_with_intercept"] = getattr(best, "with_intercept", None)
    out["auto_selected_aicc"] = candidate_aicc(best)
    out["auto_selected_converged"] = str(best_converged)

    if best_converged is True:
        chosen = best
    else:
        converged_candidates = []
        for m in candidates:
            if candidate_convergence(m) is not True:
                continue
            m_aicc = candidate_aicc(m)
            if not np.isfinite(m_aicc):
                continue
            converged_candidates.append((m, m_aicc))
        if converged_candidates:
            chosen, _ = min(converged_candidates, key=lambda t: t[1])
            out["development_fallback_used"] = True
        else:
            chosen = None

    if chosen is not None:
        out["selected_p"], out["selected_d"], out["selected_q"] = chosen.order
        out["with_intercept"] = getattr(chosen, "with_intercept", None)
        out["selected_aicc"] = candidate_aicc(chosen)
        out["development_status"] = "selected"

    return out


def _load_checkpoint(checkpoint_path: Path) -> list:
    if not checkpoint_path.exists():
        return []
    return pd.read_parquet(checkpoint_path)[COLUMN_ORDER].to_dict("records")


def _save_checkpoint(rows: list, checkpoint_path: Path) -> None:
    """중단 시 손상된 checkpoint가 남지 않도록 임시 파일에 쓰고 원자적으로 교체한다."""
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    pd.DataFrame(rows)[COLUMN_ORDER].to_parquet(tmp_path, index=False)
    os.replace(tmp_path, checkpoint_path)


def build_orders(df: pd.DataFrame, checkpoint_path: Path = CHECKPOINT_PATH,
                  checkpoint_every: int = CHECKPOINT_EVERY) -> pd.DataFrame:
    rows = _load_checkpoint(checkpoint_path)
    done_keys = {(r["center_id"], r["sku_id"]) for r in rows}
    n_since_checkpoint = 0

    grouped = df.sort_values(WEEK_COL).groupby([CENTER_COL, SKU_COL], observed=True, sort=False)
    for (center_id, sku_id), g in grouped:
        if (center_id, sku_id) in done_keys:
            continue

        qty = g[QTY_COL].to_numpy()
        n_obs = len(qty)
        is_constant = bool(len(np.unique(qty)) == 1)
        row = _empty_row(center_id, sku_id, n_obs, is_constant)

        if n_obs < MIN_HISTORY:
            row["development_status"] = "short_history"
        elif is_constant:
            row["development_status"] = "constant"
        else:
            row.update(_select_order(qty))

        rows.append(row)
        n_since_checkpoint += 1

        if n_since_checkpoint >= checkpoint_every:
            _save_checkpoint(rows, checkpoint_path)
            n_since_checkpoint = 0

    if n_since_checkpoint > 0:
        _save_checkpoint(rows, checkpoint_path)

    return pd.DataFrame(rows)[COLUMN_ORDER].sort_values([CENTER_COL, SKU_COL]).reset_index(drop=True)


def build_order_distribution(orders: pd.DataFrame) -> pd.DataFrame:
    """A/B별 selected (p,d,q) 분포."""
    sel = orders[orders["development_status"] == "selected"].copy()
    sel["order"] = list(zip(sel["selected_p"].astype(int), sel["selected_d"].astype(int), sel["selected_q"].astype(int)))
    dist = sel.groupby([CENTER_COL, "order"], observed=True).size().reset_index(name="n")
    return dist.sort_values([CENTER_COL, "n"], ascending=[True, False]).reset_index(drop=True)


def run_audit(orders: pd.DataFrame, n_input_sku: int) -> None:
    """저장 전 전수 audit. 하나라도 어긋나면 즉시 AssertionError로 중단한다."""
    dup = int(orders.duplicated(subset=[CENTER_COL, SKU_COL]).sum())
    assert dup == 0, f"(center_id, sku_id) 중복 {dup}건"

    assert len(orders) == n_input_sku, f"입력 SKU 수({n_input_sku}) != 결과 SKU 수({len(orders)})"

    unknown_status = set(orders["development_status"].unique()) - set(STATUSES)
    assert not unknown_status, f"알 수 없는 development_status: {unknown_status}"
    assert int(orders["development_status"].value_counts().sum()) == len(orders), "status 합계 != 전체 SKU 수"

    no_order_mask = orders["development_status"].isin(["short_history", "constant"])
    bad_no_order = orders.loc[no_order_mask, ["selected_p", "selected_d", "selected_q"]].notna().any(axis=1)
    assert not bad_no_order.any(), f"short_history/constant인데 selected order 존재: {int(bad_no_order.sum())}건"

    fallback_mask = orders["development_fallback_used"]
    bad_fallback = orders.loc[fallback_mask, ["selected_p", "selected_d", "selected_q"]].isna().any(axis=1)
    assert not bad_fallback.any(), f"fallback 사용인데 selected order 없음: {int(bad_fallback.sum())}건"

    sel = orders[orders["development_status"] == "selected"]
    bad_pdq = sel[["selected_p", "selected_d", "selected_q"]].isna().any(axis=1)
    assert not bad_pdq.any(), f"selected인데 p/d/q 결측: {int(bad_pdq.sum())}건"
    bad_intercept = sel["with_intercept"].isna()
    assert not bad_intercept.any(), f"selected인데 with_intercept 결측: {int(bad_intercept.sum())}건"
    bad_aicc = ~np.isfinite(sel["selected_aicc"].to_numpy(dtype=float))
    assert not bad_aicc.any(), f"selected인데 selected_aicc non-finite: {int(bad_aicc.sum())}건"


def build_summary(orders: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = {"A": orders[orders[CENTER_COL] == "A"], "B": orders[orders[CENTER_COL] == "B"], "ALL": orders}
    for scope, sub in scopes.items():
        searched_ok = sub[sub["auto_selected_converged"] != ""]
        status_counts = {s: int((sub["development_status"] == s).sum()) for s in STATUSES}
        rows.append({
            "scope": scope,
            "n_total": len(sub),
            "n_short_history": status_counts["short_history"],
            "n_constant": status_counts["constant"],
            "n_attempted": int(sub["auto_arima_attempted"].sum()),
            "n_initial_converged": int((searched_ok["auto_selected_converged"] == "True").sum()),
            "n_initial_nonconverged": int((searched_ok["auto_selected_converged"] == "False").sum()),
            "n_initial_unknown": int((searched_ok["auto_selected_converged"] == "Unknown").sum()),
            "n_fallback_used": int(sub["development_fallback_used"].sum()),
            "n_no_converged_candidate": status_counts["no_converged_candidate"],
            "n_auto_arima_failure": status_counts["auto_arima_failure"],
            "n_selected": status_counts["selected"],
        })
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame) -> None:
    print("=" * 80)
    for scope in ["A", "B", "ALL"]:
        r = summary[summary["scope"] == scope].iloc[0]
        print(f"[{scope}]")
        print(f"  전체 SKU 수: {r['n_total']:,}")
        print(f"  short_history: {r['n_short_history']:,}")
        print(f"  constant: {r['n_constant']:,}")
        print(f"  auto_arima attempted: {r['n_attempted']:,}")
        print(f"  initial converged / non-converged / unknown: "
              f"{r['n_initial_converged']:,} / {r['n_initial_nonconverged']:,} / {r['n_initial_unknown']:,}")
        print(f"  fallback 사용: {r['n_fallback_used']:,}")
        print(f"  no converged candidate: {r['n_no_converged_candidate']:,}")
        print(f"  auto_arima failure: {r['n_auto_arima_failure']:,}")
        print(f"  최종 selected_order 수: {r['n_selected']:,}")
        print()

    print("[저장 경로]")
    for p in [ORDERS_PATH, SUMMARY_PATH, ORDER_DIST_PATH]:
        print(f"  {p}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_source()
    n_input_sku = df.drop_duplicates([CENTER_COL, SKU_COL]).shape[0]

    orders = build_orders(df, CHECKPOINT_PATH, CHECKPOINT_EVERY)
    run_audit(orders, n_input_sku)
    orders.to_parquet(ORDERS_PATH, index=False)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    summary = build_summary(orders)
    summary.to_csv(SUMMARY_PATH, index=False)

    dist = build_order_distribution(orders)
    dist.to_csv(ORDER_DIST_PATH, index=False)

    print_summary(summary)


if __name__ == "__main__":
    main()
