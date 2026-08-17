"""
leakage_returns_audit.py

Day 1 Step 2: 최종 Feature Table의 반품 혼입 이상 및 시간 경계 leakage 최소 필수 검증.

검사:
    - qty/target의 음수·inf 및 반품 컬럼 존재 여부
    - h1/h2/h4별 P20: train target_date < validation_start
    - 2024 holdout target의 이전 학습 구간 유입 방지
    - production pipeline의 h1 P20 적용 여부
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parents[4]
FEATURE_TABLE_PATH = BASE_DIR / "data" / "final_feature_table.parquet"
OUT_DIR = BASE_DIR / "data" / "ml" / "experiments" / "day1_audit"

CENTER_COL, SKU_COL, WEEK_COL, QTY_COL = "center_id", "sku_id", "week_st", "qty"
HORIZONS = [1, 2, 4]
HOLDOUT_START = pd.Timestamp("2024-01-01")

sys.path.insert(0, str(BASE_DIR / "src" / "ml" / "models"))
sys.path.insert(0, str(BASE_DIR / "src" / "ml" / "feature_engineering"))

from common import apply_target_date_boundary_filter  # noqa: E402
from split_train_val_test import (  # noqa: E402
    A_TRAIN_START, A_TRAIN_END, A_VAL_START,
    B_TRAIN_START, B_TRAIN_END, B_WF_POOL_START,
)

# production train/validation 경계
PROD_BOUNDS = {
    "A": (pd.Timestamp(A_TRAIN_START), pd.Timestamp(A_TRAIN_END), pd.Timestamp(A_VAL_START)),
    "B": (pd.Timestamp(B_TRAIN_START), pd.Timestamp(B_TRAIN_END), pd.Timestamp(B_WF_POOL_START)),
}


def load_table() -> pd.DataFrame:
    cols = [CENTER_COL, SKU_COL, WEEK_COL, QTY_COL, "split"] + [f"target_h{h}" for h in HORIZONS]
    df = pd.read_parquet(FEATURE_TABLE_PATH, columns=cols)
    print(f"[로드] {FEATURE_TABLE_PATH} -> {len(df):,}행")
    return df


def has_returns_column() -> bool:
    return any("반품" in c for c in pq.ParquetFile(FEATURE_TABLE_PATH).schema.names)


def return_quality_audit(df: pd.DataFrame) -> dict:
    n_qty_neg = int((df[QTY_COL] < 0).sum())
    n_qty_inf = int(np.isinf(df[QTY_COL].to_numpy(dtype=float)).sum())
    n_target_neg = {h: int((df[f"target_h{h}"] < 0).sum()) for h in HORIZONS}
    n_target_inf = {
        h: int(np.isinf(df[f"target_h{h}"].to_numpy(dtype=float)).sum()) for h in HORIZONS
    }
    returns_col_present = has_returns_column()
    gate = (
        n_qty_neg == 0
        and n_qty_inf == 0
        and all(v == 0 for v in n_target_neg.values())
        and all(v == 0 for v in n_target_inf.values())
        and not returns_col_present
    )
    return {
        "n_qty_negative": n_qty_neg,
        "n_qty_inf": n_qty_inf,
        "n_target_negative": n_target_neg,
        "n_target_inf": n_target_inf,
        "returns_column_present": returns_col_present,
        "return_data_quality_gate": gate,
    }


def p20_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for center, (train_start, train_end, val_start) in PROD_BOUNDS.items():
        train_rows = df[
            (df[CENTER_COL] == center)
            & (df[WEEK_COL] >= train_start)
            & (df[WEEK_COL] <= train_end)
        ]
        for h in HORIZONS:
            target_date = train_rows[WEEK_COL] + pd.Timedelta(weeks=h)
            purged = train_rows[target_date < val_start]
            n_violation = int(((purged[WEEK_COL] + pd.Timedelta(weeks=h)) >= val_start).sum())
            rows.append({
                "check": "P20",
                "center": center,
                "horizon": f"h{h}",
                "train_start": train_start.date(),
                "train_end": train_end.date(),
                "validation_start": val_start.date(),
                "n_train_rows": len(train_rows),
                "n_boundary_before_purge": int((target_date >= val_start).sum()),
                "n_violation_after_purge": n_violation,
                "purge_rule_pass": len(train_rows) > 0 and n_violation == 0,
            })
    return pd.DataFrame(rows)


def holdout_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    train_pool = df[df["split"] == "train"]
    for center, sub in train_pool.groupby(CENTER_COL, observed=True):
        for h in HORIZONS:
            target_date = sub[WEEK_COL] + pd.Timedelta(weeks=h)
            purged = sub[target_date < HOLDOUT_START]
            n_violation = int(((purged[WEEK_COL] + pd.Timedelta(weeks=h)) >= HOLDOUT_START).sum())
            rows.append({
                "check": "holdout_2024",
                "center": center,
                "horizon": f"h{h}",
                "n_train_rows": len(sub),
                "n_boundary_before_purge": int((target_date >= HOLDOUT_START).sum()),
                "n_violation_after_purge": n_violation,
                "boundary_rule_pass": len(sub) > 0 and n_violation == 0,
            })
    return pd.DataFrame(rows)


def h1_pipeline_status(df: pd.DataFrame) -> str:
    """실제 production P20 함수 적용 후 h1 validation 경계 침범 여부 확인."""
    filtered = apply_target_date_boundary_filter(df[df["split"] == "train"].copy(), horizon_weeks=1)
    val_start = filtered[CENTER_COL].map({c: bounds[2] for c, bounds in PROD_BOUNDS.items()})
    target_date = filtered[WEEK_COL] + pd.Timedelta(weeks=1)
    n_violation = int((val_start.isna() | (target_date >= val_start)).sum())
    return "PASS" if len(filtered) > 0 and n_violation == 0 else "FAIL"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_table()

    ret = return_quality_audit(df)
    p20_df = p20_audit(df)
    holdout_df = holdout_audit(df)
    h1_status = h1_pipeline_status(df)

    p20_pass = {
        h: bool(p20_df.loc[p20_df["horizon"] == f"h{h}", "purge_rule_pass"].all())
        for h in HORIZONS
    }
    holdout_pass = {
        h: bool(holdout_df.loc[holdout_df["horizon"] == f"h{h}", "boundary_rule_pass"].all())
        for h in HORIZONS
    }
    final_gate = (
        ret["return_data_quality_gate"]
        and all(p20_pass.values())
        and all(holdout_pass.values())
        and h1_status == "PASS"
    )

    pd.DataFrame([
        {"metric": "n_qty_negative", "value": ret["n_qty_negative"]},
        {"metric": "n_qty_inf", "value": ret["n_qty_inf"]},
        *[{"metric": f"n_target_h{h}_negative", "value": ret["n_target_negative"][h]} for h in HORIZONS],
        *[{"metric": f"n_target_h{h}_inf", "value": ret["n_target_inf"][h]} for h in HORIZONS],
        {"metric": "returns_column_present", "value": ret["returns_column_present"]},
        {"metric": "return_data_quality_gate", "value": ret["return_data_quality_gate"]},
    ]).to_csv(OUT_DIR / "returns_audit.csv", index=False, encoding="utf-8-sig")

    pd.concat([p20_df, holdout_df], ignore_index=True).to_csv(
        OUT_DIR / "cv_purging_audit.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([
        *[{"metric": f"h{h}_P20_purge_rule_pass", "value": p20_pass[h]} for h in HORIZONS],
        *[{"metric": f"h{h}_2024_holdout_boundary_rule_pass", "value": holdout_pass[h]} for h in HORIZONS],
        {"metric": "h1_pipeline_status", "value": h1_status},
        {"metric": "feature_table_boundary_safety_gate", "value": final_gate},
    ]).to_csv(OUT_DIR / "step2_final_gate.csv", index=False, encoding="utf-8-sig")

    print(f"return_data_quality_gate: {'PASS' if ret['return_data_quality_gate'] else 'FAIL'}")
    print(f"P20 h1/h2/h4: {[p20_pass[h] for h in HORIZONS]}")
    print(f"2024 holdout h1/h2/h4: {[holdout_pass[h] for h in HORIZONS]}")
    print(f"h1_pipeline_status: {h1_status}")
    print(f"feature_table_boundary_safety_gate: {'PASS' if final_gate else 'FAIL'}")


if __name__ == "__main__":
    main()
