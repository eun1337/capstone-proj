"""
target_audit.py

Day 1 Step 1: 최종 Feature Table의 target 최소 필수 검증.

검사:
    - (center_id, sku_id, week_st) key 중복 여부
    - target_h1/h2/h4가 실제 t+1/t+2/t+4 qty와 일치하는지
    - 미래 주차 미관측 구간의 target NaN이 정상적인 right-censoring인지

최종 target은 raw target_h1/h2/h4를 사용하며,
로그 변환이 필요한 경우 학습 시 np.log1p()로 계산한다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[4]
FEATURE_TABLE_PATH = BASE_DIR / "data" / "final_feature_table.parquet"
OUT_DIR = BASE_DIR / "data" / "ml" / "experiments" / "day1_audit"
AUDIT_OUT_PATH = OUT_DIR / "target_audit.csv"

CENTER_COL, SKU_COL, WEEK_COL, QTY_COL = "center_id", "sku_id", "week_st", "qty"
HORIZONS = [1, 2, 4]
KEY_COLS = [CENTER_COL, SKU_COL, WEEK_COL]
NEEDED_COLS = KEY_COLS + [QTY_COL] + [f"target_h{h}" for h in HORIZONS]


def load_table() -> pd.DataFrame:
    df = pd.read_parquet(FEATURE_TABLE_PATH, columns=NEEDED_COLS)
    print(f"[로드] {FEATURE_TABLE_PATH} -> {len(df):,}행")
    return df


def key_uniqueness_check(df: pd.DataFrame) -> dict:
    n_duplicate_rows = int(df.duplicated(subset=KEY_COLS, keep=False).sum())
    return {"n_total_rows": len(df), "n_duplicate_rows": n_duplicate_rows, "pass": n_duplicate_rows == 0}


def self_join_validate_horizon(df: pd.DataFrame, h: int) -> dict:
    """t+h주 실제 qty를 self-join해 저장된 target_h{h}와 대조."""
    target_col = f"target_h{h}"
    left = df[KEY_COLS + [target_col]].copy()
    left["target_date_expected"] = left[WEEK_COL] + pd.Timedelta(weeks=h)

    future = df[KEY_COLS + [QTY_COL]].rename(
        columns={WEEK_COL: "target_date_expected", QTY_COL: "actual_future_qty"}
    )
    future["_future_row_exists"] = True

    merged = left.merge(
        future,
        on=[CENTER_COL, SKU_COL, "target_date_expected"],
        how="left",
        validate="one_to_one",
    )
    row_exists = merged["_future_row_exists"].eq(True)
    close_match = np.isclose(
        merged[target_col].to_numpy(dtype=float),
        merged["actual_future_qty"].to_numpy(dtype=float),
        equal_nan=True,
    )

    n_mismatch = int((row_exists & ~close_match).sum())
    # 미래 주차가 없는데 target이 있으면 잘못 생성된 값
    n_orphan = int((~row_exists & merged[target_col].notna()).sum())

    return {
        "horizon": h,
        "n_verifiable_rows": int(row_exists.sum()),
        "n_mismatch_rows": n_mismatch,
        "n_orphan_nonnull_target_no_future_row": n_orphan,
        "pass": n_mismatch == 0 and n_orphan == 0,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_table()

    key_stats = key_uniqueness_check(df)
    horizon_stats = (
        {h: self_join_validate_horizon(df, h) for h in HORIZONS}
        if key_stats["pass"]
        else {
            h: {
                "horizon": h,
                "n_verifiable_rows": 0,
                "n_mismatch_rows": np.nan,
                "n_orphan_nonnull_target_no_future_row": np.nan,
                "pass": False,
            }
            for h in HORIZONS
        }
    )

    target_audit_gate = key_stats["pass"] and all(s["pass"] for s in horizon_stats.values())
    rows = [{"section": "key_uniqueness", "horizon": None, **key_stats}]
    rows += [{"section": "self_join_horizon", **s} for s in horizon_stats.values()]
    rows.append({
        "section": "runtime_log_policy",
        "horizon": None,
        "runtime_log1p_zero_is_zero": bool(np.log1p(0.0) == 0.0),
        "legacy_log_target_used": False,
    })
    rows.append({"section": "final_gate", "horizon": None, "target_audit_gate": target_audit_gate})
    pd.DataFrame(rows).to_csv(AUDIT_OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"[key] duplicate_rows={key_stats['n_duplicate_rows']:,} PASS={key_stats['pass']}")
    for h, s in horizon_stats.items():
        print(
            f"[h{h}] verifiable={s['n_verifiable_rows']:,} mismatch={s['n_mismatch_rows']} "
            f"orphan={s['n_orphan_nonnull_target_no_future_row']} PASS={s['pass']}"
        )
    print(f"target_audit_gate: {'PASS' if target_audit_gate else 'FAIL'}")


if __name__ == "__main__":
    main()
