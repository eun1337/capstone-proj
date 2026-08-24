"""
nan_audit.py

Day 1 Step 3: 최종 Feature Table의 NaN 최소 필수 검증.

검사:
    - 전체 컬럼의 NaN 현황
    - _filled 계열의 잔여 NaN과 cold-start/warm-up 관계
    - P22 모델별 실제 입력 X와 target y의 구조적/residual NaN
    - 구조적 NaN 처리 및 fold-train-only imputation 원칙
"""

import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parents[5]
FEATURE_TABLE_PATH = BASE_DIR / "data" / "final_feature_table.parquet"
OUT_DIR = BASE_DIR / "data" / "ml" / "experiments" / "day1_audit"

CENTER_COL, SKU_COL, WEEK_COL = "center_id", "sku_id", "week_st"
HORIZONS = [1, 2, 4]
FINAL_TARGET_COLS = ["target_h1", "target_h2", "target_h4"]

P22_COMMON_FEATURES = [
    "center_id", "입수", "KAN_소분류", "ISO_주차", "평균온도", "총강수량",
    "existed_before_regime", "월", "분기", "is_warmup", "coldstart_flag",
    "adi_expanding_filled", "cv2_expanding_filled", "강수량_호우_count", "covid_flag",
    "center_is_B", "temp_x_precip", "center_temp_inter", "weeks_since_last_active_filled",
    "ccsi_lag_m1", "cpi_y1_prev", "cpi_y2_prev_yoy",
]
TREE_EXTRA_FEATURES = [
    "qty_lag1_filled_log1p", "qty_rollmean_4_filled_log1p", "qty_rollstd_4_filled_log1p"
]
# DL은 저장 qty_log1p가 아니라 runtime np.log1p(qty)를 사용
SEQ_EXTRA_FEATURES = ["qty"]
MODEL_KIND = {"RF": "tree", "LightGBM": "tree", "LSTM": "seq", "TFT": "seq", "Informer": "seq"}

DEMAND_HISTORY_FEATURES = {
    "adi_expanding_filled", "cv2_expanding_filled", "weeks_since_last_active_filled",
    "qty_lag1_filled_log1p", "qty_rollmean_4_filled_log1p", "qty_rollstd_4_filled_log1p",
}
ECONOMIC_FEATURES = {"ccsi_lag_m1", "cpi_y1_prev", "cpi_y2_prev_yoy"}


def holiday_features(h: int) -> list[str]:
    return [f"target_h{h}_공휴일_W0", f"target_h{h}_공휴일_W-1", f"target_h{h}_공휴일_W+1"]


def model_feature_set(model: str, h: int) -> list[str]:
    extra = TREE_EXTRA_FEATURES if MODEL_KIND[model] == "tree" else SEQ_EXTRA_FEATURES
    return P22_COMMON_FEATURES + holiday_features(h) + extra


def p22_needed_columns() -> list[str]:
    cols = set(P22_COMMON_FEATURES) | set(TREE_EXTRA_FEATURES) | set(SEQ_EXTRA_FEATURES) | set(FINAL_TARGET_COLS)
    for h in HORIZONS:
        cols.update(holiday_features(h))
    cols.update([CENTER_COL, SKU_COL, WEEK_COL, "sold_flag"])
    return sorted(cols)


def full_nan_summary() -> pd.DataFrame:
    """전체 컬럼 NaN 개수/비율을 batch 단위로 집계해 메모리 사용을 제한."""
    parquet = pq.ParquetFile(FEATURE_TABLE_PATH)
    columns = parquet.schema.names
    counts = pd.Series(0, index=columns, dtype="int64")
    n_rows = 0

    for batch in parquet.iter_batches(columns=columns, batch_size=100_000):
        part = batch.to_pandas()
        counts = counts.add(part.isna().sum(), fill_value=0).astype("int64")
        n_rows += len(part)

    return pd.DataFrame({
        "feature": columns,
        "n_rows": n_rows,
        "n_nan": [int(counts[c]) for c in columns],
        "nan_pct": [round(int(counts[c]) / n_rows * 100, 4) for c in columns],
    })


def load_p22_table() -> pd.DataFrame:
    df = pd.read_parquet(FEATURE_TABLE_PATH, columns=p22_needed_columns())
    print(f"[로드] {FEATURE_TABLE_PATH} -> {len(df):,}행")
    return df


def build_references(df: pd.DataFrame) -> dict:
    max_week_by_center = df.groupby(CENTER_COL, observed=True)[WEEK_COL].max()
    first_sale = (
        df.loc[df["sold_flag"] == 1]
        .groupby([CENTER_COL, SKU_COL], observed=True)[WEEK_COL]
        .min()
    )
    first_sale_frame = df[[CENTER_COL, SKU_COL, WEEK_COL]].join(
        first_sale.rename("first_sale_week"), on=[CENTER_COL, SKU_COL]
    )
    no_prior_sale = (
        (first_sale_frame[WEEK_COL] < first_sale_frame["first_sale_week"])
        | first_sale_frame["first_sale_week"].isna()
    )
    return {
        "coldstart": df["coldstart_flag"].fillna(False) | no_prior_sale,
        "warmup": df["is_warmup"].fillna(False),
        "max_week_by_center": max_week_by_center,
    }


def economic_history_prefix_mask(df: pd.DataFrame, nan_mask: pd.Series) -> pd.Series:
    """경제지표의 데이터 초반 연속 NaN 구간만 history 부족으로 인정."""
    false_mask = pd.Series(False, index=df.index)
    if not nan_mask.any():
        return false_mask

    weekly = pd.DataFrame({"week_st": df[WEEK_COL], "is_nan": nan_mask}).groupby(
        "week_st", observed=True
    )["is_nan"].agg(all_nan="all", any_nan="any").sort_index()
    if weekly.empty or weekly["all_nan"].all():
        return false_mask

    prefix_weeks = weekly.index[weekly["all_nan"].cumprod().astype(bool)]
    return df[WEEK_COL].isin(prefix_weeks) if len(prefix_weeks) else false_mask


def strict_explained_mask(df: pd.DataFrame, refs: dict, col: str, nan_mask: pd.Series) -> pd.Series:
    target_match = re.match(r"^target_h([124])$", col)
    if target_match:
        h = int(target_match.group(1))
        return (df[WEEK_COL] + pd.Timedelta(weeks=h)) > df[CENTER_COL].map(refs["max_week_by_center"])
    if col in DEMAND_HISTORY_FEATURES:
        return refs["coldstart"] | refs["warmup"]
    if col in ECONOMIC_FEATURES:
        return economic_history_prefix_mask(df, nan_mask)
    return pd.Series(False, index=df.index)


def feature_stat(df: pd.DataFrame, refs: dict, col: str) -> tuple[int, int, int]:
    nan_mask = df[col].isna()
    n_nan = int(nan_mask.sum())
    if n_nan == 0:
        return 0, 0, 0
    explained = strict_explained_mask(df, refs, col, nan_mask)
    n_structural = int((nan_mask & explained).sum())
    return n_nan, n_structural, n_nan - n_structural


def filled_column_audit(df: pd.DataFrame, refs: dict) -> pd.DataFrame:
    filled_cols = [c for c in df.columns if "_filled" in c]
    rows = []
    for col in filled_cols:
        n_nan, structural, residual = feature_stat(df, refs, col)
        rows.append({
            "feature": col,
            "n_nan": n_nan,
            "coldstart_overlap": int((df[col].isna() & refs["coldstart"]).sum()),
            "warmup_overlap": int((df[col].isna() & refs["warmup"]).sum()),
            "structural_nan_count": structural,
            "residual_nan_count": residual,
            "status": "PASS" if residual == 0 else "REVIEW",
        })
    return pd.DataFrame(rows)


def display_feature_name(col: str, model: str) -> str:
    return "qty_log1p_runtime" if col == "qty" and MODEL_KIND[model] == "seq" else col


def model_feature_nan_audit(df: pd.DataFrame, refs: dict) -> pd.DataFrame:
    rows = []
    for model in MODEL_KIND:
        for h in HORIZONS:
            for col in model_feature_set(model, h):
                n_nan, structural, residual = feature_stat(df, refs, col)
                rows.append({
                    "model": model,
                    "horizon": f"h{h}",
                    "feature": display_feature_name(col, model),
                    "source_column": col,
                    "n_rows": len(df),
                    "n_nan": n_nan,
                    "nan_pct": round(n_nan / len(df) * 100, 4),
                    "structural_nan_count": structural,
                    "residual_nan_count": residual,
                    "status": "PASS" if residual == 0 else "REVIEW",
                })
    return pd.DataFrame(rows)


def target_nan_audit(df: pd.DataFrame, refs: dict) -> pd.DataFrame:
    rows = []
    for col in FINAL_TARGET_COLS:
        n_nan, structural, residual = feature_stat(df, refs, col)
        rows.append({
            "model": "TARGET_Y",
            "horizon": col.replace("target_", ""),
            "feature": col,
            "source_column": col,
            "n_rows": len(df),
            "n_nan": n_nan,
            "nan_pct": round(n_nan / len(df) * 100, 4),
            "structural_nan_count": structural,
            "residual_nan_count": residual,
            "status": "PASS" if residual == 0 else "REVIEW",
        })
    return pd.DataFrame(rows)


def model_nan_gate(feature_audit_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in feature_audit_df.groupby("model", sort=False):
        for horizon in sorted(group["horizon"].unique()):
            subset = group[group["horizon"] == horizon]
            residual = int(subset["residual_nan_count"].sum())
            rows.append({
                "model": model,
                "horizon": horizon,
                "n_features": int(subset["feature"].nunique()),
                "n_features_with_nan": int((subset["n_nan"] > 0).sum()),
                "total_nan_count": int(subset["n_nan"].sum()),
                "residual_nan_total": residual,
                "data_quality_gate": "PASS" if residual == 0 else "REVIEW",
            })
        unique = group.sort_values("horizon").drop_duplicates(subset=["feature"])
        residual = int(unique["residual_nan_count"].sum())
        rows.append({
            "model": model,
            "horizon": "ALL",
            "n_features": int(unique["feature"].nunique()),
            "n_features_with_nan": int((unique["n_nan"] > 0).sum()),
            "total_nan_count": int(unique["n_nan"].sum()),
            "residual_nan_total": residual,
            "data_quality_gate": "PASS" if residual == 0 else "REVIEW",
        })
    return pd.DataFrame(rows)


POLICY_BY_CAUSE = {
    "cold_start": ("row 보존, 강제 0/평균 대체 금지", False, "구조적 결측으로 유지"),
    "warm_up": ("필요 시 해당 CV fold train에서만 대체값 계산", True, "fold 밖 통계 사용 금지"),
    "economic_history": ("과거 자료 자체가 없는 초기 구간은 NaN 유지", False, "임의 대체 금지"),
    "right_censoring": ("데이터 마지막 이후 target은 학습에서 제외", False, "미관측 미래값"),
    "mixed_structural": ("구성 원인별 정책 적용", True, "warm-up 포함 시 fold-local"),
    "unexplained": ("원인 확인 전 REVIEW 유지", False, "확정 전 미대체"),
}


def dominant_cause(df: pd.DataFrame, refs: dict, col: str) -> str:
    if re.match(r"^target_h([124])$", col):
        return "right_censoring"

    nan_mask = df[col].isna()
    n_nan = int(nan_mask.sum())
    if n_nan == 0:
        return "no_nan"
    if col in ECONOMIC_FEATURES:
        explained = economic_history_prefix_mask(df, nan_mask)
        return "economic_history" if int((nan_mask & explained).sum()) == n_nan else "unexplained"
    if col in DEMAND_HISTORY_FEATURES:
        warm = int((nan_mask & refs["warmup"]).sum())
        cold = int((nan_mask & refs["coldstart"]).sum())
        if warm == n_nan:
            return "warm_up"
        if cold == n_nan:
            return "cold_start"
        if int((nan_mask & (refs["warmup"] | refs["coldstart"])).sum()) == n_nan:
            return "mixed_structural"
    return "unexplained"


def build_nan_policy(df: pd.DataFrame, refs: dict, model_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    features = pd.concat([
        model_df[["feature", "source_column", "structural_nan_count", "residual_nan_count"]],
        target_df[["feature", "source_column", "structural_nan_count", "residual_nan_count"]],
    ], ignore_index=True).drop_duplicates(subset=["feature"])

    rows = []
    for _, row in features[features["structural_nan_count"] > 0].iterrows():
        cause = dominant_cause(df, refs, row["source_column"])
        handling, fold_local, leakage_risk = POLICY_BY_CAUSE[cause]
        rows.append({
            "feature": row["feature"],
            "cause": cause,
            "structural_nan_count": int(row["structural_nan_count"]),
            "residual_nan_count": int(row["residual_nan_count"]),
            "recommended_handling": handling,
            "fold_local_fit_required": fold_local,
            "leakage_risk": leakage_risk,
        })
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = full_nan_summary()
    summary_df.to_csv(OUT_DIR / "nan_feature_summary.csv", index=False, encoding="utf-8-sig")

    df = load_p22_table()
    refs = build_references(df)

    filled_df = filled_column_audit(df, refs)
    filled_df.to_csv(OUT_DIR / "filled_column_audit.csv", index=False, encoding="utf-8-sig")

    model_df = model_feature_nan_audit(df, refs)
    target_df = target_nan_audit(df, refs)
    model_df.to_csv(OUT_DIR / "model_feature_nan_audit.csv", index=False, encoding="utf-8-sig")

    gate_df = pd.concat([model_nan_gate(model_df), model_nan_gate(target_df)], ignore_index=True)
    gate_df.to_csv(OUT_DIR / "model_nan_gate.csv", index=False, encoding="utf-8-sig")

    policy_df = build_nan_policy(df, refs, model_df, target_df)
    policy_df.to_csv(OUT_DIR / "cold_start_nan_policy.csv", index=False, encoding="utf-8-sig")

    n_nan_cols = int((summary_df["n_nan"] > 0).sum())
    step3_status = "PASS" if gate_df["data_quality_gate"].eq("PASS").all() else "REVIEW"

    print(f"전체 컬럼={len(summary_df)}, NaN 존재 컬럼={n_nan_cols}")
    print(f"_filled 계열 residual 합계={int(filled_df['residual_nan_count'].sum()) if len(filled_df) else 0}")
    print(f"Step3 최종 상태(P22 기준): {step3_status}")


if __name__ == "__main__":
    main()
