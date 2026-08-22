"""
audit_features.py
Day3 RF/LightGBM Freeze된 30개 Feature Set의 dtype/NaN/Inf/target/qty 실데이터 audit.
config.get_model_feature_cols()와 folds.generate_expanding_folds()를 그대로 사용하며
전처리/encoding/학습은 하지 않는다.
"""

import numpy as np
import pandas as pd

from src.ml.day3_rf_lightgbm.common import config as cfg
from src.ml.day3_rf_lightgbm.common import folds as f
from src.ml.day3_rf_lightgbm.common.data_loader import load_development

CATEGORICAL_COLS = list(cfg.CATEGORICAL_FEATURES)
ALL_FEATURE_COLS = sorted(set().union(*(cfg.get_model_feature_cols(h) for h in cfg.HORIZONS)))


def audit_feature_set_structure() -> tuple[pd.DataFrame, bool]:
    rows = []
    common27 = {}
    for h in cfg.HORIZONS:
        feats = list(cfg.get_model_feature_cols(h))
        other_holiday = [c for hh in cfg.HORIZONS if hh != h for c in cfg.HOLIDAY_FEATURES[hh]]
        common27[h] = frozenset(feats) - set(cfg.HOLIDAY_FEATURES[h])
        checks = {
            "n_features==30": len(feats) == 30,
            "no_duplicates": len(set(feats)) == len(feats),
            "center_id_excluded": "center_id" not in feats,
            "sku_id_excluded": "sku_id" not in feats,
            "KAN_CODE_excluded": "KAN_CODE" not in feats,
            "center_is_B_included": "center_is_B" in feats,
            "KAN_대분류_included": "KAN_대분류" in feats,
            "KAN_중분류_included": "KAN_중분류" in feats,
            "KAN_소분류_included": "KAN_소분류" in feats,
            "own_holiday_included": set(cfg.HOLIDAY_FEATURES[h]).issubset(feats),
            "no_other_horizon_holiday": not any(c in feats for c in other_holiday),
        }
        rows.append({"horizon": f"h{h}", **checks, "verdict": "PASS" if all(checks.values()) else "FAIL"})
    common27_equal = len(set(common27.values())) == 1
    return pd.DataFrame(rows), common27_equal


def audit_dtypes(dev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ALL_FEATURE_COLS:
        s = dev[col]
        rows.append({
            "feature": col,
            "dtype": str(s.dtype),
            "is_numeric_or_bool": pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s),
        })
    return pd.DataFrame(rows)


def audit_center_is_b(dev: pd.DataFrame) -> dict:
    s = dev["center_is_B"]
    return {
        "dtype": str(s.dtype),
        "is_numeric": pd.api.types.is_numeric_dtype(s),
        "unique_values": sorted(s.unique().tolist()),
    }


def audit_nan(dev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ALL_FEATURE_COLS:
        s = dev[col]
        nan_mask = s.isna()
        n_nan = int(nan_mask.sum())
        row = {"feature": col, "nan_count": n_nan, "nan_ratio": n_nan / len(dev)}
        if n_nan > 0:
            sub = dev.loc[nan_mask]
            row["centers"] = sorted(sub["center_id"].unique().tolist())
            row["week_st_min"] = sub["week_st"].min()
            row["week_st_max"] = sub["week_st"].max()
        rows.append(row)
    return pd.DataFrame(rows)


def audit_inf(dev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ALL_FEATURE_COLS:
        s = dev[col]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        arr = s.to_numpy(dtype=float)
        pos_inf = int(np.isposinf(arr).sum())
        neg_inf = int(np.isneginf(arr).sum())
        if pos_inf or neg_inf:
            rows.append({"feature": col, "pos_inf": pos_inf, "neg_inf": neg_inf})
    return pd.DataFrame(rows)


def audit_categorical(dev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CATEGORICAL_COLS:
        s = dev[col]
        empty = int((s.notna() & (s.astype(str).str.strip() == "")).sum())
        rows.append({
            "feature": col, "dtype": str(s.dtype),
            "nan_count": int(s.isna().sum()),
            "unique_count": int(s.nunique(dropna=True)),
            "empty_string_count": empty,
        })
    return pd.DataFrame(rows)


def audit_target(dev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in cfg.HORIZONS:
        col = cfg.TARGET_COLS[h]
        s = dev[col]
        arr = s.to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        rows.append({
            "target": col, "dtype": str(s.dtype),
            "nan_count": int(np.isnan(arr).sum()), "inf_count": int(np.isinf(arr).sum()),
            "negative_count": int((finite < 0).sum()),
            "min": float(finite.min()) if len(finite) else np.nan,
            "max": float(finite.max()) if len(finite) else np.nan,
        })
    return pd.DataFrame(rows)


def audit_qty(dev: pd.DataFrame) -> dict:
    s = dev["qty"]
    arr = s.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    return {
        "dtype": str(s.dtype),
        "nan_count": int(np.isnan(arr).sum()),
        "pos_inf_count": int(np.isposinf(arr).sum()),
        "neg_inf_count": int(np.isneginf(arr).sum()),
        "negative_count": int((finite < 0).sum()),
        "min": float(finite.min()) if len(finite) else np.nan,
        "max": float(finite.max()) if len(finite) else np.nan,
    }


def _fold_x_audit(df: pd.DataFrame, feature_cols: list[str]) -> tuple[int, int]:
    X = df[feature_cols]
    nan_count = int(X.isna().sum().sum())
    num = X.select_dtypes(include=[np.number])
    inf_count = int(np.isinf(num.to_numpy(dtype=float)).sum()) if num.shape[1] else 0
    return nan_count, inf_count


def audit_expanding_folds(dev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage, validation_year in (("P10", 2022), ("P13", 2023)):
        for h in cfg.HORIZONS:
            feature_cols = list(cfg.get_model_feature_cols(h))
            target_col = cfg.TARGET_COLS[h]
            for fd in f.generate_expanding_folds(dev, validation_year, h):
                train_df = dev.loc[fd["train_mask"]]
                val_df = dev.loc[fd["val_mask"]]

                train_nan, train_inf = _fold_x_audit(train_df, feature_cols)
                val_nan, val_inf = _fold_x_audit(val_df, feature_cols)

                val_target = val_df[target_col].to_numpy(dtype=float)
                val_target_nan = int(np.isnan(val_target).sum())
                val_target_inf = int(np.isinf(val_target[~np.isnan(val_target)]).sum())

                rows.append({
                    "stage": stage, "horizon": f"h{h}", "fold": fd["fold"],
                    "n_train": len(train_df), "n_val": len(val_df),
                    "train_X_nan": train_nan, "train_X_inf": train_inf,
                    "val_X_nan": val_nan, "val_X_inf": val_inf,
                    "val_target_nan": val_target_nan, "val_target_inf": val_target_inf,
                })
    return pd.DataFrame(rows)


def main() -> None:
    dev = load_development()

    print("=" * 100)
    print("[1] Feature Set 구조 확인")
    structure_df, common27_equal = audit_feature_set_structure()
    print(structure_df.to_string(index=False))
    print(f"h1/h2/h4 공통 27개 feature 동일: {common27_equal}")

    print("=" * 100)
    print("[2] dtype audit (30개 feature 합집합)")
    print(audit_dtypes(dev).to_string(index=False))
    print(f"center_is_B: {audit_center_is_b(dev)}")

    print("=" * 100)
    print("[3] NaN audit")
    nan_df = audit_nan(dev)
    print(nan_df.to_string(index=False))

    print("=" * 100)
    print("[4] Inf audit (numeric feature만)")
    inf_df = audit_inf(dev)
    print(inf_df.to_string(index=False) if len(inf_df) else "Inf 없음")

    print("=" * 100)
    print("[5] categorical audit (KAN 대/중/소분류)")
    print(audit_categorical(dev).to_string(index=False))

    print("=" * 100)
    print("[6] raw target audit (target_h1/h2/h4, development 전체)")
    print(audit_target(dev).to_string(index=False))

    print("=" * 100)
    print("[7] qty audit (MASE history 원천)")
    print(audit_qty(dev))

    print("=" * 100)
    print("[8] P10/P13 실제 fold audit (generate_expanding_folds 사용)")
    fold_df = audit_expanding_folds(dev)
    print(fold_df.to_string(index=False))


if __name__ == "__main__":
    main()
