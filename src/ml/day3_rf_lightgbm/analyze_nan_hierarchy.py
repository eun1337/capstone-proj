"""
analyze_nan_hierarchy.py
RF trainer의 5개 구조적 NaN feature에 대해 KAN 계층형(소/중/대분류) median imputation이
의미 있는지 판단하기 위한 실데이터 분석. P10/P13 각 fold의 train subset만 사용하며,
imputation이나 모델 학습은 하지 않는다.
"""

import numpy as np
import pandas as pd

from src.ml.day3_rf_lightgbm.common import config as cfg
from src.ml.day3_rf_lightgbm.common import folds as f
from src.ml.day3_rf_lightgbm.common.data_loader import load_development, load_holdout_2024

NAN_FEATURES = (
    "adi_expanding_filled",
    "cv2_expanding_filled",
    "qty_lag1_filled_log1p",
    "qty_rollmean_4_filled_log1p",
    "qty_rollstd_4_filled_log1p",
)
CAT_LEVELS = ("KAN_소분류", "KAN_중분류", "KAN_대분류")
STAGES = (("P10", 2022), ("P13", 2023))


def fold_feature_rows(train_df: pd.DataFrame, feature: str, stage: str, horizon: int, fold_no: int) -> pd.DataFrame:
    nan_mask = train_df[feature].isna()
    nan_rows = train_df.loc[nan_mask]
    if len(nan_rows) == 0:
        return pd.DataFrame()

    non_nan = train_df.loc[~nan_mask]
    out = pd.DataFrame({
        "stage": stage, "horizon": horizon, "fold": fold_no, "feature": feature,
        "coldstart_flag": nan_rows["coldstart_flag"].to_numpy(),
        "is_warmup": nan_rows["is_warmup"].to_numpy(),
    }, index=nan_rows.index)

    for level, short in zip(CAT_LEVELS, ("sub", "mid", "large")):
        grp = non_nan.groupby(level, observed=True)[feature]
        cnt_map = grp.size()
        med_map = grp.median()
        out[f"count_{short}"] = nan_rows[level].map(cnt_map).fillna(0).astype(int).to_numpy()
        out[f"median_{short}"] = nan_rows[level].map(med_map).to_numpy()

    out["median_overall"] = non_nan[feature].median()
    out["n_train_non_nan"] = len(non_nan)
    return out


def collect_all(dev: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for stage, validation_year in STAGES:
        for h in (1, 2, 4):
            for fd in f.generate_expanding_folds(dev, validation_year, h):
                train_df = dev.loc[fd["train_mask"]]
                for feature in NAN_FEATURES:
                    part = fold_feature_rows(train_df, feature, stage, h, fd["fold"])
                    if len(part):
                        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def summarize_counts_by_feature_stage(all_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in NAN_FEATURES:
        for stage in ("P10", "P13"):
            sub = all_rows[(all_rows["feature"] == feature) & (all_rows["stage"] == stage)]
            if len(sub) == 0:
                continue
            n_fold_combos = sub[["horizon", "fold"]].drop_duplicates().shape[0]
            rows.append({
                "feature": feature, "stage": stage,
                "n_fold_combos": n_fold_combos,
                "n_nan_rows_pooled": len(sub),
                "nan_rows_per_combo_avg": round(len(sub) / n_fold_combos, 1),
            })
    return pd.DataFrame(rows)


def summarize_peer_count_distribution(all_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in NAN_FEATURES:
        for stage in ("P10", "P13"):
            sub = all_rows[(all_rows["feature"] == feature) & (all_rows["stage"] == stage)]
            if len(sub) == 0:
                continue
            for short, label in (("sub", "KAN_소분류"), ("mid", "KAN_중분류"), ("large", "KAN_대분류")):
                s = sub[f"count_{short}"]
                rows.append({
                    "feature": feature, "stage": stage, "level": label,
                    "min": int(s.min()), "p25": int(s.quantile(0.25)),
                    "median": int(s.median()), "p75": int(s.quantile(0.75)), "max": int(s.max()),
                })
    return pd.DataFrame(rows)


def summarize_usable_ratio(all_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in NAN_FEATURES:
        for stage in ("P10", "P13"):
            sub = all_rows[(all_rows["feature"] == feature) & (all_rows["stage"] == stage)]
            n = len(sub)
            if n == 0:
                continue
            sub_ok = sub["count_sub"] > 0
            mid_needed = (~sub_ok) & (sub["count_mid"] > 0)
            large_needed = (~sub_ok) & (~mid_needed) & (sub["count_large"] > 0)
            overall_needed = (~sub_ok) & (~mid_needed) & (~large_needed)
            rows.append({
                "feature": feature, "stage": stage, "n_nan_rows": n,
                "소분류_usable_ratio": round(sub_ok.mean(), 4),
                "중분류_필요_ratio": round(mid_needed.mean(), 4),
                "대분류_필요_ratio": round(large_needed.mean(), 4),
                "전체fallback_필요_ratio": round(overall_needed.mean(), 4),
            })
    return pd.DataFrame(rows)


def summarize_median_gap(all_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in NAN_FEATURES:
        sub = all_rows[all_rows["feature"] == feature]
        usable = sub[sub["count_sub"] > 0]
        if len(usable) == 0:
            continue
        gap_sub_overall = (usable["median_sub"] - usable["median_overall"]).abs()
        gap_sub_large = (usable["median_sub"] - usable["median_large"]).abs()
        rows.append({
            "feature": feature,
            "n_소분류usable_rows": len(usable),
            "|소분류median - 전체median|_median": round(gap_sub_overall.median(), 4),
            "|소분류median - 전체median|_p75": round(gap_sub_overall.quantile(0.75), 4),
            "|소분류median - 대분류median|_median": round(gap_sub_large.median(), 4),
            "전체median_값": round(usable["median_overall"].iloc[0], 4),
        })
    return pd.DataFrame(rows)


def summarize_coldstart_ratio(all_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in NAN_FEATURES:
        sub = all_rows[all_rows["feature"] == feature]
        rows.append({
            "feature": feature,
            "n_nan_rows_pooled": len(sub),
            "coldstart_flag==1_ratio": round((sub["coldstart_flag"] == 1).mean(), 4),
            "is_warmup==1_ratio": round((sub["is_warmup"] == 1).mean(), 4),
            "coldstart또는warmup_ratio": round(((sub["coldstart_flag"] == 1) | (sub["is_warmup"] == 1)).mean(), 4),
        })
    return pd.DataFrame(rows)


def fit_hierarchical_median(train_df: pd.DataFrame, feature: str) -> dict:
    """train_df만으로 소/중/대분류 + 전체 median 맵을 만든다(train 밖 데이터 미사용)."""
    non_nan = train_df.loc[train_df[feature].notna()]
    return {
        "sub": non_nan.groupby("KAN_소분류", observed=True)[feature].median(),
        "mid": non_nan.groupby("KAN_중분류", observed=True)[feature].median(),
        "large": non_nan.groupby("KAN_대분류", observed=True)[feature].median(),
        "overall": non_nan[feature].median(),
    }


def apply_hierarchical_median(df: pd.DataFrame, feature: str, maps: dict) -> tuple[pd.Series, pd.Series]:
    """maps(train 기준)로 df[feature]의 NaN을 소->중->대->전체 순으로 채운다. df 원본은 수정하지 않는다.
    반환: (채워진 값 Series, 각 행이 해소된 단계 Series[already_valid/sub/mid/large/global/unresolved])."""
    values = df[feature].copy()
    resolved_at = pd.Series(np.where(values.notna(), "already_valid", "unresolved"), index=df.index)

    for level_col, level_key in (("KAN_소분류", "sub"), ("KAN_중분류", "mid"), ("KAN_대분류", "large")):
        still_nan = values.isna()
        if not still_nan.any():
            break
        candidate = df[level_col].map(maps[level_key])
        fillable = still_nan & candidate.notna()
        values.loc[fillable] = candidate.loc[fillable]
        resolved_at.loc[fillable] = level_key

    still_nan = values.isna()
    if still_nan.any():
        values.loc[still_nan] = maps["overall"]
        resolved_at.loc[still_nan] = "global"

    return values, resolved_at


def summarize_resolution(resolved_at: pd.Series, original_nan_mask: pd.Series) -> dict:
    counts = resolved_at[original_nan_mask].value_counts()
    return {
        "n_original_nan": int(original_nan_mask.sum()),
        "resolved_sub": int(counts.get("sub", 0)),
        "resolved_mid": int(counts.get("mid", 0)),
        "resolved_large": int(counts.get("large", 0)),
        "resolved_global": int(counts.get("global", 0)),
        "unresolved": int(counts.get("unresolved", 0)),
    }


def _zero_summary() -> dict:
    return {"n_original_nan": 0, "resolved_sub": 0, "resolved_mid": 0,
            "resolved_large": 0, "resolved_global": 0, "unresolved": 0}


def _add(acc: dict, s: dict) -> None:
    for k in acc:
        acc[k] += s[k]


def check_b_robustness(dev: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """B walk-forward 전체 fold에서 train-fit 계층형 median이 train/validation의 NaN을
    (validation 자체 통계량 없이) 얼마나 해소하는지 horizon별로 집계한다."""
    rows = []
    problem_folds = []
    for h in cfg.HORIZONS:
        train_acc, val_acc = _zero_summary(), _zero_summary()
        for fd in f.generate_b_walkforward_folds(dev, h):
            train_df = dev.loc[fd["train_mask"]]
            val_df = dev.loc[fd["val_mask"]]
            fold_unresolved = 0
            for feature in NAN_FEATURES:
                maps = fit_hierarchical_median(train_df, feature)
                _, train_resolved = apply_hierarchical_median(train_df, feature, maps)
                _, val_resolved = apply_hierarchical_median(val_df, feature, maps)
                t = summarize_resolution(train_resolved, train_df[feature].isna())
                v = summarize_resolution(val_resolved, val_df[feature].isna())
                _add(train_acc, t)
                _add(val_acc, v)
                fold_unresolved += t["unresolved"] + v["unresolved"]
            if fold_unresolved > 0:
                problem_folds.append({"horizon": h, "fold": fd["fold"], "unresolved": fold_unresolved})
        rows.append({"구간": "B_train", "horizon": f"h{h}", "original_nan": train_acc["n_original_nan"],
                     "sub": train_acc["resolved_sub"], "mid": train_acc["resolved_mid"],
                     "large": train_acc["resolved_large"], "global": train_acc["resolved_global"],
                     "unresolved": train_acc["unresolved"]})
        rows.append({"구간": "B_val", "horizon": f"h{h}", "original_nan": val_acc["n_original_nan"],
                     "sub": val_acc["resolved_sub"], "mid": val_acc["resolved_mid"],
                     "large": val_acc["resolved_large"], "global": val_acc["resolved_global"],
                     "unresolved": val_acc["unresolved"]})
    return pd.DataFrame(rows), problem_folds


def check_final_holdout(dev: pd.DataFrame, holdout: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Final Train에서만 계층형 median을 만들어 2024 Holdout에 적용했을 때의 해소 현황과
    Final Train에 없던 2024 KAN 카테고리 수를 horizon별로 집계한다."""
    rows = []
    unseen_rows = []
    for h in cfg.HORIZONS:
        final_train = dev.loc[f.final_retrain_mask(dev, h)]
        hold_df = holdout.loc[f.holdout_2024_mask(holdout, h)]

        for level in CAT_LEVELS:
            unseen_n = len(set(hold_df[level].unique()) - set(final_train[level].unique()))
            unseen_rows.append({"horizon": f"h{h}", "level": level, "n_unseen_categories": unseen_n})

        train_acc, hold_acc = _zero_summary(), _zero_summary()
        for feature in NAN_FEATURES:
            maps = fit_hierarchical_median(final_train, feature)
            _, train_resolved = apply_hierarchical_median(final_train, feature, maps)
            _, hold_resolved = apply_hierarchical_median(hold_df, feature, maps)
            _add(train_acc, summarize_resolution(train_resolved, final_train[feature].isna()))
            _add(hold_acc, summarize_resolution(hold_resolved, hold_df[feature].isna()))

        rows.append({"구간": "FinalTrain", "horizon": f"h{h}", "original_nan": train_acc["n_original_nan"],
                     "sub": train_acc["resolved_sub"], "mid": train_acc["resolved_mid"],
                     "large": train_acc["resolved_large"], "global": train_acc["resolved_global"],
                     "unresolved": train_acc["unresolved"]})
        rows.append({"구간": "Holdout2024", "horizon": f"h{h}", "original_nan": hold_acc["n_original_nan"],
                     "sub": hold_acc["resolved_sub"], "mid": hold_acc["resolved_mid"],
                     "large": hold_acc["resolved_large"], "global": hold_acc["resolved_global"],
                     "unresolved": hold_acc["unresolved"]})
    return pd.DataFrame(rows), pd.DataFrame(unseen_rows)


def main() -> None:
    dev = load_development()
    all_rows = collect_all(dev)

    print("=" * 100)
    print("[1] feature x stage별 NaN 행 수 (fold-train 기준, pooled)")
    print(summarize_counts_by_feature_stage(all_rows).to_string(index=False))

    print("=" * 100)
    print("[2] 계층별 non-NaN peer 수 분포 (min/p25/median/p75/max)")
    print(summarize_peer_count_distribution(all_rows).to_string(index=False))

    print("=" * 100)
    print("[3] 계층별 usable 비율")
    print(summarize_usable_ratio(all_rows).to_string(index=False))

    print("=" * 100)
    print("[4] 계층별 median 값 차이 (소분류 usable한 행 기준)")
    print(summarize_median_gap(all_rows).to_string(index=False))

    print("=" * 100)
    print("[5] NaN 행의 coldstart_flag/is_warmup 비율")
    print(summarize_coldstart_ratio(all_rows).to_string(index=False))

    print("=" * 100)
    print("[6] B robustness 계층형 median 적용 가능성 (전체 fold, validation은 train-fit map만 사용)")
    b_df, b_problems = check_b_robustness(dev)
    print(b_df.to_string(index=False))
    print(f"  문제 fold(unresolved>0) 개수: {len(b_problems)}")
    if b_problems:
        print(pd.DataFrame(b_problems).to_string(index=False))

    print("=" * 100)
    print("[7] Final Train -> 2024 Holdout 계층형 median 적용 가능성 (2024 자체 통계량 미사용)")
    holdout = load_holdout_2024()
    fh_df, unseen_df = check_final_holdout(dev, holdout)
    print(fh_df.to_string(index=False))
    print("  Final Train에 없던 2024 KAN 카테고리:")
    print(unseen_df.to_string(index=False))

    a_usable_100pct = bool((summarize_usable_ratio(all_rows)["소분류_usable_ratio"] == 1.0).all())
    print("=" * 100)
    print("[최종 판정]")
    print(f"  A P10/P13 소분류 usable 100%: {a_usable_100pct}")
    print(f"  B robustness unresolved==0: {bool((b_df['unresolved'] == 0).all())}")
    print(f"  Final Train unresolved==0: {bool((fh_df.loc[fh_df['구간'] == 'FinalTrain', 'unresolved'] == 0).all())}")
    print(f"  2024 Holdout unresolved==0: {bool((fh_df.loc[fh_df['구간'] == 'Holdout2024', 'unresolved'] == 0).all())}")


if __name__ == "__main__":
    main()
