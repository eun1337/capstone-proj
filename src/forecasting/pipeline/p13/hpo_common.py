"""
hpo_common.py

P13 HPO 공통 로직.

5개 model family가 동일한 evaluation population에서 HPO되도록 common evaluation key
검증과 OOF filtering을 처리한다. GridSampler 실행 검증과
Bias guardrail → Pooled WAPE → near-tie Worst-fold WAPE 선택 규칙을 제공한다.
"""

from __future__ import annotations

from pathlib import Path

import optuna
import pandas as pd

from src.forecasting.common import evaluator as ev
from src.forecasting.common.config import BIAS_GUARDRAIL_ABS_PCT, NEAR_TIE_WAPE_PCT_POINT

EVAL_KEY_COLS = ("horizon", "fold_id", "center_id", "sku_id", "week_st", "target_date")


def _canonicalize_horizon(series: pd.Series) -> pd.Series:
    """'h1'/'h2'/'h4' 문자열 표기와 1/2/4 숫자 표기가 섞이지 않도록 정수로 통일한다."""

    def _conv(v):
        if isinstance(v, str):
            v = v.strip().lower()
            if v.startswith("h"):
                v = v[1:]
            return int(v)
        return int(v)

    return series.map(_conv).astype(int)


def normalize_key_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """EVAL_KEY_COLS 6개 컬럼만 정규화한 복사본을 반환한다(다른 컬럼은 그대로 유지)."""
    df = df.copy()
    df["horizon"] = _canonicalize_horizon(df["horizon"])
    df["fold_id"] = df["fold_id"].astype(int)
    df["center_id"] = df["center_id"].astype(str)
    df["sku_id"] = df["sku_id"].astype(str)
    df["week_st"] = pd.to_datetime(df["week_st"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    return df


def check_no_duplicate_eval_keys(df: pd.DataFrame, name: str) -> None:
    dup = int(df.duplicated(subset=list(EVAL_KEY_COLS)).sum())
    if dup:
        raise ValueError(f"{name}에 evaluation key{EVAL_KEY_COLS} duplicate {dup}건 존재함")


def load_common_eval_keys(path: Path) -> pd.DataFrame:
    """P13 common evaluation key를 로드하고 key dtype, 필수 컬럼, 중복 여부를 검증한다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"P13 common evaluation key 파일이 존재하지 않음: {path}")

    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if len(df) == 0:
        raise ValueError(f"P13 common evaluation key 파일이 비어 있음: {path}")

    if "fold" in df.columns and "fold_id" not in df.columns:
        df = df.rename(columns={"fold": "fold_id"})

    missing = [c for c in EVAL_KEY_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"P13 common evaluation key 파일에 필수 컬럼 누락: {missing} (path={path})")

    df = normalize_key_dtypes(df)
    check_no_duplicate_eval_keys(df, f"common evaluation key 파일({path})")
    return df


def check_common_keys_match_p13_period(common_keys: pd.DataFrame, folds: list, horizon: int, path: Path) -> None:
    """common evaluation key가 해당 horizon의 2023 P13 validation 기간과 겹치는지 확인한다."""
    horizon_keys = common_keys[common_keys["horizon"] == horizon]
    if len(horizon_keys) == 0:
        raise ValueError(f"h{horizon}: common evaluation key 파일({path})에 horizon={horizon} row가 전혀 없음")

    val_start = min(f["val_start"] for f in folds)
    val_end = max(f["val_end"] for f in folds)
    overlap = horizon_keys["target_date"].between(val_start, val_end)
    if not overlap.any():
        raise ValueError(
            f"h{horizon}: common evaluation key({path})의 target_date가 P13(2023) validation "
            f"구간({val_start.date()}~{val_end.date()})과 전혀 겹치지 않음 - P10(2022) 등 다른 "
            f"stage의 key 파일을 잘못 전달했을 가능성이 높음"
        )


def filter_pooled_oof_by_common_keys(pooled_native_oof: pd.DataFrame, common_keys: pd.DataFrame) -> pd.DataFrame:
    """native pooled OOF를 common evaluation key로 filtering하고 key 무결성을 검증한다."""
    key_cols = list(EVAL_KEY_COLS)
    oof_norm = normalize_key_dtypes(pooled_native_oof)
    check_no_duplicate_eval_keys(oof_norm, "trial pooled native OOF")

    merged = oof_norm.merge(common_keys[key_cols], on=key_cols, how="inner")

    if len(merged) == 0:
        raise ValueError("trial pooled OOF와 common evaluation key의 교집합이 0건임")
    if len(merged) > min(len(oof_norm), len(common_keys)):
        raise ValueError(
            f"merge 결과 row 수가 비정상적으로 증가함(row inflation 의심): "
            f"merged={len(merged)}, oof={len(oof_norm)}, common_keys={len(common_keys)}"
        )
    check_no_duplicate_eval_keys(merged, "common-key filtered OOF")
    return merged


def per_fold_metrics_common(filtered_oof: pd.DataFrame) -> dict:
    """common-key filtered OOF에서 fold별 metric을 다시 계산한다."""
    out = {}
    for fold_id, grp in filtered_oof.groupby("fold_id"):
        out[int(fold_id)] = ev.compute_metrics(
            grp["y_true"].to_numpy(), grp["y_pred"].to_numpy(), grp["mase_scale"].to_numpy(),
        )
    return out


def select_best_trial(trial_summaries: list[dict]) -> dict:
    """Bias guardrail → Pooled WAPE → near-tie Worst-fold WAPE 순으로 P13 trial을 선택한다."""
    eligible = [t for t in trial_summaries if abs(t["pooled_bias"]) <= BIAS_GUARDRAIL_ABS_PCT]
    if not eligible:
        return {
            "selected": None,
            "reason": "no_trial_passed_bias_guardrail",
            "all_trials_sorted_by_wape": sorted(trial_summaries, key=lambda t: t["pooled_wape"]),
        }

    eligible_sorted = sorted(eligible, key=lambda t: t["pooled_wape"])
    best_wape = eligible_sorted[0]["pooled_wape"]
    near_ties = [t for t in eligible_sorted if t["pooled_wape"] - best_wape <= NEAR_TIE_WAPE_PCT_POINT]

    if len(near_ties) == 1:
        return {"selected": near_ties[0], "reason": "unique_min_wape_within_bias_guardrail"}

    selected = min(near_ties, key=lambda t: t["worst_fold_wape"])
    return {
        "selected": selected,
        "reason": f"near_tie_broken_by_worst_fold_wape(n_near_ties={len(near_ties)})",
    }


def make_grid_sampler_study(grid_search_space: dict, sampler_seed: int) -> optuna.Study:
    """GridSampler와 NopPruner를 사용하는 exhaustive HPO study를 생성한다."""
    sampler = optuna.samplers.GridSampler(search_space=grid_search_space, seed=sampler_seed)
    return optuna.create_study(direction="minimize", sampler=sampler, pruner=optuna.pruners.NopPruner())


def check_grid_fully_and_uniquely_evaluated(trial_summaries: list[dict], grid_search_space: dict, horizon: int) -> None:
    """GridSampler가 동일 configuration을 중복 evaluate하지 않았는지 확인한다."""
    grid_keys = list(grid_search_space)
    seen = [tuple(t["params"][k] for k in grid_keys) for t in trial_summaries]
    if len(seen) != len(set(seen)):
        raise RuntimeError(f"h{horizon}: GridSampler가 동일 configuration을 중복 evaluate함: {seen}")
