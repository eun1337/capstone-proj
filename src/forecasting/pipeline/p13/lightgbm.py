# -*- coding: utf-8 -*-
"""
lightgbm.py

LightGBM P13 HPO runner.

A센터 2023 expanding 4-fold에서 num_leaves와 min_child_samples의 4개 조합을
GridSampler로 모두 평가한다. n_estimators는 100으로 고정한다.
5-family common evaluation key로 filtering한 OOF를 사용해 horizon별 hyperparameter를 선택한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna
import pandas as pd

from src.forecasting.common import evaluator as ev
from src.forecasting.common import folds as day3_folds
from src.forecasting.common.config import (
    BIAS_GUARDRAIL_ABS_PCT,
    HORIZONS,
    NEAR_TIE_WAPE_PCT_POINT,
    P13_MODEL_SEED,
    P13_SAMPLER_SEED,
)
from src.forecasting.common.data_loader import load_development
from src.forecasting.machine_learning.lightgbm import trainer as lgbm_trainer
from src.forecasting.machine_learning.lightgbm.config import (
    BAGGING_FRACTION,
    BAGGING_FREQ,
    FEATURE_FRACTION,
    LAMBDA_L1,
    LAMBDA_L2,
    MIN_CHILD_SAMPLES_CHOICES,
    NUM_LEAVES_CHOICES,
    P13_FIXED_PARAMS,
    P13_GRID_SEARCH_SPACE,
)

FAMILY = "lightgbm"
EVALUATION_POPULATION = "five_family_common_keys"
EVAL_KEY_COLS = ("horizon", "fold_id", "center_id", "sku_id", "week_st", "target_date")


def suggest_lgbm_hp(trial: optuna.Trial) -> dict:
    """GridSampler가 num_leaves/min_child_samples 조합 하나를 고르고, 나머지는 고정값이다."""
    return {
        "num_leaves": trial.suggest_categorical("num_leaves", list(NUM_LEAVES_CHOICES)),
        "min_child_samples": trial.suggest_categorical("min_child_samples", list(MIN_CHILD_SAMPLES_CHOICES)),
        "feature_fraction": FEATURE_FRACTION,
        "bagging_fraction": BAGGING_FRACTION,
        "bagging_freq": BAGGING_FREQ,
        "lambda_l1": LAMBDA_L1,
        "lambda_l2": LAMBDA_L2,
    }


# Common evaluation key 처리
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


def _normalize_key_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """EVAL_KEY_COLS 6개 컬럼만 정규화한 복사본을 반환한다(다른 컬럼은 그대로 유지)."""
    df = df.copy()
    df["horizon"] = _canonicalize_horizon(df["horizon"])
    df["fold_id"] = df["fold_id"].astype(int)
    df["center_id"] = df["center_id"].astype(str)
    df["sku_id"] = df["sku_id"].astype(str)
    df["week_st"] = pd.to_datetime(df["week_st"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    return df


def _check_no_duplicate_eval_keys(df: pd.DataFrame, name: str) -> None:
    dup = int(df.duplicated(subset=list(EVAL_KEY_COLS)).sum())
    if dup:
        raise ValueError(f"{name}에 evaluation key{EVAL_KEY_COLS} duplicate {dup}건 존재함")


def load_common_eval_keys(path: Path) -> pd.DataFrame:
    """P13 common evaluation key 파일을 로드/정규화/검증한다(존재/비어있음/필수컬럼/duplicate fail-fast)."""
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

    df = _normalize_key_dtypes(df)
    _check_no_duplicate_eval_keys(df, f"common evaluation key 파일({path})")
    return df


def _check_common_keys_match_p13_period(common_keys: pd.DataFrame, folds: list, horizon: int, path: Path) -> None:
    """P10(2022) 등 다른 stage의 key 파일이 잘못 전달되지 않았는지 target_date 겹침으로 확인한다."""
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


def _filter_pooled_oof_by_common_keys(pooled_native_oof: pd.DataFrame, common_keys: pd.DataFrame) -> pd.DataFrame:
    """native pooled OOF를 common_keys와 exact-key inner merge한다(duplicate/inflation/빈 교집합 fail-fast)."""
    key_cols = list(EVAL_KEY_COLS)
    oof_norm = _normalize_key_dtypes(pooled_native_oof)
    _check_no_duplicate_eval_keys(oof_norm, "trial pooled native OOF")

    merged = oof_norm.merge(common_keys[key_cols], on=key_cols, how="inner")

    if len(merged) == 0:
        raise ValueError("trial pooled OOF와 common evaluation key의 교집합이 0건임")
    if len(merged) > min(len(oof_norm), len(common_keys)):
        raise ValueError(
            f"merge 결과 row 수가 비정상적으로 증가함(row inflation 의심): "
            f"merged={len(merged)}, oof={len(oof_norm)}, common_keys={len(common_keys)}"
        )
    _check_no_duplicate_eval_keys(merged, "common-key filtered OOF")
    return merged


def _per_fold_metrics_common(filtered_oof: pd.DataFrame) -> dict:
    """common-key filter 후 OOF를 fold_id별로 나눠 compute_metrics를 다시 호출한다."""
    out = {}
    for fold_id, grp in filtered_oof.groupby("fold_id"):
        out[int(fold_id)] = ev.compute_metrics(
            grp["y_true"].to_numpy(), grp["y_pred"].to_numpy(), grp["mase_scale"].to_numpy(),
        )
    return out


# Fold 실행
def _run_folds(hp: dict, fold_data: list, horizon: int, seed: int, stage: str, config_id: str) -> dict:
    """하나의 LightGBM configuration을 4-fold에서 고정 iteration으로 학습·평가한다."""
    oof_frames, fold_metrics, best_iteration_diagnostic, used_iterations = [], [], [], []
    for fold_id, train_df, val_df in fold_data:
        try:
            result = lgbm_trainer.train_and_evaluate_fold(
                train_df, val_df, horizon, **hp,
                stage=stage, model_family=FAMILY, config_id=config_id,
                seed=seed, fold_id=fold_id, use_best_iteration=False,
                fixed_params=P13_FIXED_PARAMS,
            )
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "fold_id": fold_id}
        if result["used_iteration"] != P13_FIXED_PARAMS["n_estimators"]:
            return {
                "status": "failed", "fold_id": fold_id,
                "error": f"used_iteration={result['used_iteration']}가 fixed n_estimators="
                         f"{P13_FIXED_PARAMS['n_estimators']}과 다름 - adaptive 선택이 남아있을 가능성",
            }
        oof_frames.append(result["oof"])
        fold_metrics.append(result["metrics"])
        best_iteration_diagnostic.append(result["best_iteration"])
        used_iterations.append(result["used_iteration"])
    return {
        "status": "ok", "oof_frames": oof_frames,
        "native_fold_metrics": fold_metrics,
        "best_iteration_diagnostic": best_iteration_diagnostic,
        "used_iterations": used_iterations,
    }


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


def run_p13_hpo(
    sub_a: pd.DataFrame, horizon: int, n_trials: int, seed: int, sampler_seed: int,
    common_eval_keys: pd.DataFrame, common_eval_keys_path: Path,
) -> dict:
    folds = day3_folds.generate_expanding_folds(sub_a, 2023, horizon)
    fold_data = [
        (fold["fold"], sub_a.loc[fold["train_mask"]], sub_a.loc[fold["val_mask"]])
        for fold in folds
    ]
    for fold_id, train_df, val_df in fold_data:
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError(f"h{horizon} fold {fold_id}: train/val row 0개")

    _check_common_keys_match_p13_period(common_eval_keys, folds, horizon, common_eval_keys_path)
    horizon_common_keys = common_eval_keys[common_eval_keys["horizon"] == horizon]

    grid_size = len(NUM_LEAVES_CHOICES) * len(MIN_CHILD_SAMPLES_CHOICES)
    if n_trials != grid_size:
        raise ValueError(
            f"n_trials={n_trials}가 P13_GRID_SEARCH_SPACE의 Cartesian product 크기 "
            f"{grid_size}와 다름 - GridSampler는 search space 크기만큼만 정확히 evaluate해야 함"
        )
    sampler = optuna.samplers.GridSampler(search_space=P13_GRID_SEARCH_SPACE, seed=sampler_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=optuna.pruners.NopPruner())
    trial_summaries: list[dict] = []
    filtered_oof_by_trial: dict[int, pd.DataFrame] = {}

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_lgbm_hp(trial)
        run = _run_folds(hp, fold_data, horizon, seed, "p13_final_hpo", f"trial{trial.number}")

        if run["status"] != "ok":
            trial_summaries.append({
                "trial_id": trial.number, "params": hp, "status": "failed",
                "error": run["error"], "fold_id": run["fold_id"],
                "pooled_wape": float("inf"), "pooled_bias": float("nan"),
                "worst_fold_wape": float("inf"),
            })
            return float("inf")

        pooled_native_oof = pd.concat(run["oof_frames"], ignore_index=True)
        filtered_oof = _filter_pooled_oof_by_common_keys(pooled_native_oof, horizon_common_keys)
        pooled_common = ev.compute_metrics(
            filtered_oof["y_true"].to_numpy(), filtered_oof["y_pred"].to_numpy(),
            filtered_oof["mase_scale"].to_numpy(),
        )
        per_fold_common = _per_fold_metrics_common(filtered_oof)
        worst_fold_wape = max(m["wape"] for m in per_fold_common.values())

        trial_summaries.append({
            "trial_id": trial.number, "params": hp, "status": "ok",
            "pooled_wape": pooled_common["wape"], "pooled_bias": pooled_common["bias"],
            "pooled_rmse": pooled_common["rmse"], "pooled_mae": pooled_common["mae"],
            "pooled_mase": pooled_common["mase"], "worst_fold_wape": worst_fold_wape,
            "per_fold_metrics_common": per_fold_common,
            "native_fold_metrics_diagnostic": run["native_fold_metrics"],
            "best_iteration_diagnostic": run["best_iteration_diagnostic"],
            "used_iterations": run["used_iterations"],
            "common_eval_key_count": len(horizon_common_keys),
        })
        # 선택된 trial의 기존 OOF를 재사용하기 위해 보관한다.
        filtered_oof_by_trial[trial.number] = filtered_oof
        return pooled_common["wape"]

    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    seen_params = [t["params"] for t in trial_summaries]
    grid_keys = list(P13_GRID_SEARCH_SPACE)
    seen_grid_tuples = [tuple(p[k] for k in grid_keys) for p in seen_params]
    if len(seen_grid_tuples) != len(set(seen_grid_tuples)):
        raise RuntimeError(f"h{horizon}: GridSampler가 동일 configuration을 중복 evaluate함: {seen_grid_tuples}")

    ok_trials = [t for t in trial_summaries if t["status"] == "ok"]
    selection = select_best_trial(ok_trials)

    result = {
        "horizon": horizon, "n_trials": n_trials, "seed": seed, "sampler_seed": sampler_seed,
        "n_failed_trials": sum(1 for t in trial_summaries if t["status"] != "ok"),
        "evaluation_population": EVALUATION_POPULATION,
        "common_eval_key_path": str(common_eval_keys_path),
        "common_eval_key_count": len(horizon_common_keys),
        "n_estimators_fixed": P13_FIXED_PARAMS["n_estimators"],
        "selection": {k: v for k, v in selection.items()},
        "all_trials": trial_summaries,
    }

    sel = selection["selected"]
    if sel is not None:
        result["pooled_metrics_common"] = {
            "wape": sel["pooled_wape"], "bias": sel["pooled_bias"], "rmse": sel["pooled_rmse"],
            "mae": sel["pooled_mae"], "mase": sel["pooled_mase"],
        }
        result["per_fold_metrics_common"] = sel["per_fold_metrics_common"]
        result["native_fold_metrics_diagnostic"] = sel["native_fold_metrics_diagnostic"]
        result["best_iteration_diagnostic"] = sel["best_iteration_diagnostic"]
        result["selected_oof_common_filtered"] = filtered_oof_by_trial[sel["trial_id"]]

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM P13 Final HPO (2023, A center)")
    parser.add_argument("--output-dir", default="outputs/hpo/lgbm")
    _grid_size = len(NUM_LEAVES_CHOICES) * len(MIN_CHILD_SAMPLES_CHOICES)
    parser.add_argument("--n-trials", type=int, default=_grid_size,
                        help=f"GridSampler search space 크기와 정확히 같아야 함(기본값={_grid_size}, "
                             f"lgbm.config.P13_GRID_SEARCH_SPACE 기준)")
    parser.add_argument("--seed", type=int, default=P13_MODEL_SEED, help="P13 모델 고정 seed (기본값: common.config.P13_MODEL_SEED)")
    parser.add_argument("--sampler-seed", type=int, default=P13_SAMPLER_SEED, help="P13 Optuna GridSampler 순회 순서 seed (기본값: common.config.P13_SAMPLER_SEED, 모델 seed와 분리)")
    parser.add_argument(
        "--common-eval-keys-path", required=True,
        help="필수. 5-family P13(2023) common evaluation key parquet/csv 경로. "
             "P10(2022) 등 다른 stage의 key 파일을 전달하면 fail-fast한다. "
             "이 옵션 없이 실행되는 경로는 없다.",
    )
    parser.add_argument("--horizon", type=int, choices=list(HORIZONS), default=None,
                        help="지정 시 해당 horizon만 실행(생략 시 전체 HORIZONS를 기존 순서대로 실행)")
    args = parser.parse_args()

    common_eval_keys_path = Path(args.common_eval_keys_path)
    common_eval_keys = load_common_eval_keys(common_eval_keys_path)

    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()

    out_dir = Path(args.output_dir)
    oof_dir = out_dir / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)

    horizons_to_run = [args.horizon] if args.horizon is not None else list(HORIZONS)
    for h in horizons_to_run:
        print(f"[P13][lightgbm] horizon={h} Final HPO 시작 (n_trials={args.n_trials}, common_eval_keys={common_eval_keys_path})")
        result = run_p13_hpo(sub_a, h, args.n_trials, args.seed, args.sampler_seed, common_eval_keys, common_eval_keys_path)

        sel = result["selection"]["selected"]
        if sel is None:
            print(f"[P13][lightgbm] h{h}: guardrail(|bias|<={BIAS_GUARDRAIL_ABS_PCT}%p) 통과 trial 없음 - 확인 필요")
        else:
            print(
                f"[P13][lightgbm] h{h}: best trial={sel['trial_id']} "
                f"WAPE={sel['pooled_wape']:.4f} Bias={sel['pooled_bias']:.4f} "
                f"WorstFoldWAPE={sel['worst_fold_wape']:.4f} reason={result['selection']['reason']} "
                f"n_estimators_fixed={result['n_estimators_fixed']}"
            )
            result["selected_oof_common_filtered"].to_parquet(
                oof_dir / f"oof_{FAMILY}_h{h}_trial{sel['trial_id']}_seed{args.seed}_common.parquet",
                index=False,
            )

        out_path = out_dir / f"p13_{FAMILY}_h{h}.json"
        serializable = {k: v for k, v in result.items() if k != "selected_oof_common_filtered"}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
        print(f"[P13][lightgbm] 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
