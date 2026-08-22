# -*- coding: utf-8 -*-
"""
p13_lgbm.py
P13: A센터 2021~2022 train → 2023 Q1~Q4 Final HPO(4-fold).
P12에서 동결한 Final Search Space(현재는 lgbm/config.py의 Initial Search Space와 동일 —
P12 조정이 있었다면 lgbm/config.py를 먼저 갱신할 것) 안에서만 Optuna(TPE)로 최종 HP를
선택한다.

선택 규칙 (common.config의 P21/P16 선택 기준 - BIAS_GUARDRAIL_ABS_PCT/NEAR_TIE_WAPE_PCT_POINT):
  |Pooled OOF Bias| <= 20%p guardrail → Pooled OOF WAPE 최소
  → 최저 WAPE 대비 1%p 이내 near-tie → Worst-fold WAPE tie-break

common.evaluator.compute_metrics()의 wape/bias는 이미 %(0~100) 단위이므로 guardrail/near-tie
임계값도 같은 %p 단위인 common.config.BIAS_GUARDRAIL_ABS_PCT/NEAR_TIE_WAPE_PCT_POINT를 그대로
재사용한다(로컬에서 0.20/0.01 같은 0~1 스케일 상수를 새로 만들면 guardrail이 사실상 항상
막혀버리는 단위 불일치 버그가 된다).

common.folds.generate_expanding_folds / common.evaluator.compute_metrics를 그대로
재사용해 pooled OOF 지표를 계산하므로, "같은 evaluator를 쓰는가"는 함수 identity로
자동 보장된다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna
import pandas as pd

from src.ml.day3_rf_lightgbm.common import evaluator as ev
from src.ml.day3_rf_lightgbm.common import folds as day3_folds
from src.ml.day3_rf_lightgbm.common.config import (
    BIAS_GUARDRAIL_ABS_PCT,
    HORIZONS,
    NEAR_TIE_WAPE_PCT_POINT,
    P13_MODEL_SEED,
    P13_SAMPLER_SEED,
)
from src.ml.day3_rf_lightgbm.common.data_loader import load_development
from src.ml.day3_rf_lightgbm.lgbm import trainer as lgbm_trainer
from src.ml.day3_rf_lightgbm.lgbm.p10_lgbm import suggest_lgbm_hp  # 동일 search space 재사용
from src.ml.day3_rf_lightgbm.lgbm.config import OPTUNA_N_STARTUP_TRIALS

FAMILY = "lightgbm"


def _pooled_metrics(oof_frames: list[pd.DataFrame]) -> dict:
    """여러 fold의 OOF를 이어붙여 common.evaluator.compute_metrics로 Pooled 지표 계산
    (evaluator_match 요건과 동일한 함수를 그대로 재사용)."""
    pooled = pd.concat(oof_frames, ignore_index=True)
    return ev.compute_metrics(
        pooled["y_true"].to_numpy(), pooled["y_pred"].to_numpy(),
        pooled["mase_scale"].to_numpy(),
    )


def select_best_trial(trial_summaries: list[dict]) -> dict:
    """P21/P16: Bias guardrail → Pooled WAPE 최소 → 1%p 이내 near-tie는 Worst-fold WAPE."""
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


def run_p13_hpo(sub_a: pd.DataFrame, horizon: int, n_trials: int, seed: int, sampler_seed: int) -> dict:
    folds = day3_folds.generate_expanding_folds(sub_a, 2023, horizon)
    fold_data = [
        (fold["fold"], sub_a.loc[fold["train_mask"]], sub_a.loc[fold["val_mask"]])
        for fold in folds
    ]
    for fold_id, train_df, val_df in fold_data:
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError(f"h{horizon} fold {fold_id}: train/val row 0개")

    sampler = optuna.samplers.TPESampler(seed=sampler_seed, n_startup_trials=OPTUNA_N_STARTUP_TRIALS)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    trial_summaries = []
    oof_by_trial: dict[int, pd.DataFrame] = {}

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_lgbm_hp(trial)
        oof_frames, fold_metrics, failed = [], [], False

        for fold_id, train_df, val_df in fold_data:
            try:
                result = lgbm_trainer.train_and_evaluate_fold(
                    train_df, val_df, horizon, **hp,
                    stage="p13_final_hpo", model_family=FAMILY,
                    config_id=f"trial{trial.number}", seed=seed, fold_id=fold_id,
                )
            except Exception as exc:  # noqa: BLE001
                failed = True
                trial_summaries.append({
                    "trial_id": trial.number, "params": hp, "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}", "fold_id": fold_id,
                    "pooled_wape": float("inf"), "pooled_bias": float("nan"),
                    "worst_fold_wape": float("inf"),
                })
                break
            oof_frames.append(result["oof"])
            fold_metrics.append(result["metrics"])

        if failed:
            return float("inf")

        pooled = _pooled_metrics(oof_frames)
        worst_fold_wape = max(m["wape"] for m in fold_metrics)

        trial_summaries.append({
            "trial_id": trial.number, "params": hp, "status": "ok",
            "pooled_wape": pooled["wape"], "pooled_bias": pooled["bias"],
            "pooled_rmse": pooled["rmse"], "pooled_mae": pooled["mae"],
            "pooled_mase": pooled["mase"], "worst_fold_wape": worst_fold_wape,
            "per_fold_metrics": fold_metrics,
        })
        oof_by_trial[trial.number] = pd.concat(oof_frames, ignore_index=True)

        return pooled["wape"]

    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    ok_trials = [t for t in trial_summaries if t["status"] == "ok"]
    selection = select_best_trial(ok_trials)

    return {
        "horizon": horizon, "n_trials": n_trials, "seed": seed, "sampler_seed": sampler_seed,
        "n_failed_trials": sum(1 for t in trial_summaries if t["status"] != "ok"),
        "selection": {k: v for k, v in selection.items()},
        "all_trials": trial_summaries,
        "_oof_by_trial": oof_by_trial,  # 저장 시 별도 처리 (json 직렬화 대상 아님)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM P13 Final HPO (2023, A center)")
    parser.add_argument("--output-dir", default="outputs/hpo/lgbm")
    parser.add_argument("--n-trials", type=int, required=True, help="P12에서 확정한 공통 Optuna trial 수 N")
    parser.add_argument("--seed", type=int, default=P13_MODEL_SEED, help="P13 모델 고정 seed (기본값: common.config.P13_MODEL_SEED)")
    parser.add_argument("--sampler-seed", type=int, default=P13_SAMPLER_SEED, help="P13 Optuna TPE sampler seed (기본값: common.config.P13_SAMPLER_SEED, 모델 seed와 분리)")
    args = parser.parse_args()

    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()

    out_dir = Path(args.output_dir)
    oof_dir = out_dir / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)

    for h in HORIZONS:
        print(f"[P13][lightgbm] horizon={h} Final HPO 시작 (n_trials={args.n_trials})")
        result = run_p13_hpo(sub_a, h, args.n_trials, args.seed, args.sampler_seed)

        sel = result["selection"]["selected"]
        if sel is None:
            print(f"[P13][lightgbm] h{h}: ⚠️ guardrail(|bias|<={BIAS_GUARDRAIL_ABS_PCT}%p) 통과 trial 없음 — 확인 필요")
        else:
            print(
                f"[P13][lightgbm] h{h}: best trial={sel['trial_id']} "
                f"WAPE={sel['pooled_wape']:.4f} Bias={sel['pooled_bias']:.4f} "
                f"WorstFoldWAPE={sel['worst_fold_wape']:.4f} reason={result['selection']['reason']}"
            )
            best_oof = result["_oof_by_trial"][sel["trial_id"]]
            best_oof.to_parquet(
                oof_dir / f"oof_{FAMILY}_h{h}_trial{sel['trial_id']}_seed{args.seed}.parquet",
                index=False,
            )

        out_path = out_dir / f"p13_{FAMILY}_h{h}.json"
        serializable = {k: v for k, v in result.items() if k != "_oof_by_trial"}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
        print(f"[P13][lightgbm] 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
