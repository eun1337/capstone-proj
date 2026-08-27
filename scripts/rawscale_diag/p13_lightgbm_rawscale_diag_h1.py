# -*- coding: utf-8 -*-
"""
run_diagnostic.py

진단용 1회성 실험: log1p -> expm1 재변환 편향(retransformation bias, Jensen's inequality)
가설 검증. 기존 P13 lightgbm h1 결과(outputs/hpo/lgbm/p13_lightgbm_h1.json)는
pooled_bias가 -31%~-35%로 hidden bias guardrail(|bias|<=20%p)을 통과하지 못했다.
동일한 feature/fold(4-fold expanding)/hyperparameter(4 trial grid)/evaluation
population(5-family common evaluation keys)을 유지한 채, target만 log1p 변환 없이
raw scale(qty)로 RMSE 학습해서 pooled_bias가 줄어드는지(=재변환 편향 가설을 지지하는지)
확인한다.

기존 파이프라인 코드(machine_learning/lightgbm/trainer.py, common/evaluator.py,
pipeline/p13/lightgbm.py 등)는 전혀 수정하지 않고 import만 한다. 이 실험과 관련된
모든 산출물은 outputs/hpo/_diag_rawscale_lightgbm_h1/ 안에만 저장한다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# WSL->Windows python.exe 호출 시 PYTHONPATH가 전달되지 않아, repo root를 직접 sys.path에 추가한다.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.forecasting.common import config as cfg
from src.forecasting.common import evaluator as ev
from src.forecasting.common import folds as day3_folds
from src.forecasting.common import oof as oo
from src.forecasting.common.data_loader import load_development
from src.forecasting.machine_learning.lightgbm.config import P13_FIXED_PARAMS
from src.forecasting.machine_learning.lightgbm.preprocessing import LGBMPreprocessor
from src.forecasting.pipeline.p13 import lightgbm as p13_lgbm

HORIZON = 1
FAMILY = "lightgbm"
STAGE = "p13_diag_rawscale_h1"
SEED = cfg.P13_MODEL_SEED  # 42

OUT_DIR = _REPO_ROOT / "outputs" / "hpo" / "_diag_rawscale_lightgbm_h1"
OOF_DIR = OUT_DIR / "oof"

EXISTING_RESULT_PATH = cfg.PROJECT_ROOT / "outputs" / "hpo" / "lgbm" / "p13_lightgbm_h1.json"
COMMON_EVAL_KEYS_PATH = cfg.PROJECT_ROOT / "outputs" / "audits" / "p13" / "five_family_common_evaluation_keys.parquet"


def train_and_evaluate_fold_raw(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    horizon: int,
    *,
    num_leaves: int,
    min_child_samples: int,
    feature_fraction: float,
    bagging_fraction: float,
    bagging_freq: int,
    lambda_l1: float,
    lambda_l2: float,
    stage: str,
    model_family: str,
    config_id: str,
    seed: int,
    fold_id,
    fixed_params: dict,
) -> dict:
    """machine_learning/lightgbm/trainer.py의 train_and_evaluate_fold와 동일 구조이지만,
    target을 log1p 변환하지 않고 raw scale(qty)로 그대로 RMSE 학습/예측한다
    (evaluator.inverse_transform_prediction 호출 없음)."""
    preprocessor = LGBMPreprocessor()
    X_train = preprocessor.fit_transform(train_df, horizon)
    X_val = preprocessor.transform(val_df)

    target_col = cfg.TARGET_COLS[horizon]
    y_train_raw = train_df[target_col].to_numpy(dtype=float)
    y_val_raw = val_df[target_col].to_numpy(dtype=float)

    model = LGBMRegressor(
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        colsample_bytree=feature_fraction,
        subsample=bagging_fraction,
        subsample_freq=bagging_freq,
        reg_alpha=lambda_l1,
        reg_lambda=lambda_l2,
        random_state=seed,
        **fixed_params,
    )

    model.fit(
        X_train, y_train_raw,
        eval_set=[(X_val, y_val_raw)],
        eval_metric=["l2"],
        categorical_feature=preprocessor.categorical_feature_names,
    )

    l2_curve = model.evals_result_["valid_0"]["l2"]
    best_iteration = int(np.argmin(l2_curve)) + 1
    final_iteration = len(l2_curve)  # EarlyStopping 미사용

    # P13은 use_best_iteration=False로 고정 n_estimators 전체를 사용한다(원본 trainer와 동일 정책).
    pred_raw = model.predict(X_val, num_iteration=None)
    used_iteration = final_iteration

    n_negative_clipped = int((pred_raw < 0).sum())
    pred_raw = np.clip(pred_raw, 0.0, None)

    val_keys = val_df[["center_id", "sku_id", "week_st"]].copy()
    val_keys["target_date"] = val_keys["week_st"] + pd.Timedelta(weeks=horizon)

    mase_scale = ev.build_mase_scale(train_df[["center_id", "sku_id", "week_st", "qty"]], val_keys)
    metrics = ev.compute_metrics(y_val_raw, pred_raw, mase_scale)

    # 이 실험엔 log-scale 예측이 없으므로 y_pred_log 컬럼은 raw 예측을 그대로 채운다(진단용 메타데이터일 뿐 metric 계산엔 사용되지 않음).
    oof_frame = oo.build_oof_frame(
        val_keys, y_val_raw, pred_raw, pred_raw, mase_scale,
        stage=stage, model_family=model_family, config_id=config_id,
        seed=seed, horizon=horizon, fold_id=fold_id,
    )

    return {
        "metrics": metrics,
        "oof": oof_frame,
        "best_iteration": best_iteration,
        "final_iteration": final_iteration,
        "used_iteration": used_iteration,
        "n_negative_clipped": n_negative_clipped,
        "n_train": len(train_df),
        "n_val": len(val_df),
    }


def _run_folds_raw(hp: dict, fold_data: list, horizon: int, seed: int, stage: str, config_id: str) -> dict:
    """pipeline/p13/lightgbm.py의 _run_folds와 동일 구조(고정 n_estimators 검증 포함)."""
    oof_frames, fold_metrics, best_iteration_diagnostic, used_iterations, n_negative_clipped_list = [], [], [], [], []
    for fold_id, train_df, val_df in fold_data:
        try:
            result = train_and_evaluate_fold_raw(
                train_df, val_df, horizon, **hp,
                stage=stage, model_family=FAMILY, config_id=config_id,
                seed=seed, fold_id=fold_id, fixed_params=P13_FIXED_PARAMS,
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
        n_negative_clipped_list.append(result["n_negative_clipped"])
    return {
        "status": "ok", "oof_frames": oof_frames,
        "native_fold_metrics": fold_metrics,
        "best_iteration_diagnostic": best_iteration_diagnostic,
        "used_iterations": used_iterations,
        "n_negative_clipped_diagnostic": n_negative_clipped_list,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)

    common_eval_keys = p13_lgbm.load_common_eval_keys(COMMON_EVAL_KEYS_PATH)

    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()

    folds = day3_folds.generate_expanding_folds(sub_a, 2023, HORIZON)
    fold_data = [
        (fold["fold"], sub_a.loc[fold["train_mask"]], sub_a.loc[fold["val_mask"]])
        for fold in folds
    ]
    for fold_id, train_df, val_df in fold_data:
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError(f"h{HORIZON} fold {fold_id}: train/val row 0개")

    p13_lgbm._check_common_keys_match_p13_period(common_eval_keys, folds, HORIZON, COMMON_EVAL_KEYS_PATH)
    horizon_common_keys = common_eval_keys[common_eval_keys["horizon"] == HORIZON]

    with open(EXISTING_RESULT_PATH, "r", encoding="utf-8") as f:
        existing_result = json.load(f)
    existing_trials = existing_result["all_trials"]

    print(f"[diag][lightgbm-rawscale] h{HORIZON} 시작: {len(existing_trials)}개 trial (기존 grid 재사용), "
          f"common_eval_keys={COMMON_EVAL_KEYS_PATH}")

    trial_summaries = []
    filtered_oof_by_trial = {}

    for existing_trial in existing_trials:
        trial_id = existing_trial["trial_id"]
        hp = dict(existing_trial["params"])

        print(f"[diag][lightgbm-rawscale] trial{trial_id} params={hp} 학습 시작...")
        run = _run_folds_raw(hp, fold_data, HORIZON, SEED, STAGE, f"trial{trial_id}")

        if run["status"] != "ok":
            trial_summaries.append({
                "trial_id": trial_id, "params": hp, "status": "failed",
                "error": run["error"], "fold_id": run["fold_id"],
                "pooled_wape": float("inf"), "pooled_bias": float("nan"),
                "worst_fold_wape": float("inf"),
            })
            print(f"[diag][lightgbm-rawscale] trial{trial_id} 실패: {run['error']}")
            continue

        pooled_native_oof = pd.concat(run["oof_frames"], ignore_index=True)
        filtered_oof = p13_lgbm._filter_pooled_oof_by_common_keys(pooled_native_oof, horizon_common_keys)
        pooled_common = ev.compute_metrics(
            filtered_oof["y_true"].to_numpy(), filtered_oof["y_pred"].to_numpy(),
            filtered_oof["mase_scale"].to_numpy(),
        )
        per_fold_common = p13_lgbm._per_fold_metrics_common(filtered_oof)
        worst_fold_wape = max(m["wape"] for m in per_fold_common.values())

        trial_summaries.append({
            "trial_id": trial_id, "params": hp, "status": "ok",
            "pooled_wape": pooled_common["wape"], "pooled_bias": pooled_common["bias"],
            "pooled_rmse": pooled_common["rmse"], "pooled_mae": pooled_common["mae"],
            "pooled_mase": pooled_common["mase"], "worst_fold_wape": worst_fold_wape,
            "per_fold_metrics_common": per_fold_common,
            "native_fold_metrics_diagnostic": run["native_fold_metrics"],
            "best_iteration_diagnostic": run["best_iteration_diagnostic"],
            "used_iterations": run["used_iterations"],
            "n_negative_clipped_diagnostic": run["n_negative_clipped_diagnostic"],
            "common_eval_key_count": len(horizon_common_keys),
        })
        filtered_oof_by_trial[trial_id] = filtered_oof
        print(f"[diag][lightgbm-rawscale] trial{trial_id} 완료: "
              f"WAPE={pooled_common['wape']:.4f} Bias={pooled_common['bias']:.4f}")

    ok_trials = [t for t in trial_summaries if t["status"] == "ok"]
    selection = p13_lgbm.select_best_trial(ok_trials)

    result = {
        "horizon": HORIZON,
        "n_trials": len(existing_trials),
        "seed": SEED,
        "sampler_seed": existing_result.get("sampler_seed"),
        "n_failed_trials": sum(1 for t in trial_summaries if t["status"] != "ok"),
        "evaluation_population": p13_lgbm.EVALUATION_POPULATION,
        "common_eval_key_path": str(COMMON_EVAL_KEYS_PATH),
        "common_eval_key_count": len(horizon_common_keys),
        "n_estimators_fixed": P13_FIXED_PARAMS["n_estimators"],
        "target_transform": "raw_scale_no_log1p",
        "diagnostic_note": (
            "log1p/expm1 재변환 편향(retransformation bias) 가설 검증용 1회성 진단 스크립트. "
            "outputs/hpo/lgbm/p13_lightgbm_h1.json과 동일한 feature/fold/hyperparameter grid/"
            "evaluation population을 사용하되, target을 log1p 변환하지 않고 raw scale(qty)로 "
            "RMSE 학습함(inverse_transform_prediction 미호출)."
        ),
        "reference_result_path": str(EXISTING_RESULT_PATH),
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
        filtered_oof_by_trial[sel["trial_id"]].to_parquet(
            OOF_DIR / f"oof_{FAMILY}_h{HORIZON}_trial{sel['trial_id']}_seed{SEED}_common_rawscale.parquet",
            index=False,
        )

    out_path = OUT_DIR / "p13_lightgbm_h1_rawscale.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"[diag][lightgbm-rawscale] 저장 완료: {out_path}")

    # 기존 best trial(pooled_wape 최솟값, trial3)과 동일 trial_id로 매칭해 비교 리포트 출력
    existing_ok_trials = [t for t in existing_trials if t["status"] == "ok"]
    existing_best = min(existing_ok_trials, key=lambda t: t["pooled_wape"])
    new_matched = next((t for t in trial_summaries if t["trial_id"] == existing_best["trial_id"]), None)

    print("\n=== log1p 재변환 편향 가설 검증 결과 (lightgbm h1) ===")
    header = f"| {'구분':<28} | {'pooled_wape':>12} | {'pooled_bias':>12} |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    print(f"| {'기존 (log1p, trial' + str(existing_best['trial_id']) + ', best)':<28} "
          f"| {existing_best['pooled_wape']:>12.2f} | {existing_best['pooled_bias']:>12.2f} |")

    if new_matched is not None and new_matched["status"] == "ok":
        print(f"| {'이번 (raw scale, trial' + str(new_matched['trial_id']) + ', 동일 trial)':<28} "
              f"| {new_matched['pooled_wape']:>12.2f} | {new_matched['pooled_bias']:>12.2f} |")

        bias_delta = new_matched["pooled_bias"] - existing_best["pooled_bias"]
        sign_flip = (existing_best["pooled_bias"] < 0 < new_matched["pooled_bias"]) or \
                    (existing_best["pooled_bias"] > 0 > new_matched["pooled_bias"])
        print(f"\nbias 변화량: {bias_delta:+.2f}%p (부호 반전: {sign_flip})")
        if abs(bias_delta) >= 20.0 or sign_flip:
            print("=> bias가 유의미하게(20%p 이상) 줄었거나 부호가 바뀜: "
                  "log1p/expm1 재변환 편향 가설을 지지하는 강한 증거")
        else:
            print("=> bias가 거의 그대로임: log1p/expm1 재변환이 주원인이 아닐 가능성이 높음 - "
                  "evaluator, 데이터 누수, train/val 분할 등 다른 공유 원인을 확인 필요")
    else:
        err = new_matched.get("error") if new_matched is not None else "trial_id 매칭 실패"
        print(f"이번 실험의 동일 trial이 실패함: {err}")

    print(f"\n결과 파일: {out_path}")


if __name__ == "__main__":
    main()
