"""
p13_lightgbm_hurdle_h1_hpo.py

Hurdle-LightGBM 정식 HPO (후속 개선 실험, 기존 P13 공식 결과와 완전히 분리된 트랙).
--horizon {1,2,4} (기본값 1, h1 기존 결과와 동일 재현)로 horizon을 선택한다.

구조: classifier(P(y>0), objective=binary) x conditional regressor(y>0 행만, log1p target,
objective=regression, predict 후 expm1 복원). 최종 예측 = P(y>0) * conditional_pred (soft,
hard threshold/class weighting/probability calibration/smearing 없음).

Search space: classifier와 conditional regressor 각각 num_leaves x min_child_samples
(기존 P13 LightGBM 그리드와 동일한 값)를 탐색해 4 x 4 = 16개 combined configuration을 평가한다.
나머지 하이퍼파라미터는 기존 P13 LightGBM(P13_FIXED_PARAMS/FEATURE_FRACTION 등)과 동일하게 고정.

효율: fold마다 classifier 4개, regressor 4개를 각 1회만 fit(총 4 fold x 8 = 32 fit)하고,
16개 조합은 이 예측값을 조합만 해서 평가한다(재학습 없음).

fold/common evaluation key/evaluator/OOF 스키마/guardrail-WAPE-worst_fold selection 로직은
기존 P13 공통 모듈(day3_folds, hpo_common, evaluator, oof)을 그대로 재사용한다. 2024 holdout은
사용하지 않는다.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from src.forecasting.common import config as cfg
from src.forecasting.common import evaluator as ev
from src.forecasting.common import folds as day3_folds
from src.forecasting.common import oof as oo
from src.forecasting.common.data_loader import load_development
from src.forecasting.machine_learning.lightgbm.config import (
    BAGGING_FRACTION,
    BAGGING_FREQ,
    FEATURE_FRACTION,
    LAMBDA_L1,
    LAMBDA_L2,
    MIN_CHILD_SAMPLES_CHOICES,
    NUM_LEAVES_CHOICES,
    P13_FIXED_PARAMS,
)
from src.forecasting.machine_learning.lightgbm.preprocessing import LGBMPreprocessor
from src.forecasting.pipeline.p13 import hpo_common as hc

SEED = cfg.P13_MODEL_SEED
SAMPLER_SEED = cfg.P13_SAMPLER_SEED
COMMON_KEYS_PATH = Path("outputs/audits/p13/five_family_common_evaluation_keys.parquet")
OUT_DIR = Path("outputs/hurdle_hpo/lgbm_hurdle")

CLS_FIXED_PARAMS = {**P13_FIXED_PARAMS, "objective": "binary"}
HURDLE_GRID_SEARCH_SPACE = {
    "cls_num_leaves": list(NUM_LEAVES_CHOICES),
    "cls_min_child_samples": list(MIN_CHILD_SAMPLES_CHOICES),
    "reg_num_leaves": list(NUM_LEAVES_CHOICES),
    "reg_min_child_samples": list(MIN_CHILD_SAMPLES_CHOICES),
}
QUARTILE_LABELS = ["Q1", "Q2", "Q3", "Q4"]


def _fit_cls(X_train, y_train, num_leaves, min_child_samples, categorical_feature):
    model = LGBMClassifier(
        num_leaves=num_leaves, min_child_samples=min_child_samples,
        colsample_bytree=FEATURE_FRACTION, subsample=BAGGING_FRACTION, subsample_freq=BAGGING_FREQ,
        reg_alpha=LAMBDA_L1, reg_lambda=LAMBDA_L2, random_state=SEED, **CLS_FIXED_PARAMS,
    )
    model.fit(X_train, y_train, categorical_feature=categorical_feature)
    return model


def _fit_reg(X_train, y_train, num_leaves, min_child_samples, categorical_feature):
    model = LGBMRegressor(
        num_leaves=num_leaves, min_child_samples=min_child_samples,
        colsample_bytree=FEATURE_FRACTION, subsample=BAGGING_FRACTION, subsample_freq=BAGGING_FREQ,
        reg_alpha=LAMBDA_L1, reg_lambda=LAMBDA_L2, random_state=SEED, **P13_FIXED_PARAMS,
    )
    model.fit(X_train, y_train, categorical_feature=categorical_feature)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Hurdle-LightGBM 정식 HPO")
    parser.add_argument("--horizon", type=int, choices=[1, 2, 4], default=1)
    horizon = parser.parse_args().horizon

    common_eval_keys = hc.load_common_eval_keys(COMMON_KEYS_PATH)
    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()
    folds = day3_folds.generate_expanding_folds(sub_a, 2023, horizon)
    hc.check_common_keys_match_p13_period(common_eval_keys, folds, horizon, COMMON_KEYS_PATH)
    horizon_common_keys = common_eval_keys[common_eval_keys["horizon"] == horizon]

    target_col = cfg.TARGET_COLS[horizon]

    fold_contexts = {}
    for fold in folds:
        fold_id = fold["fold"]
        train_df = sub_a.loc[fold["train_mask"]]
        val_df = sub_a.loc[fold["val_mask"]]
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError(f"h{horizon} fold {fold_id}: train/val row 0개")

        preprocessor = LGBMPreprocessor()
        X_train = preprocessor.fit_transform(train_df, horizon)
        X_val = preprocessor.transform(val_df)
        cat_features = preprocessor.categorical_feature_names

        y_train_raw = train_df[target_col].to_numpy(dtype=float)
        y_val_raw = val_df[target_col].to_numpy(dtype=float)
        pos_mask = y_train_raw > 0

        cls_pred = {}
        for nl in NUM_LEAVES_CHOICES:
            for mcs in MIN_CHILD_SAMPLES_CHOICES:
                model = _fit_cls(X_train, pos_mask.astype(int), nl, mcs, cat_features)
                cls_pred[(nl, mcs)] = model.predict_proba(X_val)[:, 1]

        X_train_pos = X_train.iloc[pos_mask]
        y_train_pos_log = np.log1p(y_train_raw[pos_mask])
        reg_pred = {}
        for nl in NUM_LEAVES_CHOICES:
            for mcs in MIN_CHILD_SAMPLES_CHOICES:
                model = _fit_reg(X_train_pos, y_train_pos_log, nl, mcs, cat_features)
                reg_pred[(nl, mcs)] = ev.inverse_transform_prediction(model.predict(X_val))

        val_keys = val_df[["center_id", "sku_id", "week_st"]].copy()
        val_keys["target_date"] = val_keys["week_st"] + pd.Timedelta(weeks=horizon)
        mase_scale = ev.build_mase_scale(train_df[["center_id", "sku_id", "week_st", "qty"]], val_keys)

        sku_avg = train_df.groupby(["center_id", "sku_id"])["qty"].mean()
        demand_quartile_map = pd.qcut(sku_avg, q=4, labels=QUARTILE_LABELS, duplicates="drop").to_dict()

        fold_contexts[fold_id] = {
            "val_keys": val_keys, "y_val_raw": y_val_raw, "mase_scale": mase_scale,
            "cls_pred": cls_pred, "reg_pred": reg_pred, "demand_quartile_map": demand_quartile_map,
        }

    study = hc.make_grid_sampler_study(HURDLE_GRID_SEARCH_SPACE, SAMPLER_SEED)
    trial_summaries: list[dict] = []
    filtered_oof_by_trial: dict[int, pd.DataFrame] = {}

    def objective(trial) -> float:
        cls_key = (
            trial.suggest_categorical("cls_num_leaves", list(NUM_LEAVES_CHOICES)),
            trial.suggest_categorical("cls_min_child_samples", list(MIN_CHILD_SAMPLES_CHOICES)),
        )
        reg_key = (
            trial.suggest_categorical("reg_num_leaves", list(NUM_LEAVES_CHOICES)),
            trial.suggest_categorical("reg_min_child_samples", list(MIN_CHILD_SAMPLES_CHOICES)),
        )

        oof_frames = []
        native_fold_metrics = []
        for fold_id, ctx in fold_contexts.items():
            pred = ctx["cls_pred"][cls_key] * ctx["reg_pred"][reg_key]
            pred_log = np.log1p(pred)
            oof_frame = oo.build_oof_frame(
                ctx["val_keys"], ctx["y_val_raw"], pred_log, pred, ctx["mase_scale"],
                stage="p13_hurdle_final_hpo", model_family="LIGHTGBM_HURDLE",
                config_id=f"trial{trial.number}", seed=SEED, horizon=horizon, fold_id=fold_id,
            )
            oof_frames.append(oof_frame)
            native_fold_metrics.append(ev.compute_metrics(ctx["y_val_raw"], pred, ctx["mase_scale"]))

        pooled_native_oof = pd.concat(oof_frames, ignore_index=True)
        filtered_oof = hc.filter_pooled_oof_by_common_keys(pooled_native_oof, horizon_common_keys)
        pooled_common = ev.compute_metrics(
            filtered_oof["y_true"].to_numpy(), filtered_oof["y_pred"].to_numpy(), filtered_oof["mase_scale"].to_numpy(),
        )
        per_fold_common = hc.per_fold_metrics_common(filtered_oof)
        worst_fold_wape = max(m["wape"] for m in per_fold_common.values())

        trial_summaries.append({
            "trial_id": trial.number,
            "params": {
                "cls_num_leaves": cls_key[0], "cls_min_child_samples": cls_key[1],
                "reg_num_leaves": reg_key[0], "reg_min_child_samples": reg_key[1],
            },
            "status": "ok",
            "pooled_wape": pooled_common["wape"], "pooled_bias": pooled_common["bias"],
            "pooled_rmse": pooled_common["rmse"], "pooled_mae": pooled_common["mae"],
            "pooled_mase": pooled_common["mase"], "worst_fold_wape": worst_fold_wape,
            "guardrail_pass": abs(pooled_common["bias"]) <= cfg.BIAS_GUARDRAIL_ABS_PCT,
            "per_fold_metrics_common": per_fold_common,
            "native_fold_metrics_diagnostic": native_fold_metrics,
        })
        filtered_oof_by_trial[trial.number] = filtered_oof
        return pooled_common["wape"]

    study.optimize(objective, n_trials=16, n_jobs=1)
    hc.check_grid_fully_and_uniquely_evaluated(trial_summaries, HURDLE_GRID_SEARCH_SPACE, horizon)

    ok_trials = [t for t in trial_summaries if t["status"] == "ok"]
    selection = hc.select_best_trial(ok_trials)

    result = {
        "horizon": horizon, "family": "lightgbm_hurdle", "n_trials": 16, "n_fits_total": 32,
        "seed": SEED, "sampler_seed": SAMPLER_SEED,
        "grid_search_space": HURDLE_GRID_SEARCH_SPACE,
        "fixed_params_classifier": CLS_FIXED_PARAMS,
        "fixed_params_regressor": P13_FIXED_PARAMS,
        "shared_fixed_params": {
            "feature_fraction": FEATURE_FRACTION, "bagging_fraction": BAGGING_FRACTION,
            "bagging_freq": BAGGING_FREQ, "lambda_l1": LAMBDA_L1, "lambda_l2": LAMBDA_L2,
        },
        "evaluation_population": "five_family_common_keys",
        "common_eval_key_path": str(COMMON_KEYS_PATH),
        "common_eval_key_count": len(horizon_common_keys),
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

        sel_filtered_oof = filtered_oof_by_trial[sel["trial_id"]]

        quartile_rows = [
            (fold_id, center_id, sku_id, q)
            for fold_id, ctx in fold_contexts.items()
            for (center_id, sku_id), q in ctx["demand_quartile_map"].items()
        ]
        quartile_df = pd.DataFrame(quartile_rows, columns=["fold_id", "center_id", "sku_id", "demand_quartile"])
        diag_df = sel_filtered_oof.merge(quartile_df, on=["fold_id", "center_id", "sku_id"], how="left")
        diag_df["demand_quartile"] = diag_df["demand_quartile"].fillna("no_train_history")

        def _wape_bias(df: pd.DataFrame) -> dict:
            m = ev.compute_metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy(), df["mase_scale"].to_numpy())
            return {"wape": m["wape"], "bias": m["bias"]}

        zero = diag_df[diag_df["y_true"] == 0]
        nonzero = diag_df[diag_df["y_true"] > 0]
        result["selected_diagnostics"] = {
            "note": "selection rule에는 사용되지 않은 진단용 breakdown",
            "zero_rows": {"mean_pred": float(zero["y_pred"].mean()), "sum_pred": float(zero["y_pred"].sum())},
            "nonzero_rows": _wape_bias(nonzero),
            "by_demand_quartile": {name: _wape_bias(grp) for name, grp in diag_df.groupby("demand_quartile")},
        }

        oof_dir = OUT_DIR / "oof"
        oof_dir.mkdir(parents=True, exist_ok=True)
        sel_filtered_oof.to_parquet(
            oof_dir / f"oof_lightgbm_hurdle_h{horizon}_trial{sel['trial_id']}_seed{SEED}_common.parquet",
            index=False,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"p13_hurdle_lightgbm_h{horizon}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"저장 완료: {out_path}")


if __name__ == "__main__":
    main()
