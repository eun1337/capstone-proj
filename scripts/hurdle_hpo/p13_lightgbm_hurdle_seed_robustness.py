"""
p13_lightgbm_hurdle_seed_robustness.py

Hurdle-LightGBM h1/h2/h4 정식 HPO에서 선택된 HP를 고정하고 seed만 42(기존 결과 재사용)/123/456으로
바꿔 WAPE/Bias 안정성을 확인한다. 새 HPO/search space 탐색 없음. 기존 outputs/hurdle_hpo 결과는
읽기만 하고 수정하지 않는다. 구조는 기존 정식 Hurdle과 동일(classifier binary x conditional
regressor log1p+expm1, soft 결합).
"""

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
from src.forecasting.machine_learning.lightgbm.preprocessing import LGBMPreprocessor
from src.forecasting.pipeline.p13 import hpo_common as hc
from scripts.hurdle_hpo.p13_lightgbm_hurdle_h1_hpo import (
    BAGGING_FRACTION,
    BAGGING_FREQ,
    CLS_FIXED_PARAMS,
    COMMON_KEYS_PATH,
    FEATURE_FRACTION,
    LAMBDA_L1,
    LAMBDA_L2,
    OUT_DIR,
    P13_FIXED_PARAMS,
)

HORIZON_HP = {
    1: {"cls": (31, 100), "reg": (31, 100)},
    2: {"cls": (31, 1000), "reg": (31, 100)},
    4: {"cls": (31, 1000), "reg": (31, 100)},
}
SEEDS_TO_RUN = (123, 456)
OFFICIAL_SEED = cfg.P13_MODEL_SEED
OUT_PATH = OUT_DIR / "robustness" / "seed_stability.json"


def _fit_cls(X_train, y_train, num_leaves, min_child_samples, categorical_feature, seed):
    model = LGBMClassifier(
        num_leaves=num_leaves, min_child_samples=min_child_samples,
        colsample_bytree=FEATURE_FRACTION, subsample=BAGGING_FRACTION, subsample_freq=BAGGING_FREQ,
        reg_alpha=LAMBDA_L1, reg_lambda=LAMBDA_L2, random_state=seed, **CLS_FIXED_PARAMS,
    )
    model.fit(X_train, y_train, categorical_feature=categorical_feature)
    return model


def _fit_reg(X_train, y_train, num_leaves, min_child_samples, categorical_feature, seed):
    model = LGBMRegressor(
        num_leaves=num_leaves, min_child_samples=min_child_samples,
        colsample_bytree=FEATURE_FRACTION, subsample=BAGGING_FRACTION, subsample_freq=BAGGING_FREQ,
        reg_alpha=LAMBDA_L1, reg_lambda=LAMBDA_L2, random_state=seed, **P13_FIXED_PARAMS,
    )
    model.fit(X_train, y_train, categorical_feature=categorical_feature)
    return model


def _run_seed(sub_a, horizon, folds, horizon_common_keys, target_col, hp, seed):
    oof_frames = []
    for fold in folds:
        fold_id = fold["fold"]
        train_df = sub_a.loc[fold["train_mask"]]
        val_df = sub_a.loc[fold["val_mask"]]

        preprocessor = LGBMPreprocessor()
        X_train = preprocessor.fit_transform(train_df, horizon)
        X_val = preprocessor.transform(val_df)
        cat_features = preprocessor.categorical_feature_names

        y_train_raw = train_df[target_col].to_numpy(dtype=float)
        y_val_raw = val_df[target_col].to_numpy(dtype=float)
        pos_mask = y_train_raw > 0

        cls_model = _fit_cls(X_train, pos_mask.astype(int), hp["cls"][0], hp["cls"][1], cat_features, seed)
        prob_val = cls_model.predict_proba(X_val)[:, 1]

        X_train_pos = X_train.iloc[pos_mask]
        y_train_pos_log = np.log1p(y_train_raw[pos_mask])
        reg_model = _fit_reg(X_train_pos, y_train_pos_log, hp["reg"][0], hp["reg"][1], cat_features, seed)
        cond_pred = ev.inverse_transform_prediction(reg_model.predict(X_val))
        pred = prob_val * cond_pred
        pred_log = np.log1p(pred)

        val_keys = val_df[["center_id", "sku_id", "week_st"]].copy()
        val_keys["target_date"] = val_keys["week_st"] + pd.Timedelta(weeks=horizon)
        mase_scale = ev.build_mase_scale(train_df[["center_id", "sku_id", "week_st", "qty"]], val_keys)

        oof_frames.append(oo.build_oof_frame(
            val_keys, y_val_raw, pred_log, pred, mase_scale,
            stage="p13_hurdle_seed_robustness", model_family="LIGHTGBM_HURDLE",
            config_id=f"seed{seed}", seed=seed, horizon=horizon, fold_id=fold_id,
        ))

    pooled_native_oof = pd.concat(oof_frames, ignore_index=True)
    filtered_oof = hc.filter_pooled_oof_by_common_keys(pooled_native_oof, horizon_common_keys)
    pooled = ev.compute_metrics(
        filtered_oof["y_true"].to_numpy(), filtered_oof["y_pred"].to_numpy(), filtered_oof["mase_scale"].to_numpy(),
    )
    per_fold = hc.per_fold_metrics_common(filtered_oof)
    return (
        {k: pooled[k] for k in ("wape", "bias", "rmse", "mae", "mase")},
        {str(k): {"wape": v["wape"], "bias": v["bias"]} for k, v in per_fold.items()},
    )


def main() -> None:
    results = {}
    for horizon, hp in HORIZON_HP.items():
        official_path = OUT_DIR / f"p13_hurdle_lightgbm_h{horizon}.json"
        with open(official_path, encoding="utf-8") as f:
            official = json.load(f)
        sel_params = official["selection"]["selected"]["params"]
        expected = {
            "cls_num_leaves": hp["cls"][0], "cls_min_child_samples": hp["cls"][1],
            "reg_num_leaves": hp["reg"][0], "reg_min_child_samples": hp["reg"][1],
        }
        if sel_params != expected:
            raise ValueError(f"h{horizon}: 지정된 고정 HP가 기존 선택 결과와 다름: {sel_params} vs {expected}")

        seed_results = {
            OFFICIAL_SEED: {
                "pooled": official["pooled_metrics_common"],
                "per_fold": {
                    str(k): {"wape": v["wape"], "bias": v["bias"]}
                    for k, v in official["per_fold_metrics_common"].items()
                },
            },
        }

        common_eval_keys = hc.load_common_eval_keys(COMMON_KEYS_PATH)
        dev = load_development()
        sub_a = dev[dev["center_id"] == "A"].copy()
        folds = day3_folds.generate_expanding_folds(sub_a, 2023, horizon)
        hc.check_common_keys_match_p13_period(common_eval_keys, folds, horizon, COMMON_KEYS_PATH)
        horizon_common_keys = common_eval_keys[common_eval_keys["horizon"] == horizon]
        target_col = cfg.TARGET_COLS[horizon]

        for seed in SEEDS_TO_RUN:
            pooled, per_fold = _run_seed(sub_a, horizon, folds, horizon_common_keys, target_col, hp, seed)
            seed_results[seed] = {"pooled": pooled, "per_fold": per_fold}

        all_seeds = (OFFICIAL_SEED, *SEEDS_TO_RUN)
        wapes = [seed_results[s]["pooled"]["wape"] for s in all_seeds]
        biases = [seed_results[s]["pooled"]["bias"] for s in all_seeds]

        results[str(horizon)] = {
            "fixed_hp": hp,
            "seeds": {str(s): seed_results[s] for s in all_seeds},
            "wape_mean": float(np.mean(wapes)), "wape_std": float(np.std(wapes)),
            "wape_min": float(np.min(wapes)), "wape_max": float(np.max(wapes)),
            "bias_mean": float(np.mean(biases)), "bias_std": float(np.std(biases)),
            "bias_min": float(np.min(biases)), "bias_max": float(np.max(biases)),
            "bias_mean_guardrail_pass": abs(float(np.mean(biases))) <= cfg.BIAS_GUARDRAIL_ABS_PCT,
            "guardrail_pass_by_seed": {
                str(s): abs(seed_results[s]["pooled"]["bias"]) <= cfg.BIAS_GUARDRAIL_ABS_PCT for s in all_seeds
            },
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
