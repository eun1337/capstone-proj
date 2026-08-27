"""
p13_lgbm_h1_hurdle_tweedie_diagnostic.py

LightGBM h1 soft-Hurdle의 conditional regressor로 Tweedie(objective="tweedie")만 최소 진단(READ-ONLY).
classifier/fold/feature/seed/HP/common evaluation keys는 p13_lgbm_h1_hurdle_target_scale_diagnostic.py와
동일하게 고정하고, conditional regressor의 objective만 tweedie_variance_power in {1.1,1.2,1.5,1.8}로
바꿔 4개 후보를 비교한다. target은 train의 y>0 raw demand. 그 외 HP/HPO/threshold/class weight/
calibration 변경 없음. Tweedie predict()는 이미 raw(response) scale이라 별도 inverse transform이 없다.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.forecasting.common import config as cfg
from src.forecasting.common import evaluator as ev
from src.forecasting.common import folds as day3_folds
from src.forecasting.common.data_loader import load_development
from src.forecasting.machine_learning.lightgbm.config import P13_FIXED_PARAMS
from src.forecasting.machine_learning.lightgbm.preprocessing import LGBMPreprocessor
from src.forecasting.pipeline.p13 import hpo_common as hc
from scripts.diagnostics.p13_lgbm_h1_demand_quartile_bias_diagnostic import QUARTILE_LABELS
from scripts.diagnostics.p13_lgbm_h1_hurdle_diagnostic import _fit_lgbm_classifier
from scripts.diagnostics.p13_lgbm_h1_target_transform_diagnostic import (
    COMMON_KEYS_PATH,
    HORIZON,
    SEED,
    SOURCE_JSON_PATH,
    _load_best_trial,
)

TWEEDIE_VARIANCE_POWERS = (1.1, 1.2, 1.5, 1.8)
TWEEDIE_FIXED_PARAMS = {**P13_FIXED_PARAMS, "objective": "tweedie"}
OUTPUT_PATH = Path("outputs/diagnostics/target_transform_bias/lightgbm_h1_hurdle_tweedie_diagnostic.json")


def _fit_lgbm_tweedie(X_train, y_train, hp: dict, categorical_feature: list, variance_power: float) -> LGBMRegressor:
    model = LGBMRegressor(
        num_leaves=hp["num_leaves"],
        min_child_samples=hp["min_child_samples"],
        colsample_bytree=hp["feature_fraction"],
        subsample=hp["bagging_fraction"],
        subsample_freq=hp["bagging_freq"],
        reg_alpha=hp["lambda_l1"],
        reg_lambda=hp["lambda_l2"],
        random_state=SEED,
        tweedie_variance_power=variance_power,
        **TWEEDIE_FIXED_PARAMS,
    )
    model.fit(X_train, y_train, categorical_feature=categorical_feature)
    return model


def main() -> None:
    best_trial = _load_best_trial(SOURCE_JSON_PATH)
    hp = best_trial["params"]

    common_eval_keys = hc.load_common_eval_keys(COMMON_KEYS_PATH)
    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()
    folds = day3_folds.generate_expanding_folds(sub_a, 2023, HORIZON)
    hc.check_common_keys_match_p13_period(common_eval_keys, folds, HORIZON, COMMON_KEYS_PATH)
    horizon_common_keys = common_eval_keys[common_eval_keys["horizon"] == HORIZON]

    target_col = cfg.TARGET_COLS[HORIZON]
    pred_cols = [f"pred_tweedie_{p}" for p in TWEEDIE_VARIANCE_POWERS]

    fold_frames = []
    for fold in folds:
        fold_id = fold["fold"]
        train_df = sub_a.loc[fold["train_mask"]]
        val_df = sub_a.loc[fold["val_mask"]]

        preprocessor = LGBMPreprocessor()
        X_train = preprocessor.fit_transform(train_df, HORIZON)
        X_val = preprocessor.transform(val_df)
        cat_features = preprocessor.categorical_feature_names

        y_train_raw = train_df[target_col].to_numpy(dtype=float)
        y_val_raw = val_df[target_col].to_numpy(dtype=float)

        pos_mask = y_train_raw > 0
        cls_model = _fit_lgbm_classifier(X_train, pos_mask.astype(int), hp, cat_features)
        prob_val = cls_model.predict_proba(X_val)[:, 1]

        X_train_pos = X_train.iloc[pos_mask]
        y_train_pos_raw = y_train_raw[pos_mask]

        fold_data = {}
        for power, col in zip(TWEEDIE_VARIANCE_POWERS, pred_cols):
            tweedie_model = _fit_lgbm_tweedie(X_train_pos, y_train_pos_raw, hp, cat_features, power)
            cond_pred = np.clip(tweedie_model.predict(X_val), 0.0, None)
            fold_data[col] = prob_val * cond_pred

        sku_avg = train_df.groupby(["center_id", "sku_id"])["qty"].mean()
        demand_quartile_map = pd.qcut(sku_avg, q=4, labels=QUARTILE_LABELS, duplicates="drop").to_dict()

        val_keys = val_df[["center_id", "sku_id", "week_st"]].copy()
        val_keys["target_date"] = val_keys["week_st"] + pd.Timedelta(weeks=HORIZON)
        mase_scale = ev.build_mase_scale(train_df[["center_id", "sku_id", "week_st", "qty"]], val_keys)
        sku_idx = pd.MultiIndex.from_frame(val_keys[["center_id", "sku_id"]])

        fold_frames.append(pd.DataFrame({
            "horizon": HORIZON,
            "fold_id": fold_id,
            "center_id": val_keys["center_id"].to_numpy(),
            "sku_id": val_keys["sku_id"].to_numpy(),
            "week_st": val_keys["week_st"].to_numpy(),
            "target_date": val_keys["target_date"].to_numpy(),
            "y_true": y_val_raw,
            "mase_scale": mase_scale,
            **fold_data,
            "demand_quartile": np.asarray(sku_idx.map(demand_quartile_map), dtype=object),
        }))

    pooled = pd.concat(fold_frames, ignore_index=True)
    filtered = hc.filter_pooled_oof_by_common_keys(pooled, horizon_common_keys)

    def _metrics(df: pd.DataFrame, col: str) -> dict:
        m = ev.compute_metrics(df["y_true"].to_numpy(), df[col].to_numpy(), df["mase_scale"].to_numpy())
        return {"wape": m["wape"], "bias": m["bias"], "rmse": m["rmse"], "mae": m["mae"], "mase": m["mase"]}

    zero = filtered[filtered["y_true"] == 0]
    nonzero = filtered[filtered["y_true"] > 0]

    result = {
        "family": "lightgbm",
        "horizon": HORIZON,
        "source_trial_id": best_trial["trial_id"],
        "source_trial_params": hp,
        "tweedie_variance_powers": list(TWEEDIE_VARIANCE_POWERS),
        "common_eval_key_count": len(horizon_common_keys),
        "pooled_metrics_common": {col: _metrics(filtered, col) for col in pred_cols},
        "zero_rows": {
            col: {"mean_pred": float(zero[col].mean()), "sum_pred": float(zero[col].sum())} for col in pred_cols
        },
        "nonzero_rows": {
            col: {k: v for k, v in _metrics(nonzero, col).items() if k in ("wape", "bias")} for col in pred_cols
        },
        "by_demand_quartile": {
            name: {col: {k: v for k, v in _metrics(grp, col).items() if k in ("wape", "bias")} for col in pred_cols}
            for name, grp in filtered.groupby("demand_quartile")
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
