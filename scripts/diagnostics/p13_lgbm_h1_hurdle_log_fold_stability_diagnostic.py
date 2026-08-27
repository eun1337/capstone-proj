"""
p13_lgbm_h1_hurdle_log_fold_stability_diagnostic.py

LightGBM h1 soft-Hurdle(classifier + conditional log1p regressor) 결과의 fold별 안정성 진단(READ-ONLY).
p13_lgbm_h1_hurdle_target_scale_diagnostic.py와 동일한 classifier/fold/feature/HP/common evaluation keys를
그대로 사용해 conditional=log1p 조합만 재현하고, pooled 결과가 fold별로도 일관된지 확인한다.
새 구조/HP/threshold/class weight/calibration 없음.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.common import config as cfg
from src.forecasting.common import evaluator as ev
from src.forecasting.common import folds as day3_folds
from src.forecasting.common.data_loader import load_development
from src.forecasting.machine_learning.lightgbm.preprocessing import LGBMPreprocessor
from src.forecasting.pipeline.p13 import hpo_common as hc
from scripts.diagnostics.p13_lgbm_h1_demand_quartile_bias_diagnostic import QUARTILE_LABELS
from scripts.diagnostics.p13_lgbm_h1_hurdle_diagnostic import _fit_lgbm_classifier
from scripts.diagnostics.p13_lgbm_h1_target_transform_diagnostic import (
    COMMON_KEYS_PATH,
    HORIZON,
    SOURCE_JSON_PATH,
    _fit_lgbm,
    _load_best_trial,
)

OUTPUT_PATH = Path("outputs/diagnostics/target_transform_bias/lightgbm_h1_hurdle_log_fold_stability.json")


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
        y_train_pos_log = np.log1p(y_train_raw[pos_mask])
        reg_log_model = _fit_lgbm(X_train_pos, y_train_pos_log, hp, cat_features)
        cond_pred_log = ev.inverse_transform_prediction(reg_log_model.predict(X_val))
        pred_hurdle_log = prob_val * cond_pred_log

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
            "pred_hurdle_log": pred_hurdle_log,
            "demand_quartile": np.asarray(sku_idx.map(demand_quartile_map), dtype=object),
        }))

    pooled = pd.concat(fold_frames, ignore_index=True)
    filtered = hc.filter_pooled_oof_by_common_keys(pooled, horizon_common_keys)

    def _wape_bias(df: pd.DataFrame) -> dict:
        m = ev.compute_metrics(df["y_true"].to_numpy(), df["pred_hurdle_log"].to_numpy(), df["mase_scale"].to_numpy())
        return {"wape": m["wape"], "bias": m["bias"]}

    fold_results = {}
    for fold_id, grp in filtered.groupby("fold_id"):
        q4 = grp[grp["demand_quartile"] == "Q4"]
        zero = grp[grp["y_true"] == 0]
        nonzero = grp[grp["y_true"] > 0]
        fold_results[int(fold_id)] = {
            "overall": _wape_bias(grp),
            "q4": _wape_bias(q4),
            "zero_rows_sum_pred": float(zero["pred_hurdle_log"].sum()),
            "nonzero_rows": _wape_bias(nonzero),
        }

    result = {
        "family": "lightgbm",
        "horizon": HORIZON,
        "source_trial_id": best_trial["trial_id"],
        "source_trial_params": hp,
        "common_eval_key_count": len(horizon_common_keys),
        "pooled_overall": _wape_bias(filtered),
        "folds": fold_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
