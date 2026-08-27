"""
p13_lgbm_h1_hurdle_diagnostic.py

LightGBM h1 soft-Hurdle(분류 x 조건부회귀) 진단(READ-ONLY, 그룹별 보정 없음).
classifier: P(y>0), trial 3 구조적 HP를 objective만 binary로 바꿔 고정 사용(HPO/threshold 탐색 없음).
conditional regressor: train의 y>0 행만, raw target, trial 3 HP 그대로 고정.
최종 예측 = P(y>0) * max(conditional_pred, 0) (soft 결합, hard threshold 미사용).
기존 log1p+naive expm1, raw-target(전체 행) baseline과 동일 fold/평가로 비교한다.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.forecasting.common import config as cfg
from src.forecasting.common import evaluator as ev
from src.forecasting.common import folds as day3_folds
from src.forecasting.common.data_loader import load_development
from src.forecasting.machine_learning.lightgbm.config import P13_FIXED_PARAMS
from src.forecasting.machine_learning.lightgbm.preprocessing import LGBMPreprocessor
from src.forecasting.pipeline.p13 import hpo_common as hc
from scripts.diagnostics.p13_lgbm_h1_demand_quartile_bias_diagnostic import QUARTILE_LABELS
from scripts.diagnostics.p13_lgbm_h1_target_transform_diagnostic import (
    COMMON_KEYS_PATH,
    HORIZON,
    SEED,
    SOURCE_JSON_PATH,
    _fit_lgbm,
    _load_best_trial,
)

CLS_FIXED_PARAMS = {**P13_FIXED_PARAMS, "objective": "binary"}
PRED_COLS = ("pred_naive", "pred_raw", "pred_hurdle")
OUTPUT_PATH = Path("outputs/diagnostics/target_transform_bias/lightgbm_h1_hurdle_diagnostic.json")


def _fit_lgbm_classifier(X_train, y_train, hp: dict, categorical_feature: list) -> LGBMClassifier:
    model = LGBMClassifier(
        num_leaves=hp["num_leaves"],
        min_child_samples=hp["min_child_samples"],
        colsample_bytree=hp["feature_fraction"],
        subsample=hp["bagging_fraction"],
        subsample_freq=hp["bagging_freq"],
        reg_alpha=hp["lambda_l1"],
        reg_lambda=hp["lambda_l2"],
        random_state=SEED,
        **CLS_FIXED_PARAMS,
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
        y_train_log = np.log1p(y_train_raw)

        log_model = _fit_lgbm(X_train, y_train_log, hp, cat_features)
        pred_naive = ev.inverse_transform_prediction(log_model.predict(X_val))

        raw_model = _fit_lgbm(X_train, y_train_raw, hp, cat_features)
        pred_raw = np.clip(raw_model.predict(X_val), 0.0, None)

        pos_mask = y_train_raw > 0
        cls_model = _fit_lgbm_classifier(X_train, pos_mask.astype(int), hp, cat_features)
        prob_val = cls_model.predict_proba(X_val)[:, 1]
        cond_reg_model = _fit_lgbm(X_train.iloc[pos_mask], y_train_raw[pos_mask], hp, cat_features)
        cond_pred_val = np.clip(cond_reg_model.predict(X_val), 0.0, None)
        pred_hurdle = prob_val * cond_pred_val

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
            "pred_naive": pred_naive,
            "pred_raw": pred_raw,
            "pred_hurdle": pred_hurdle,
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
        "classifier_note": "trial 3 구조적 HP를 objective만 binary로 바꿔 고정 사용한 진단용 값 - 최적 분류기 HP 아님",
        "common_eval_key_count": len(horizon_common_keys),
        "pooled_metrics_common": {col: _metrics(filtered, col) for col in PRED_COLS},
        "zero_rows": {
            col: {"mean_pred": float(zero[col].mean()), "sum_pred": float(zero[col].sum())} for col in PRED_COLS
        },
        "nonzero_rows": {
            col: {k: v for k, v in _metrics(nonzero, col).items() if k in ("wape", "bias")} for col in PRED_COLS
        },
        "by_demand_quartile": {
            name: {col: {k: v for k, v in _metrics(grp, col).items() if k in ("wape", "bias")} for col in PRED_COLS}
            for name, grp in filtered.groupby("demand_quartile")
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
