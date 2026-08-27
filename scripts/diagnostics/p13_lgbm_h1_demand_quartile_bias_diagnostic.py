"""
p13_lgbm_h1_demand_quartile_bias_diagnostic.py

LightGBM h1 과소예측 Bias가 특정 수요 규모 SKU에 집중되는지 진단(READ-ONLY, 그룹별 보정 없음).
p13_lgbm_h1_target_transform_diagnostic.py와 동일한 fold/HP/모델 fit을 재사용하되,
각 validation row를 "그 fold의 TRAIN 데이터에서 계산한 SKU별 평균 주간수요" 4분위(Q1~Q4)로 나눠
naive/raw/smearing 3가지 예측을 그룹별로 비교한다. 그룹 분류에 validation 실제값은 쓰지 않는다.
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
from scripts.diagnostics.p13_lgbm_h1_target_transform_diagnostic import (
    COMMON_KEYS_PATH,
    HORIZON,
    SOURCE_JSON_PATH,
    _fit_lgbm,
    _load_best_trial,
)

QUARTILE_LABELS = ["Q1", "Q2", "Q3", "Q4"]
OUTPUT_PATH = Path("outputs/diagnostics/target_transform_bias/lightgbm_h1_demand_quartile_bias.json")


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
        pred_log_train = log_model.predict(X_train)
        pred_log_val = log_model.predict(X_val)

        residual_log = y_train_log - pred_log_train
        smearing_factor = float(np.mean(np.exp(residual_log)))

        pred_naive = ev.inverse_transform_prediction(pred_log_val)
        pred_smearing = np.clip(np.exp(pred_log_val) * smearing_factor - 1.0, 0.0, None)

        raw_model = _fit_lgbm(X_train, y_train_raw, hp, cat_features)
        pred_raw = np.clip(raw_model.predict(X_val), 0.0, None)

        # train fold의 SKU별 과거 평균 주간수요로만 4분위를 나눈다(미래 validation 실제값 미사용).
        sku_avg = train_df.groupby(["center_id", "sku_id"])["qty"].mean()
        quartile = pd.qcut(sku_avg, q=4, labels=QUARTILE_LABELS, duplicates="drop")
        quartile_map = quartile.to_dict()

        val_keys = val_df[["center_id", "sku_id", "week_st"]].copy()
        val_keys["target_date"] = val_keys["week_st"] + pd.Timedelta(weeks=HORIZON)
        mase_scale = ev.build_mase_scale(train_df[["center_id", "sku_id", "week_st", "qty"]], val_keys)
        demand_quartile = pd.MultiIndex.from_frame(val_keys[["center_id", "sku_id"]]).map(quartile_map)

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
            "pred_smearing": pred_smearing,
            "pred_raw": pred_raw,
            "demand_quartile": np.asarray(demand_quartile, dtype=object),
        }))

    pooled = pd.concat(fold_frames, ignore_index=True)
    filtered = hc.filter_pooled_oof_by_common_keys(pooled, horizon_common_keys)
    filtered["demand_quartile"] = filtered["demand_quartile"].fillna("no_train_history")

    def _group_metrics(grp: pd.DataFrame, pred_col: str) -> dict:
        m = ev.compute_metrics(grp["y_true"].to_numpy(), grp[pred_col].to_numpy(), grp["mase_scale"].to_numpy())
        return {"wape": m["wape"], "bias": m["bias"], "rmse": m["rmse"], "mae": m["mae"], "mase": m["mase"]}

    groups = {}
    for group_name, grp in filtered.groupby("demand_quartile"):
        groups[group_name] = {
            "n_sku": int(grp["sku_id"].nunique()),
            "n_rows": int(len(grp)),
            "sum_y_true": float(grp["y_true"].sum()),
            "sum_pred_naive": float(grp["pred_naive"].sum()),
            "sum_pred_raw": float(grp["pred_raw"].sum()),
            "sum_pred_smearing": float(grp["pred_smearing"].sum()),
            "naive": _group_metrics(grp, "pred_naive"),
            "raw": _group_metrics(grp, "pred_raw"),
            "smearing": _group_metrics(grp, "pred_smearing"),
        }

    result = {
        "family": "lightgbm",
        "horizon": HORIZON,
        "source_trial_id": best_trial["trial_id"],
        "source_trial_params": hp,
        "common_eval_key_count": len(horizon_common_keys),
        "quartile_definition": "각 fold의 TRAIN 데이터에서 계산한 SKU별 평균 주간수요(qty) 4분위(Q1=최저~Q4=최고), fold마다 재계산",
        "groups": groups,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
