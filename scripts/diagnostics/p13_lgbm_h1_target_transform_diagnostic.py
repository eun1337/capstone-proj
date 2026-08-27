"""
p13_lgbm_h1_target_transform_diagnostic.py

LightGBM h1 target-transform 진단(READ-ONLY 실험). 기존 P13 h1 HPO에서 pooled WAPE가
가장 낮았던 trial(configuration)을 고정해 재사용하고, 다음 3가지 variant를 같은
2023 expanding 4-fold/같은 common evaluation key로 비교한다.

1. log1p 학습 + naive expm1 역변환 (기존 P13과 동일 fit, 이 스크립트에서 재현)
2. raw target 학습 (log1p 없이 직접 fit, inverse transform 없음)
3. log1p 학습 + Duan smearing 역변환 (동일 log1p fit을 재사용, train-fold residual로만 factor 계산)

기존 outputs/hpo 결과는 읽기만 하고 수정하지 않으며, HPO/모델 선택 로직은 다시 만들지 않는다.
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

HORIZON = 1
SEED = cfg.P13_MODEL_SEED
SOURCE_JSON_PATH = Path("outputs/hpo/lgbm/p13_lightgbm_h1.json")
COMMON_KEYS_PATH = Path("outputs/audits/p13/five_family_common_evaluation_keys.parquet")
OUTPUT_PATH = Path("outputs/diagnostics/target_transform_bias/lightgbm_h1_smearing.json")


def _load_best_trial(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    ok_trials = [t for t in data["all_trials"] if t["status"] == "ok"]
    return min(ok_trials, key=lambda t: t["pooled_wape"])


def _fit_lgbm(X_train, y_train, hp: dict, categorical_feature: list) -> LGBMRegressor:
    model = LGBMRegressor(
        num_leaves=hp["num_leaves"],
        min_child_samples=hp["min_child_samples"],
        colsample_bytree=hp["feature_fraction"],
        subsample=hp["bagging_fraction"],
        subsample_freq=hp["bagging_freq"],
        reg_alpha=hp["lambda_l1"],
        reg_lambda=hp["lambda_l2"],
        random_state=SEED,
        **P13_FIXED_PARAMS,
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
    fold_smearing_records = []
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

        # smearing factor는 train-fold in-sample residual로만 계산한다(validation 정보 미사용 - leakage 방지).
        residual_log = y_train_log - pred_log_train
        smearing_factor = float(np.mean(np.exp(residual_log)))

        pred_naive = ev.inverse_transform_prediction(pred_log_val)
        pred_smearing = np.clip(np.exp(pred_log_val) * smearing_factor - 1.0, 0.0, None)

        raw_model = _fit_lgbm(X_train, y_train_raw, hp, cat_features)
        pred_raw = np.clip(raw_model.predict(X_val), 0.0, None)

        val_keys = val_df[["center_id", "sku_id", "week_st"]].copy()
        val_keys["target_date"] = val_keys["week_st"] + pd.Timedelta(weeks=HORIZON)
        mase_scale = ev.build_mase_scale(train_df[["center_id", "sku_id", "week_st", "qty"]], val_keys)

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
        }))
        fold_smearing_records.append({"fold_id": fold_id, "smearing_factor": smearing_factor})

    pooled = pd.concat(fold_frames, ignore_index=True)
    filtered = hc.filter_pooled_oof_by_common_keys(pooled, horizon_common_keys)

    def _metrics(pred_col: str) -> dict:
        return ev.compute_metrics(filtered["y_true"].to_numpy(), filtered[pred_col].to_numpy(), filtered["mase_scale"].to_numpy())

    pooled_naive_recomputed = _metrics("pred_naive")
    pooled_raw = _metrics("pred_raw")
    pooled_smearing = _metrics("pred_smearing")

    for rec in fold_smearing_records:
        grp = filtered[filtered["fold_id"] == rec["fold_id"]]
        naive_fold = ev.compute_metrics(grp["y_true"].to_numpy(), grp["pred_naive"].to_numpy(), grp["mase_scale"].to_numpy())
        smearing_fold = ev.compute_metrics(grp["y_true"].to_numpy(), grp["pred_smearing"].to_numpy(), grp["mase_scale"].to_numpy())
        rec["naive_wape"] = naive_fold["wape"]
        rec["naive_bias"] = naive_fold["bias"]
        rec["smearing_wape"] = smearing_fold["wape"]
        rec["smearing_bias"] = smearing_fold["bias"]

    result = {
        "family": "lightgbm",
        "horizon": HORIZON,
        "seed": SEED,
        "source_trial_id": best_trial["trial_id"],
        "source_trial_params": hp,
        "source_json_path": str(SOURCE_JSON_PATH),
        "common_eval_key_path": str(COMMON_KEYS_PATH),
        "common_eval_key_count": len(horizon_common_keys),
        "fold_smearing": fold_smearing_records,
        "pooled_metrics_common": {
            "log1p_naive_existing_p13": {
                "wape": best_trial["pooled_wape"], "bias": best_trial["pooled_bias"],
                "rmse": best_trial["pooled_rmse"], "mae": best_trial["pooled_mae"], "mase": best_trial["pooled_mase"],
                "source": "read_only_from_outputs/hpo/lgbm/p13_lightgbm_h1.json",
            },
            "log1p_naive_recomputed_this_run": pooled_naive_recomputed,
            "raw_target": pooled_raw,
            "log1p_smearing": pooled_smearing,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
