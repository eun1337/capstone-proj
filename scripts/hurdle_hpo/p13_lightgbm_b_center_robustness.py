"""
p13_lightgbm_b_center_robustness.py

B센터 2023 post-regime walk-forward robustness (재튜닝 없음, model selection 없음).
A센터에서 고정한 설정을 그대로 적용해 single LightGBM(P13 최저-WAPE trial, 공식 selected 아님)과
Hurdle-LightGBM(정식 HPO selected)을 동일 B fold/population/preprocessing/evaluator/MASE scale로 비교한다.
2024 holdout은 사용하지 않는다(load_development()만 사용).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

import src.forecasting.pipeline.p13.lightgbm as p13_lgbm
from src.forecasting.common import config as cfg
from src.forecasting.common import evaluator as ev
from src.forecasting.common import folds as day3_folds
from src.forecasting.common import oof as oo
from src.forecasting.common.data_loader import load_development
from src.forecasting.machine_learning.lightgbm.preprocessing import LGBMPreprocessor
from src.forecasting.pipeline.robustness import frozen_checks_ml as row_checks
from scripts.hurdle_hpo.p13_lightgbm_hurdle_h1_hpo import (
    BAGGING_FRACTION,
    BAGGING_FREQ,
    CLS_FIXED_PARAMS,
    FEATURE_FRACTION,
    LAMBDA_L1,
    LAMBDA_L2,
    P13_FIXED_PARAMS,
)

SEED = cfg.P13_MODEL_SEED
OUT_DIR = Path("outputs/hurdle_hpo/lgbm_hurdle/b_center")
OOF_DIR = OUT_DIR / "oof"

SINGLE_HP = {
    "num_leaves": 31, "min_child_samples": 100, "feature_fraction": 0.4,
    "bagging_fraction": 0.4, "bagging_freq": 1, "lambda_l1": 0.0, "lambda_l2": 0.0,
}
HURDLE_HP = {
    1: {"cls": (31, 100), "reg": (31, 100)},
    2: {"cls": (31, 1000), "reg": (31, 100)},
    4: {"cls": (31, 1000), "reg": (31, 100)},
}
def _fit_hurdle_cls(X_train, y_train, num_leaves, min_child_samples, categorical_feature):
    model = LGBMClassifier(
        num_leaves=num_leaves, min_child_samples=min_child_samples,
        colsample_bytree=FEATURE_FRACTION, subsample=BAGGING_FRACTION, subsample_freq=BAGGING_FREQ,
        reg_alpha=LAMBDA_L1, reg_lambda=LAMBDA_L2, random_state=SEED, **CLS_FIXED_PARAMS,
    )
    model.fit(X_train, y_train, categorical_feature=categorical_feature)
    return model


def _fit_hurdle_reg(X_train, y_train, num_leaves, min_child_samples, categorical_feature):
    model = LGBMRegressor(
        num_leaves=num_leaves, min_child_samples=min_child_samples,
        colsample_bytree=FEATURE_FRACTION, subsample=BAGGING_FRACTION, subsample_freq=BAGGING_FREQ,
        reg_alpha=LAMBDA_L1, reg_lambda=LAMBDA_L2, random_state=SEED, **P13_FIXED_PARAMS,
    )
    model.fit(X_train, y_train, categorical_feature=categorical_feature)
    return model


def _metrics_dict(m: dict) -> dict:
    return {k: m[k] for k in ("wape", "bias", "rmse", "mae", "mase")}


def run_single(dev: pd.DataFrame, horizon: int, folds: list) -> dict:
    fold_data = [(f["fold"], dev.loc[f["train_mask"]], dev.loc[f["val_mask"]]) for f in folds]
    run = p13_lgbm._run_folds(SINGLE_HP, fold_data, horizon, SEED, "b_robustness", f"b_robustness_h{horizon}_lightgbm_single")
    if run["status"] != "ok":
        raise RuntimeError(f"h{horizon} single LightGBM B robustness 실패: {run['error']}")

    pooled_oof = pd.concat(run["oof_frames"], ignore_index=True)
    pooled = ev.compute_metrics(pooled_oof["y_true"].to_numpy(), pooled_oof["y_pred"].to_numpy(), pooled_oof["mase_scale"].to_numpy())
    per_fold = {
        str(f["fold"]): _metrics_dict(m) for f, m in zip(folds, run["native_fold_metrics"])
    }

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    pooled_oof.to_parquet(OOF_DIR / f"oof_single_lightgbm_h{horizon}_b_center.parquet", index=False)

    return {
        "family": "lightgbm_single", "horizon": horizon,
        "note": "A-center P13 최저-WAPE trial(trial_id=3, 공식 selected 아님) HP를 B에 재튜닝 없이 고정 적용",
        "hp": SINGLE_HP, "fixed_params": P13_FIXED_PARAMS, "seed": SEED,
        "n_folds": len(folds),
        "pooled_metrics": _metrics_dict(pooled),
        "per_fold_metrics": per_fold,
    }


def run_hurdle(dev: pd.DataFrame, horizon: int, folds: list) -> dict:
    hp = HURDLE_HP[horizon]
    target_col = cfg.TARGET_COLS[horizon]

    oof_frames = []
    per_fold = {}
    quartile_rows = []
    for fold in folds:
        fold_id = fold["fold"]
        train_df = dev.loc[fold["train_mask"]]
        val_df = dev.loc[fold["val_mask"]]

        preprocessor = LGBMPreprocessor()
        X_train = preprocessor.fit_transform(train_df, horizon)
        X_val = preprocessor.transform(val_df)
        cat_features = preprocessor.categorical_feature_names

        y_train_raw = train_df[target_col].to_numpy(dtype=float)
        y_val_raw = val_df[target_col].to_numpy(dtype=float)
        pos_mask = y_train_raw > 0

        cls_model = _fit_hurdle_cls(X_train, pos_mask.astype(int), hp["cls"][0], hp["cls"][1], cat_features)
        prob_val = cls_model.predict_proba(X_val)[:, 1]

        X_train_pos = X_train.iloc[pos_mask]
        y_train_pos_log = np.log1p(y_train_raw[pos_mask])
        reg_model = _fit_hurdle_reg(X_train_pos, y_train_pos_log, hp["reg"][0], hp["reg"][1], cat_features)
        cond_pred = ev.inverse_transform_prediction(reg_model.predict(X_val))
        pred = prob_val * cond_pred
        pred_log = np.log1p(pred)

        val_keys = val_df[["center_id", "sku_id", "week_st"]].copy()
        val_keys["target_date"] = val_keys["week_st"] + pd.Timedelta(weeks=horizon)
        mase_scale = ev.build_mase_scale(train_df[["center_id", "sku_id", "week_st", "qty"]], val_keys)

        oof_frame = oo.build_oof_frame(
            val_keys, y_val_raw, pred_log, pred, mase_scale,
            stage="b_robustness", model_family="LIGHTGBM_HURDLE",
            config_id=f"b_robustness_h{horizon}_lightgbm_hurdle", seed=SEED, horizon=horizon, fold_id=fold_id,
        )
        oof_frames.append(oof_frame)
        fold_metrics = ev.compute_metrics(y_val_raw, pred, mase_scale)
        per_fold[str(fold_id)] = _metrics_dict(fold_metrics)

        # B post-regime train-only(B만) SKU 평균 주간수요 4분위 - Hurdle 진단 전용, selection에는 미사용.
        b_train = train_df[train_df["center_id"] == "B"]
        if len(b_train) > 0:
            sku_avg = b_train.groupby("sku_id")["qty"].mean()
            # 초기 fold는 B 이력이 짧아 qcut이 4분위보다 적게 나올 수 있어 label 개수를 고정하지 않는다.
            codes = pd.qcut(sku_avg, q=4, duplicates="drop").cat.codes
            for sku_id, code in codes.items():
                if code >= 0:
                    quartile_rows.append((fold_id, sku_id, f"Q{code + 1}"))

    pooled_oof = pd.concat(oof_frames, ignore_index=True)
    pooled = ev.compute_metrics(pooled_oof["y_true"].to_numpy(), pooled_oof["y_pred"].to_numpy(), pooled_oof["mase_scale"].to_numpy())

    quartile_df = pd.DataFrame(quartile_rows, columns=["fold_id", "sku_id", "demand_quartile"])
    diag_df = pooled_oof.merge(quartile_df, on=["fold_id", "sku_id"], how="left")
    diag_df["demand_quartile"] = diag_df["demand_quartile"].fillna("no_train_history")

    def _wape_bias(df: pd.DataFrame) -> dict:
        m = ev.compute_metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy(), df["mase_scale"].to_numpy())
        return {"wape": m["wape"], "bias": m["bias"]}

    zero = diag_df[diag_df["y_true"] == 0]
    nonzero = diag_df[diag_df["y_true"] > 0]

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    pooled_oof.to_parquet(OOF_DIR / f"oof_hurdle_lightgbm_h{horizon}_b_center.parquet", index=False)

    return {
        "family": "lightgbm_hurdle", "horizon": horizon,
        "note": "A-center 정식 Hurdle HPO selected HP를 B에 재튜닝/재선택 없이 고정 적용",
        "classifier_hp": {"num_leaves": hp["cls"][0], "min_child_samples": hp["cls"][1]},
        "regressor_hp": {"num_leaves": hp["reg"][0], "min_child_samples": hp["reg"][1]},
        "fixed_params_classifier": CLS_FIXED_PARAMS, "fixed_params_regressor": P13_FIXED_PARAMS,
        "seed": SEED, "n_folds": len(folds),
        "pooled_metrics": _metrics_dict(pooled),
        "per_fold_metrics": per_fold,
        "diagnostics": {
            "note": "selection에는 사용되지 않은 진단용 breakdown",
            "quartile_definition": "각 fold의 B post-regime TRAIN 데이터(B만)로 계산한 SKU별 평균 주간수요 4분위",
            "zero_rows": {"mean_pred": float(zero["y_pred"].mean()), "sum_pred": float(zero["y_pred"].sum())},
            "nonzero_rows": _wape_bias(nonzero),
            "by_demand_quartile": {name: _wape_bias(grp) for name, grp in diag_df.groupby("demand_quartile")},
        },
    }


def main() -> None:
    dev = load_development()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for horizon in cfg.HORIZONS:
        row_checks.verify_b_robustness_row_fold_count(dev, horizon)
        folds = day3_folds.generate_b_walkforward_folds(dev, horizon)

        single_result = run_single(dev, horizon, folds)
        with open(OUT_DIR / f"single_lightgbm_h{horizon}.json", "w", encoding="utf-8") as f:
            json.dump(single_result, f, ensure_ascii=False, indent=2, default=str)
        print(f"[B-ROBUSTNESS] h{horizon} single: {single_result['pooled_metrics']}")

        hurdle_result = run_hurdle(dev, horizon, folds)
        with open(OUT_DIR / f"hurdle_lightgbm_h{horizon}.json", "w", encoding="utf-8") as f:
            json.dump(hurdle_result, f, ensure_ascii=False, indent=2, default=str)
        print(f"[B-ROBUSTNESS] h{horizon} hurdle: {hurdle_result['pooled_metrics']}")


if __name__ == "__main__":
    main()
