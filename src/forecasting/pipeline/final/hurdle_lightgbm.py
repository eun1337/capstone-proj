"""
hurdle_lightgbm.py

Hurdle-LightGBM(classifier P(y>0) + conditional regressor log1p) Final retrain / 2024 Holdout.
P21 winner 경로를 거치지 않고, A-center Hurdle HPO로 확정된 horizon별 HP를 그대로 고정해 사용한다.

retrain: final_retrain_mask(pre-2024, A 전체 + B post-regime pooled)로 classifier/conditional
regressor를 1세트씩 학습해 preprocessor와 함께 저장한다(validation/early stopping/재선택 없음).

holdout: 저장된 artifact로 holdout_2024_mask 대상 행만 predict-only 평가한다(2024 내부 재학습 없음).

--stage {retrain,holdout}로 분리 실행하며, 모듈 import만으로는 데이터 로드나 학습이 시작되지 않는다.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from src.forecasting.common import config as cfg
from src.forecasting.common import evaluator as ev
from src.forecasting.common import folds as day3_folds
from src.forecasting.common.data_loader import load_development, load_holdout_2024
from src.forecasting.machine_learning.lightgbm.config import (
    BAGGING_FRACTION,
    BAGGING_FREQ,
    FEATURE_FRACTION,
    LAMBDA_L1,
    LAMBDA_L2,
    P13_FIXED_PARAMS,
)
from src.forecasting.machine_learning.lightgbm.preprocessing import LGBMPreprocessor
from src.forecasting.pipeline.final import model_artifact as final_model_artifact
from src.forecasting.pipeline.robustness import frozen_checks_ml as row_checks

SEED = cfg.P13_MODEL_SEED
CLS_FIXED_PARAMS = {**P13_FIXED_PARAMS, "objective": "binary"}

# A-center 정식 Hurdle HPO(outputs/hurdle_hpo/lgbm_hurdle/p13_hurdle_lightgbm_h{h}.json) selected 값.
HORIZON_HP = {
    1: {"cls_num_leaves": 31, "cls_min_child_samples": 100, "reg_num_leaves": 31, "reg_min_child_samples": 100},
    2: {"cls_num_leaves": 31, "cls_min_child_samples": 1000, "reg_num_leaves": 31, "reg_min_child_samples": 100},
    4: {"cls_num_leaves": 31, "cls_min_child_samples": 1000, "reg_num_leaves": 31, "reg_min_child_samples": 100},
}

OUT_RETRAIN_DIR = Path("outputs/final_retrain/lightgbm_hurdle")
OUT_HOLDOUT_DIR = Path("outputs/final_holdout/lightgbm_hurdle")


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


def run_retrain(dev: pd.DataFrame, horizon: int) -> dict:
    """final_retrain_mask population(A 전체 + B post-regime, pre-2024 pooled)으로
    classifier/conditional regressor를 1세트 학습해 artifact + metadata를 저장한다."""
    hp = HORIZON_HP[horizon]
    row_checks.verify_final_retrain_row_workload(dev, horizon)
    full_df = dev.loc[day3_folds.final_retrain_mask(dev, horizon)]
    target_col = cfg.TARGET_COLS[horizon]

    t0 = time.perf_counter()
    preprocessor = LGBMPreprocessor()
    X = preprocessor.fit_transform(full_df, horizon)
    cat_features = preprocessor.categorical_feature_names

    y_raw = full_df[target_col].to_numpy(dtype=float)
    pos_mask = y_raw > 0

    cls_model = _fit_cls(X, pos_mask.astype(int), hp["cls_num_leaves"], hp["cls_min_child_samples"], cat_features)
    X_pos = X.iloc[pos_mask]
    reg_model = _fit_reg(X_pos, np.log1p(y_raw[pos_mask]), hp["reg_num_leaves"], hp["reg_min_child_samples"], cat_features)
    fit_time_sec = time.perf_counter() - t0

    OUT_RETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    model_artifact_path = OUT_RETRAIN_DIR / f"lightgbm_hurdle_h{horizon}_final.joblib"
    joblib.dump({"preprocessor": preprocessor, "classifier": cls_model, "regressor": reg_model}, model_artifact_path)

    config = {
        "classifier_hp": {"num_leaves": hp["cls_num_leaves"], "min_child_samples": hp["cls_min_child_samples"]},
        "regressor_hp": {"num_leaves": hp["reg_num_leaves"], "min_child_samples": hp["reg_min_child_samples"]},
        "fixed_params_classifier": CLS_FIXED_PARAMS,
        "fixed_params_regressor": P13_FIXED_PARAMS,
        "shared_fixed_params": {
            "feature_fraction": FEATURE_FRACTION, "bagging_fraction": BAGGING_FRACTION,
            "bagging_freq": BAGGING_FREQ, "lambda_l1": LAMBDA_L1, "lambda_l2": LAMBDA_L2,
        },
        "classifier_target": "y>0",
        "conditional_target_transform": "log1p",
        "final_combination": "probability_x_conditional_prediction",
        "total_train_rows": len(full_df),
        "a_train_rows": int((full_df["center_id"] == "A").sum()),
        "b_train_rows": int((full_df["center_id"] == "B").sum()),
        "train_week_st_min": str(full_df["week_st"].min()),
        "train_week_st_max": str(full_df["week_st"].max()),
        "positive_row_regressor_train_count": int(pos_mask.sum()),
        "target_cutoff": str(cfg.FINAL_TRAIN_CUTOFF),
        "b_history_start": str(cfg.B_HISTORY_START),
        "target_transform": "log1p",
    }
    metadata_path = OUT_RETRAIN_DIR / f"lightgbm_hurdle_h{horizon}_final_metadata.json"
    final_model_artifact.save_metadata(
        metadata_path, family="lightgbm_hurdle", horizon=horizon, config=config, seed=SEED,
        model_artifact_path=model_artifact_path, training_population_size=len(full_df),
        population_unit="rows", fit_time_sec=fit_time_sec,
    )
    return {"metadata_path": str(metadata_path), "model_artifact_path": str(model_artifact_path)}


def run_holdout(holdout_df: pd.DataFrame, dev: pd.DataFrame, horizon: int, metadata_path: Path) -> dict:
    """저장된 Final retrain artifact로 holdout_2024_mask 대상만 predict-only 평가한다(재학습 없음)."""
    metadata = final_model_artifact.load_metadata(metadata_path)
    if metadata["horizon"] != horizon:
        raise ValueError(f"artifact horizon={metadata['horizon']}이 요청 horizon={horizon}과 다름")
    if metadata["family"] != "lightgbm_hurdle":
        raise ValueError(f"artifact family={metadata['family']!r}이 'lightgbm_hurdle'이 아님")
    if metadata["seed"] != SEED:
        raise ValueError(f"artifact seed={metadata['seed']}가 SEED={SEED}와 다름")
    hp = HORIZON_HP[horizon]
    expected_cls_hp = {"num_leaves": hp["cls_num_leaves"], "min_child_samples": hp["cls_min_child_samples"]}
    expected_reg_hp = {"num_leaves": hp["reg_num_leaves"], "min_child_samples": hp["reg_min_child_samples"]}
    if metadata["config"]["classifier_hp"] != expected_cls_hp:
        raise ValueError(f"artifact classifier_hp={metadata['config']['classifier_hp']}가 HORIZON_HP[{horizon}]={expected_cls_hp}와 다름")
    if metadata["config"]["regressor_hp"] != expected_reg_hp:
        raise ValueError(f"artifact regressor_hp={metadata['config']['regressor_hp']}가 HORIZON_HP[{horizon}]={expected_reg_hp}와 다름")
    if metadata["config"]["target_transform"] != "log1p":
        raise ValueError(f"artifact target_transform={metadata['config']['target_transform']!r}이 'log1p'가 아님")
    for flag in ("validation_used", "early_stopping_used", "best_epoch_selection_used"):
        if metadata[flag] is not False:
            raise ValueError(f"artifact {flag}={metadata[flag]!r}가 False가 아님")

    eligible_mask = day3_folds.holdout_2024_mask(holdout_df, horizon)
    expected_n = int(eligible_mask.sum())
    eligible_df = holdout_df.loc[eligible_mask]
    target_col = cfg.TARGET_COLS[horizon]

    bundle = joblib.load(metadata["model_artifact_path"])
    preprocessor, cls_model, reg_model = bundle["preprocessor"], bundle["classifier"], bundle["regressor"]

    X = preprocessor.transform(eligible_df)
    prob = cls_model.predict_proba(X)[:, 1]
    cond_pred = ev.inverse_transform_prediction(reg_model.predict(X))
    pred = prob * cond_pred

    keys = eligible_df[["center_id", "sku_id", "week_st"]].copy()
    keys["target_date"] = keys["week_st"] + pd.Timedelta(weeks=horizon)
    y_true = eligible_df[target_col].to_numpy(dtype=float)
    mase_scale = ev.build_mase_scale(dev[["center_id", "sku_id", "week_st", "qty"]], keys)

    pred_df = pd.DataFrame({
        "center_id": keys["center_id"].to_numpy(), "sku_id": keys["sku_id"].to_numpy(),
        "week_st": keys["week_st"].to_numpy(), "target_date": keys["target_date"].to_numpy(),
        "horizon": horizon, "y_true": y_true, "y_pred": pred,
        "sale_probability": prob, "conditional_prediction": cond_pred,
    })
    if len(pred_df) != expected_n:
        raise ValueError(f"prediction row 수={len(pred_df)}가 holdout_2024_mask expected row 수={expected_n}와 다름")

    def _metrics(df: pd.DataFrame) -> dict:
        m = ev.compute_metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy(), df["_mase_scale"].to_numpy())
        return {k: m[k] for k in ("wape", "bias", "rmse", "mae", "mase")}

    def _wape_bias(df: pd.DataFrame) -> dict:
        m = _metrics(df)
        return {"wape": m["wape"], "bias": m["bias"]}

    calc_df = pred_df.copy()
    calc_df["_mase_scale"] = mase_scale

    # Q1~Q4: Final retrain TRAIN(=final_retrain_mask population, 2024 미사용)의 센터별 SKU 평균 주간 qty 기준
    # (A/B 각각 내부에서 4분위를 계산 - 센터별 해석 유지).
    train_df = dev.loc[day3_folds.final_retrain_mask(dev, horizon)]
    quartile_map = {}
    for center in ("A", "B"):
        center_train = train_df[train_df["center_id"] == center]
        if len(center_train) == 0:
            continue
        sku_avg = center_train.groupby("sku_id")["qty"].mean()
        codes = pd.qcut(sku_avg, q=4, duplicates="drop").cat.codes
        for sku_id, code in codes.items():
            if code >= 0:
                quartile_map[(center, sku_id)] = f"Q{code + 1}"
    demand_quartile = pd.MultiIndex.from_frame(keys[["center_id", "sku_id"]]).map(quartile_map)
    calc_df["_demand_quartile"] = pd.Series(np.asarray(demand_quartile, dtype=object)).fillna("no_train_history")

    zero = calc_df[calc_df["y_true"] == 0]
    nonzero = calc_df[calc_df["y_true"] > 0]
    quarter = calc_df["target_date"].dt.quarter

    summary = {
        "family": "lightgbm_hurdle", "horizon": horizon, "seed": SEED,
        "final_retrain_metadata_path": str(metadata_path),
        "n_expected_predictions": expected_n,
        "n_predictions": len(calc_df),
        "pooled_metrics": _metrics(calc_df),
        "a_only_metrics": _metrics(calc_df[calc_df["center_id"] == "A"]),
        "b_only_metrics": _metrics(calc_df[calc_df["center_id"] == "B"]),
        "per_quarter_metrics": {int(q): _metrics(calc_df[quarter == q]) for q in sorted(quarter.unique())},
        "diagnostics": {
            "zero_rows": {
                "n": int(len(zero)),
                "mean_pred": float(zero["y_pred"].mean()) if len(zero) else None,
                "sum_pred": float(zero["y_pred"].sum()) if len(zero) else None,
            },
            "nonzero_rows": _wape_bias(nonzero),
            "by_demand_quartile": {
                name: _wape_bias(grp) for name, grp in calc_df.groupby("_demand_quartile")
            },
        },
    }

    OUT_HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(OUT_HOLDOUT_DIR / f"prediction_lightgbm_hurdle_h{horizon}.parquet", index=False)
    with open(OUT_HOLDOUT_DIR / f"summary_lightgbm_hurdle_h{horizon}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Hurdle-LightGBM Final retrain / 2024 holdout")
    parser.add_argument("--stage", required=True, choices=["retrain", "holdout"])
    parser.add_argument("--horizon", type=int, choices=list(cfg.HORIZONS), default=None)
    args = parser.parse_args()
    horizons = [args.horizon] if args.horizon is not None else list(cfg.HORIZONS)

    if args.stage == "retrain":
        dev = load_development()
        for horizon in horizons:
            result = run_retrain(dev, horizon)
            print(f"[FINAL-RETRAIN][lightgbm_hurdle] h{horizon}: {result['metadata_path']}")
    else:
        dev = load_development()
        holdout_df = load_holdout_2024()
        for horizon in horizons:
            metadata_path = OUT_RETRAIN_DIR / f"lightgbm_hurdle_h{horizon}_final_metadata.json"
            result = run_holdout(holdout_df, dev, horizon, metadata_path)
            print(f"[FINAL-HOLDOUT][lightgbm_hurdle] h{horizon}: pooled={result['pooled_metrics']}")


if __name__ == "__main__":
    main()
