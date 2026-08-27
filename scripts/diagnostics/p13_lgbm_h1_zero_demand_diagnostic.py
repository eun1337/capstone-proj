"""
p13_lgbm_h1_zero_demand_diagnostic.py

LightGBM h1 저수요군(Q1/Q2) 성능 악화가 실제 수요 0인 행에 대한 과대예측 때문인지 진단(READ-ONLY).
p13_lgbm_h1_target_transform_diagnostic.py의 fold/HP/모델 fit과
p13_lgbm_h1_demand_quartile_bias_diagnostic.py의 train-only 수요 4분위 로직을 그대로 재사용하고,
target=0/>0 행을 나눠 naive/raw/smearing 예측을 비교한다. 추가로 train-only 판매빈도(0이 아닌 주 비율)
4분위로도 같은 방식으로 비교한다.
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
from scripts.diagnostics.p13_lgbm_h1_target_transform_diagnostic import (
    COMMON_KEYS_PATH,
    HORIZON,
    SOURCE_JSON_PATH,
    _fit_lgbm,
    _load_best_trial,
)

FREQ_LABELS = ["F1", "F2", "F3", "F4"]
PRED_COLS = ("pred_naive", "pred_raw", "pred_smearing")
OUTPUT_PATH = Path("outputs/diagnostics/target_transform_bias/lightgbm_h1_zero_demand_diagnostic.json")


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

        sku_avg = train_df.groupby(["center_id", "sku_id"])["qty"].mean()
        demand_quartile_map = pd.qcut(sku_avg, q=4, labels=QUARTILE_LABELS, duplicates="drop").to_dict()

        # train-only 판매빈도(0이 아닌 주 비율) - 간헐수요 여부를 수요 규모와 분리해서 보기 위함.
        sku_freq = train_df.groupby(["center_id", "sku_id"])["qty"].apply(lambda s: (s > 0).mean())
        freq_bucket_map = pd.qcut(sku_freq, q=4, labels=FREQ_LABELS, duplicates="drop").to_dict()

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
            "pred_smearing": pred_smearing,
            "pred_raw": pred_raw,
            "demand_quartile": np.asarray(sku_idx.map(demand_quartile_map), dtype=object),
            "freq_bucket": np.asarray(sku_idx.map(freq_bucket_map), dtype=object),
        }))

    pooled = pd.concat(fold_frames, ignore_index=True)
    filtered = hc.filter_pooled_oof_by_common_keys(pooled, horizon_common_keys)

    def _summarize(df: pd.DataFrame, group_col: str) -> dict:
        out = {}
        for name, grp in df.groupby(group_col):
            zero = grp[grp["y_true"] == 0]
            nonzero = grp[grp["y_true"] > 0]
            entry = {
                "n_rows": int(len(grp)),
                "n_zero_rows": int(len(zero)),
                "zero_rate": float(len(zero) / len(grp)),
                "zero_rows": {},
                "nonzero_rows": {},
            }
            for col in PRED_COLS:
                entry["zero_rows"][col] = {
                    "mean_pred": float(zero[col].mean()) if len(zero) else None,
                    "sum_pred": float(zero[col].sum()) if len(zero) else None,
                }
                if len(nonzero):
                    m = ev.compute_metrics(
                        nonzero["y_true"].to_numpy(), nonzero[col].to_numpy(), nonzero["mase_scale"].to_numpy(),
                    )
                    entry["nonzero_rows"][col] = {"wape": m["wape"], "bias": m["bias"]}
                else:
                    entry["nonzero_rows"][col] = None
            out[name] = entry
        return out

    result = {
        "family": "lightgbm",
        "horizon": HORIZON,
        "source_trial_id": best_trial["trial_id"],
        "source_trial_params": hp,
        "common_eval_key_count": len(horizon_common_keys),
        "by_demand_quartile": _summarize(filtered, "demand_quartile"),
        "by_train_sale_frequency_bucket": _summarize(filtered, "freq_bucket"),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
