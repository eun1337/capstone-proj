from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.forecasting.common import config as cfg
from src.forecasting.common import evaluator as ev
from src.forecasting.common import folds as day3_folds
from src.forecasting.common import oof as oo
from src.forecasting.common.data_loader import load_development
from src.forecasting.machine_learning.rf.config import FIXED_PARAMS as RF_FIXED_PARAMS
from src.forecasting.machine_learning.rf.preprocessing import RFPreprocessor
from src.forecasting.pipeline.p13 import hpo_common as hc

SEED = cfg.P13_MODEL_SEED
FAMILY = "rf_hurdle"
STAGE = "p13_followup_hurdle_rf"

CLS_GRID = [(mf, msl) for mf in (0.1, 1 / 3) for msl in (100, 500)]
REG_GRID = [(mf, msl, tr) for mf in (0.1, 1 / 3) for msl in (100, 500) for tr in ("raw", "log1p")]

CLS_FIXED = {**RF_FIXED_PARAMS, "criterion": "gini", "class_weight": None}
REG_FIXED = dict(RF_FIXED_PARAMS)

OUT_DIR = _REPO_ROOT / "outputs" / "hurdle_hpo" / "rf_hurdle"
OOF_DIR = OUT_DIR / "oof"
COMMON_EVAL_KEYS_PATH = _REPO_ROOT / "outputs" / "audits" / "p13" / "five_family_common_evaluation_keys.parquet"


def run_horizon(horizon: int, common_eval_keys: pd.DataFrame) -> dict:
    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()
    target_col = cfg.TARGET_COLS[horizon]

    folds = day3_folds.generate_expanding_folds(sub_a, 2023, horizon)
    hc.check_common_keys_match_p13_period(common_eval_keys, folds, horizon, COMMON_EVAL_KEYS_PATH)
    horizon_common_keys = common_eval_keys[common_eval_keys["horizon"] == horizon]

    fold_records = []
    for fold in folds:
        fold_id = fold["fold"]
        train_df = sub_a.loc[fold["train_mask"]]
        val_df = sub_a.loc[fold["val_mask"]]

        preprocessor = RFPreprocessor()
        X_train = preprocessor.fit_transform(train_df, horizon)
        X_val = preprocessor.transform(val_df)

        y_train_raw = train_df[target_col].to_numpy(dtype=float)
        y_cls_train = (y_train_raw > 0).astype(int)
        pos_mask = y_train_raw > 0
        X_train_pos = X_train[pos_mask]
        y_train_pos_raw = y_train_raw[pos_mask]

        y_val_true = val_df[target_col].to_numpy(dtype=float)
        val_keys = val_df[["center_id", "sku_id", "week_st"]].copy()
        val_keys["target_date"] = val_keys["week_st"] + pd.Timedelta(weeks=horizon)
        mase_scale = ev.build_mase_scale(train_df[["center_id", "sku_id", "week_st", "qty"]], val_keys)

        sku_avg_qty = train_df.groupby(["center_id", "sku_id"])["qty"].mean().rename("avg_qty").reset_index()
        sku_avg_qty["quartile"] = pd.qcut(sku_avg_qty["avg_qty"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        quartile_table = sku_avg_qty[["center_id", "sku_id", "quartile"]]

        cls_probas = {}
        for mf, msl in CLS_GRID:
            clf = RandomForestClassifier(
                n_estimators=100, max_features=mf, min_samples_leaf=msl, random_state=SEED, **CLS_FIXED,
            )
            clf.fit(X_train, y_cls_train)
            cls_probas[(mf, msl)] = clf.predict_proba(X_val)[:, 1]

        reg_preds = {}
        for mf, msl, tr in REG_GRID:
            y_reg = y_train_pos_raw if tr == "raw" else np.log1p(y_train_pos_raw)
            reg = RandomForestRegressor(
                n_estimators=100, max_features=mf, min_samples_leaf=msl, random_state=SEED, **REG_FIXED,
            )
            reg.fit(X_train_pos, y_reg)
            pred = reg.predict(X_val)
            pred_raw = np.clip(pred, 0.0, None) if tr == "raw" else ev.inverse_transform_prediction(pred)
            reg_preds[(mf, msl, tr)] = pred_raw

        fold_records.append({
            "fold_id": fold_id, "val_keys": val_keys, "y_val_true": y_val_true,
            "mase_scale": mase_scale, "cls_probas": cls_probas, "reg_preds": reg_preds,
            "quartile_table": quartile_table,
        })
        print(f"[hurdle-rf] h{horizon} fold{fold_id} 학습 완료 (train={len(train_df)} val={len(val_df)} pos_rate={pos_mask.mean():.4f})")

    combos = []
    for mf_c, msl_c in CLS_GRID:
        for mf_r, msl_r, tr in REG_GRID:
            frames = []
            for rec in fold_records:
                final_pred = rec["cls_probas"][(mf_c, msl_c)] * rec["reg_preds"][(mf_r, msl_r, tr)]
                frames.append(pd.DataFrame({
                    "horizon": horizon, "fold_id": rec["fold_id"],
                    "center_id": rec["val_keys"]["center_id"].to_numpy(),
                    "sku_id": rec["val_keys"]["sku_id"].to_numpy(),
                    "week_st": rec["val_keys"]["week_st"].to_numpy(),
                    "target_date": rec["val_keys"]["target_date"].to_numpy(),
                    "y_true": rec["y_val_true"], "y_pred": final_pred,
                    "mase_scale": rec["mase_scale"],
                }))
            pooled = pd.concat(frames, ignore_index=True)
            filtered = hc.filter_pooled_oof_by_common_keys(pooled, horizon_common_keys)
            pooled_common = ev.compute_metrics(
                filtered["y_true"].to_numpy(), filtered["y_pred"].to_numpy(), filtered["mase_scale"].to_numpy(),
            )
            per_fold_common = hc.per_fold_metrics_common(filtered)
            worst_fold_wape = max(m["wape"] for m in per_fold_common.values())
            combos.append({
                "trial_id": len(combos),
                "params": {
                    "classifier": {"max_features": mf_c, "min_samples_leaf": msl_c},
                    "regressor": {"max_features": mf_r, "min_samples_leaf": msl_r, "target_transform": tr},
                },
                "pooled_wape": pooled_common["wape"], "pooled_bias": pooled_common["bias"],
                "pooled_rmse": pooled_common["rmse"], "pooled_mae": pooled_common["mae"],
                "pooled_mase": pooled_common["mase"], "worst_fold_wape": worst_fold_wape,
                "per_fold_metrics_common": per_fold_common,
                "guardrail_pass": bool(abs(pooled_common["bias"]) <= cfg.BIAS_GUARDRAIL_ABS_PCT),
                "common_eval_key_count": len(horizon_common_keys),
            })
        print(f"[hurdle-rf] h{horizon} classifier(max_features={mf_c}, min_samples_leaf={msl_c}) x 8 regressor 조합 평가 완료")

    selection = hc.select_best_trial(combos)
    n_guardrail_pass = sum(1 for c in combos if c["guardrail_pass"])

    result = {
        "horizon": horizon, "seed": SEED,
        "search_space": {
            "classifier": {"max_features": [0.1, 1 / 3], "min_samples_leaf": [100, 500]},
            "regressor": {"max_features": [0.1, 1 / 3], "min_samples_leaf": [100, 500], "target_transform": ["raw", "log1p"]},
        },
        "fixed_params": {
            "classifier": {**CLS_FIXED, "n_estimators": 100, "random_state": SEED},
            "regressor": {**REG_FIXED, "n_estimators": 100, "random_state": SEED},
        },
        "n_classifier_configs": len(CLS_GRID), "n_regressor_configs": len(REG_GRID),
        "n_combinations": len(combos), "n_fits": len(CLS_GRID) * 4 + len(REG_GRID) * 4,
        "evaluation_population": "five_family_common_keys",
        "common_eval_key_path": str(COMMON_EVAL_KEYS_PATH),
        "common_eval_key_count": len(horizon_common_keys),
        "n_guardrail_pass": n_guardrail_pass,
        "y_pred_log_note": "y_pred_log는 raw arm 포함 항상 np.log1p(final_pred)를 저장한 스키마 호환용 파생 필드이며, "
                            "모델이 log space로 학습됐다는 의미가 아니다.",
        "selection": {k: v for k, v in selection.items()},
        "all_combinations": combos,
    }

    sel = selection["selected"]
    if sel is not None:
        mf_c = sel["params"]["classifier"]["max_features"]
        msl_c = sel["params"]["classifier"]["min_samples_leaf"]
        mf_r = sel["params"]["regressor"]["max_features"]
        msl_r = sel["params"]["regressor"]["min_samples_leaf"]
        tr = sel["params"]["regressor"]["target_transform"]

        oof_frames, quartile_frames = [], []
        for rec in fold_records:
            final_pred = rec["cls_probas"][(mf_c, msl_c)] * rec["reg_preds"][(mf_r, msl_r, tr)]
            oof_frames.append(oo.build_oof_frame(
                rec["val_keys"], rec["y_val_true"], np.log1p(final_pred), final_pred, rec["mase_scale"],
                stage=STAGE, model_family=FAMILY,
                config_id=f"cls_mf{mf_c}_msl{msl_c}__reg_mf{mf_r}_msl{msl_r}_{tr}",
                seed=SEED, horizon=horizon, fold_id=rec["fold_id"],
            ))
            quartile_frames.append(rec["quartile_table"].assign(fold_id=rec["fold_id"]))

        selected_oof = pd.concat(oof_frames, ignore_index=True)
        selected_filtered = hc.filter_pooled_oof_by_common_keys(selected_oof, horizon_common_keys)
        fold_diag = hc.per_fold_metrics_common(selected_filtered)

        y_true_d = selected_filtered["y_true"].to_numpy()
        y_pred_d = selected_filtered["y_pred"].to_numpy()
        mase_scale_d = selected_filtered["mase_scale"].to_numpy()
        zero_mask = y_true_d == 0
        pos_mask_d = ~zero_mask

        zero_diag = {
            "n": int(zero_mask.sum()),
            "mean_pred": float(y_pred_d[zero_mask].mean()) if zero_mask.any() else None,
            "sum_pred": float(y_pred_d[zero_mask].sum()) if zero_mask.any() else None,
        }
        pos_diag = ev.compute_metrics(y_true_d[pos_mask_d], y_pred_d[pos_mask_d], mase_scale_d[pos_mask_d])

        quartile_df = pd.concat(quartile_frames, ignore_index=True)
        merged_q = selected_filtered.merge(quartile_df, on=["fold_id", "center_id", "sku_id"], how="left")
        quartile_diag = {}
        for q in ("Q1", "Q2", "Q3", "Q4"):
            sub = merged_q[merged_q["quartile"] == q]
            if len(sub) == 0:
                continue
            quartile_diag[q] = ev.compute_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(), sub["mase_scale"].to_numpy())

        result["selected_diagnostics"] = {
            "fold_metrics": fold_diag, "target_zero": zero_diag, "target_positive": pos_diag,
            "demand_quartile": quartile_diag,
        }

        OOF_DIR.mkdir(parents=True, exist_ok=True)
        selected_filtered.to_parquet(
            OOF_DIR / f"oof_hurdle_rf_h{horizon}_cls_mf{mf_c}_msl{msl_c}__reg_mf{mf_r}_msl{msl_r}_{tr}_seed{SEED}_common.parquet",
            index=False,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"p13_hurdle_rf_h{horizon}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"[hurdle-rf] h{horizon} 저장 완료: {out_path} (guardrail_pass={n_guardrail_pass}/32, reason={selection['reason']})")
    return result


def main() -> None:
    common_eval_keys = hc.load_common_eval_keys(COMMON_EVAL_KEYS_PATH)
    for h in (1, 2, 4):
        run_horizon(h, common_eval_keys)


if __name__ == "__main__":
    main()
