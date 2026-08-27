# -*- coding: utf-8 -*-
"""
verify_mase_trial3.py

사용자 질문에 대한 사후 검증 1회성 스크립트: lightgbm h1 raw-scale 진단에서
pooled_mase가 trial3 기준 0.6082(log1p) -> 1.6352(raw scale)로 뛴 게

  (a) 계산상 정상인지(버그 아닌지)
  (b) 기존 P13과 동일한 MASE denominator/evaluator를 썼는지

를 row-level로 직접 확인한다. log1p(원본 trainer.py, 미수정) 경로와 raw scale
(p13_lightgbm_rawscale_diag_h1.py의 train_and_evaluate_fold_raw, 미수정) 경로를 trial3 하나에
대해 각각 다시 돌려서 같은 (center_id, sku_id, week_st) 키에 대해 mase_scale이
bit-identical한지, err/mase_scale 비율의 어떤 row들이 pooled_mase 증가를
주도하는지 확인한다. 기존 파이프라인 코드는 이번에도 전혀 수정하지 않는다.
산출물은 outputs/hpo/_diag_rawscale_lightgbm_h1/ 안에만 남긴다(이 스크립트 자체 +
verify_mase_trial3_report.txt).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from src.forecasting.common import config as cfg
from src.forecasting.common import evaluator as ev
from src.forecasting.common import folds as day3_folds
from src.forecasting.common.data_loader import load_development
from src.forecasting.machine_learning.lightgbm import trainer as lgbm_trainer
from src.forecasting.machine_learning.lightgbm.config import P13_FIXED_PARAMS
from src.forecasting.pipeline.p13 import lightgbm as p13_lgbm

HORIZON = 1
SEED = cfg.P13_MODEL_SEED
COMMON_EVAL_KEYS_PATH = cfg.PROJECT_ROOT / "outputs" / "audits" / "p13" / "five_family_common_evaluation_keys.parquet"

TRIAL3_HP = {
    "num_leaves": 31, "min_child_samples": 100,
    "feature_fraction": 0.4, "bagging_fraction": 0.4, "bagging_freq": 1,
    "lambda_l1": 0.0, "lambda_l2": 0.0,
}

# 같은 폴더의 p13_lightgbm_rawscale_diag_h1.py에서 train_and_evaluate_fold_raw를 그대로 재사용한다
# (파일 내용을 바꾸지 않고 import만 함).
_spec = importlib.util.spec_from_file_location(
    "_diag_lgbm_run_diagnostic", Path(__file__).resolve().parent / "p13_lightgbm_rawscale_diag_h1.py"
)
_diag_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_diag_module)
train_and_evaluate_fold_raw = _diag_module.train_and_evaluate_fold_raw


def run_log1p_trial3(fold_data):
    oof_frames = []
    for fold_id, train_df, val_df in fold_data:
        result = lgbm_trainer.train_and_evaluate_fold(
            train_df, val_df, HORIZON, **TRIAL3_HP,
            stage="verify_mase_log1p", model_family="lightgbm", config_id="trial3_verify",
            seed=SEED, fold_id=fold_id, use_best_iteration=False,
            fixed_params=P13_FIXED_PARAMS,
        )
        if result["used_iteration"] != P13_FIXED_PARAMS["n_estimators"]:
            raise RuntimeError(f"fold {fold_id}: used_iteration 불일치")
        oof_frames.append(result["oof"])
    return pd.concat(oof_frames, ignore_index=True)


def run_raw_trial3(fold_data):
    oof_frames = []
    for fold_id, train_df, val_df in fold_data:
        result = train_and_evaluate_fold_raw(
            train_df, val_df, HORIZON, **TRIAL3_HP,
            stage="verify_mase_raw", model_family="lightgbm", config_id="trial3_verify",
            seed=SEED, fold_id=fold_id, fixed_params=P13_FIXED_PARAMS,
        )
        if result["used_iteration"] != P13_FIXED_PARAMS["n_estimators"]:
            raise RuntimeError(f"fold {fold_id}: used_iteration 불일치")
        oof_frames.append(result["oof"])
    return pd.concat(oof_frames, ignore_index=True)


def main() -> None:
    common_eval_keys = p13_lgbm.load_common_eval_keys(COMMON_EVAL_KEYS_PATH)
    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()

    folds = day3_folds.generate_expanding_folds(sub_a, 2023, HORIZON)
    fold_data = [
        (fold["fold"], sub_a.loc[fold["train_mask"]], sub_a.loc[fold["val_mask"]])
        for fold in folds
    ]
    horizon_common_keys = common_eval_keys[common_eval_keys["horizon"] == HORIZON]

    print("[verify] trial3 log1p(원본 trainer.py, 미수정) 재실행...")
    oof_log1p = run_log1p_trial3(fold_data)
    filtered_log1p = p13_lgbm._filter_pooled_oof_by_common_keys(oof_log1p, horizon_common_keys)
    metrics_log1p = ev.compute_metrics(
        filtered_log1p["y_true"].to_numpy(), filtered_log1p["y_pred"].to_numpy(), filtered_log1p["mase_scale"].to_numpy(),
    )
    print(f"[verify] log1p 재현: wape={metrics_log1p['wape']:.4f} bias={metrics_log1p['bias']:.4f} "
          f"mase={metrics_log1p['mase']:.4f} (기존 JSON: wape=59.5044 bias=-31.6307 mase=0.6082)")

    print("[verify] trial3 raw scale(p13_lightgbm_rawscale_diag_h1.py, 미수정) 재실행...")
    oof_raw = run_raw_trial3(fold_data)
    filtered_raw = p13_lgbm._filter_pooled_oof_by_common_keys(oof_raw, horizon_common_keys)
    metrics_raw = ev.compute_metrics(
        filtered_raw["y_true"].to_numpy(), filtered_raw["y_pred"].to_numpy(), filtered_raw["mase_scale"].to_numpy(),
    )
    print(f"[verify] raw scale 재현: wape={metrics_raw['wape']:.4f} bias={metrics_raw['bias']:.4f} "
          f"mase={metrics_raw['mase']:.4f} (진단 JSON: wape=67.8351 bias=3.4496 mase=1.6352)")

    key_cols = ["center_id", "sku_id", "week_st", "target_date", "fold_id"]
    merged = filtered_log1p[key_cols + ["y_true", "y_pred", "mase_scale"]].merge(
        filtered_raw[key_cols + ["y_true", "y_pred", "mase_scale"]],
        on=key_cols, suffixes=("_log1p", "_raw"), how="inner",
    )
    print(f"\n[verify] merge된 row 수: {len(merged)} (log1p={len(filtered_log1p)}, raw={len(filtered_raw)})")

    # (a) mase_scale(denominator)가 두 실행에서 bit-identical한지 확인
    #     - 두 실행 모두 동일한 fold train_df 이력(qty)에서 ev.build_mase_scale로 계산되므로
    #       모델/타겟 변환과 무관하게 같아야 한다.
    scale_diff = (merged["mase_scale_log1p"] - merged["mase_scale_raw"]).abs()
    print(f"[verify] mase_scale(log1p) vs mase_scale(raw) 최대 절대 차이: {scale_diff.max()} "
          f"(0이어야 정상 - 동일 fold train history로 계산되므로 target 변환과 무관함)")
    assert (merged["y_true_log1p"] == merged["y_true_raw"]).all(), "y_true가 두 실행에서 다름 - key 정합성 문제"

    # (b) row-level ratio = |err| / mase_scale 비교
    merged["err_log1p"] = merged["y_pred_log1p"] - merged["y_true_log1p"]
    merged["err_raw"] = merged["y_pred_raw"] - merged["y_true_raw"]
    valid = np.isfinite(merged["mase_scale_log1p"]) & (merged["mase_scale_log1p"] > 0)
    m = merged.loc[valid].copy()
    m["ratio_log1p"] = m["err_log1p"].abs() / m["mase_scale_log1p"]
    m["ratio_raw"] = m["err_raw"].abs() / m["mase_scale_log1p"]
    m["ratio_delta"] = m["ratio_raw"] - m["ratio_log1p"]

    print(f"\n[verify] mase_n(유효 denominator row 수) = {len(m)} "
          f"(기존 JSON mase_n 합계 = 124393+132312+138949+145168 = {124393+132312+138949+145168})")
    print(f"[verify] mean(ratio_log1p) = {m['ratio_log1p'].mean():.4f}  (== pooled_mase 재현값과 일치해야 함)")
    print(f"[verify] mean(ratio_raw)   = {m['ratio_raw'].mean():.4f}  (== pooled_mase 재현값과 일치해야 함)")

    # mase_scale 분포 - 작은 denominator일수록 ratio가 크게 튈 수 있음을 보여준다
    print("\n[verify] mase_scale(denominator) 분포 (log1p run 기준, raw와 동일):")
    print(m["mase_scale_log1p"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())

    # ratio_delta가 가장 큰 상위 20개 row - MASE 증가를 주도하는 row 특성 확인
    top = m.sort_values("ratio_delta", ascending=False).head(20)
    print("\n[verify] pooled_mase 증가 기여도 상위 20개 row (ratio_raw - ratio_log1p 기준):")
    cols = ["center_id", "sku_id", "week_st", "y_true_log1p", "y_pred_log1p", "y_pred_raw",
            "mase_scale_log1p", "ratio_log1p", "ratio_raw", "ratio_delta"]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(top[cols].to_string(index=False))

    # 상위 1% 최소 mase_scale row를 제외했을 때 평균이 얼마나 바뀌는지 - 극단값 민감도 확인
    low_scale_cutoff = m["mase_scale_log1p"].quantile(0.01)
    trimmed = m[m["mase_scale_log1p"] > low_scale_cutoff]
    print(f"\n[verify] mase_scale 하위 1%(<= {low_scale_cutoff:.4f}) row 제외 시:")
    print(f"  전체:   mean(ratio_log1p)={m['ratio_log1p'].mean():.4f}  mean(ratio_raw)={m['ratio_raw'].mean():.4f}  "
          f"delta={m['ratio_raw'].mean() - m['ratio_log1p'].mean():.4f}")
    print(f"  trim1%: mean(ratio_log1p)={trimmed['ratio_log1p'].mean():.4f}  mean(ratio_raw)={trimmed['ratio_raw'].mean():.4f}  "
          f"delta={trimmed['ratio_raw'].mean() - trimmed['ratio_log1p'].mean():.4f}")
    print(f"  (하위 1% row가 제외된 만큼 delta가 크게 줄어들면 -> 소수의 저변동 시계열이 MASE 증가를 주도한다는 뜻)")

    # 참고: WAPE는 sum(|err|)/sum(y_true) 형태라 개별 row의 denominator 크기와 무관하게
    # 절대오차를 전체 합으로 나누므로, MASE(row-level 비율의 평균)보다 outlier row에 훨씬 둔감하다.
    wape_log1p = m["err_log1p"].abs().sum() / m["y_true_log1p"].sum() * 100
    wape_raw = m["err_raw"].abs().sum() / m["y_true_raw"].sum() * 100
    print(f"\n[verify] (참고) 동일 row subset 기준 WAPE: log1p={wape_log1p:.2f} raw={wape_raw:.2f} "
          f"-> WAPE는 전역 합 비율이라 MASE(row 평균 비율)보다 outlier에 덜 민감함을 보여줌")


if __name__ == "__main__":
    main()
