"""
run_a_expanding_cv_lgbm_multih.py
"A센터 CV를 왜 2023년 안에서만 했는지"를 검증하기 위한 LightGBM 확장 실험.

folds.A_EXPANDING_H2_FOLDS(Fold1: train 2021 1년 -> val 2022 전체, Fold2~4: 6개월씩
Expanding)로 A센터 단독 Hurdle 모델을 h=1/2/4 각각 새로 학습해, "train 1년(Fold1)이
train 2~3년(Fold2~4)보다 Val이 눈에 띄게 불안정한가"를 직접 관찰한다. 전 구간
best_hyperparams_pure.json(2024 완전 미참조) 사용, early-stop val은 학습구간 마지막
4주 내부 val(폴드의 실제 val과는 별개).

h=2는 이 코드베이스에 존재하지 않던 타겟이라 이 스크립트 안에서 직접 생성한다
(target_h1/target_h4는 A_train/val/test_feat.parquet에 이미 있음 - A_HORIZONS=[1,4,8]로 만들어졌던 산출물). h=2 계산은 split_train_val_test.add_horizon_targets와 정확히 같은 공식(그룹별 -h shift, global_end 넘어가면 NaN)을 재현한다.

A 전용 모델이라 get_base_model_excluded_cols(A/B 공통 피처만 남기는 함수) 대신 get_excluded_cols(전체 피처, qty_lag2/4 포함)를 쓴다 — train_center_base_models.py의 A전용 모델 관례와 동일.

최종 재학습(2021~2023 전체) 후 2024 전체를 각 horizon 타겟으로 순수 holdout 평가.

python run_a_expanding_cv_lgbm_multih.py

산출물: data/ml/cv_experiment/reports/a_expanding_cv_lgbm_multih_report.{csv,md}
프로덕션 파일은 전혀 건드리지 않는다(A_train/val/test_feat.parquet은 read-only로만 읽음).
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as cc  # noqa: E402
import folds  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(BASE_DIR / "src" / "forecasting" / "archive" / "models_hurdle"))
from common import (  # noqa: E402
    CENTER_COL,
    WEEK_COL,
    compute_metrics,
    get_excluded_cols,
    prepare_X,
    predict_base_hurdle,
    select_feature_cols,
)

SKU_COL = "sku_id"
QTY_COL = "qty"
HORIZONS = [1, 2, 4]
N_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 30
RANDOM_STATE = 42
FALLBACK_LR = 0.03
FALLBACK_LEAVES = 63
FIXED_PARAMS = dict(subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1)

REPORT_DIR = cc.CV_DIR / "reports"
ROUND = 3


def add_target_h2(df: pd.DataFrame) -> pd.DataFrame:
    """split_train_val_test.add_horizon_targets와 동일 공식으로 target_h2만 추가 생성
    (원본 파이프라인이 A_HORIZONS=[1,4,8]로 만들어져 h=2는 애초에 없었음)."""
    df = df.copy()
    global_end = df[WEEK_COL].max()
    grouped = df.groupby(SKU_COL, sort=False)
    shifted_qty = grouped[QTY_COL].shift(-2)
    beyond_end = (df[WEEK_COL] + pd.Timedelta(weeks=2)) > global_end
    df["target_h2"] = np.where(beyond_end, np.nan, shifted_qty)
    return df


def get_feature_cols_a(df: pd.DataFrame) -> list[str]:
    return select_feature_cols(df, get_excluded_cols(df))


def train_hurdle_h(train_df: pd.DataFrame, es_val_df: pd.DataFrame, feature_cols: list[str],
                    target_col: str, label: str) -> tuple[dict, dict]:
    """cv_common.train_hurdle을 A 단독/임의 horizon 타겟에 맞게 다시 쓴 버전(B 가중치 없음,
    target_col을 인자로 받음). 하이퍼파라미터는 항상 use_pure=True(best_hyperparams_pure.json) —
    h=2/h=4용으로 별도 재탐색된 값이 없어 h=1 튜닝값을 그대로 재사용한다는 근사임을
    명시(한계로 리포트에 남김)."""
    tuned_cls = cc.load_tuned_params("classifier", use_pure=True)
    tuned_reg = cc.load_tuned_params("regressor", use_pure=True)

    y_cls_train = (train_df[target_col] > 0).astype(int)
    y_cls_val = (es_val_df[target_col] > 0).astype(int)
    n_pos, n_neg = int((y_cls_train == 1).sum()), int((y_cls_train == 0).sum())
    natural_spw = n_neg / max(n_pos, 1)

    X_cls_train = prepare_X(train_df, feature_cols)
    X_cls_val = prepare_X(es_val_df, feature_cols)

    if tuned_cls is not None:
        scale_pos_weight = natural_spw * tuned_cls["scale_pos_weight_multiplier"]
        lgb_params = {k: tuned_cls[k] for k in
                      ["num_leaves", "max_depth", "learning_rate", "min_child_samples",
                       "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]}
    else:
        scale_pos_weight = natural_spw
        lgb_params = dict(learning_rate=FALLBACK_LR, num_leaves=FALLBACK_LEAVES, **FIXED_PARAMS)
    cls_model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS, scale_pos_weight=scale_pos_weight, subsample_freq=1,
        verbose=-1, random_state=RANDOM_STATE, n_jobs=-1, **lgb_params,
    )
    cls_model.fit(
        X_cls_train, y_cls_train, eval_set=[(X_cls_val, y_cls_val)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"    [{label}] 분류기 best_iter={cls_model.best_iteration_} scale_pos_weight={scale_pos_weight:.3f}")

    pos_train = train_df[train_df[target_col] > 0]
    pos_val = es_val_df[es_val_df[target_col] > 0]
    X_reg_train = prepare_X(pos_train, feature_cols)
    y_reg_train = np.log1p(pos_train[target_col])
    X_reg_val = prepare_X(pos_val, feature_cols)
    y_reg_val = np.log1p(pos_val[target_col])

    if tuned_reg is not None:
        lgb_params_r = {k: tuned_reg[k] for k in
                        ["num_leaves", "max_depth", "learning_rate", "min_child_samples",
                         "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]}
    else:
        lgb_params_r = dict(learning_rate=FALLBACK_LR, num_leaves=FALLBACK_LEAVES, **FIXED_PARAMS)
    reg_model = lgb.LGBMRegressor(
        n_estimators=N_ESTIMATORS, objective="regression", subsample_freq=1,
        verbose=-1, random_state=RANDOM_STATE, n_jobs=-1, **lgb_params_r,
    )
    reg_model.fit(
        X_reg_train, y_reg_train, eval_set=[(X_reg_val, y_reg_val)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"    [{label}] 회귀기 best_iter={reg_model.best_iteration_}")

    return {"model": cls_model, "feature_cols": feature_cols}, {"model": reg_model, "feature_cols": feature_cols}


def score(df: pd.DataFrame, cls_bundle: dict, reg_bundle: dict, target_col: str) -> dict:
    d = df[df[target_col].notna()]
    if len(d) == 0:
        return {"n": 0, "RMSE": np.nan, "MAE": np.nan, "WAPE": np.nan, "MASE": np.nan, "Bias(%)": np.nan}
    y_true = d[target_col].to_numpy()
    y_pred = predict_base_hurdle(d, cls_bundle, reg_bundle, mode="soft")
    naive_pred = d[QTY_COL].to_numpy()
    m = compute_metrics(y_true, y_pred, naive_pred)
    return {"n": len(d), **m}


def main():
    print("=== A_full 로드 + target_h2 생성 ===")
    a_full = cc.load_a_full()
    a_full = add_target_h2(a_full)
    print(f"  A_full: {len(a_full):,}행, {a_full[WEEK_COL].min().date()} ~ {a_full[WEEK_COL].max().date()}")
    for h in HORIZONS:
        print(f"  target_h{h} notna 개수: {a_full[f'target_h{h}'].notna().sum():,}")

    fold_rows = []
    for f in folds.A_EXPANDING_H2_FOLDS:
        print("=" * 80)
        print(f"[Fold {f['fold']}] train {f['train_start']}~{f['train_end']} -> val {f['val_start']}~{f['val_end']}")
        train_window_raw = cc.slice_by_date(a_full, f["train_start"], f["train_end"])
        val_window = cc.slice_by_date(a_full, f["val_start"], f["val_end"])

        for h in HORIZONS:
            target_col = f"target_h{h}"
            train_window = cc.apply_boundary_filter(train_window_raw, f["train_end"], horizon_weeks=h)
            train_window = train_window[train_window[target_col].notna()]
            train_fit, es_val = cc.carve_internal_es_val(train_window)
            feature_cols = get_feature_cols_a(train_fit)

            label = f"A_ext_fold{f['fold']}_h{h}"
            cls_b, reg_b = train_hurdle_h(train_fit, es_val, feature_cols, target_col, label)
            m = score(val_window, cls_b, reg_b, target_col)
            print(f"  [Fold {f['fold']} h={h}] Val WAPE={m['WAPE']:.2f}% RMSE={m['RMSE']:.2f} n={m['n']:,}")
            fold_rows.append({"fold": f["fold"], "train_start": f["train_start"], "train_end": f["train_end"],
                               "val_start": f["val_start"], "val_end": f["val_end"], "horizon": f"h{h}",
                               **{k: round(v, ROUND) if isinstance(v, float) else v for k, v in m.items()}})

    fold_df = pd.DataFrame(fold_rows)
    print()
    print("=" * 80)
    print("[Fold별 상세]")
    print(fold_df.to_string(index=False))

    summary = (
        fold_df.groupby("horizon")["WAPE"]
        .agg(WAPE_mean="mean", WAPE_std="std")
        .reset_index()
    )
    fold1_wape = fold_df[fold_df["fold"] == 1][["horizon", "WAPE"]].rename(columns={"WAPE": "Fold1_WAPE(train 1yr)"})
    rest_mean = (
        fold_df[fold_df["fold"] != 1].groupby("horizon")["WAPE"].mean()
        .reset_index().rename(columns={"WAPE": "Fold2~4_WAPE_mean(train 2~3yr)"})
    )
    stability = fold1_wape.merge(rest_mean, on="horizon")
    print()
    print("[Fold1(train 1년) vs Fold2~4(train 2~3년) 평균 비교 - 데이터 부족 가설 검증]")
    print(stability.to_string(index=False))

    print()
    print("=== 최종 재학습(2021~2023 전체) -> 2024 전체 Pure Holdout ===")
    final_rows = []
    for h in HORIZONS:
        target_col = f"target_h{h}"
        window = cc.slice_by_date(a_full, folds.A_FINAL_TRAIN_START, folds.A_FINAL_TRAIN_END)
        window = cc.apply_boundary_filter(window, folds.A_FINAL_TRAIN_END, horizon_weeks=h)
        window = window[window[target_col].notna()]
        train_fit, es_val = cc.carve_internal_es_val(window)
        feature_cols = get_feature_cols_a(train_fit)
        cls_b, reg_b = train_hurdle_h(train_fit, es_val, feature_cols, target_col, f"A_final_h{h}")

        test_df = cc.slice_by_date(a_full, folds.FINAL_TEST_START, folds.FINAL_TEST_END)
        m = score(test_df, cls_b, reg_b, target_col)
        print(f"  [최종 h={h}] 2024 Test WAPE={m['WAPE']:.2f}% RMSE={m['RMSE']:.2f} n={m['n']:,}")
        final_rows.append({"horizon": f"h{h}", **{k: round(v, ROUND) if isinstance(v, float) else v for k, v in m.items()}})

    final_df = pd.DataFrame(final_rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(REPORT_DIR / "a_expanding_cv_lgbm_multih_folds.csv", index=False, encoding="utf-8-sig")
    final_df.to_csv(REPORT_DIR / "a_expanding_cv_lgbm_multih_final.csv", index=False, encoding="utf-8-sig")

    md_lines = ["# A센터 LightGBM Expanding CV 확장 실험 (h=1,2,4)", "",
                "## Fold별 상세 (Val WAPE)", "",
                "| " + " | ".join(fold_df.columns) + " |",
                "| " + " | ".join(["---"] * len(fold_df.columns)) + " |"]
    for _, r in fold_df.iterrows():
        md_lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    md_lines += ["", "## Fold1(train 1년) vs Fold2~4(train 2~3년) 평균", "",
                 "| " + " | ".join(stability.columns) + " |",
                 "| " + " | ".join(["---"] * len(stability.columns)) + " |"]
    for _, r in stability.iterrows():
        md_lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    md_lines += ["", "## 최종 재학습(2021~2023) -> 2024 Pure Holdout", "",
                 "| " + " | ".join(final_df.columns) + " |",
                 "| " + " | ".join(["---"] * len(final_df.columns)) + " |"]
    for _, r in final_df.iterrows():
        md_lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    md_lines += ["", "각주: h=1은 기존 target_h1 재사용, h=4는 기존 target_h4(A_HORIZONS=[1,4,8] 산출물) 재사용, "
                      "h=2는 이 스크립트에서 add_horizon_targets와 동일 공식으로 신규 생성. 전 horizon 공통으로 "
                      "best_hyperparams_pure.json(target_h1 기준 튜닝값)을 그대로 재사용했다 — h=2/h=4 전용 재탐색은 "
                      "하지 않았으므로 절대 수치보다 Fold 간 상대 비교(안정성) 위주로 해석할 것."]
    (REPORT_DIR / "a_expanding_cv_lgbm_multih_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n저장 완료 -> {REPORT_DIR}/a_expanding_cv_lgbm_multih_*.{{csv,md}}")


if __name__ == "__main__":
    main()
