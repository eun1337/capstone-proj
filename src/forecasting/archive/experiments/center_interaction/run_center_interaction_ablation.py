"""
run_center_interaction_ablation.py
센터 상호작용 피처(center_qty_lag1_inter/center_temp_inter) 및 Hurdle soft 결합 효과를
서로 독립적으로 검증(Ablation)하는 실험 스크립트.

배경: feature_center_interaction_experiment.py가 "B 회귀 WAPE 61%(재검토 조건)"를 확인해
상호작용 피처(data/ml/feature_engineering_experiment/feature_table_with_center_interactions.parquet)
를 만들었으나, 실제 성능 개선 여부를 확인하기 전이라 아직 프로덕션(feature_external_interaction_concat.py)
에 반영되지 않은 상태다. 이 스크립트가 그 ablation을 수행해 반영 여부 판단 근거를 만든다.

핵심 설계 (팀 협의 확정):
    1) 프로덕션 산출물(base_model_cls/reg.pkl, feature_table_final.parquet,
       train_base_model.py, common.py, evaluate_pipeline.py)은 전부 read-only로만
       import/참조한다 — 이 스크립트가 직접 수정하거나 덮어쓰지 않는다. 재학습 모델과
       리포트는 전부 data/ml/experiments/center_interaction_ablation/ 아래에만 저장한다.
    2) 타겟 정의는 train_base_model.py와 완전히 동일하게 target_h1 기준을 쓴다
       (1단계 분류=(target_h1>0), 2단계 회귀=log1p(target_h1), target_h1>0인 행만).
       sold_flag/qty_log1p(당일 시점 컬럼)는 쓰지 않는다 — h1주 뒤 미래를 보는
       이 프로젝트의 타겟과 시점이 다르기 때문(common.py 모듈 docstring 참고).
    3) 하이퍼파라미터는 원인을 격리(feature 효과 vs soft 전환 효과)해서 보기 위해
       tune_hyperparameters.py 산출물(data/ml/models/tuning/best_hyperparams.json)을
       Baseline과 동일하게 그대로 적용한다(train_base_model.load_tuned_params 재사용).
    4) 비교 3종(전부 target_h1 기준 Hurdle: 분류+조건부회귀):
         - Baseline: 기존 프로덕션 모델(재학습 없음, base_model_cls/reg.pkl 그대로 로드)
           + Hard 모드 + feature_table_final.parquet
         - Ablation 1(피처 효과 격리): 상호작용 피처 포함 재학습 + Hard 모드
           + feature_table_with_center_interactions.parquet
         - Ablation 2(피처+soft 결합 효과): Ablation 1과 동일 모델 + Soft 모드
           (prob x expm1(회귀예측), common.predict_base_hurdle(mode="soft") 그대로 재사용)
       Ablation 1/2는 같은 모델을 한 번만 학습하고 평가만 Hard/Soft 두 번 수행한다
       (모델을 두 번 학습하지 않음 — 결합 모드 차이만 격리하기 위함).
    5) 평가는 evaluate_pipeline.py의 evaluate_a/evaluate_b_walkforward를 그대로 재사용한다
       (A: A_test 2024-07~12 홀드아웃 / B: B_walkforward_folds.json 53-fold Pool 검증,
       공식 파이프라인과 동일한 정의로 숫자를 신뢰할 수 있게 함). 세 시나리오 모두
       "n"(평가 행수)이 같은지 리포트에서 눈으로 확인 가능 — 다르면 무언가 잘못된 것.
    6) 검증 목표: B WAPE가 통계 트랙(ARIMA no_exog) Baseline 수준(h1 기준 약 17.3%,
       data/ml/models/sarima/performance_comparison.csv의 pool_2024/B/arima/h1)에
       근접하는지, A 성능 저하(Degradation)가 없는지 A/B 분리 표로 확인한다.
    7) 반품(returns) 데이터는 이 실험 어디에서도 참조하지 않는다(qty/target_h1 기준).
"""

from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(BASE_DIR / "src" / "forecasting" / "archive" / "models_hurdle"))
sys.path.insert(0, str(BASE_DIR / "src" / "forecasting" / "archive" / "feature_engineering"))

from common import (  # noqa: E402
    BASE_CLS_MODEL_PATH,
    BASE_MODEL_PATH,
    FEATURE_TABLE_PATH,
    TARGET_COL,
    apply_target_date_boundary_filter,
    get_base_model_excluded_cols,
    load_model_bundle,
    prepare_X,
    save_model_bundle,
    select_feature_cols,
)
from evaluate_pipeline import evaluate_a, evaluate_b_walkforward  # noqa: E402
import train_base_model as tbm  # noqa: E402

EXP_DIR = BASE_DIR / "data" / "ml" / "experiments" / "center_interaction_ablation"
MODEL_DIR = EXP_DIR / "models"
REPORT_DIR = EXP_DIR / "reports"
INTERACTION_TABLE_PATH = (
    BASE_DIR / "data" / "ml" / "feature_engineering_experiment" / "feature_table_with_center_interactions.parquet"
)
ABL_CLS_PATH = MODEL_DIR / "ablation_cls.pkl"
ABL_REG_PATH = MODEL_DIR / "ablation_reg.pkl"

# data/ml/models/sarima/performance_comparison.csv, eval_split=pool_2024, center=B, variant=arima, horizon=h1.
# ARIMA/ARIMAX 트랙 최종 채택 baseline(no_exog) 값 — 이번 재검토 트리거였던 B 회귀 WAPE 61%가
# 이 수준까지 개선되는지 참고용으로만 비교(하드 기준선 아님, 팀 확정 결론은 project_arimax_track_concluded 참고).
SARIMA_B_H1_WAPE = 17.342

REPORT_COLS = ["scenario", "center", "n", "RMSE", "MAE", "WAPE", "MASE", "Bias(%)"]


def train_ablation_hurdle(df: pd.DataFrame) -> tuple[dict, dict]:
    """train_base_model.py의 Hurdle 학습 절차(target_h1 기준 1단계 분류 + 2단계 조건부회귀,
    best_hyperparams.json 적용)를 상호작용 피처 포함 feature table에 동일하게 재현한다.
    프로덕션 base_model_*.pkl은 절대 덮어쓰지 않고, 학습된 번들만 반환한다(저장은 호출부 책임)."""
    train_df = df[df["split"] == "train"].copy()
    print(f"[Ablation] train 행수(피처 포함, 필터 전): {len(train_df):,}")
    train_df = apply_target_date_boundary_filter(train_df, horizon_weeks=1)
    train_df = train_df[train_df[TARGET_COL].notna()]
    print(f"[Ablation] target NaN 제외 후: {len(train_df):,}행")

    val_df = df[df["split"] == "val"].copy()
    val_df = val_df[val_df[TARGET_COL].notna()]
    print(f"[Ablation] val(A만 존재) 행수: {len(val_df):,}")

    excluded = get_base_model_excluded_cols(train_df)
    feature_cols = select_feature_cols(train_df, excluded)
    has_new_feats = "center_qty_lag1_inter" in feature_cols and "center_temp_inter" in feature_cols
    print(f"[Ablation] feature 개수: {len(feature_cols)}개 (신규 상호작용 피처 포함: {has_new_feats})")
    if not has_new_feats:
        raise RuntimeError("center_qty_lag1_inter/center_temp_inter가 feature_cols에 없음 — 입력 파일 확인 필요")

    tuned_cls_params = tbm.load_tuned_params("classifier")
    tuned_reg_params = tbm.load_tuned_params("regressor")
    if tuned_cls_params is None or tuned_reg_params is None:
        raise RuntimeError(
            f"{tbm.BEST_HYPERPARAMS_PATH} 에서 classifier/regressor best_params를 찾지 못함 — "
            "이 실험은 Baseline과 동일 하이퍼파라미터 적용이 전제 조건(원인 격리 목적)이라 폴백 그리드로 진행하지 않음"
        )

    # ---- 1단계: 분류기 (target_h1 > 0) ----
    y_cls_train = (train_df[TARGET_COL] > 0).astype(int)
    y_cls_val = (val_df[TARGET_COL] > 0).astype(int)
    n_pos, n_neg = int((y_cls_train == 1).sum()), int((y_cls_train == 0).sum())
    natural_scale_pos_weight = n_neg / n_pos
    print(f"[Ablation 1단계-분류] 양성비율={y_cls_train.mean():.1%} natural_scale_pos_weight={natural_scale_pos_weight:.3f}")

    X_cls_train = prepare_X(train_df, feature_cols)
    X_cls_val = prepare_X(val_df, feature_cols)

    print(f"[Ablation 1단계-분류] best_hyperparams.json 적용: {tuned_cls_params}")
    scale_pos_weight = natural_scale_pos_weight * tuned_cls_params["scale_pos_weight_multiplier"]
    sw_cls_train = tbm.b_sample_weight(train_df, tuned_cls_params["b_center_weight"])
    cls_lgb_params = {k: tuned_cls_params[k] for k in
                       ["num_leaves", "max_depth", "learning_rate", "min_child_samples",
                        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]}
    cls_model = lgb.LGBMClassifier(
        n_estimators=tbm.N_ESTIMATORS, scale_pos_weight=scale_pos_weight, subsample_freq=1,
        verbose=-1, random_state=tbm.RANDOM_STATE, n_jobs=-1, **cls_lgb_params,
    )
    cls_model.fit(
        X_cls_train, y_cls_train, sample_weight=sw_cls_train, eval_set=[(X_cls_val, y_cls_val)],
        callbacks=[lgb.early_stopping(tbm.EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )

    # ---- 2단계: 조건부회귀기 (target_h1 > 0인 행만, log1p(target_h1)) ----
    pos_train = train_df[train_df[TARGET_COL] > 0]
    pos_val = val_df[val_df[TARGET_COL] > 0]
    print(f"[Ablation 2단계-조건부회귀] train 대상행: {len(pos_train):,} / val 대상행: {len(pos_val):,}")
    X_reg_train = prepare_X(pos_train, feature_cols)
    y_reg_train = np.log1p(pos_train[TARGET_COL])
    X_reg_val = prepare_X(pos_val, feature_cols)
    y_reg_val = np.log1p(pos_val[TARGET_COL])

    print(f"[Ablation 2단계-조건부회귀] best_hyperparams.json 적용: {tuned_reg_params}")
    sw_reg_train = tbm.b_sample_weight(pos_train, tuned_reg_params["b_center_weight"])
    reg_lgb_params = {k: tuned_reg_params[k] for k in
                       ["num_leaves", "max_depth", "learning_rate", "min_child_samples",
                        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]}
    reg_model = lgb.LGBMRegressor(
        n_estimators=tbm.N_ESTIMATORS, objective="regression", subsample_freq=1,
        verbose=-1, random_state=tbm.RANDOM_STATE, n_jobs=-1, **reg_lgb_params,
    )
    reg_model.fit(
        X_reg_train, y_reg_train, sample_weight=sw_reg_train, eval_set=[(X_reg_val, y_reg_val)],
        callbacks=[lgb.early_stopping(tbm.EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )

    # ---- Hard 모드용 threshold 탐색 (val 전체, train_base_model.py와 동일 절차) ----
    cls_bundle_tmp = {"model": cls_model, "feature_cols": feature_cols}
    reg_bundle_tmp = {"model": reg_model, "feature_cols": feature_cols}
    best_threshold, threshold_table = tbm.search_threshold(val_df, cls_bundle_tmp, reg_bundle_tmp)
    print(f"[Ablation] Hard 모드 threshold 채택(WAPE 최소): {best_threshold:.2f}")
    print(threshold_table.to_string(index=False))

    cls_bundle = {
        "model": cls_model, "feature_cols": feature_cols, "threshold": float(best_threshold),
        "scale_pos_weight": scale_pos_weight, "tuned_params": tuned_cls_params,
    }
    reg_bundle = {"model": reg_model, "feature_cols": feature_cols, "tuned_params": tuned_reg_params}

    print()
    print("[Ablation 1단계-분류] Feature Importance Top 10 (신규 상호작용 피처 순위 확인용)")
    print(pd.Series(cls_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10).to_string())
    print("[Ablation 2단계-조건부회귀] Feature Importance Top 10")
    print(pd.Series(reg_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10).to_string())

    return cls_bundle, reg_bundle


def build_report_rows(scenario: str, result_a: dict, result_b: dict) -> list[dict]:
    return [
        {"scenario": scenario, **result_a},
        {"scenario": scenario, **result_b},
    ]


def main():
    print("=" * 80)
    print("[1/3] Baseline: 기존 프로덕션 모델(base_model_cls/reg.pkl, 재학습 없음) + Hard 모드 "
          "+ feature_table_final.parquet")
    base_df = pd.read_parquet(FEATURE_TABLE_PATH)
    base_cls_bundle = load_model_bundle(BASE_CLS_MODEL_PATH)
    base_reg_bundle = load_model_bundle(BASE_MODEL_PATH)
    baseline_a = evaluate_a(base_df, base_cls_bundle, base_reg_bundle, mode="hard")
    baseline_b, _ = evaluate_b_walkforward(base_df, base_cls_bundle, base_reg_bundle, mode="hard")

    print()
    print("=" * 80)
    print("[2/3] Ablation 모델 학습: 상호작용 피처 포함 + best_hyperparams.json 적용 "
          "(feature_table_with_center_interactions.parquet)")
    interaction_df = pd.read_parquet(INTERACTION_TABLE_PATH)
    abl_cls_bundle, abl_reg_bundle = train_ablation_hurdle(interaction_df)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_model_bundle(
        ABL_CLS_PATH, abl_cls_bundle["model"], abl_cls_bundle["feature_cols"],
        threshold=abl_cls_bundle["threshold"], scale_pos_weight=abl_cls_bundle["scale_pos_weight"],
        tuned_params=abl_cls_bundle["tuned_params"],
    )
    save_model_bundle(
        ABL_REG_PATH, abl_reg_bundle["model"], abl_reg_bundle["feature_cols"],
        tuned_params=abl_reg_bundle["tuned_params"],
    )

    print()
    print("=" * 80)
    print("[3/3] Ablation 평가: Hard 모드(Ablation 1, 피처 효과) / Soft 모드(Ablation 2, 피처+결합모드 효과)")
    abl1_a = evaluate_a(interaction_df, abl_cls_bundle, abl_reg_bundle, mode="hard")
    abl1_b, _ = evaluate_b_walkforward(interaction_df, abl_cls_bundle, abl_reg_bundle, mode="hard")
    abl2_a = evaluate_a(interaction_df, abl_cls_bundle, abl_reg_bundle, mode="soft")
    abl2_b, _ = evaluate_b_walkforward(interaction_df, abl_cls_bundle, abl_reg_bundle, mode="soft")

    rows = []
    rows += build_report_rows("Baseline (production model, Hard)", baseline_a, baseline_b)
    rows += build_report_rows("Ablation1 (+interaction feat, Hard)", abl1_a, abl1_b)
    rows += build_report_rows("Ablation2 (+interaction feat, Soft)", abl2_a, abl2_b)
    report = pd.DataFrame(rows)[REPORT_COLS]

    print()
    print("=" * 80)
    print("[최종 비교 리포트] Baseline vs Ablation1(피처 효과) vs Ablation2(피처+Soft 결합 효과)")
    print(report.to_string(index=False))

    n_by_center = report.groupby("center")["n"].nunique()
    mismatched = n_by_center[n_by_center > 1]
    if len(mismatched) > 0:
        print(f"  ⚠ 센터별 평가 행수(n)가 시나리오 간 다름(정상이면 안 됨): {mismatched.to_dict()}")
    else:
        print("  (3개 시나리오 모두 센터별 평가 행수(n) 동일 확인 - 정상)")

    print()
    print(f"참고: 통계 트랙(ARIMA no_exog) B h1 WAPE Baseline = {SARIMA_B_H1_WAPE:.2f}% "
          f"(data/ml/models/sarima/performance_comparison.csv, pool_2024 — 하드 기준선 아닌 참고용)")
    b_rows = report[report["center"] == "B"]
    best_b_row = b_rows.loc[b_rows["WAPE"].idxmin()]
    print(f"  B WAPE 최소: {best_b_row['scenario']} = {best_b_row['WAPE']:.2f}% "
          f"({'통계 트랙 Baseline 수준 근접(1.1배 이내)' if best_b_row['WAPE'] <= SARIMA_B_H1_WAPE * 1.1 else '아직 격차 있음'})")

    a_rows = report[report["center"] == "A"]
    base_a_wape = a_rows.loc[a_rows["scenario"].str.startswith("Baseline"), "WAPE"].iloc[0]
    print(f"  A WAPE Baseline = {base_a_wape:.2f}% 대비:")
    for _, r in a_rows[~a_rows["scenario"].str.startswith("Baseline")].iterrows():
        delta = r["WAPE"] - base_a_wape
        flag = "부작용(Degradation) 의심" if delta > 0.5 else "저하 없음"
        print(f"    {r['scenario']}: WAPE={r['WAPE']:.2f}% (Baseline 대비 {delta:+.2f}%p) - {flag}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "ablation_report.csv"
    report.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[저장 완료] {out_path}")
    print(f"[저장 완료] {ABL_CLS_PATH}")
    print(f"[저장 완료] {ABL_REG_PATH}")


if __name__ == "__main__":
    main()
