"""
hpo_pure.py
tune_hyperparameters.py(프로덕션)의 3단계 Optuna HPO(Classifier -> Regressor ->
Tweedie)를 "2024가 어떤 결정에도 1%도 유출되지 않도록" 다시 수행한다.

프로덕션 tune_hyperparameters.py의 유일한 문제 지점은 load_data()가
`a_val_df = df[df["split"]=="val"]`(A의 2024-01~06)를 Optuna 목적함수에 직접
쓴다는 것(load_data():116). 이 스크립트는 그 지점만 "A train(2021~2023)의
시간순 마지막 4주"로 바꾸고, 그 외 모든 로직(목적함수, 탐색공간, 3단계 구조,
Stage1->Stage2 순차 고정 방식)은 tune_hyperparameters.py에서 그대로 import해서
재사용한다(프로덕션 파일 수정 없음, 이 함수들은 val 데이터를 인자로만 받아
내부에서 split 컬럼을 직접 참조하지 않는 순수 함수라 재사용 안전함을 확인함).

python hpo_pure.py

핵심 설계:
    1) A/B 둘 다 "train split(2021~2023 A, 2023-07~12 B)의 시간순 마지막 4주"를
       내부 val로 쓰는 대칭 구조(B는 원래도 이랬음, A만 새로 대칭 적용).
    2) Stage 1(분류기 탐색)이 "고정 기준 회귀기"로 쓰는 fixed_reg_bundle을
       production base_model_reg.pkl(과거에 이미 tune_hyperparameters.py로
       튜닝된, 즉 2024를 간접 참조한 값) 대신 **이 스크립트 안에서 fixed
       하이퍼파라미터(lr=0.03/num_leaves=63, HPO 도입 이전 프로덕션 기본값)로
       새로 학습한 순수 회귀기**로 교체한다 — "100% 완벽하게 2024 미유출"을
       위해 Stage1의 평가 파트너조차 과거 HPO 산출물에 의존하지 않도록 함.
    3) 완전히 별도의 Optuna SQLite study(STUDY_DB_PURE)와 study_name(*_pure)을
       써서, 기존 production optuna_studies.db에 이미 저장된 40회 완료 study를
       load_if_exists=True가 그대로 재사용해버리는 사고(즉 시행 0회로 스킵)를
       방지한다 — 이게 제일 위험한 실수 포인트라 별도 DB로 원천 차단.
    4) N_TRIALS=40(스테이지당, 프로덕션과 동일)을 그대로 유지 — 시간이 걸리더라도
       프로덕션과 동등한 탐색 강도로 비교 가능한 결과를 만들기 위함(품질을
       위해 trial 수를 줄이지 않음).
    5) 결과: data/ml/cv_experiment/best_hyperparams_pure.json,
       data/ml/cv_experiment/optuna_studies_pure.db(재개 가능),
       data/ml/cv_experiment/models/classifier_best_candidate_pure.pkl
       (Stage1 best로 재적합한 분류기, Stage2 고정 기준용 중간 산출물).
       프로덕션 data/ml/models/, data/ml/models/tuning/은 전혀 건드리지 않음.
"""

from pathlib import Path
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(BASE_DIR / "src" / "forecasting" / "archive" / "models_hurdle"))

from common import (  # noqa: E402
    CENTER_COL,
    FEATURE_TABLE_PATH,
    TARGET_COL,
    WEEK_COL,
    apply_target_date_boundary_filter,
    get_base_model_excluded_cols,
    prepare_X,
    save_model_bundle,
    select_feature_cols,
)
from tune_hyperparameters import (  # noqa: E402
    BIAS_ALPHA,
    EARLY_STOPPING_ROUNDS,
    FIXED_PARAMS,
    N_ESTIMATORS_CAP,
    N_TRIALS,
    RANDOM_STATE,
    _run_study,
    combined_score,
    fit_classifier_with_params,
    make_classifier_objective,
    make_regressor_objective,
    make_tweedie_objective,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as cc  # noqa: E402

A_INTERNAL_VAL_WEEKS = 4
B_INTERNAL_VAL_WEEKS = 4

TUNING_DIR = cc.CV_DIR
MODEL_OUT_DIR = cc.CV_DIR / "models"
STUDY_DB_PURE = TUNING_DIR / "optuna_studies_pure.db"
BEST_PARAMS_PURE_PATH = cc.PURE_HYPERPARAMS_PATH

# Stage1 고정 기준 회귀기용 — HPO 도입 이전 프로덕션 기본값(train_center_base_models.py와 동일)
BASELINE_LEARNING_RATE = 0.03
BASELINE_NUM_LEAVES = 63


def load_data_pure():
    """A/B 둘 다 train split의 시간순 마지막 4주를 내부 val로 뗀다 — 2024
    (split=='val'/'test'/'pool')는 이 함수가 반환하는 어떤 데이터프레임에도
    포함되지 않는다."""
    df = pd.read_parquet(FEATURE_TABLE_PATH)
    train_df = df[df["split"] == "train"].copy()
    train_df = apply_target_date_boundary_filter(train_df, horizon_weeks=1)
    train_df = train_df[train_df[TARGET_COL].notna()]

    b_weeks = sorted(train_df.loc[train_df[CENTER_COL] == "B", WEEK_COL].unique())
    b_val_weeks = set(b_weeks[-B_INTERNAL_VAL_WEEKS:])
    is_b_val = (train_df[CENTER_COL] == "B") & (train_df[WEEK_COL].isin(b_val_weeks))

    a_weeks = sorted(train_df.loc[train_df[CENTER_COL] == "A", WEEK_COL].unique())
    a_val_weeks = set(a_weeks[-A_INTERNAL_VAL_WEEKS:])
    is_a_val = (train_df[CENTER_COL] == "A") & (train_df[WEEK_COL].isin(a_val_weeks))

    b_val_df = train_df[is_b_val].copy()
    a_val_df = train_df[is_a_val].copy()
    hpo_train_df = train_df[~is_b_val & ~is_a_val].copy()

    feature_cols = select_feature_cols(hpo_train_df, get_base_model_excluded_cols(hpo_train_df))
    print(
        f"[HPO-pure] hpo_train={len(hpo_train_df):,}행  "
        f"a_val(A train 마지막{A_INTERNAL_VAL_WEEKS}주, {pd.Timestamp(min(a_val_weeks)).date()}~"
        f"{pd.Timestamp(max(a_val_weeks)).date()})={len(a_val_df):,}행  "
        f"b_val(B train 마지막{B_INTERNAL_VAL_WEEKS}주)={len(b_val_df):,}행  "
        f"feature={len(feature_cols)}개  (2024 데이터 전혀 미포함)"
    )
    return hpo_train_df, a_val_df, b_val_df, feature_cols


def fit_pure_baseline_regressor(hpo_train_df, a_val_df, feature_cols):
    """Stage1(분류기 탐색)의 고정 평가 기준 회귀기 — production base_model_reg.pkl
    (과거 HPO로 이미 튜닝돼 2024를 간접 참조한 값)을 쓰지 않고, HPO 이전 프로덕션
    기본 하이퍼파라미터(lr=0.03/num_leaves=63)로 이 pure 데이터셋 위에서 새로
    학습한다 — Stage1이 어떤 경로로도 2024에 의존하지 않도록 하기 위함."""
    pos_train = hpo_train_df[hpo_train_df[TARGET_COL] > 0]
    pos_val = a_val_df[a_val_df[TARGET_COL] > 0]
    X_train = prepare_X(pos_train, feature_cols)
    y_train = np.log1p(pos_train[TARGET_COL])
    X_val = prepare_X(pos_val, feature_cols)
    y_val = np.log1p(pos_val[TARGET_COL])

    model = lgb.LGBMRegressor(
        n_estimators=N_ESTIMATORS_CAP, objective="regression",
        learning_rate=BASELINE_LEARNING_RATE, num_leaves=BASELINE_NUM_LEAVES,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, subsample_freq=1,
        subsample=0.8, colsample_bytree=0.8,
    )
    model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"[HPO-pure] Stage1 고정 기준 회귀기(순수 baseline) 학습 완료 best_iter={model.best_iteration_}")
    return {"model": model, "feature_cols": feature_cols}


def _save_results(results: dict):
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    with open(BEST_PARAMS_PURE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  [저장] {BEST_PARAMS_PURE_PATH}", flush=True)


def _run_study_pure(name: str, objective_fn, n_trials: int):
    """production _run_study와 동일 로직이나 STUDY_DB_PURE(별도 SQLite 파일)를 쓴다
    — production optuna_studies.db의 기존 완료 study를 잘못 재사용해 시행 0회로
    스킵되는 사고를 방지하기 위한 핵심 안전장치."""
    import optuna
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize", study_name=f"{name}_pure",
        storage=f"sqlite:///{STUDY_DB_PURE}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    n_done = len(study.trials)
    n_remaining = max(0, n_trials - n_done)
    if n_remaining == 0:
        print(f"  [{name}_pure] 이미 {n_done}회 완료 -> 재사용(추가 시행 없음)", flush=True)
    else:
        print(f"  [{name}_pure] 기존 {n_done}회 + 신규 {n_remaining}회 시행 예정", flush=True)
        study.optimize(objective_fn, n_trials=n_remaining, catch=(Exception,))
    return study


def main():
    t0 = time.time()
    print("=" * 80)
    print("[HPO-pure] 데이터 로드 중 (A/B 둘 다 train 마지막 4주 내부val, 2024 완전 배제)...", flush=True)
    hpo_train_df, a_val_df, b_val_df, feature_cols = load_data_pure()
    results = {}

    print("=" * 80)
    print(f"[Stage 0/3] Stage1 고정 기준 회귀기(순수 baseline) 준비", flush=True)
    fixed_reg_bundle = fit_pure_baseline_regressor(hpo_train_df, a_val_df, feature_cols)

    print("=" * 80)
    print(f"[Stage 1/3] Classifier 탐색 (목표 {N_TRIALS}회)", flush=True)
    study_cls = _run_study_pure(
        "classifier",
        make_classifier_objective(hpo_train_df, a_val_df, b_val_df, feature_cols, fixed_reg_bundle),
        N_TRIALS,
    )
    print(f"[Stage 1/3] 완료. best score={study_cls.best_value:.3f}  best_params={study_cls.best_params}", flush=True)
    results["classifier"] = {
        "best_score": study_cls.best_value, "best_params": study_cls.best_params,
        "best_attrs": study_cls.best_trial.user_attrs, "n_trials": len(study_cls.trials),
    }
    _save_results(results)

    print("  [Stage 1/3] best 파라미터로 classifier 재적합(Stage2 고정 기준 생성용)...", flush=True)
    best_cls_model = fit_classifier_with_params(hpo_train_df, a_val_df, feature_cols, study_cls.best_params)
    best_cls_bundle = {"model": best_cls_model, "feature_cols": feature_cols}
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_model_bundle(MODEL_OUT_DIR / "classifier_best_candidate_pure.pkl", best_cls_model, feature_cols,
                       **study_cls.best_params)

    print("=" * 80)
    print(f"[Stage 2/3] Regressor 탐색 (목표 {N_TRIALS}회, Stage1 best classifier 고정)", flush=True)
    study_reg = _run_study_pure(
        "regressor",
        make_regressor_objective(hpo_train_df, a_val_df, b_val_df, feature_cols, best_cls_bundle),
        N_TRIALS,
    )
    print(f"[Stage 2/3] 완료. best score={study_reg.best_value:.3f}  best_params={study_reg.best_params}", flush=True)
    results["regressor"] = {
        "best_score": study_reg.best_value, "best_params": study_reg.best_params,
        "best_attrs": study_reg.best_trial.user_attrs, "n_trials": len(study_reg.trials),
    }
    _save_results(results)

    print("=" * 80)
    print(f"[Stage 3/3] Tweedie 탐색 (목표 {N_TRIALS}회, 독립 단일모델)", flush=True)
    study_tw = _run_study_pure(
        "tweedie",
        make_tweedie_objective(hpo_train_df, a_val_df, b_val_df, feature_cols),
        N_TRIALS,
    )
    print(f"[Stage 3/3] 완료. best score={study_tw.best_value:.3f}  best_params={study_tw.best_params}", flush=True)
    results["tweedie"] = {
        "best_score": study_tw.best_value, "best_params": study_tw.best_params,
        "best_attrs": study_tw.best_trial.user_attrs, "n_trials": len(study_tw.trials),
    }
    _save_results(results)

    elapsed_min = (time.time() - t0) / 60
    print("=" * 80)
    print(f"[HPO-pure 전체 완료] 총 소요 {elapsed_min:.1f}분 (2024 데이터는 어떤 단계에도 미참조)")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
