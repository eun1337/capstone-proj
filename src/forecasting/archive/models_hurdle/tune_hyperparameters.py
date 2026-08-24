"""
tune_hyperparameters.py
Optuna 기반 Hurdle Classifier / Hurdle Regressor / Tweedie 하이퍼파라미터 + 가중치 탐색

목표: (1) LightGBM 8종 파라미터(num_leaves/max_depth/learning_rate/min_child_samples/
subsample/colsample_bytree/reg_alpha/reg_lambda) 자동 탐색, (2) 분류기 scale_pos_weight
보정(자연 비율 대비 배수), (3) B센터 sample weight(1.0~3.0배)로 B 과소표본 문제 완화,
(4) 이 전부를 "Validation WAPE + α*|Bias|" 복합 목적함수로 한 번에 최적화.

핵심 설계:
    1) 목적함수가 A와 B를 "행 수 비례"가 아니라 "센터 평균"으로 결합한다
       (score = (WAPE_A+WAPE_B)/2 + ALPHA*(|Bias_A|+|Bias_B|)/2). A가 B보다 행 수가
       훨씬 많아(1.5M vs 170K) 단순 concat WAPE를 쓰면 B센터 개선분이 지표에 거의
       반영되지 않아 "B센터 집중 케어" 목표 자체가 무의미해지기 때문 — B를 동등한
       비중으로 넣어야 b_center_weight 탐색이 실제로 보상받는다.
    2) B에는 val split이 없어(train/pool만 존재), train_center_base_models.py와 동일한
       관례를 그대로 재사용한다 — B train의 시간순 마지막 4주(B_INTERNAL_VAL_WEEKS)를
       "내부 검증용"으로 떼어 HPO 학습에서 제외하고 평가에만 쓴다. B_pool(진짜 최종
       홀드아웃)은 튜닝 과정에 절대 사용하지 않는다(모델 선택에 test 정보가 새면 안 됨).
    3) 3단계 순차 탐색(분류기 -> 회귀기 -> Tweedie). 분류기/회귀기는 서로를 거쳐야만
       최종 WAPE가 나오는 2단계 파이프라인이라 완전한 동시 최적화는 비현실적 —
       1단계에서는 "현재 저장된 production 회귀기"를 고정 기준으로 분류기를 탐색하고,
       1단계 best로 실제 재적합한 분류기를 2단계의 고정 기준으로 삼아 회귀기를 탐색한다
       (표준적인 alternating/coordinate 방식 근사). Tweedie는 단일모델이라 독립적으로 탐색.
    4) 각 study는 SQLite(optuna_studies.db)에 저장(load_if_exists=True) — 오래 걸리는
       백그라운드 작업이라 중간에 끊겨도 이어서 볼 수 있고, 시행 1개가 실패해도
       (catch=(Exception,)) 전체 study가 죽지 않는다.
    5) subsample(=bagging_fraction)은 LightGBM에서 subsample_freq(=bagging_freq)>0이
       아니면 조용히 무시되는 잘 알려진 함정이 있어 subsample_freq=1을 고정으로 같이 준다.
    6) 여기서 찾은 best_hyperparams.json은 이 스크립트가 직접 production bundle을
       덮어쓰지 않는다 — train_base_model.py/train_tweedie_model.py가 이 파일을 읽어
       "전체 train 데이터(B 마지막 4주 제외 없이 전부)"로 최종 재학습하는 별도 단계를 거친다
       (튜닝 단계와 최종 학습 단계의 데이터 범위가 다른 것은 표준적인 관례).
"""

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from common import (
    BASE_MODEL_PATH,
    CENTER_COL,
    FEATURE_TABLE_PATH,
    MODEL_DIR,
    TARGET_COL,
    WEEK_COL,
    apply_target_date_boundary_filter,
    compute_metrics,
    get_base_model_excluded_cols,
    load_model_bundle,
    predict_base_hurdle,
    prepare_X,
    save_model_bundle,
    select_feature_cols,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 40
RANDOM_STATE = 42
N_ESTIMATORS_CAP = 1000
EARLY_STOPPING_ROUNDS = 30
BIAS_ALPHA = 1.0  # 목적함수 WAPE + ALPHA*|Bias| 에서 |Bias| 가중치
B_INTERNAL_VAL_WEEKS = 4  # train_center_base_models.py와 동일 관례

TUNING_DIR = MODEL_DIR / "tuning"
STUDY_DB = TUNING_DIR / "optuna_studies.db"
BEST_PARAMS_PATH = TUNING_DIR / "best_hyperparams.json"

FIXED_PARAMS = dict(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, subsample_freq=1)


def suggest_lgb_params(trial: optuna.Trial) -> dict:
    return dict(
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        max_depth=trial.suggest_int("max_depth", 3, 15),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 200),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )


def b_sample_weight(df: pd.DataFrame, b_weight: float) -> np.ndarray:
    return np.where(df[CENTER_COL].to_numpy() == "B", b_weight, 1.0)


def combined_score(a_true, a_pred, a_naive, b_true, b_pred, b_naive, alpha: float = BIAS_ALPHA):
    m_a = compute_metrics(a_true, a_pred, a_naive)
    m_b = compute_metrics(b_true, b_pred, b_naive)
    wape = (m_a["WAPE"] + m_b["WAPE"]) / 2
    bias = (abs(m_a["Bias(%)"]) + abs(m_b["Bias(%)"])) / 2
    return wape + alpha * bias, m_a, m_b


def load_data():
    df = pd.read_parquet(FEATURE_TABLE_PATH)
    train_df = df[df["split"] == "train"].copy()
    train_df = apply_target_date_boundary_filter(train_df, horizon_weeks=1)
    train_df = train_df[train_df[TARGET_COL].notna()]

    b_weeks = sorted(train_df.loc[train_df[CENTER_COL] == "B", WEEK_COL].unique())
    b_val_weeks = b_weeks[-B_INTERNAL_VAL_WEEKS:]
    is_b_internal_val = (train_df[CENTER_COL] == "B") & (train_df[WEEK_COL].isin(b_val_weeks))

    b_val_df = train_df[is_b_internal_val].copy()
    hpo_train_df = train_df[~is_b_internal_val].copy()

    a_val_df = df[df["split"] == "val"].copy()
    a_val_df = a_val_df[a_val_df[TARGET_COL].notna()]

    feature_cols = select_feature_cols(hpo_train_df, get_base_model_excluded_cols(hpo_train_df))
    return hpo_train_df, a_val_df, b_val_df, feature_cols


def _log_trial(stage: str, trial: optuna.Trial, score: float, m_a: dict, m_b: dict, extra: str = ""):
    print(
        f"  [{stage} trial {trial.number}] score={score:.3f}  "
        f"WAPE_A={m_a['WAPE']:.2f}% Bias_A={m_a['Bias(%)']:+.2f}%  "
        f"WAPE_B={m_b['WAPE']:.2f}% Bias_B={m_b['Bias(%)']:+.2f}% {extra}",
        flush=True,
    )


def make_classifier_objective(hpo_train_df, a_val_df, b_val_df, feature_cols, fixed_reg_bundle):
    y_train = (hpo_train_df[TARGET_COL] > 0).astype(int)
    y_a_val = (a_val_df[TARGET_COL] > 0).astype(int)
    n_pos, n_neg = int((y_train == 1).sum()), int((y_train == 0).sum())
    natural_ratio = n_neg / n_pos
    X_train = prepare_X(hpo_train_df, feature_cols)
    X_a_val = prepare_X(a_val_df, feature_cols)
    center_arr = hpo_train_df[CENTER_COL].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = suggest_lgb_params(trial)
        scale_pos_multiplier = trial.suggest_float("scale_pos_weight_multiplier", 0.5, 3.0)
        b_weight = trial.suggest_float("b_center_weight", 1.0, 3.0)
        sw_train = np.where(center_arr == "B", b_weight, 1.0)

        model = lgb.LGBMClassifier(
            n_estimators=N_ESTIMATORS_CAP,
            scale_pos_weight=natural_ratio * scale_pos_multiplier,
            **FIXED_PARAMS,
            **params,
        )
        model.fit(
            X_train, y_train, sample_weight=sw_train,
            eval_set=[(X_a_val, y_a_val)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
        )

        cls_bundle_tmp = {"model": model, "feature_cols": feature_cols}
        y_pred_a = predict_base_hurdle(a_val_df, cls_bundle_tmp, fixed_reg_bundle, mode="soft")
        y_pred_b = predict_base_hurdle(b_val_df, cls_bundle_tmp, fixed_reg_bundle, mode="soft")
        score, m_a, m_b = combined_score(
            a_val_df[TARGET_COL].to_numpy(), y_pred_a, a_val_df["qty"].to_numpy(),
            b_val_df[TARGET_COL].to_numpy(), y_pred_b, b_val_df["qty"].to_numpy(),
        )
        for k, v in {"WAPE_A": m_a["WAPE"], "Bias_A": m_a["Bias(%)"], "WAPE_B": m_b["WAPE"],
                     "Bias_B": m_b["Bias(%)"], "best_iteration": model.best_iteration_,
                     "natural_scale_pos_weight": natural_ratio}.items():
            trial.set_user_attr(k, v)
        _log_trial("Classifier", trial, score, m_a, m_b, extra=f"best_iter={model.best_iteration_}")
        return score

    return objective


def fit_classifier_with_params(hpo_train_df, a_val_df, feature_cols, params: dict):
    y_train = (hpo_train_df[TARGET_COL] > 0).astype(int)
    y_a_val = (a_val_df[TARGET_COL] > 0).astype(int)
    n_pos, n_neg = int((y_train == 1).sum()), int((y_train == 0).sum())
    natural_ratio = n_neg / n_pos
    sw_train = b_sample_weight(hpo_train_df, params["b_center_weight"])
    lgb_params = {k: params[k] for k in
                  ["num_leaves", "max_depth", "learning_rate", "min_child_samples",
                   "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]}
    model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS_CAP,
        scale_pos_weight=natural_ratio * params["scale_pos_weight_multiplier"],
        **FIXED_PARAMS, **lgb_params,
    )
    model.fit(
        prepare_X(hpo_train_df, feature_cols), y_train, sample_weight=sw_train,
        eval_set=[(prepare_X(a_val_df, feature_cols), y_a_val)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def make_regressor_objective(hpo_train_df, a_val_df, b_val_df, feature_cols, fixed_cls_bundle):
    pos_train = hpo_train_df[hpo_train_df[TARGET_COL] > 0]
    pos_a_val = a_val_df[a_val_df[TARGET_COL] > 0]
    X_train = prepare_X(pos_train, feature_cols)
    y_train = np.log1p(pos_train[TARGET_COL])
    X_a_val = prepare_X(pos_a_val, feature_cols)
    y_a_val = np.log1p(pos_a_val[TARGET_COL])
    center_arr = pos_train[CENTER_COL].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = suggest_lgb_params(trial)
        b_weight = trial.suggest_float("b_center_weight", 1.0, 3.0)
        sw_train = np.where(center_arr == "B", b_weight, 1.0)

        model = lgb.LGBMRegressor(
            n_estimators=N_ESTIMATORS_CAP, objective="regression", **FIXED_PARAMS, **params,
        )
        model.fit(
            X_train, y_train, sample_weight=sw_train,
            eval_set=[(X_a_val, y_a_val)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
        )

        reg_bundle_tmp = {"model": model, "feature_cols": feature_cols}
        y_pred_a = predict_base_hurdle(a_val_df, fixed_cls_bundle, reg_bundle_tmp, mode="soft")
        y_pred_b = predict_base_hurdle(b_val_df, fixed_cls_bundle, reg_bundle_tmp, mode="soft")
        score, m_a, m_b = combined_score(
            a_val_df[TARGET_COL].to_numpy(), y_pred_a, a_val_df["qty"].to_numpy(),
            b_val_df[TARGET_COL].to_numpy(), y_pred_b, b_val_df["qty"].to_numpy(),
        )
        for k, v in {"WAPE_A": m_a["WAPE"], "Bias_A": m_a["Bias(%)"], "WAPE_B": m_b["WAPE"],
                     "Bias_B": m_b["Bias(%)"], "best_iteration": model.best_iteration_}.items():
            trial.set_user_attr(k, v)
        _log_trial("Regressor", trial, score, m_a, m_b, extra=f"best_iter={model.best_iteration_}")
        return score

    return objective


def make_tweedie_objective(hpo_train_df, a_val_df, b_val_df, feature_cols):
    X_train = prepare_X(hpo_train_df, feature_cols)
    y_train = hpo_train_df[TARGET_COL]
    X_a_val = prepare_X(a_val_df, feature_cols)
    y_a_val = a_val_df[TARGET_COL]
    X_b_val = prepare_X(b_val_df, feature_cols)
    center_arr = hpo_train_df[CENTER_COL].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = suggest_lgb_params(trial)
        variance_power = trial.suggest_float("tweedie_variance_power", 1.05, 1.95)
        b_weight = trial.suggest_float("b_center_weight", 1.0, 3.0)
        sw_train = np.where(center_arr == "B", b_weight, 1.0)

        model = lgb.LGBMRegressor(
            n_estimators=N_ESTIMATORS_CAP, objective="tweedie",
            tweedie_variance_power=variance_power, **FIXED_PARAMS, **params,
        )
        model.fit(
            X_train, y_train, sample_weight=sw_train,
            eval_set=[(X_a_val, y_a_val)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
        )

        y_pred_a = np.clip(model.predict(X_a_val), 0, None)
        y_pred_b = np.clip(model.predict(X_b_val), 0, None)
        score, m_a, m_b = combined_score(
            a_val_df[TARGET_COL].to_numpy(), y_pred_a, a_val_df["qty"].to_numpy(),
            b_val_df[TARGET_COL].to_numpy(), y_pred_b, b_val_df["qty"].to_numpy(),
        )
        for k, v in {"WAPE_A": m_a["WAPE"], "Bias_A": m_a["Bias(%)"], "WAPE_B": m_b["WAPE"],
                     "Bias_B": m_b["Bias(%)"], "best_iteration": model.best_iteration_}.items():
            trial.set_user_attr(k, v)
        _log_trial("Tweedie", trial, score, m_a, m_b, extra=f"best_iter={model.best_iteration_} vp={variance_power:.3f}")
        return score

    return objective


def _save_results(results: dict):
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  [저장] {BEST_PARAMS_PATH}", flush=True)


def _run_study(name: str, objective_fn, n_trials: int) -> optuna.Study:
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize", study_name=name,
        storage=f"sqlite:///{STUDY_DB}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    n_done = len(study.trials)
    n_remaining = max(0, n_trials - n_done)
    if n_remaining == 0:
        print(f"  [{name}] 이미 {n_done}회 완료된 study 발견 -> 재사용(추가 시행 없음)", flush=True)
    else:
        print(f"  [{name}] 기존 {n_done}회 + 신규 {n_remaining}회 시행 예정", flush=True)
        study.optimize(objective_fn, n_trials=n_remaining, catch=(Exception,))
    return study


def main():
    t0 = time.time()
    print("=" * 80)
    print("[HPO] 데이터 로드 중...", flush=True)
    hpo_train_df, a_val_df, b_val_df, feature_cols = load_data()
    print(
        f"[HPO] hpo_train={len(hpo_train_df):,}행  a_val={len(a_val_df):,}행  "
        f"b_val(internal,마지막{B_INTERNAL_VAL_WEEKS}주)={len(b_val_df):,}행  feature={len(feature_cols)}개",
        flush=True,
    )
    results = {}

    print("=" * 80)
    print(f"[Stage 1/3] Classifier 탐색 (목표 {N_TRIALS}회, 기존 production 회귀기를 고정 기준으로 사용)", flush=True)
    fixed_reg_bundle = load_model_bundle(BASE_MODEL_PATH)
    study_cls = _run_study(
        "classifier",
        make_classifier_objective(hpo_train_df, a_val_df, b_val_df, feature_cols, fixed_reg_bundle),
        N_TRIALS,
    )
    print(f"[Stage 1/3] 완료. best score={study_cls.best_value:.3f}  best_params={study_cls.best_params}", flush=True)
    results["classifier"] = {
        "best_score": study_cls.best_value,
        "best_params": study_cls.best_params,
        "best_attrs": study_cls.best_trial.user_attrs,
        "n_trials": len(study_cls.trials),
    }
    _save_results(results)

    print("  [Stage 1/3] best 파라미터로 classifier 재적합(2단계 고정 기준 생성용)...", flush=True)
    best_cls_model = fit_classifier_with_params(hpo_train_df, a_val_df, feature_cols, study_cls.best_params)
    best_cls_bundle = {"model": best_cls_model, "feature_cols": feature_cols}
    save_model_bundle(TUNING_DIR / "classifier_best_candidate.pkl", best_cls_model, feature_cols,
                       **study_cls.best_params)

    print("=" * 80)
    print(f"[Stage 2/3] Regressor 탐색 (목표 {N_TRIALS}회, Stage1 best classifier 고정)", flush=True)
    study_reg = _run_study(
        "regressor",
        make_regressor_objective(hpo_train_df, a_val_df, b_val_df, feature_cols, best_cls_bundle),
        N_TRIALS,
    )
    print(f"[Stage 2/3] 완료. best score={study_reg.best_value:.3f}  best_params={study_reg.best_params}", flush=True)
    results["regressor"] = {
        "best_score": study_reg.best_value,
        "best_params": study_reg.best_params,
        "best_attrs": study_reg.best_trial.user_attrs,
        "n_trials": len(study_reg.trials),
    }
    _save_results(results)

    print("=" * 80)
    print(f"[Stage 3/3] Tweedie 탐색 (목표 {N_TRIALS}회, 독립 단일모델)", flush=True)
    study_tw = _run_study(
        "tweedie",
        make_tweedie_objective(hpo_train_df, a_val_df, b_val_df, feature_cols),
        N_TRIALS,
    )
    print(f"[Stage 3/3] 완료. best score={study_tw.best_value:.3f}  best_params={study_tw.best_params}", flush=True)
    results["tweedie"] = {
        "best_score": study_tw.best_value,
        "best_params": study_tw.best_params,
        "best_attrs": study_tw.best_trial.user_attrs,
        "n_trials": len(study_tw.trials),
    }
    _save_results(results)

    elapsed_min = (time.time() - t0) / 60
    print("=" * 80)
    print(f"[HPO 전체 완료] 총 소요 {elapsed_min:.1f}분")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
