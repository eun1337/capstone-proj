"""
train_base_model.py
Hurdle Base Model 고도화 & 튜닝 (분류기 + 조건부회귀기, 잔차 스테이지 없음)

Ablation Test에서 잔차(Residual) 보정이 in-sample/OOF 두 버전 모두 A_test 성능을
악화시킴을 확인해 잔차 스테이지를 완전히 제거했다(common.py 모듈 docstring 참고).
대신 Hurdle Model(1단계 분류 + 2단계 조건부회귀) 자체를 튜닝해 정확도를 끌어올린다.

핵심 설계:
    1) target_date<=train_end 경계 필터(공통 팀 컨벤션, common.py)를 train_df에만
       적용한다 — val_df(split=='val', A만 존재)는 train_end 기준 필터를 적용하면
       전량 탈락하므로(모든 val 행의 target_date가 train_end 이후) 적용하지 않는다.
    2) scale_pos_weight = train_df 기준 (수요==0 행수)/(수요>0 행수). 수요 0 비율이
       73%대라 분류기가 다수 클래스(0)로 쏠리지 않도록 양성 클래스 가중치를 보정한다.
    3) 분류기/회귀기 각각 learning_rate∈{0.03,0.05} x num_leaves∈{31,63} 4조합을
       n_estimators=1000 + early_stopping_rounds=30으로 학습(eval_set=A val split)해
       검증 성능(분류: logloss, 회귀: RMSE) 최저 조합을 채택한다. val split이 A만
       존재하므로 이 튜닝은 A 분포 기준 캘리브레이션이라는 한계가 있으나, Base Model이
       A+B 공유 단일 모델이라 다른 대안이 없다(팀 컨벤션 그대로 수용).
    4) 분류 threshold(hard 모드 P(수요>0)>=threshold 판정 기준)를 0.20~0.50(0.05
       step)에서 탐색해 A val split 기준 WAPE가 최소가 되는 값을 채택한다(수요가
       희소할 때 threshold를 낮추면 과소추정이 줄어든다는 요청 취지 반영). 동률이면
       |Bias|가 더 작은 쪽을 선택. 채택된 threshold는 cls_bundle["threshold"]로 저장돼
       common.predict_base_hurdle(mode="hard")이 자동으로 사용한다.
    5) feature: 두 단계 모두 get_base_model_excluded_cols(df) + 'split' 수동 제외를
       동일하게 사용(qty_lag2/4는 B에 구조적으로 없어 A+B 통합 모델에서 계속 제외).
    6) 하이퍼파라미터/가중치: tune_hyperparameters.py(Optuna, WAPE+α|Bias| 복합 목적함수로
       분류기/회귀기/Tweedie를 A val + B 내부val 기준 탐색)가 만든
       data/ml/models/tuning/best_hyperparams.json이 있으면 그 값(num_leaves 등 8종 +
       classifier의 scale_pos_weight 배수 + B센터 sample_weight 배수)을 그대로 써서
       단일 설정으로 학습한다(그리드서치 생략). 파일이 없으면 기존 GRID_LEARNING_RATE/
       GRID_NUM_LEAVES 좁은 그리드로 폴백 — 이 스크립트만으로도 항상 동작해야 하므로
       tuning 산출물은 선택적 의존성으로 둔다. B센터 sample_weight는 튜닝 단계와 달리
       "B 마지막 4주 제외" 없이 전체 train 데이터에 적용한다(최종 프로덕션 학습이므로
       가용 데이터를 전부 씀 — tune_hyperparameters.py 모듈 docstring 6번 참고).
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import (
    BASE_CLS_MODEL_PATH,
    BASE_MODEL_PATH,
    CENTER_COL,
    FEATURE_TABLE_PATH,
    MODEL_DIR,
    TARGET_COL,
    apply_target_date_boundary_filter,
    compute_metrics,
    get_base_model_excluded_cols,
    predict_base_hurdle,
    prepare_X,
    save_model_bundle,
    select_feature_cols,
)

BEST_HYPERPARAMS_PATH = MODEL_DIR / "tuning" / "best_hyperparams.json"


def load_tuned_params(key: str) -> dict | None:
    """tune_hyperparameters.py 산출물이 있으면 해당 모델(key='classifier'|'regressor')의
    best_params를 반환, 없으면 None(폴백 신호)."""
    if not BEST_HYPERPARAMS_PATH.exists():
        return None
    with open(BEST_HYPERPARAMS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get(key, {}).get("best_params")


def b_sample_weight(df: pd.DataFrame, b_weight: float) -> np.ndarray:
    return np.where(df[CENTER_COL].to_numpy() == "B", b_weight, 1.0)

# SBC(ADI/CV² expanding) feature 추가 후 재학습 - 기존 그리드서치(4x4, 이후 2x1로 축소)에서
# lr=0.03/num_leaves=63이 이미 classifier/regressor 모두 최적으로 확인됐고, 이번 재학습은
# "새 feature가 실제로 도움되는지" ablation이 목적이라 그 결과를 재검증할 필요가 없어
# 단일 설정으로 고정(학습 1회씩만, 그리드서치 재반복 없음 — 시간 절약).
GRID_LEARNING_RATE = [0.03]
GRID_NUM_LEAVES = [63]
N_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 30
THRESHOLD_GRID = np.round(np.arange(0.20, 0.51, 0.05), 2)
RANDOM_STATE = 42

FIXED_PARAMS = dict(subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1)


def grid_search_classifier(X_train, y_train, X_val, y_val, scale_pos_weight: float):
    rows = []
    best = None
    for lr in GRID_LEARNING_RATE:
        for leaves in GRID_NUM_LEAVES:
            model = lgb.LGBMClassifier(
                n_estimators=N_ESTIMATORS, learning_rate=lr, num_leaves=leaves,
                scale_pos_weight=scale_pos_weight, verbose=-1, **FIXED_PARAMS,
            )
            model.fit(
                X_train, y_train, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
            )
            val_logloss = model.best_score_["valid_0"]["binary_logloss"]
            best_iter = model.best_iteration_
            rows.append({"learning_rate": lr, "num_leaves": leaves, "best_iteration": best_iter, "val_logloss": val_logloss})
            print(f"  [분류기 grid] lr={lr} leaves={leaves} best_iter={best_iter} val_logloss={val_logloss:.5f}")
            if best is None or val_logloss < best[1]:
                best = (model, val_logloss, lr, leaves)
    grid_table = pd.DataFrame(rows)
    return best[0], best[2], best[3], grid_table


def grid_search_regressor(X_train, y_train, X_val, y_val):
    rows = []
    best = None
    for lr in GRID_LEARNING_RATE:
        for leaves in GRID_NUM_LEAVES:
            model = lgb.LGBMRegressor(
                n_estimators=N_ESTIMATORS, learning_rate=lr, num_leaves=leaves,
                objective="regression", verbose=-1, **FIXED_PARAMS,
            )
            model.fit(
                X_train, y_train, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
            )
            val_rmse = model.best_score_["valid_0"]["l2"] ** 0.5
            best_iter = model.best_iteration_
            rows.append({"learning_rate": lr, "num_leaves": leaves, "best_iteration": best_iter, "val_rmse_log": val_rmse})
            print(f"  [회귀기 grid] lr={lr} leaves={leaves} best_iter={best_iter} val_rmse(log)={val_rmse:.5f}")
            if best is None or val_rmse < best[1]:
                best = (model, val_rmse, lr, leaves)
    grid_table = pd.DataFrame(rows)
    return best[0], best[2], best[3], grid_table


def search_threshold(val_df: pd.DataFrame, cls_bundle: dict, reg_bundle: dict) -> tuple:
    y_true = val_df[TARGET_COL].to_numpy()
    naive_pred = val_df["qty"].to_numpy()
    rows = []
    best = None
    for th in THRESHOLD_GRID:
        y_pred = predict_base_hurdle(val_df, cls_bundle, reg_bundle, mode="hard", threshold=th)
        m = compute_metrics(y_true, y_pred, naive_pred)
        rows.append({"threshold": th, **{k: round(v, 3) for k, v in m.items()}})
        print(f"  [threshold 탐색] th={th:.2f} WAPE={m['WAPE']:.3f}% Bias={m['Bias(%)']:+.3f}%")
        key = (m["WAPE"], abs(m["Bias(%)"]))
        if best is None or key < best[0]:
            best = (key, th)
    return best[1], pd.DataFrame(rows)


def main():
    df = pd.read_parquet(FEATURE_TABLE_PATH)
    train_df = df[df["split"] == "train"].copy()
    print(f"[Base Model] train 행수(A+B 통합, 필터 전): {len(train_df):,}")
    train_df = apply_target_date_boundary_filter(train_df, horizon_weeks=1)

    valid = train_df[TARGET_COL].notna()
    train_df = train_df[valid]
    print(f"[Base Model] target NaN 제외 후: {len(train_df):,}행")

    val_df = df[df["split"] == "val"].copy()
    val_df = val_df[val_df[TARGET_COL].notna()]
    print(f"[Base Model] val(A만 존재) 행수: {len(val_df):,}")

    excluded = get_base_model_excluded_cols(train_df)
    feature_cols = select_feature_cols(train_df, excluded)
    lag24_leak = [c for c in feature_cols if c.startswith(("qty_lag2", "qty_lag4"))]
    print(
        f"[Base Model] feature 개수: {len(feature_cols)}개 "
        f"(qty_lag2/4 계열 잔존: {lag24_leak if lag24_leak else '없음(정상 제외됨)'})"
    )

    tuned_cls_params = load_tuned_params("classifier")
    tuned_reg_params = load_tuned_params("regressor")

    # ---- 1단계: 분류기 ----
    y_cls_train = (train_df[TARGET_COL] > 0).astype(int)
    y_cls_val = (val_df[TARGET_COL] > 0).astype(int)
    n_pos, n_neg = int((y_cls_train == 1).sum()), int((y_cls_train == 0).sum())
    natural_scale_pos_weight = n_neg / n_pos
    print(f"[Base Model 1단계-분류] 양성비율={y_cls_train.mean():.1%} natural_scale_pos_weight={natural_scale_pos_weight:.3f}")

    X_cls_train = prepare_X(train_df, feature_cols)
    X_cls_val = prepare_X(val_df, feature_cols)

    if tuned_cls_params is not None:
        print(f"[Base Model 1단계-분류] tune_hyperparameters.py 결과 사용: {tuned_cls_params}")
        scale_pos_weight = natural_scale_pos_weight * tuned_cls_params["scale_pos_weight_multiplier"]
        sw_cls_train = b_sample_weight(train_df, tuned_cls_params["b_center_weight"])
        lgb_params = {k: tuned_cls_params[k] for k in
                      ["num_leaves", "max_depth", "learning_rate", "min_child_samples",
                       "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]}
        cls_model = lgb.LGBMClassifier(
            n_estimators=N_ESTIMATORS, scale_pos_weight=scale_pos_weight, subsample_freq=1,
            verbose=-1, random_state=RANDOM_STATE, n_jobs=-1, **lgb_params,
        )
        cls_model.fit(
            X_cls_train, y_cls_train, sample_weight=sw_cls_train, eval_set=[(X_cls_val, y_cls_val)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
        )
        best_cls_lr, best_cls_leaves = tuned_cls_params["learning_rate"], tuned_cls_params["num_leaves"]
        cls_grid_table = pd.DataFrame([{**tuned_cls_params, "scale_pos_weight": scale_pos_weight,
                                         "best_iteration": cls_model.best_iteration_}])
    else:
        print("[Base Model 1단계-분류] best_hyperparams.json 없음 -> 기존 좁은 그리드로 폴백")
        scale_pos_weight = natural_scale_pos_weight
        cls_model, best_cls_lr, best_cls_leaves, cls_grid_table = grid_search_classifier(
            X_cls_train, y_cls_train, X_cls_val, y_cls_val, natural_scale_pos_weight
        )
    print(f"[Base Model 1단계-분류] 채택: lr={best_cls_lr} num_leaves={best_cls_leaves}")

    # ---- 2단계: 조건부회귀기 (target_h1>0만) ----
    pos_train = train_df[train_df[TARGET_COL] > 0]
    pos_val = val_df[val_df[TARGET_COL] > 0]
    print(f"[Base Model 2단계-조건부회귀] train 대상행: {len(pos_train):,} / val 대상행: {len(pos_val):,}")
    X_reg_train = prepare_X(pos_train, feature_cols)
    y_reg_train = np.log1p(pos_train[TARGET_COL])
    X_reg_val = prepare_X(pos_val, feature_cols)
    y_reg_val = np.log1p(pos_val[TARGET_COL])

    if tuned_reg_params is not None:
        print(f"[Base Model 2단계-조건부회귀] tune_hyperparameters.py 결과 사용: {tuned_reg_params}")
        sw_reg_train = b_sample_weight(pos_train, tuned_reg_params["b_center_weight"])
        lgb_params = {k: tuned_reg_params[k] for k in
                      ["num_leaves", "max_depth", "learning_rate", "min_child_samples",
                       "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]}
        reg_model = lgb.LGBMRegressor(
            n_estimators=N_ESTIMATORS, objective="regression", subsample_freq=1,
            verbose=-1, random_state=RANDOM_STATE, n_jobs=-1, **lgb_params,
        )
        reg_model.fit(
            X_reg_train, y_reg_train, sample_weight=sw_reg_train, eval_set=[(X_reg_val, y_reg_val)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
        )
        best_reg_lr, best_reg_leaves = tuned_reg_params["learning_rate"], tuned_reg_params["num_leaves"]
        reg_grid_table = pd.DataFrame([{**tuned_reg_params, "best_iteration": reg_model.best_iteration_}])
    else:
        print("[Base Model 2단계-조건부회귀] best_hyperparams.json 없음 -> 기존 좁은 그리드로 폴백")
        reg_model, best_reg_lr, best_reg_leaves, reg_grid_table = grid_search_regressor(
            X_reg_train, y_reg_train, X_reg_val, y_reg_val
        )
    print(f"[Base Model 2단계-조건부회귀] 채택: lr={best_reg_lr} num_leaves={best_reg_leaves}")

    # ---- Threshold 탐색 (val 전체, hard 모드) ----
    cls_bundle_tmp = {"model": cls_model, "feature_cols": feature_cols}
    reg_bundle_tmp = {"model": reg_model, "feature_cols": feature_cols}
    best_threshold, threshold_table = search_threshold(val_df, cls_bundle_tmp, reg_bundle_tmp)
    print(f"[Threshold 탐색] 채택된 threshold(WAPE 최소): {best_threshold:.2f}")

    save_model_bundle(
        BASE_CLS_MODEL_PATH, cls_model, feature_cols,
        threshold=float(best_threshold), scale_pos_weight=scale_pos_weight,
        learning_rate=best_cls_lr, num_leaves=best_cls_leaves,
        tuned_params=tuned_cls_params,
    )
    save_model_bundle(
        BASE_MODEL_PATH, reg_model, feature_cols,
        learning_rate=best_reg_lr, num_leaves=best_reg_leaves,
        tuned_params=tuned_reg_params,
    )

    print()
    print("=" * 80)
    print("[분류기 grid 결과]")
    print(cls_grid_table.to_string(index=False))
    print()
    print("[회귀기 grid 결과]")
    print(reg_grid_table.to_string(index=False))
    print()
    print("[Threshold 탐색 결과]")
    print(threshold_table.to_string(index=False))

    print()
    print("[Base Model 1단계-분류] Feature Importance Top 10")
    print(pd.Series(cls_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10).to_string())
    print("[Base Model 2단계-조건부회귀] Feature Importance Top 10")
    print(pd.Series(reg_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
