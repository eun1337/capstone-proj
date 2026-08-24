"""
train_center_base_models.py
센터별 분리 학습 Hurdle Soft 모델 (A전용 / B전용)

통합(A+B) Hurdle Base Model은 qty_lag2/4가 B에 구조적으로 없어(B_LAG_WEEKS=[1]) 항상
get_base_model_excluded_cols로 제외해야 했다. 센터별로 따로 학습하면 그 제약이
사라진다 — A전용 모델은 A_LAG_WEEKS=[1,2,4]로 실제 값이 있는 qty_lag2/4를 그대로
활용하도록 get_excluded_cols(전체 feature set, qty_lag2/4 포함)를 쓴다.

핵심 설계:
    1) target_date<=train_end 경계 필터를 두 모델 모두에 적용(공통 팀 컨벤션).
    2) 하이퍼파라미터: 통합 모델 튜닝에서 이미 확인된 lr=0.03/num_leaves=63을 그대로
       재사용한다(센터별 재그리드서치는 하지 않음 — 이전 세션에서 4x4 그리드가
       3~5시간+ 걸림을 실측 확인해, 센터별로 또 반복하면 시간이 감당 안 됨. 검증된
       값 재사용이 현실적 절충안). n_estimators=1000 + early stopping(30)으로 각
       센터 데이터에 맞는 트리 개수는 자동으로 맞춘다.
    3) A: A의 val split(2024-01~06, 기존 존재)으로 early stopping.
    4) B: 별도 val split이 없음(train/pool만 존재) — pool은 최종 walk-forward 평가용
       이라 조기종료에 쓰면 안 됨(정보 누수). 대신 B train(2023-07~12, 26주) 안에서
       시간 순서를 지켜 마지막 4주를 내부 검증용으로 떼어 early stopping에 쓴다.
    5) soft 모드(Prob x Quantity) 비교가 이번 목적이라 threshold는 저장하지 않는다
       (hard 모드 센터별 threshold 탐색은 후속 작업으로 남김).
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import (
    A_BASE_CLS_PATH,
    A_BASE_REG_PATH,
    B_BASE_CLS_PATH,
    B_BASE_REG_PATH,
    FEATURE_TABLE_PATH,
    TARGET_COL,
    apply_target_date_boundary_filter,
    get_excluded_cols,
    prepare_X,
    save_model_bundle,
    select_feature_cols,
)

LEARNING_RATE = 0.03
NUM_LEAVES = 63
N_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 30
B_INTERNAL_VAL_WEEKS = 4
FIXED_PARAMS = dict(subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)


def train_one_center(center: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
                      feature_cols: list[str], cls_path, reg_path):
    y_cls_train = (train_df[TARGET_COL] > 0).astype(int)
    y_cls_val = (val_df[TARGET_COL] > 0).astype(int)
    n_pos, n_neg = int((y_cls_train == 1).sum()), int((y_cls_train == 0).sum())
    scale_pos_weight = n_neg / n_pos
    print(f"[{center}] 양성비율={y_cls_train.mean():.1%} scale_pos_weight={scale_pos_weight:.3f}")

    X_cls_train = prepare_X(train_df, feature_cols)
    X_cls_val = prepare_X(val_df, feature_cols)
    cls_model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, num_leaves=NUM_LEAVES,
        scale_pos_weight=scale_pos_weight, verbose=-1, **FIXED_PARAMS,
    )
    cls_model.fit(
        X_cls_train, y_cls_train, eval_set=[(X_cls_val, y_cls_val)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"[{center}] 분류기 학습 완료 best_iter={cls_model.best_iteration_}")
    save_model_bundle(cls_path, cls_model, feature_cols, learning_rate=LEARNING_RATE,
                       num_leaves=NUM_LEAVES, scale_pos_weight=scale_pos_weight)

    pos_train = train_df[train_df[TARGET_COL] > 0]
    pos_val = val_df[val_df[TARGET_COL] > 0]
    print(f"[{center}] 회귀기 대상행 train={len(pos_train):,} val={len(pos_val):,}")
    X_reg_train = prepare_X(pos_train, feature_cols)
    y_reg_train = np.log1p(pos_train[TARGET_COL])
    X_reg_val = prepare_X(pos_val, feature_cols)
    y_reg_val = np.log1p(pos_val[TARGET_COL])
    reg_model = lgb.LGBMRegressor(
        n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, num_leaves=NUM_LEAVES,
        objective="regression", verbose=-1, **FIXED_PARAMS,
    )
    reg_model.fit(
        X_reg_train, y_reg_train, eval_set=[(X_reg_val, y_reg_val)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"[{center}] 회귀기 학습 완료 best_iter={reg_model.best_iteration_}")
    save_model_bundle(reg_path, reg_model, feature_cols, learning_rate=LEARNING_RATE, num_leaves=NUM_LEAVES)


def main():
    df = pd.read_parquet(FEATURE_TABLE_PATH)

    print("=" * 80)
    print("[A 전용 모델]")
    a_train = df[(df["center_id"] == "A") & (df["split"] == "train")].copy()
    print(f"[A] train 행수(필터 전): {len(a_train):,}")
    a_train = apply_target_date_boundary_filter(a_train, horizon_weeks=1)
    a_train = a_train[a_train[TARGET_COL].notna()]
    a_val = df[(df["center_id"] == "A") & (df["split"] == "val")].copy()
    a_val = a_val[a_val[TARGET_COL].notna()]
    print(f"[A] val 행수: {len(a_val):,}")

    a_feature_cols = select_feature_cols(a_train, get_excluded_cols(a_train))
    lag24_present = [c for c in a_feature_cols if c.startswith(("qty_lag2", "qty_lag4"))]
    print(f"[A] feature 개수: {len(a_feature_cols)}개 (qty_lag2/4 포함: {lag24_present})")
    train_one_center("A", a_train, a_val, a_feature_cols, A_BASE_CLS_PATH, A_BASE_REG_PATH)

    print("=" * 80)
    print("[B 전용 모델]")
    b_train_full = df[(df["center_id"] == "B") & (df["split"] == "train")].copy()
    print(f"[B] train 행수(필터 전): {len(b_train_full):,}")
    b_train_full = apply_target_date_boundary_filter(b_train_full, horizon_weeks=1)
    b_train_full = b_train_full[b_train_full[TARGET_COL].notna()]

    b_weeks = sorted(b_train_full["week_st"].unique())
    val_weeks_b = b_weeks[-B_INTERNAL_VAL_WEEKS:]
    b_val = b_train_full[b_train_full["week_st"].isin(val_weeks_b)]
    b_train = b_train_full[~b_train_full["week_st"].isin(val_weeks_b)]
    print(
        f"[B] 내부 분할(pool은 최종평가용이라 조기종료에 사용 불가): "
        f"train {len(b_train):,}행({len(b_weeks) - B_INTERNAL_VAL_WEEKS}주) / "
        f"val {len(b_val):,}행({B_INTERNAL_VAL_WEEKS}주, {pd.Timestamp(val_weeks_b[0]).date()}~{pd.Timestamp(val_weeks_b[-1]).date()})"
    )

    b_feature_cols = select_feature_cols(b_train, get_excluded_cols(b_train))
    print(f"[B] feature 개수: {len(b_feature_cols)}개")
    train_one_center("B", b_train, b_val, b_feature_cols, B_BASE_CLS_PATH, B_BASE_REG_PATH)


if __name__ == "__main__":
    main()
