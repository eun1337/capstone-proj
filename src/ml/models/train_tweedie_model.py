"""
train_tweedie_model.py
Tweedie 단일모델 비교 실험 (Hurdle 2단계 구조의 필요성 검증용)

Hurdle Model(분류기+조건부회귀기 2단계)이 아니라 단일 LightGBM Tweedie 회귀모델
하나로 같은 예측(target_h1)을 했을 때 성능이 어떻게 나오는지 비교한다. 최종 모델
교체를 전제로 하지 않고, "이 복잡도가 실제로 필요한가"를 검증하는 별도 실험이라
production 파일(train_base_model.py, base_model_cls/reg.pkl)과 완전히 분리한다.

핵심 설계:
    1) 전체 train 행(0 포함) 그대로 사용, target_h1 원값(raw qty)을 그대로 타겟으로
       씀 — Tweedie objective가 로그링크를 내부에서 처리하므로 Hurdle 회귀기처럼
       수동으로 log1p/expm1 변환할 필요가 없음(LightGBM tweedie의 predict()는 이미
       원래 스케일로 복원된 값을 반환).
    2) feature: Hurdle 회귀기와 동일하게 get_base_model_excluded_cols 사용(qty_lag2/4는
       B에 구조적으로 없어 A+B 통합 모델에서 계속 제외 — 공정 비교를 위해 동일 feature셋).
    3) tweedie_variance_power(LightGBM 파라미터명, "variance_power" 아님) 1.3/1.5/1.7만
       1차로 스윕. num_leaves=63/learning_rate=0.03은 기존 Hurdle 회귀기에서 이미
       검증된 값을 그대로 시작점으로 사용(재그리드서치 없음 — 이 데이터에서 LightGBM
       조합 1개당 40~75분 걸림을 여러 차례 실측했으므로, 결과가 Hurdle과 근접할 때만
       추가로 좁게 재탐색).
    4) 검증(A val split)에서 실제 WAPE(compute_metrics) 기준으로 최적 variance_power를
       선택 — 내부 tweedie deviance는 참고만 하고, 실제 비즈니스 지표로 채택 여부 결정.
    5) 공정 비교 기준: Tweedie는 연속값 기댓값 하나만 내는 구조라 Hurdle Hard(명시적
       확률 임계값으로 0을 만드는 방식)와는 메커니즘 자체가 다르다. 그래서 Tweedie의
       진짜 비교 상대는 Hurdle Soft(둘 다 "연속적 기댓값" 방식)이고, Hurdle Hard는
       "임계값 컷오프를 쓰면 추가로 얼마나 더 좋아지는지"를 보여주는 별도 참고선으로만
       본다(evaluate_tweedie.py에서 비교표 출력 시 이 구분을 명시).
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import (
    FEATURE_TABLE_PATH,
    MODEL_DIR,
    TARGET_COL,
    apply_target_date_boundary_filter,
    compute_metrics,
    get_base_model_excluded_cols,
    prepare_X,
    save_model_bundle,
    select_feature_cols,
)

TWEEDIE_MODEL_PATH = MODEL_DIR / "tweedie_reg.pkl"

VARIANCE_POWER_GRID = [1.3, 1.5, 1.7]
LEARNING_RATE = 0.03
NUM_LEAVES = 63
N_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 30
FIXED_PARAMS = dict(subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)


def main():
    df = pd.read_parquet(FEATURE_TABLE_PATH)
    train_df = df[df["split"] == "train"].copy()
    print(f"[Tweedie] train 행수(A+B 통합, 필터 전): {len(train_df):,}")
    train_df = apply_target_date_boundary_filter(train_df, horizon_weeks=1)
    train_df = train_df[train_df[TARGET_COL].notna()]

    val_df = df[df["split"] == "val"].copy()
    val_df = val_df[val_df[TARGET_COL].notna()]
    print(f"[Tweedie] val(A만 존재) 행수: {len(val_df):,}")

    feature_cols = select_feature_cols(train_df, get_base_model_excluded_cols(train_df))
    print(f"[Tweedie] feature 개수: {len(feature_cols)}개 (Hurdle 회귀기와 동일 feature셋)")

    X_train = prepare_X(train_df, feature_cols)
    y_train = train_df[TARGET_COL]
    X_val = prepare_X(val_df, feature_cols)
    y_val = val_df[TARGET_COL]
    naive_val = val_df["qty"].to_numpy()

    rows = []
    best = None
    for vp in VARIANCE_POWER_GRID:
        print(f"[Tweedie] variance_power={vp} 학습 시작")
        model = lgb.LGBMRegressor(
            objective="tweedie", tweedie_variance_power=vp,
            n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, num_leaves=NUM_LEAVES,
            verbose=-1, **FIXED_PARAMS,
        )
        model.fit(
            X_train, y_train, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
        )
        pred_val = np.clip(model.predict(X_val), 0, None)
        m = compute_metrics(y_val.to_numpy(), pred_val, naive_val)
        best_iter = model.best_iteration_
        print(f"  best_iter={best_iter}  val WAPE={m['WAPE']:.3f}%  val RMSE={m['RMSE']:.3f}  val Bias={m['Bias(%)']:+.3f}%")
        rows.append({"variance_power": vp, "best_iteration": best_iter, **{k: round(v, 3) for k, v in m.items()}})
        if best is None or m["WAPE"] < best[1]:
            best = (model, m["WAPE"], vp)

    grid_table = pd.DataFrame(rows)
    print()
    print("=" * 80)
    print("[Tweedie variance_power 스윕 결과 (A val 기준)]")
    print(grid_table.to_string(index=False))

    best_model, best_wape, best_vp = best
    print(f"\n[채택] variance_power={best_vp} (val WAPE={best_wape:.3f}%)")

    save_model_bundle(TWEEDIE_MODEL_PATH, best_model, feature_cols, variance_power=best_vp)

    print("[Tweedie] Feature Importance Top 15")
    print(pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15).to_string())


if __name__ == "__main__":
    main()
