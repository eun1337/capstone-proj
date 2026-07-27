"""
evaluate_pipeline.py
최종 추론 & Rolling-Origin 검증 (Hurdle Base Model 단독)

Hurdle Base Model(분류+조건부회귀, train_base_model.py에서 튜닝됨) 하나로 A/B 센터를
동일하게 추론하고 분리 평가한다. A센터 전용 잔차(Residual) 보정 스테이지는 제거됨
(Ablation Test 결과 in-sample/OOF 두 버전 모두 Base 단독보다 성능이 나빠 폐기 —
common.py 모듈 docstring 참고).

핵심 설계:
    1) 추론 로직: mode="soft"(기본, 파이프라인 공식 기준점) = P(수요>0) x E[수량|수요>0].
       mode="hard"(`python evaluate_pipeline.py hard`)는 cls_bundle["threshold"] 이상일
       때만 조건부회귀값을 채택 — hard 모드 threshold 탐색에서 WAPE와 Bias가 서로 다른
       지점을 최적으로 가리키는 트레이드오프가 확인되어(threshold 탐색 로그 참고),
       하나의 지표로 단일 최적값을 정하기 어려움. 그래서 soft(확률 기댓값, 임의
       하이퍼파라미터인 threshold가 없어 편향 없이 연속적인 값을 내는 방식)를 공식
       파이프라인 기준점으로 확정함.
    2) A 평가: A_test split(2024-07~12, 최종 홀드아웃)에서 target_h1 기준.
    3) B 평가: B_walkforward_folds.json의 53-fold(각 fold=1주짜리 val 구간)를 그대로
       순회하며 Pool(2024 전체) 구간을 주 단위로 채점한다. "walk-forward"는 매 fold마다
       모델을 재학습한다는 뜻이 아니다 — lag/rolling 등 feature는 이미 Day2에서 실제
       과거 실측치로 정확히 계산되어 있으므로(미래 참조 없음), 이미 학습된 단일 Base
       Model을 각 fold의 검증 주(val_start~val_end)에 그대로 적용해 fold별 오차를
       측정하고 전체 53-fold를 합쳐 최종 성능을 낸다.
    4) target_h1이 NaN인 행(데이터셋 마지막 주의 우측절단, "정답 없음")은 평가에서
       자연스럽게 제외한다.
    5) 평가 지표 5종(RMSE/MAE/WAPE/MASE/Bias%) 정의는 common.compute_metrics 참고.
"""

import json
import sys

import numpy as np
import pandas as pd

from common import (
    BASE_CLS_MODEL_PATH,
    BASE_MODEL_PATH,
    FEATURE_TABLE_PATH,
    TARGET_COL,
    WALKFORWARD_FOLDS_PATH,
    compute_metrics,
    load_model_bundle,
    predict_base_hurdle,
)


def load_bundles():
    cls_bundle = load_model_bundle(BASE_CLS_MODEL_PATH)
    reg_bundle = load_model_bundle(BASE_MODEL_PATH)
    return cls_bundle, reg_bundle


def predict_base(df: pd.DataFrame, cls_bundle: dict, reg_bundle: dict, mode: str = "soft") -> np.ndarray:
    """Hurdle 추론. mode="soft"(기본, 파이프라인 기준점): P(수요>0) x E[수량|수요>0].
    mode="hard": P(수요>0)>=threshold(cls_bundle["threshold"])면 E[수량|수요>0], 아니면 0."""
    return predict_base_hurdle(df, cls_bundle, reg_bundle, mode=mode)


def evaluate_a(df: pd.DataFrame, cls_bundle, reg_bundle, mode: str = "soft") -> dict:
    print("=" * 80)
    print(f"[A센터 최종 평가] A_test(2024-07~12, 최종 홀드아웃) / Base(Hurdle, mode={mode})")
    a_test = df[(df["center_id"] == "A") & (df["split"] == "test")].copy()
    valid = a_test[TARGET_COL].notna()
    print(f"  A_test 행수: {len(a_test):,} -> target NaN(우측절단) 제외 후 {int(valid.sum()):,}행")
    a_test = a_test[valid]

    y_true = a_test[TARGET_COL].to_numpy()
    y_pred = predict_base(a_test, cls_bundle, reg_bundle, mode=mode)
    naive_pred = a_test["qty"].to_numpy()
    m = compute_metrics(y_true, y_pred, naive_pred)
    print(
        f"  RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  WAPE={m['WAPE']:.2f}%  "
        f"MASE={m['MASE']:.3f}  Bias={m['Bias(%)']:+.2f}%  (n={len(a_test):,})"
    )
    return {"center": "A", "n": len(a_test), **{k: round(v, 3) for k, v in m.items()}}


def evaluate_b_walkforward(df: pd.DataFrame, cls_bundle, reg_bundle, mode: str = "soft") -> tuple:
    print("=" * 80)
    print(f"[B센터 최종 평가] B_walkforward_folds.json(53-fold) 기반 Pool Rolling-Origin 검증 / Base(Hurdle, mode={mode})")
    with open(WALKFORWARD_FOLDS_PATH, encoding="utf-8") as f:
        folds = json.load(f)
    print(f"  fold 개수: {len(folds)}")

    b_pool = df[(df["center_id"] == "B") & (df["split"] == "pool")].copy()

    fold_rows = []
    all_true, all_pred, all_naive = [], [], []
    for i, (train_start, train_end, val_start, val_end) in enumerate(folds):
        mask = (b_pool["week_st"] >= val_start) & (b_pool["week_st"] <= val_end)
        fold_df = b_pool[mask]
        valid = fold_df[TARGET_COL].notna()
        fold_df = fold_df[valid]
        if len(fold_df) == 0:
            fold_rows.append({
                "fold": i + 1, "val_start": val_start, "n": 0,
                "RMSE": np.nan, "MAE": np.nan, "WAPE": np.nan, "MASE": np.nan, "Bias(%)": np.nan,
            })
            continue
        y_true = fold_df[TARGET_COL].to_numpy()
        y_pred = predict_base(fold_df, cls_bundle, reg_bundle, mode=mode)
        naive_pred = fold_df["qty"].to_numpy()
        m = compute_metrics(y_true, y_pred, naive_pred)
        fold_rows.append({"fold": i + 1, "val_start": val_start, "n": len(fold_df), **{k: round(v, 3) for k, v in m.items()}})
        all_true.append(y_true)
        all_pred.append(y_pred)
        all_naive.append(naive_pred)

    fold_table = pd.DataFrame(fold_rows)
    print("  fold별 성능 (처음 5개 / 마지막 5개)")
    print(fold_table.head(5).to_string(index=False))
    print("  ...")
    print(fold_table.tail(5).to_string(index=False))

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    naive_all = np.concatenate(all_naive)
    m = compute_metrics(y_true_all, y_pred_all, naive_all)
    print(
        f"  [B 전체 53-fold 합산] RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  WAPE={m['WAPE']:.2f}%  "
        f"MASE={m['MASE']:.3f}  Bias={m['Bias(%)']:+.2f}%  (n={len(y_true_all):,})"
    )
    return {"center": "B", "n": len(y_true_all), **{k: round(v, 3) for k, v in m.items()}}, fold_table


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "soft"
    df = pd.read_parquet(FEATURE_TABLE_PATH)
    cls_bundle, reg_bundle = load_bundles()
    print(f"[평가] mode={mode}" + (f" (threshold={cls_bundle.get('threshold')})" if mode == "hard" else " (파이프라인 기준점, Prob x Quantity)"))

    result_a = evaluate_a(df, cls_bundle, reg_bundle, mode=mode)
    result_b, fold_table = evaluate_b_walkforward(df, cls_bundle, reg_bundle, mode=mode)

    print()
    print("=" * 80)
    print(f"[최종 성능 리포트] (A/B 분리, mode={mode}, RMSE/MAE/WAPE/MASE/Bias(%))")
    report = pd.DataFrame([result_a, result_b])
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
