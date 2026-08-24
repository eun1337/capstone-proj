"""
compare_baselines.py
Simple Baseline 3종 vs SBC 반영 Hurdle Model 최종 비교

동일한 홀드아웃(A_test / B 53-fold walk-forward)에서 전통적 Simple Baseline과
현재 최종 Hurdle Model(soft / hard th=0.80)을 5대 지표로 나란히 비교해, ML 모델이
가장 단순한 벤치마크 대비 실제로 개선됐는지 검증한다.

Baseline 3종(전부 기존 컬럼으로 바로 계산, 별도 학습 불필요):
    1) Zero-Predictor: 항상 0 예측. (WAPE는 정의상 정확히 100%, Bias는 -100%가
       되는 게 정상 — sum(|true-0|)=sum(true)이므로 sanity check로도 쓸 수 있음)
    2) Naive(t-1): qty(그 행 자기 주 실측)를 target_h1 예측값으로 사용 — 이 프로젝트가
       MASE 계산에 쓰는 naive_pred와 정확히 같은 방법론이라 MASE=1.000이 나와야 정상.
    3) MA-4: qty_rollmean_4_filled(현재 주 포함 트레일링 4주 평균, fallback 채움 버전)
       사용 — 원본 qty_rollmean_4는 SKU 초반 warmup 구간에서 NaN이라, 그대로 쓰면
       baseline마다 평가 대상 행 수(n)가 달라져 "한눈에 대조"가 깨짐. ML 모델 평가와
       동일한 행 전체(target_h1 결측만 제외) 커버리지를 유지하기 위해 filled 버전 사용.
"""

import json

import numpy as np
import pandas as pd

from common import (
    BASE_CLS_MODEL_PATH,
    BASE_MODEL_PATH,
    FEATURE_TABLE_PATH,
    HURDLE_THRESHOLD,
    TARGET_COL,
    WALKFORWARD_FOLDS_PATH,
    compute_metrics,
    load_model_bundle,
    predict_base_hurdle,
)

BASE_MODEL_NAMES = ["Zero-Predictor", "Naive(t-1)", "MA-4", "Hurdle Soft"]


def build_preds(sub_df: pd.DataFrame, cls_bundle: dict, reg_bundle: dict, hurdle_hard_label: str) -> dict:
    naive_ref = sub_df["qty"].to_numpy()
    return {
        "Zero-Predictor": np.zeros(len(sub_df)),
        "Naive(t-1)": naive_ref,
        "MA-4": sub_df["qty_rollmean_4_filled"].to_numpy(),
        "Hurdle Soft": predict_base_hurdle(sub_df, cls_bundle, reg_bundle, mode="soft"),
        hurdle_hard_label: predict_base_hurdle(sub_df, cls_bundle, reg_bundle, mode="hard"),
    }


def evaluate_a_all(df: pd.DataFrame, cls_bundle, reg_bundle, model_names: list[str], hurdle_hard_label: str) -> pd.DataFrame:
    a_test = df[(df["center_id"] == "A") & (df["split"] == "test")].copy()
    a_test = a_test[a_test[TARGET_COL].notna()]
    y_true = a_test[TARGET_COL].to_numpy()
    naive_ref = a_test["qty"].to_numpy()

    preds = build_preds(a_test, cls_bundle, reg_bundle, hurdle_hard_label)
    rows = []
    for name in model_names:
        m = compute_metrics(y_true, preds[name], naive_ref)
        rows.append({"model": name, "n": len(a_test), **{k: round(v, 3) for k, v in m.items()}})
    return pd.DataFrame(rows)


def evaluate_b_all(df: pd.DataFrame, cls_bundle, reg_bundle, model_names: list[str], hurdle_hard_label: str) -> pd.DataFrame:
    with open(WALKFORWARD_FOLDS_PATH, encoding="utf-8") as f:
        folds = json.load(f)
    b_pool = df[(df["center_id"] == "B") & (df["split"] == "pool")].copy()

    accum = {name: {"true": [], "pred": [], "naive": []} for name in model_names}

    for train_start, train_end, val_start, val_end in folds:
        mask = (b_pool["week_st"] >= val_start) & (b_pool["week_st"] <= val_end)
        fold_df = b_pool[mask]
        fold_df = fold_df[fold_df[TARGET_COL].notna()]
        if len(fold_df) == 0:
            continue
        y_true = fold_df[TARGET_COL].to_numpy()
        naive_ref = fold_df["qty"].to_numpy()
        preds = build_preds(fold_df, cls_bundle, reg_bundle, hurdle_hard_label)
        for name in model_names:
            accum[name]["true"].append(y_true)
            accum[name]["pred"].append(preds[name])
            accum[name]["naive"].append(naive_ref)

    rows = []
    for name in model_names:
        y_true_all = np.concatenate(accum[name]["true"])
        y_pred_all = np.concatenate(accum[name]["pred"])
        naive_all = np.concatenate(accum[name]["naive"])
        m = compute_metrics(y_true_all, y_pred_all, naive_all)
        rows.append({"model": name, "n": len(y_true_all), **{k: round(v, 3) for k, v in m.items()}})
    return pd.DataFrame(rows)


def main():
    df = pd.read_parquet(FEATURE_TABLE_PATH)
    cls_bundle = load_model_bundle(BASE_CLS_MODEL_PATH)
    reg_bundle = load_model_bundle(BASE_MODEL_PATH)
    # predict_base_hurdle(mode="hard")이 실제로 쓰는 threshold 결정 로직과 동일하게 맞춰야
    # 라벨과 계산값이 항상 일치한다. threshold_by_center가 있으면(센터별 threshold 적용
    # 중) 라벨에 A/B 값을 각각 명시 — 없는 옛 bundle이면 스칼라 threshold(그마저 없으면
    # HURDLE_THRESHOLD)로 폴백해 하나의 값만 표시.
    threshold_by_center = cls_bundle.get("threshold_by_center")
    if threshold_by_center:
        hurdle_hard_label = f"Hurdle Hard(th_A={threshold_by_center['A']:.2f},th_B={threshold_by_center['B']:.2f})"
        print(f"[Hurdle Hard threshold] {threshold_by_center}")
    else:
        threshold = cls_bundle.get("threshold", HURDLE_THRESHOLD)
        hurdle_hard_label = f"Hurdle Hard(th={threshold:.2f})"
        print(f"[Hurdle Hard threshold] {threshold}")
    model_names = BASE_MODEL_NAMES + [hurdle_hard_label]

    print("=" * 80)
    print("[A센터] Simple Baseline 3종 vs SBC Hurdle Model (A_test 홀드아웃)")
    report_a = evaluate_a_all(df, cls_bundle, reg_bundle, model_names, hurdle_hard_label)
    print(report_a.to_string(index=False))

    print()
    print("=" * 80)
    print("[B센터] Simple Baseline 3종 vs SBC Hurdle Model (53-fold walk-forward 합산)")
    report_b = evaluate_b_all(df, cls_bundle, reg_bundle, model_names, hurdle_hard_label)
    print(report_b.to_string(index=False))

    print()
    print("=" * 80)
    print(f"[WAPE 개선율(%) vs Simple Baseline] ({hurdle_hard_label} 기준, 음수=개선)")
    for label, report in [("A", report_a), ("B", report_b)]:
        ml_row = report[report["model"] == hurdle_hard_label].iloc[0]
        for base_name in ["Zero-Predictor", "Naive(t-1)", "MA-4"]:
            base_row = report[report["model"] == base_name].iloc[0]
            wape_improve = (ml_row["WAPE"] - base_row["WAPE"]) / base_row["WAPE"] * 100
            print(f"  [{label}] vs {base_name}: WAPE {base_row['WAPE']:.2f}% -> {ml_row['WAPE']:.2f}% ({wape_improve:+.1f}%)")
        print(f"  [{label}] Hurdle Hard MASE = {ml_row['MASE']:.3f} (<1.0이면 Naive(t-1)보다 우수)")


if __name__ == "__main__":
    main()
