"""
search_threshold_sbc_groups.py
SBC(ADI축) 2그룹별 threshold 분리 탐색 (통합 모델, 협업자 제안 조정판)

원 제안은 "A/B Validation을 물동량 기준으로 합쳐서" 그룹당 1개 threshold(총 2개)를
찾자는 것이었으나, 다음 두 가지 문제로 조정함:
    1) B에는 val split이 아예 없다(실측 확인: split별 행수에서 B/val=0). 통합 모델의
       기존 threshold=0.80도 A val만으로 찾은 것과 동일하게, B는 train 마지막 4주를
       내부 검증셋으로 재구성해서 쓴다(train_center_base_models.py와 동일 로직).
    2) "물동량 합산" WAPE는 A(train 150만 행)가 B(17만 행)보다 훨씬 커서 사실상 A
       위주로 결정되고 B 전용 신호가 무의미해진다. 그래서 물동량 가중 합산 대신
       "A WAPE와 B WAPE의 단순평균"을 목적함수로 써서 두 센터를 동등하게 반영한다.

핵심 설계:
    1) 그룹 분리 기준: adi_expanding_filled(fallback 채움 버전, 결측 없음)로 모든 행을
       ADI<1.32(Smooth+Erratic) / ADI>=1.32(Intermittent+Lumpy) 2그룹으로 나눈다.
       원본(비-filled) adi_expanding을 쓰면 첫 판매 전 행(NaN)의 그룹이 정해지지 않으므로
       fallback 채움 버전을 사용(모델이 실제로 보는 값과도 일치).
    2) 그룹별 threshold를 A val(그룹별 부분집합)과 B 내부 4주 검증셋(그룹별 부분집합)
       양쪽에서 각각 탐색해 (A_WAPE+B_WAPE)/2가 최소인 조합을 채택 — 특정 센터가
       일방적으로 손해보지 않는지 A_WAPE/B_WAPE/B_Bias를 개별로 함께 출력해 확인한다.
    3) 최종 채택된 2개 threshold는 A_test/B 53-fold walk-forward에 "1회만" 적용해
       기존 단일 threshold(0.80)와 비교한다(테스트 보고 재탐색 금지).
"""

import json

import numpy as np
import pandas as pd

from common import (
    BASE_CLS_MODEL_PATH,
    BASE_MODEL_PATH,
    FEATURE_TABLE_PATH,
    TARGET_COL,
    WALKFORWARD_FOLDS_PATH,
    apply_target_date_boundary_filter,
    compute_metrics,
    load_model_bundle,
    prepare_X,
)

ADI_CUTOFF = 1.32
THRESHOLD_GRID = np.round(np.arange(0.20, 0.96, 0.05), 2)
B_INTERNAL_VAL_WEEKS = 4
GROUPS = ["SE", "IL"]  # SE=Smooth+Erratic(ADI<1.32), IL=Intermittent+Lumpy(ADI>=1.32)


def group_mask(df: pd.DataFrame, group: str) -> np.ndarray:
    adi = df["adi_expanding_filled"].to_numpy()
    return (adi < ADI_CUTOFF) if group == "SE" else (adi >= ADI_CUTOFF)


def base_predict_parts(df: pd.DataFrame, cls_bundle: dict, reg_bundle: dict) -> tuple:
    """분류 확률 + 조건부회귀값을 한 번만 계산해두고 threshold만 바꿔 재사용."""
    Xc = prepare_X(df, cls_bundle["feature_cols"])
    prob = cls_bundle["model"].predict_proba(Xc)[:, 1]
    Xr = prepare_X(df, reg_bundle["feature_cols"])
    cond_qty = np.clip(np.expm1(reg_bundle["model"].predict(Xr)), 0, None)
    return prob, cond_qty


def sweep_group(df: pd.DataFrame, prob: np.ndarray, cond_qty: np.ndarray, group: str) -> pd.DataFrame:
    mask = group_mask(df, group)
    y_true = df.loc[mask, TARGET_COL].to_numpy()
    naive_pred = df.loc[mask, "qty"].to_numpy()
    p, q = prob[mask], cond_qty[mask]
    rows = []
    for th in THRESHOLD_GRID:
        pred = np.where(p >= th, q, 0.0)
        m = compute_metrics(y_true, pred, naive_pred)
        rows.append({"threshold": float(th), "n": int(mask.sum()), **{k: round(v, 3) for k, v in m.items()}})
    return pd.DataFrame(rows)


def load_b_internal_val(df: pd.DataFrame) -> pd.DataFrame:
    b_train_full = df[(df["center_id"] == "B") & (df["split"] == "train")].copy()
    b_train_full = apply_target_date_boundary_filter(b_train_full, horizon_weeks=1)
    b_train_full = b_train_full[b_train_full[TARGET_COL].notna()]
    b_weeks = sorted(b_train_full["week_st"].unique())
    val_weeks_b = b_weeks[-B_INTERNAL_VAL_WEEKS:]
    return b_train_full[b_train_full["week_st"].isin(val_weeks_b)]


def choose_group_thresholds(a_val: pd.DataFrame, b_val: pd.DataFrame, cls_bundle, reg_bundle) -> dict:
    a_prob, a_cond = base_predict_parts(a_val, cls_bundle, reg_bundle)
    b_prob, b_cond = base_predict_parts(b_val, cls_bundle, reg_bundle)

    chosen = {}
    for group in GROUPS:
        a_table = sweep_group(a_val, a_prob, a_cond, group)
        b_table = sweep_group(b_val, b_prob, b_cond, group)
        merged = a_table.merge(b_table, on="threshold", suffixes=("_A", "_B"))
        merged["combined_wape"] = (merged["WAPE_A"] + merged["WAPE_B"]) / 2
        best = merged.loc[merged["combined_wape"].idxmin()]
        print(f"\n[{group}] A n={int(a_table['n'].iloc[0]):,} / B n={int(b_table['n'].iloc[0]):,}")
        print(merged[["threshold", "WAPE_A", "Bias(%)_A", "WAPE_B", "Bias(%)_B", "combined_wape"]].to_string(index=False))
        print(f"  -> 채택 threshold={best['threshold']:.2f}  "
              f"(A WAPE={best['WAPE_A']:.2f}% Bias={best['Bias(%)_A']:+.2f}% / "
              f"B WAPE={best['WAPE_B']:.2f}% Bias={best['Bias(%)_B']:+.2f}%)")
        chosen[group] = float(best["threshold"])
    return chosen


def predict_group_threshold(df: pd.DataFrame, cls_bundle, reg_bundle, thresholds: dict) -> np.ndarray:
    prob, cond_qty = base_predict_parts(df, cls_bundle, reg_bundle)
    is_il = group_mask(df, "IL")
    th = np.where(is_il, thresholds["IL"], thresholds["SE"])
    return np.where(prob >= th, cond_qty, 0.0)


def final_eval(df: pd.DataFrame, cls_bundle, reg_bundle, thresholds: dict) -> pd.DataFrame:
    a_test = df[(df["center_id"] == "A") & (df["split"] == "test")].copy()
    a_test = a_test[a_test[TARGET_COL].notna()]
    y_true_a = a_test[TARGET_COL].to_numpy()
    naive_a = a_test["qty"].to_numpy()
    pred_a = predict_group_threshold(a_test, cls_bundle, reg_bundle, thresholds)
    m_a = compute_metrics(y_true_a, pred_a, naive_a)

    with open(WALKFORWARD_FOLDS_PATH, encoding="utf-8") as f:
        folds = json.load(f)
    b_pool = df[(df["center_id"] == "B") & (df["split"] == "pool")].copy()
    all_true, all_pred, all_naive = [], [], []
    for train_start, train_end, val_start, val_end in folds:
        mask = (b_pool["week_st"] >= val_start) & (b_pool["week_st"] <= val_end)
        fold_df = b_pool[mask]
        fold_df = fold_df[fold_df[TARGET_COL].notna()]
        if len(fold_df) == 0:
            continue
        all_true.append(fold_df[TARGET_COL].to_numpy())
        all_pred.append(predict_group_threshold(fold_df, cls_bundle, reg_bundle, thresholds))
        all_naive.append(fold_df["qty"].to_numpy())
    m_b = compute_metrics(np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_naive))

    return pd.DataFrame([
        {"center": "A", "n": len(a_test), **{k: round(v, 3) for k, v in m_a.items()}},
        {"center": "B", "n": len(np.concatenate(all_true)), **{k: round(v, 3) for k, v in m_b.items()}},
    ])


def main():
    df = pd.read_parquet(FEATURE_TABLE_PATH)
    cls_bundle = load_model_bundle(BASE_CLS_MODEL_PATH)
    reg_bundle = load_model_bundle(BASE_MODEL_PATH)
    print(f"[기존 단일 threshold] {cls_bundle.get('threshold')}")

    a_val = df[(df["center_id"] == "A") & (df["split"] == "val")].copy()
    a_val = a_val[a_val[TARGET_COL].notna()]
    b_val = load_b_internal_val(df)
    print(f"A val: {len(a_val):,}행 / B 내부(마지막 {B_INTERNAL_VAL_WEEKS}주) val: {len(b_val):,}행")

    print("=" * 80)
    print("[그룹별 threshold 탐색] (A_WAPE + B_WAPE)/2 최소 기준")
    chosen = choose_group_thresholds(a_val, b_val, cls_bundle, reg_bundle)
    print(f"\n[채택된 그룹별 threshold] {chosen}")

    print()
    print("=" * 80)
    print("[최종 홀드아웃 1회 적용] SBC 그룹별 threshold")
    report_grouped = final_eval(df, cls_bundle, reg_bundle, chosen)
    print(report_grouped.to_string(index=False))

    print()
    print("[비교: 기존 단일 threshold=0.80]")
    single_th = {"SE": cls_bundle.get("threshold", 0.80), "IL": cls_bundle.get("threshold", 0.80)}
    report_single = final_eval(df, cls_bundle, reg_bundle, single_th)
    print(report_single.to_string(index=False))

    print()
    print("=" * 80)
    print("[개선폭] 그룹별 threshold - 단일 threshold")
    for center in ["A", "B"]:
        g = report_grouped[report_grouped["center"] == center].iloc[0]
        s = report_single[report_single["center"] == center].iloc[0]
        for metric in ["RMSE", "MAE", "WAPE", "MASE"]:
            delta = g[metric] - s[metric]
            direction = "개선" if delta < 0 else "악화"
            print(f"  [{center}] {metric}: {s[metric]:.3f} -> {g[metric]:.3f} ({direction})")
        print(f"  [{center}] Bias: {s['Bias(%)']:+.3f}% -> {g['Bias(%)']:+.3f}%")


if __name__ == "__main__":
    main()
