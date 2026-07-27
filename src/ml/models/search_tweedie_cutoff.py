"""
search_tweedie_cutoff.py
Tweedie 단일모델에 post-hoc 하드 컷오프 적용 실험

Hurdle Hard의 WAPE 우위가 "2단계 구조"가 아니라 "명시적 컷오프" 메커니즘 때문인지
분리 검증한다. Tweedie(vp=1.3)의 연속 예측값 자체에 pred<cutoff -> 0 컷오프를 걸어
WAPE가 얼마나 개선되는지 본다. 재학습 불필요(이미 저장된 tweedie_reg.pkl 재사용).

주의: Hurdle의 컷오프는 분류기가 학습한 확률(0~1, "팔릴 확률"만 전담)에 거는 반면,
Tweedie 컷오프는 "확률 x 크기"가 뒤섞인 예측 수량 자체에 거는 것이라 원천적으로 더
거친 도구다 — Hurdle Hard 수준(60%대) 도달을 기대하기보다, Tweedie 단독 대비 개선폭을
보는 것이 이번 실험의 목적.
"""

import json

import numpy as np
import pandas as pd

from common import (
    FEATURE_TABLE_PATH,
    MODEL_DIR,
    TARGET_COL,
    WALKFORWARD_FOLDS_PATH,
    apply_target_date_boundary_filter,
    compute_metrics,
    load_model_bundle,
    prepare_X,
)

TWEEDIE_PATH = MODEL_DIR / "tweedie_reg.pkl"
CUTOFF_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]
B_INTERNAL_VAL_WEEKS = 4


def sweep(y_true, raw_pred, naive_pred, label):
    rows = []
    for c in CUTOFF_GRID:
        pred = np.where(raw_pred < c, 0.0, raw_pred)
        m = compute_metrics(y_true, pred, naive_pred)
        rows.append({"cutoff": c, **{k: round(v, 3) for k, v in m.items()}})
        print(f"  [{label}] cutoff={c:.2f} WAPE={m['WAPE']:.3f}% Bias={m['Bias(%)']:+.3f}%")
    table = pd.DataFrame(rows)
    best = table.loc[table["WAPE"].idxmin()]
    print(f"  [{label}] 최적 cutoff={best['cutoff']:.2f} (WAPE={best['WAPE']:.3f}%, Bias={best['Bias(%)']:+.3f}%)")
    return table, float(best["cutoff"])


def main():
    tw = load_model_bundle(TWEEDIE_PATH)
    df = pd.read_parquet(FEATURE_TABLE_PATH)

    a_val = df[(df["center_id"] == "A") & (df["split"] == "val")].copy()
    a_val = a_val[a_val[TARGET_COL].notna()]
    Xa = prepare_X(a_val, tw["feature_cols"])
    pred_a_val = np.clip(tw["model"].predict(Xa), 0, None)

    b_train_full = df[(df["center_id"] == "B") & (df["split"] == "train")].copy()
    b_train_full = apply_target_date_boundary_filter(b_train_full, horizon_weeks=1)
    b_train_full = b_train_full[b_train_full[TARGET_COL].notna()]
    b_weeks = sorted(b_train_full["week_st"].unique())
    b_val = b_train_full[b_train_full["week_st"].isin(b_weeks[-B_INTERNAL_VAL_WEEKS:])]
    Xb = prepare_X(b_val, tw["feature_cols"])
    pred_b_val = np.clip(tw["model"].predict(Xb), 0, None)

    print("=" * 80)
    print(f"[A val, n={len(a_val):,}] cutoff 스윕")
    table_a, best_cutoff_a = sweep(a_val[TARGET_COL].to_numpy(), pred_a_val, a_val["qty"].to_numpy(), "A")

    print()
    print(f"[B 내부 마지막{B_INTERNAL_VAL_WEEKS}주, n={len(b_val):,}] cutoff 스윕")
    table_b, best_cutoff_b = sweep(b_val[TARGET_COL].to_numpy(), pred_b_val, b_val["qty"].to_numpy(), "B")

    print()
    print("=" * 80)
    print("[A cutoff 전체 곡선]")
    print(table_a.to_string(index=False))
    print()
    print("[B cutoff 전체 곡선]")
    print(table_b.to_string(index=False))

    # 최종 홀드아웃 1회 적용 (센터별 자기 cutoff)
    a_test = df[(df["center_id"] == "A") & (df["split"] == "test")].copy()
    a_test = a_test[a_test[TARGET_COL].notna()]
    Xa_test = prepare_X(a_test, tw["feature_cols"])
    pred_a_test_raw = np.clip(tw["model"].predict(Xa_test), 0, None)
    pred_a_test = np.where(pred_a_test_raw < best_cutoff_a, 0.0, pred_a_test_raw)
    m_a_raw = compute_metrics(a_test[TARGET_COL].to_numpy(), pred_a_test_raw, a_test["qty"].to_numpy())
    m_a_cut = compute_metrics(a_test[TARGET_COL].to_numpy(), pred_a_test, a_test["qty"].to_numpy())

    with open(WALKFORWARD_FOLDS_PATH, encoding="utf-8") as f:
        folds = json.load(f)
    b_pool = df[(df["center_id"] == "B") & (df["split"] == "pool")].copy()
    all_true, all_raw, all_cut, all_naive = [], [], [], []
    for train_start, train_end, val_start, val_end in folds:
        mask = (b_pool["week_st"] >= val_start) & (b_pool["week_st"] <= val_end)
        fold_df = b_pool[mask]
        fold_df = fold_df[fold_df[TARGET_COL].notna()]
        if len(fold_df) == 0:
            continue
        Xf = prepare_X(fold_df, tw["feature_cols"])
        raw = np.clip(tw["model"].predict(Xf), 0, None)
        cut = np.where(raw < best_cutoff_b, 0.0, raw)
        all_true.append(fold_df[TARGET_COL].to_numpy())
        all_raw.append(raw)
        all_cut.append(cut)
        all_naive.append(fold_df["qty"].to_numpy())
    y_true_b = np.concatenate(all_true)
    naive_b = np.concatenate(all_naive)
    m_b_raw = compute_metrics(y_true_b, np.concatenate(all_raw), naive_b)
    m_b_cut = compute_metrics(y_true_b, np.concatenate(all_cut), naive_b)

    print()
    print("=" * 80)
    print(f"[최종 홀드아웃] A cutoff={best_cutoff_a:.2f} / B cutoff={best_cutoff_b:.2f} (센터별 자기 val 기준 채택)")
    report = pd.DataFrame([
        {"center": "A", "variant": "Tweedie(raw, cutoff없음)", "n": len(a_test), **{k: round(v, 3) for k, v in m_a_raw.items()}},
        {"center": "A", "variant": "Tweedie+cutoff", "n": len(a_test), **{k: round(v, 3) for k, v in m_a_cut.items()}},
        {"center": "B", "variant": "Tweedie(raw, cutoff없음)", "n": len(y_true_b), **{k: round(v, 3) for k, v in m_b_raw.items()}},
        {"center": "B", "variant": "Tweedie+cutoff", "n": len(y_true_b), **{k: round(v, 3) for k, v in m_b_cut.items()}},
    ])
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
