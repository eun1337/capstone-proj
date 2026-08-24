"""
search_threshold_by_center.py
센터별 분리 Hurdle 모델의 hard 모드 threshold 탐색 (A전용/B전용)

통합 모델 threshold 탐색에서 WAPE와 Bias가 서로 다른 지점을 최적으로 가리키는
트레이드오프를 확인했고(0.85=WAPE 최저, 0.50=Bias 최적), 진짜 WAPE 최저점을 찾으려면
0.5~0.95까지 넓게 봐야 함을 배웠다. 이번엔 A전용/B전용 각각에 대해 처음부터
0.20~0.95 전 구간을 한 번에 훑는다(예측만 필요해 재학습 없이 빠르게 가능).

핵심 설계:
    1) A: A val split(2024-01~06, train_center_base_models.py의 early stopping에도
       쓰인 것과 동일)로 탐색.
    2) B: 별도 val split이 없어, train_center_base_models.py가 조기종료에 썼던 것과
       똑같은 내부 검증셋(B train 마지막 4주, pool 비침범)을 동일 로직으로 재구성해
       탐색한다 — pool(최종 walk-forward 평가용)은 절대 여기 쓰지 않는다.
    3) 각 센터마다 WAPE 최저 threshold와 |Bias| 최소(가장 균형 잡힌) threshold를
       모두 표시해 트레이드오프를 그대로 보여준다. 최종 저장은 WAPE 최저 기준
       (기존 통합 모델 threshold 선정 관례와 동일한 기준 유지).
"""

import numpy as np
import pandas as pd
import joblib

from common import (
    A_BASE_CLS_PATH,
    A_BASE_REG_PATH,
    B_BASE_CLS_PATH,
    B_BASE_REG_PATH,
    FEATURE_TABLE_PATH,
    MODEL_DIR,
    TARGET_COL,
    apply_target_date_boundary_filter,
    compute_metrics,
    load_model_bundle,
    predict_base_hurdle,
)

THRESHOLD_GRID = np.round(np.arange(0.20, 0.96, 0.05), 2)
B_INTERNAL_VAL_WEEKS = 4


def sweep(val_df: pd.DataFrame, cls_bundle: dict, reg_bundle: dict, label: str) -> pd.DataFrame:
    y_true = val_df[TARGET_COL].to_numpy()
    naive_pred = val_df["qty"].to_numpy()
    rows = []
    for th in THRESHOLD_GRID:
        y_pred = predict_base_hurdle(val_df, cls_bundle, reg_bundle, mode="hard", threshold=th)
        m = compute_metrics(y_true, y_pred, naive_pred)
        rows.append({"threshold": float(th), **{k: round(v, 3) for k, v in m.items()}})
        print(f"  [{label}] th={th:.2f} WAPE={m['WAPE']:.3f}% Bias={m['Bias(%)']:+.3f}%")
    return pd.DataFrame(rows)


def summarize_and_save(table: pd.DataFrame, cls_path, label: str):
    best_wape_row = table.loc[table["WAPE"].idxmin()]
    best_bias_row = table.loc[table["Bias(%)"].abs().idxmin()]
    print()
    print(f"[{label}] WAPE 최저: threshold={best_wape_row['threshold']:.2f} "
          f"(WAPE={best_wape_row['WAPE']:.3f}%, Bias={best_wape_row['Bias(%)']:+.3f}%)")
    print(f"[{label}] |Bias| 최소(균형): threshold={best_bias_row['threshold']:.2f} "
          f"(WAPE={best_bias_row['WAPE']:.3f}%, Bias={best_bias_row['Bias(%)']:+.3f}%)")

    cls_bundle = load_model_bundle(cls_path)
    cls_bundle["threshold"] = float(best_wape_row["threshold"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(cls_bundle, cls_path)
    print(f"[{label}] {cls_path.name} threshold 저장 완료 -> {best_wape_row['threshold']:.2f} (WAPE 최저 기준)")
    return best_wape_row, best_bias_row


def main():
    df = pd.read_parquet(FEATURE_TABLE_PATH)

    print("=" * 80)
    print("[A전용] threshold 탐색 (A val split)")
    a_cls = load_model_bundle(A_BASE_CLS_PATH)
    a_reg = load_model_bundle(A_BASE_REG_PATH)
    a_val = df[(df["center_id"] == "A") & (df["split"] == "val")].copy()
    a_val = a_val[a_val[TARGET_COL].notna()]
    print(f"  val 행수: {len(a_val):,}")
    a_table = sweep(a_val, a_cls, a_reg, "A")
    a_best_wape, a_best_bias = summarize_and_save(a_table, A_BASE_CLS_PATH, "A전용")

    print("=" * 80)
    print("[B전용] threshold 탐색 (B train 마지막 4주 내부 검증셋, pool 비침범)")
    b_cls = load_model_bundle(B_BASE_CLS_PATH)
    b_reg = load_model_bundle(B_BASE_REG_PATH)
    b_train_full = df[(df["center_id"] == "B") & (df["split"] == "train")].copy()
    b_train_full = apply_target_date_boundary_filter(b_train_full, horizon_weeks=1)
    b_train_full = b_train_full[b_train_full[TARGET_COL].notna()]
    b_weeks = sorted(b_train_full["week_st"].unique())
    val_weeks_b = b_weeks[-B_INTERNAL_VAL_WEEKS:]
    b_val = b_train_full[b_train_full["week_st"].isin(val_weeks_b)]
    print(f"  내부 val 행수: {len(b_val):,} ({B_INTERNAL_VAL_WEEKS}주, "
          f"{pd.Timestamp(val_weeks_b[0]).date()}~{pd.Timestamp(val_weeks_b[-1]).date()})")
    b_table = sweep(b_val, b_cls, b_reg, "B")
    b_best_wape, b_best_bias = summarize_and_save(b_table, B_BASE_CLS_PATH, "B전용")

    print()
    print("=" * 80)
    print("[전체 threshold 곡선] A전용")
    print(a_table.to_string(index=False))
    print()
    print("[전체 threshold 곡선] B전용")
    print(b_table.to_string(index=False))

    print()
    print("=" * 80)
    print("[센터별 threshold 요약]")
    summary = pd.DataFrame([
        {"center": "A", "criterion": "WAPE 최저", **a_best_wape.to_dict()},
        {"center": "A", "criterion": "Bias 균형", **a_best_bias.to_dict()},
        {"center": "B", "criterion": "WAPE 최저", **b_best_wape.to_dict()},
        {"center": "B", "criterion": "Bias 균형", **b_best_bias.to_dict()},
    ])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
