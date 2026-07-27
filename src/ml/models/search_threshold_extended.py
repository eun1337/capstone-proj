"""
search_threshold_extended.py
통합 Hurdle Model의 hard 모드 threshold 전 구간(0.20~0.95) 탐색

train_base_model.py는 0.20~0.50 구간만 탐색하는데, 그 구간에서 WAPE가 단조
개선되며 상한 0.50이 채택돼 진짜 최적점이 범위 밖에 있을 가능성이 있었다. 이
스크립트는 재학습 없이(이미 저장된 base_model_cls.pkl/base_model_reg.pkl 재사용,
threshold 탐색은 예측만 필요해 학습 비용이 들지 않음) 0.20~0.95 전 구간을 그때그때
새로 계산해 진짜 WAPE 최저점을 찾고, base_model_cls.pkl의 threshold를 그 값으로
갱신한다. (과거 특정 시점의 탐색 결과를 하드코딩해 재사용하지 않음 — 모델이 재학습될
때마다(예: SBC feature 추가 후) 이 스크립트를 다시 돌리면 항상 그 시점 모델 기준
최신 결과가 나온다.)
"""

import numpy as np
import pandas as pd
import joblib

from common import (
    BASE_CLS_MODEL_PATH,
    BASE_MODEL_PATH,
    FEATURE_TABLE_PATH,
    MODEL_DIR,
    TARGET_COL,
    compute_metrics,
    load_model_bundle,
    predict_base_hurdle,
)

THRESHOLD_GRID = np.round(np.arange(0.20, 0.96, 0.05), 2)


def main():
    df = pd.read_parquet(FEATURE_TABLE_PATH)
    val_df = df[df["split"] == "val"].copy()
    val_df = val_df[val_df[TARGET_COL].notna()]
    print(f"[Threshold 전 구간 탐색] val(A만 존재) 행수: {len(val_df):,}")

    cls_bundle = load_model_bundle(BASE_CLS_MODEL_PATH)
    reg_bundle = load_model_bundle(BASE_MODEL_PATH)
    print(f"  현재 저장된 threshold: {cls_bundle.get('threshold')}")

    y_true = val_df[TARGET_COL].to_numpy()
    naive_pred = val_df["qty"].to_numpy()

    rows = []
    for th in THRESHOLD_GRID:
        y_pred = predict_base_hurdle(val_df, cls_bundle, reg_bundle, mode="hard", threshold=th)
        m = compute_metrics(y_true, y_pred, naive_pred)
        row = {"threshold": float(th), **{k: round(v, 3) for k, v in m.items()}}
        rows.append(row)
        print(f"  th={th:.2f} WAPE={m['WAPE']:.3f}% Bias={m['Bias(%)']:+.3f}%")

    table = pd.DataFrame(rows)
    print()
    print("=" * 80)
    print("[Threshold 전 구간 탐색 결과] (0.20~0.95)")
    print(table.to_string(index=False))

    best_wape_row = table.loc[table["WAPE"].idxmin()]
    best_bias_row = table.loc[table["Bias(%)"].abs().idxmin()]
    print()
    print(f"[WAPE 최저] threshold={best_wape_row['threshold']:.2f} "
          f"(WAPE={best_wape_row['WAPE']:.3f}%, Bias={best_wape_row['Bias(%)']:+.3f}%)")
    print(f"[Bias 균형] threshold={best_bias_row['threshold']:.2f} "
          f"(WAPE={best_bias_row['WAPE']:.3f}%, Bias={best_bias_row['Bias(%)']:+.3f}%)")

    best_threshold = float(best_wape_row["threshold"])
    if best_threshold != cls_bundle.get("threshold"):
        cls_bundle["threshold"] = best_threshold
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(cls_bundle, BASE_CLS_MODEL_PATH)
        print(f"  base_model_cls.pkl threshold 갱신 완료 -> {best_threshold:.2f} (WAPE 최저 기준)")
    else:
        print("  기존 threshold와 동일 — 갱신 불필요")


if __name__ == "__main__":
    main()
