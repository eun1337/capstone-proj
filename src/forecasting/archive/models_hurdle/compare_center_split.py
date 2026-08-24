"""
compare_center_split.py
통합 Hurdle Model vs 센터별 분리 Hurdle Model (soft 모드) 비교

evaluate_pipeline.py의 evaluate_a/evaluate_b_walkforward를 그대로 재사용해
(A_test 홀드아웃 / B 53-fold walk-forward 평가 로직은 통합·분리 모델 공통) 통합
모델과 A전용/B전용 모델을 동일한 조건(soft 모드)으로 나란히 비교한다.
"""

import pandas as pd

from common import (
    A_BASE_CLS_PATH,
    A_BASE_REG_PATH,
    B_BASE_CLS_PATH,
    B_BASE_REG_PATH,
    BASE_CLS_MODEL_PATH,
    BASE_MODEL_PATH,
    FEATURE_TABLE_PATH,
    load_model_bundle,
)
from evaluate_pipeline import evaluate_a, evaluate_b_walkforward


def main():
    df = pd.read_parquet(FEATURE_TABLE_PATH)

    unified_cls = load_model_bundle(BASE_CLS_MODEL_PATH)
    unified_reg = load_model_bundle(BASE_MODEL_PATH)
    a_cls = load_model_bundle(A_BASE_CLS_PATH)
    a_reg = load_model_bundle(A_BASE_REG_PATH)
    b_cls = load_model_bundle(B_BASE_CLS_PATH)
    b_reg = load_model_bundle(B_BASE_REG_PATH)

    print("#" * 80)
    print("[통합 모델] A 평가")
    result_unified_a = evaluate_a(df, unified_cls, unified_reg, mode="soft")
    result_unified_a["model"] = "통합"

    print("#" * 80)
    print("[A전용 모델] A 평가")
    result_a_only = evaluate_a(df, a_cls, a_reg, mode="soft")
    result_a_only["model"] = "센터별 분리"

    print("#" * 80)
    print("[통합 모델] B 평가")
    result_unified_b, _ = evaluate_b_walkforward(df, unified_cls, unified_reg, mode="soft")
    result_unified_b["model"] = "통합"

    print("#" * 80)
    print("[B전용 모델] B 평가")
    result_b_only, _ = evaluate_b_walkforward(df, b_cls, b_reg, mode="soft")
    result_b_only["model"] = "센터별 분리"

    print()
    print("=" * 80)
    print("[통합 vs 센터별 분리] A센터 비교 (soft 모드)")
    report_a = pd.DataFrame([result_unified_a, result_a_only])[["model", "center", "n", "RMSE", "MAE", "WAPE", "MASE", "Bias(%)"]]
    print(report_a.to_string(index=False))

    print()
    print("[통합 vs 센터별 분리] B센터 비교 (soft 모드)")
    report_b = pd.DataFrame([result_unified_b, result_b_only])[["model", "center", "n", "RMSE", "MAE", "WAPE", "MASE", "Bias(%)"]]
    print(report_b.to_string(index=False))

    print()
    print("=" * 80)
    print("[개선폭 요약] (센터별 분리 - 통합, 음수면 개선인 지표: RMSE/MAE/WAPE/MASE / Bias는 절대값 감소가 개선)")
    for center, u, s in [("A", result_unified_a, result_a_only), ("B", result_unified_b, result_b_only)]:
        for metric in ["RMSE", "MAE", "WAPE", "MASE"]:
            delta = s[metric] - u[metric]
            pct = delta / u[metric] * 100 if u[metric] else float("nan")
            direction = "개선" if delta < 0 else "악화"
            print(f"  [{center}] {metric}: {u[metric]:.3f} -> {s[metric]:.3f} ({pct:+.1f}%, {direction})")
        bias_delta = abs(s["Bias(%)"]) - abs(u["Bias(%)"])
        direction = "개선(0에 근접)" if bias_delta < 0 else "악화(0에서 멀어짐)"
        print(f"  [{center}] |Bias|: {abs(u['Bias(%)']):.3f}% -> {abs(s['Bias(%)']):.3f}% ({direction})")


if __name__ == "__main__":
    main()
