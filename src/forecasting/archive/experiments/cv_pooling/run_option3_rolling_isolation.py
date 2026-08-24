"""
run_option3_rolling_isolation.py
후속 과제: "옵션3 기각 사유가 Rolling Window 때문인지 A+B Pooling 때문인지 분리되지
않는다"는 cv_policy_decision_report.md 3.1/3.3의 caveat을 검증하기 위한 추가 실험.

run_option3()의 "B pooled 최종모델" 블록은 AB_POOLED_FINAL_TRAIN_START~END(2021~2023
전체, rolling 제한 없음)로 재학습한 것이라 실제로는 "Rolling"이 아니라 "Full History
Pooling"이었다. 이 스크립트는 동일한 파이프라인(A+B pooled, B만 평가)을 그대로 두고
학습 구간만 "2024 기준 24개월 직전(2022-01-01~2023-12-31)"으로 좁혀 진짜 Rolling
Window 최종모델을 하나 더 만든 뒤, 기존 방식 / Full History Pooling과 2024 Test에서
3-way 비교한다.

python run_option3_rolling_isolation.py

설계:
    1) 기존 방식(A+B 통합 프로덕션 모델)만 재학습 없이 read-only로 로드해서 평가한다.
    2) A+B Pooled Full History(2021~2023)와 A+B Pooled 24개월 Rolling(2022~2023)은
       둘 다 이번에 새로 학습한다. 기존에 저장된 b_pooled_final은 2026-08-02
       추가된 center_qty_lag1_inter/center_temp_inter 피처보다 오래된
       B_full_feat.parquet 기준으로 학습된 것이라 재사용하지 않는다(재학습 사유는
       build_b_full_history.py 재실행 여부와 무관하게 두 변형을 동일 피처 조건에서
       비교하기 위함).
    3) 세 모델 모두 정확히 같은 평가 데이터프레임(B_full의 2024 슬라이스)에 대해
       채점해 1:1 비교를 보장한다(run_b_only_experiment.py와 동일 원칙).
    4) 산출물은 기존 리포트를 덮어쓰지 않도록 별도 파일명으로 저장한다.

사전 조건: B_full_feat.parquet을 build_b_full_history.py로 최신 피처 코드 기준으로
재생성해 두어야 한다(2026-08-02 이전에 생성된 캐시는 위 두 상호작용 컬럼이 없어
predict_and_score에서 KeyError가 난다).
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as cc  # noqa: E402
import folds  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(BASE_DIR / "src" / "forecasting" / "archive" / "models_hurdle"))
from common import BASE_CLS_MODEL_PATH, BASE_MODEL_PATH, TARGET_COL  # noqa: E402

REPORT_DIR = cc.CV_DIR / "reports"
MODEL_OUT_DIR = cc.CV_DIR / "models"

# 2024-01-01 기준 24개월 직전 = 2022-01-01 (OPTION3_FOLDS의 "val_start 24개월 이전"
# 규칙을 최종 holdout(=사실상 val_start) 기준으로 그대로 연장한 값).
ROLLING_TRAIN_START = "2022-01-01"
ROLLING_TRAIN_END = folds.AB_POOLED_FINAL_TRAIN_END  # "2023-12-31"

ROUND = 3
METRIC_COLS = ["WAPE", "RMSE", "MAE", "MASE", "Bias(%)", "MAPE"]


def _train_pooled_window(a_full: pd.DataFrame, b_full: pd.DataFrame, train_start: str, train_end: str,
                          label: str, out_stub: str):
    print(f"=== A+B Pooled [{label}] 학습 ===")
    print(f"  학습 구간: {train_start} ~ {train_end}")

    tuned_reg = cc.load_tuned_params("regressor")
    b_weight = tuned_reg["b_center_weight"] if tuned_reg else None
    print(f"  B sample_weight: {b_weight if b_weight is not None else '(튜닝값 없음, 1.0 균등)'}")

    a_window = cc.slice_by_date(a_full, train_start, train_end)
    b_window = cc.slice_by_date(b_full, train_start, train_end)
    pooled = pd.concat([a_window, b_window], ignore_index=True)
    print(f"  pooled 원본: A {len(a_window):,}행 + B {len(b_window):,}행 = {len(pooled):,}행")

    pooled = cc.apply_boundary_filter(pooled, train_end, horizon_weeks=1)
    pooled = pooled[pooled[TARGET_COL].notna()]

    train_fit, es_val = cc.carve_internal_es_val(pooled)
    print(f"  내부 early-stop val: {len(es_val):,}행 / 나머지 학습: {len(train_fit):,}행")

    feature_cols = cc.get_feature_cols(train_fit)
    cls_bundle, reg_bundle = cc.train_hurdle(
        train_fit, es_val, feature_cols, label=label, b_weight=b_weight
    )

    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cc.save_model_bundle(MODEL_OUT_DIR / f"{out_stub}_cls.pkl", cls_bundle["model"], feature_cols)
    cc.save_model_bundle(MODEL_OUT_DIR / f"{out_stub}_reg.pkl", reg_bundle["model"], feature_cols)
    return cls_bundle, reg_bundle


def train_rolling_pooled(a_full: pd.DataFrame, b_full: pd.DataFrame):
    return _train_pooled_window(a_full, b_full, ROLLING_TRAIN_START, ROLLING_TRAIN_END,
                                 label="B_pooled_rolling24_final", out_stub="b_pooled_rolling24")


def train_full_pooled(a_full: pd.DataFrame, b_full: pd.DataFrame):
    """b_pooled_final(2026-07-29 학습본)은 2026-08-02 추가된 center_qty_lag1_inter/
    center_temp_inter 피처보다 오래된 B_full_feat.parquet 기준이라 재사용하지 않고,
    최신 피처로 다시 학습해 Rolling 변형과 동일한 조건에서 비교한다."""
    return _train_pooled_window(a_full, b_full, folds.AB_POOLED_FINAL_TRAIN_START, folds.AB_POOLED_FINAL_TRAIN_END,
                                 label="B_pooled_full_final_refresh", out_stub="b_pooled_full_refresh")


def main():
    b_full = cc.load_b_full()
    test_df = cc.slice_by_date(b_full, folds.FINAL_TEST_START, folds.FINAL_TEST_END)
    test_df = test_df[test_df[TARGET_COL].notna()]
    print(f"[2024 Test] 평가 대상(B만): {len(test_df):,}행 "
          f"({test_df[cc.WEEK_COL].min().date()} ~ {test_df[cc.WEEK_COL].max().date()})")
    print()

    a_full = cc.load_a_full()

    cls_full, reg_full = train_full_pooled(a_full, b_full)
    full_m = cc.predict_and_score(test_df, cls_full, reg_full)
    print(f"  [A+B Pooled Full History] WAPE={full_m['WAPE']:.2f}% RMSE={full_m['RMSE']:.2f} "
          f"MAE={full_m['MAE']:.2f} MAPE={full_m['MAPE']:.2f}%")

    print()
    cls_roll, reg_roll = train_rolling_pooled(a_full, b_full)
    rolling_m = cc.predict_and_score(test_df, cls_roll, reg_roll)
    print(f"  [A+B Pooled 24M Rolling] WAPE={rolling_m['WAPE']:.2f}% RMSE={rolling_m['RMSE']:.2f} "
          f"MAE={rolling_m['MAE']:.2f} MAPE={rolling_m['MAPE']:.2f}%")

    print()
    print("=== 기존 방식(프로덕션 A+B 통합 모델, read-only) 동일 데이터로 평가 ===")
    existing_cls = cc.load_model_bundle(BASE_CLS_MODEL_PATH)
    existing_reg = cc.load_model_bundle(BASE_MODEL_PATH)
    existing_m = cc.predict_and_score(test_df, existing_cls, existing_reg)
    print(f"  [기존 방식] WAPE={existing_m['WAPE']:.2f}% RMSE={existing_m['RMSE']:.2f} "
          f"MAE={existing_m['MAE']:.2f} MAPE={existing_m['MAPE']:.2f}%")

    rows = [
        {"실험 구분": "기존 방식(A+B 통합)", **{k: round(existing_m[k], ROUND) for k in METRIC_COLS}, "n": existing_m["n"]},
        {"실험 구분": "A+B Pooled Full History(2021~2023)", **{k: round(full_m[k], ROUND) for k in METRIC_COLS}, "n": full_m["n"]},
        {"실험 구분": f"A+B Pooled 24개월 Rolling({ROLLING_TRAIN_START}~{ROLLING_TRAIN_END})",
         **{k: round(rolling_m[k], ROUND) for k in METRIC_COLS}, "n": rolling_m["n"]},
    ]
    report_df = pd.DataFrame(rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_DIR / "option3_rolling_isolation_report.csv", index=False, encoding="utf-8-sig")
    md_lines = [
        "| " + " | ".join(report_df.columns) + " |",
        "| " + " | ".join(["---"] * len(report_df.columns)) + " |",
    ]
    for _, r in report_df.iterrows():
        md_lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    md_lines.append("")
    md_lines.append(
        "각주: 세 모델 모두 정확히 같은 평가셋(B_full_feat.parquet의 2024-01-01~2024-12-31 슬라이스, "
        f"n={rolling_m['n']:,})에 대해 채점했다. '기존 방식'은 재학습 없이 read-only 평가, "
        "'Full History'와 '24개월 Rolling'은 둘 다 이번에 최신 피처(center_qty_lag1_inter/"
        "center_temp_inter 포함) 기준으로 신규 학습(train_hurdle, early-stopping val=마지막 4주)했다 — "
        "기존 저장돼 있던 b_pooled_final(2026-07-29 학습, 위 두 피처 없음)은 재사용하지 않았다. "
        "이 비교로 Pooling 효과(기존 방식 -> Full History)와 Rolling 효과(Full History -> 24M Rolling)가 "
        "분리된다 — cv_policy_decision_report.md 2.3/3.3에서 지적된 confound(rolling 효과와 pooling 효과가 "
        "뒤섞여 있다는 caveat)에 대한 후속 검증."
    )
    (REPORT_DIR / "option3_rolling_isolation_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print()
    print("=" * 80)
    print("[Pooling 효과 vs Rolling 효과 분리 — 2024 Test 3-way 비교]")
    print(report_df.to_string(index=False))
    print(f"\n저장 완료 -> {REPORT_DIR}/option3_rolling_isolation_report.{{csv,md}}")


if __name__ == "__main__":
    main()
