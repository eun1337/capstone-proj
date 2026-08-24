"""
run_b_only_experiment.py
B센터 단독 모델링 실험 — A센터 데이터를 완전히 제외하고 B센터 자체 데이터(pre+post
레짐 전체이력, build_b_full_history.py 산출물)만으로 독립 Hurdle 모델을 학습해,
기존 방식(A+B 통합 프로덕션 모델)과 2024년 전체 Test에서 1:1 비교한다.

python run_b_only_experiment.py

핵심 설계:
    1) A 데이터는 로드조차 하지 않는다(cv_common.load_a_full 호출 없음) — 완전 배제.
    2) 학습 데이터: B_full_feat.parquet(2020-12-28~2023-12-31 구간 전체, pre 레짐
       포함 — 앞선 Option3 CV 실험에서 동일하게 "B 본연의
       데이터 전부"를 쓴다). 2024년은 다른 실험과 동일하게 순수 holdout으로 격리.
    3) 조기종료(early-stopping) val은 학습 구간 마지막 4주를 내부 분리
       (cv_common.carve_internal_es_val, 기존 B_INTERNAL_VAL_WEEKS 관례).
    4) "기존 방식"과 "B단독"을 정확히 같은 2024 평가 데이터프레임(B_full의 2024
       슬라이스)에 대해 평가해 1:1 비교의 공정성을 보장한다(프로덕션
       feature_table_final.parquet의 B/pool 2024 구간과 수치상 동일해야 정상 —
       두 소스 모두 같은 Day2/Day3 함수로 계산되고 2024 구간은 레짐 이슈가 없음).
    5) 반품 데이터는 이 실험에서도 전혀 쓰지 않음(B_full_feat.parquet 자체가
       반품수량을 배제하고 만들어짐, build_b_full_history.py 참고).
    6) 산출물은 기존과 동일하게 data/ml/cv_experiment/ 아래에만 저장하고, 이전
       CV 비교 리포트(cv_comparison_report.*)는 덮어쓰지 않도록 별도 파일명을 쓴다.
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

B_ONLY_TRAIN_END = folds.AB_POOLED_FINAL_TRAIN_END  # "2023-12-31" — 2024는 holdout

ROUND = 3
METRIC_COLS = ["WAPE", "MAE", "RMSE", "MAPE"]


def train_b_only(b_full: pd.DataFrame):
    print("=== B센터 단독 모델 학습 (A 데이터 미사용) ===")
    window = b_full[b_full[cc.WEEK_COL] <= pd.Timestamp(B_ONLY_TRAIN_END)]
    print(f"  학습 대상 원본 구간: {window[cc.WEEK_COL].min().date()} ~ {window[cc.WEEK_COL].max().date()} "
          f"({len(window):,}행, pre+post 레짐 전체)")
    window = cc.apply_boundary_filter(window, B_ONLY_TRAIN_END, horizon_weeks=1)
    window = window[window[TARGET_COL].notna()]

    train_fit, es_val = cc.carve_internal_es_val(window)
    print(f"  내부 early-stop val: {es_val[cc.WEEK_COL].min().date()} ~ {es_val[cc.WEEK_COL].max().date()} "
          f"({len(es_val):,}행) / 나머지 학습: {len(train_fit):,}행")

    feature_cols = cc.get_feature_cols(train_fit)
    cls_bundle, reg_bundle = cc.train_hurdle(train_fit, es_val, feature_cols, label="B_only_final")

    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cc.save_model_bundle(MODEL_OUT_DIR / "b_only_cls.pkl", cls_bundle["model"], feature_cols)
    cc.save_model_bundle(MODEL_OUT_DIR / "b_only_reg.pkl", reg_bundle["model"], feature_cols)
    return cls_bundle, reg_bundle


def main():
    b_full = cc.load_b_full()
    test_df = cc.slice_by_date(b_full, folds.FINAL_TEST_START, folds.FINAL_TEST_END)
    test_df = test_df[test_df[TARGET_COL].notna()]
    print(f"[2024 Test] 평가 대상: {len(test_df):,}행 "
          f"({test_df[cc.WEEK_COL].min().date()} ~ {test_df[cc.WEEK_COL].max().date()})")

    cls_only, reg_only = train_b_only(b_full)
    b_only_m = cc.predict_and_score(test_df, cls_only, reg_only)
    print(f"  [B단독] WAPE={b_only_m['WAPE']:.2f}% RMSE={b_only_m['RMSE']:.2f} "
          f"MAE={b_only_m['MAE']:.2f} MAPE={b_only_m['MAPE']:.2f}% "
          f"(MAPE 대상 {b_only_m['n_mape']:,}행, actual=0 제외 {b_only_m['n_mape_excluded_zero']:,}행)")

    print()
    print("=== 기존 방식(프로덕션 A+B 통합 모델, read-only) 동일 데이터로 평가 ===")
    existing_cls = cc.load_model_bundle(BASE_CLS_MODEL_PATH)
    existing_reg = cc.load_model_bundle(BASE_MODEL_PATH)
    existing_m = cc.predict_and_score(test_df, existing_cls, existing_reg)
    print(f"  [기존] WAPE={existing_m['WAPE']:.2f}% RMSE={existing_m['RMSE']:.2f} "
          f"MAE={existing_m['MAE']:.2f} MAPE={existing_m['MAPE']:.2f}% "
          f"(MAPE 대상 {existing_m['n_mape']:,}행, actual=0 제외 {existing_m['n_mape_excluded_zero']:,}행)")

    rows = [
        {"실험 구분": "기존 방식(A+B 통합)", **{k: round(existing_m[k], ROUND) for k in METRIC_COLS},
         "n": existing_m["n"], "MAPE 대상행": existing_m["n_mape"], "MAPE 제외행(actual=0)": existing_m["n_mape_excluded_zero"]},
        {"실험 구분": "B단독(A 미사용)", **{k: round(b_only_m[k], ROUND) for k in METRIC_COLS},
         "n": b_only_m["n"], "MAPE 대상행": b_only_m["n_mape"], "MAPE 제외행(actual=0)": b_only_m["n_mape_excluded_zero"]},
    ]
    delta = {"실험 구분": "차이(기존 - B단독, 양수=B단독이 더 좋음)"}
    for k in METRIC_COLS:
        delta[k] = round(existing_m[k] - b_only_m[k], ROUND)
    delta["n"] = ""
    delta["MAPE 대상행"] = ""
    delta["MAPE 제외행(actual=0)"] = ""
    rows.append(delta)

    report_df = pd.DataFrame(rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_DIR / "b_only_vs_existing_report.csv", index=False, encoding="utf-8-sig")
    md_lines = [
        "| " + " | ".join(report_df.columns) + " |",
        "| " + " | ".join(["---"] * len(report_df.columns)) + " |",
    ]
    for _, r in report_df.iterrows():
        md_lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    md_lines.append("")
    md_lines.append(
        f"각주: 학습 구간은 B_full_feat.parquet 전체(pre+post 레짐, ~2023-12-31까지, A 데이터 미사용). "
        f"2024-01-01~2024-12-31을 공통 holdout test로 사용. MAPE는 actual=0인 행을 분모 계산에서 "
        f"제외하는 방식(zero_handling='exclude')으로 계산 — epsilon 방식은 실제 0행이 전체의 "
        f"상당 비율(희소 수요 데이터)이라 그 값들이 MAPE를 지배해버려 지표로서 무의미해지므로 채택하지 않음. "
        f"반품 데이터는 이 실험에서도 전혀 사용하지 않음."
    )
    (REPORT_DIR / "b_only_vs_existing_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print()
    print("=" * 80)
    print("[B센터 단독 vs 기존 방식 — 2024 Test 종합 비교]")
    print(report_df.to_string(index=False))
    print(f"\n저장 완료 -> {REPORT_DIR}/b_only_vs_existing_report.{{csv,md}}")


if __name__ == "__main__":
    main()
