"""
ab_final_report.py
확정된 CV 정책(A: 3개월 Expanding CV로 검증 후 2021~2023 전체 재학습 / B: A+B Pooling
통합 Base Model 유지 + 53주 Walk-forward CV) 기준으로 A/B 2024 최종 성능을
WAPE/MAE/RMSE/MAPE 4개 지표로 한 번에 정리한다. 재학습은 하지 않는다 — 이미
저장돼 있는 모델(A: 이전 세션에서 만든 a_final_cv, B: 프로덕션 base_model)을
그대로 불러와 추론만 한다.

python ab_final_report.py

핵심 설계:
    1) A: a_final_cv_cls/reg.pkl(2021~2023만으로 학습, early-stop도 그 구간 내부
       마지막 4주만 사용해 2024를 전혀 안 봄)을 cv_common.load_a_full()의 2024
       전체 슬라이스로 평가한다 — 이 모델이 애초에 그 소스로 학습됐으므로 평가도
       같은 소스를 써야 값이 일관된다.
    2) B: 프로덕션 base_model_cls/reg.pkl(재학습 없음, "A+B Pooling 통합 유지"
       원칙 그대로)을 프로덕션 feature_table_final.parquet의 B/pool(2024)로
       평가한다. WAPE/RMSE/MAE는 evaluate_pipeline.evaluate_b_walkforward()를
       그대로 재사용(공식 53-fold 방식과 완전히 동일한 숫자 보장, 프로덕션 파일
       미수정 — import만 함). 그 함수가 MAPE를 계산하지 않으므로 같은 B/pool
       데이터에 대해 cv_common.compute_mape()로 별도 계산해 병합한다.
    3) 반품 데이터는 애초에 이 스크립트가 참조하는 어떤 컬럼에도 없음(qty/target_h1
       기준으로만 동작) — 별도 처리 불필요.
    4) 산출물은 data/ml/cv_experiment/ 아래에만 저장, 프로덕션 파일은 read-only.
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as cc  # noqa: E402
import folds  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BASE_DIR / "src" / "ml" / "models"))
from common import BASE_CLS_MODEL_PATH, BASE_MODEL_PATH, FEATURE_TABLE_PATH, TARGET_COL  # noqa: E402
from evaluate_pipeline import evaluate_b_walkforward  # noqa: E402

MODEL_DIR = cc.CV_DIR / "models"
REPORT_DIR = cc.CV_DIR / "reports"
A_FINAL_CLS_PATH = MODEL_DIR / "a_final_cv_cls.pkl"
A_FINAL_REG_PATH = MODEL_DIR / "a_final_cv_reg.pkl"

ROUND = 3
METRIC_COLS = ["WAPE", "MAE", "RMSE", "MAPE"]


def eval_a() -> dict:
    print("=== A센터: a_final_cv(2021~2023 재학습, 2024 미참조) -> 2024 전체 평가 ===")
    if not A_FINAL_CLS_PATH.exists():
        raise FileNotFoundError(
            f"{A_FINAL_CLS_PATH} 없음 — run_cv_experiment.py(option1/option2/all)를 먼저 실행해야 함"
        )
    cls_bundle = cc.load_model_bundle(A_FINAL_CLS_PATH)
    reg_bundle = cc.load_model_bundle(A_FINAL_REG_PATH)

    a_full = cc.load_a_full()
    a_2024 = cc.slice_by_date(a_full, folds.FINAL_TEST_START, folds.FINAL_TEST_END)
    m = cc.predict_and_score(a_2024, cls_bundle, reg_bundle)
    print(f"  n={m['n']:,}  WAPE={m['WAPE']:.2f}%  RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  "
          f"MAPE={m['MAPE']:.2f}%(대상 {m['n_mape']:,}행, actual=0 제외 {m['n_mape_excluded_zero']:,}행)")
    return m


def eval_b() -> dict:
    print()
    print("=== B센터: 프로덕션 base_model(A+B 통합, 재학습 없음) -> 53주 Walk-forward 2024 평가 ===")
    cls_bundle = cc.load_model_bundle(BASE_CLS_MODEL_PATH)
    reg_bundle = cc.load_model_bundle(BASE_MODEL_PATH)
    df = pd.read_parquet(FEATURE_TABLE_PATH)

    summary, fold_table = evaluate_b_walkforward(df, cls_bundle, reg_bundle, mode="soft")

    b_pool = df[(df["center_id"] == "B") & (df["split"] == "pool")].copy()
    b_pool = b_pool[b_pool[TARGET_COL].notna()]
    mape_only = cc.predict_and_score(b_pool, cls_bundle, reg_bundle)

    m = {
        "n": summary["n"], "WAPE": summary["WAPE"], "RMSE": summary["RMSE"], "MAE": summary["MAE"],
        "MAPE": mape_only["MAPE"], "n_mape": mape_only["n_mape"], "n_mape_excluded_zero": mape_only["n_mape_excluded_zero"],
    }
    print(f"  [53-fold 합산 WAPE/RMSE/MAE + 전체 MAPE 병합] n={m['n']:,}  WAPE={m['WAPE']:.2f}%  "
          f"RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  MAPE={m['MAPE']:.2f}%"
          f"(대상 {m['n_mape']:,}행, actual=0 제외 {m['n_mape_excluded_zero']:,}행)")
    return m


def main():
    m_a = eval_a()
    m_b = eval_b()

    rows = [
        {"센터": "A", "검증체계": "3개월 Expanding CV -> 2021~2023 재학습(a_final_cv)",
         **{k: round(m_a[k], ROUND) for k in METRIC_COLS}, "n": m_a["n"]},
        {"센터": "B", "검증체계": "A+B Pooling 통합 유지, 53주 Walk-forward",
         **{k: round(m_b[k], ROUND) for k in METRIC_COLS}, "n": m_b["n"]},
    ]
    report_df = pd.DataFrame(rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_DIR / "ab_final_report_confirmed_policy.csv", index=False, encoding="utf-8-sig")
    md_lines = [
        "| " + " | ".join(report_df.columns) + " |",
        "| " + " | ".join(["---"] * len(report_df.columns)) + " |",
    ]
    for _, r in report_df.iterrows():
        md_lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    md_lines.append("")
    md_lines.append(
        "각주: A는 이전 세션에서 학습된 a_final_cv(2021~2023 전체, early-stop도 그 구간 "
        "내부 마지막 4주만 사용 — 2024를 전혀 참조하지 않음)를 재사용. B는 프로덕션 "
        "base_model_cls/reg.pkl(A+B 통합, 재학습 없음)을 evaluate_pipeline.py와 동일한 "
        "53-fold walk-forward 방식으로 평가. 두 센터 모두 2024는 이 평가 1회 외에 어떤 "
        "학습/탐색 과정에도 쓰이지 않음. MAPE는 actual=0 행을 제외(zero_handling='exclude')."
    )
    (REPORT_DIR / "ab_final_report_confirmed_policy.md").write_text("\n".join(md_lines), encoding="utf-8")

    print()
    print("=" * 80)
    print("[A/B 확정 정책 기준 2024 최종 성능]")
    print(report_df.to_string(index=False))
    print(f"\n저장 완료 -> {REPORT_DIR}/ab_final_report_confirmed_policy.{{csv,md}}")


if __name__ == "__main__":
    main()
