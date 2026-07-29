"""
run_cv_experiment.py
A/B 센터 CV 구조 개편 실험 오케스트레이터: 기존 방식 vs Option1(A 반기 Expanding) vs
Option2(A 분기 Expanding) vs Option3(A+B Pooling 24개월 Rolling) 비교.

python run_cv_experiment.py {existing|option1|option2|option3|all}

이 실험은 기존 프로덕션 산출물(data/ml/splits/*.parquet, data/ml/models/*.pkl,
feature_table_final.parquet)을 전혀 덮어쓰지 않는다 — 전부 read-only로만 읽고,
새로 학습한 모델/리포트는 data/ml/cv_experiment/ 아래에만 쓴다. 반품 데이터도
전 과정에서 쓰지 않는다(cv_common.py / build_b_full_history.py 참고).

핵심 설계:
    1) 2024-01~12 전체는 모든 옵션 공통 최종 holdout test — CV/튜닝에 전혀 안 씀.
    2) Option1/Option2의 최종 재학습 데이터 범위(2021~2023 A 전체)가 서로 동일해서
       최종모델을 한 번만 학습해 두 옵션 리포트 행에 공유한다(중복 학습 방지).
    3) "기존 방식" 행은 재학습 없이 현재 프로덕션 번들을 그대로 평가만 한다. B의 Val
       평균/std는 evaluate_pipeline.evaluate_b_walkforward()가 만드는 53-fold
       테이블을 그대로 재사용(기존 검증 결과와 완전히 동일한 숫자 보장). A의
       "2024 Test"는 기존 evaluate_pipeline의 A_test(24H2 전용)와 달리 이 실험
       공통 정의(24년 전체)로 다시 계산하므로 기존 리포트 수치와 다를 수 있음
       (각주로 명시).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as cc  # noqa: E402
import folds  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BASE_DIR / "src" / "ml" / "models"))
from common import (  # noqa: E402
    BASE_CLS_MODEL_PATH,
    BASE_MODEL_PATH,
    FEATURE_TABLE_PATH,
    TARGET_COL,
)
from evaluate_pipeline import evaluate_b_walkforward  # noqa: E402

REPORT_DIR = cc.CV_DIR / "reports"
MODEL_OUT_DIR = cc.CV_DIR / "models"

ROUND = 3


def _mean_std(values: list[float]) -> tuple[float, float | None]:
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if len(arr) == 0:
        return np.nan, None
    if len(arr) == 1:
        return round(float(arr[0]), ROUND), None
    return round(float(arr.mean()), ROUND), round(float(arr.std(ddof=1)), ROUND)


def _report_row(experiment: str, center: str, cv_method: str, n_fold,
                 val_wape: list[float], val_rmse: list[float],
                 test_metrics: dict) -> dict:
    wape_mean, wape_std = _mean_std(val_wape)
    rmse_mean, rmse_std = _mean_std(val_rmse)
    return {
        "실험 구분": experiment,
        "대상 센터": center,
        "CV 방식": cv_method,
        "Val WAPE 평균": wape_mean,
        "Val WAPE Std": wape_std if wape_std is not None else "-",
        "Val RMSE 평균": rmse_mean,
        "Val RMSE Std": rmse_std if rmse_std is not None else "-",
        "2024 Test WAPE": round(test_metrics["WAPE"], ROUND),
        "2024 Test RMSE": round(test_metrics["RMSE"], ROUND),
        "2024 Test MAE": round(test_metrics["MAE"], ROUND),
        "n(fold)": n_fold,
    }


# ---------------------------------------------------------------------------
# 기존 방식 (재학습 없음, 프로덕션 번들 read-only 평가)
# ---------------------------------------------------------------------------

def run_existing(fold_detail_rows: list[dict]) -> list[dict]:
    print("=" * 80)
    print("[기존 방식] 프로덕션 번들 read-only 평가 (재학습 없음)")
    cls_bundle = cc.load_model_bundle(BASE_CLS_MODEL_PATH)
    reg_bundle = cc.load_model_bundle(BASE_MODEL_PATH)
    df = pd.read_parquet(FEATURE_TABLE_PATH)

    a_val = df[(df["center_id"] == "A") & (df["split"] == "val")]
    a_val = a_val[a_val[TARGET_COL].notna()]
    a_val_m = cc.predict_and_score(a_val, cls_bundle, reg_bundle)
    print(f"  [A] 기존 val(2024H1, 단일 측정치) WAPE={a_val_m['WAPE']:.2f}%")

    a_2024 = cc.slice_by_date(df, folds.FINAL_TEST_START, folds.FINAL_TEST_END)
    a_2024 = a_2024[a_2024["center_id"] == "A"]
    a_test_m = cc.predict_and_score(a_2024, cls_bundle, reg_bundle)
    print(f"  [A] 2024 전체 Test WAPE={a_test_m['WAPE']:.2f}% "
          f"(참고: 기존 evaluate_pipeline.py의 A_test는 24H2만 봄 — 이 실험은 24년 전체 기준으로 재정의)")

    b_summary, b_fold_table = evaluate_b_walkforward(df, cls_bundle, reg_bundle, mode="soft")
    b_val_wape = b_fold_table["WAPE"].dropna().tolist()
    b_val_rmse = b_fold_table["RMSE"].dropna().tolist()
    for _, r in b_fold_table.iterrows():
        fold_detail_rows.append({"experiment": "existing", "center": "B", "fold": r["fold"],
                                  "val_start": r["val_start"], "WAPE": r["WAPE"], "RMSE": r["RMSE"]})
    print(f"  [B] 기존 53-fold walk-forward(재학습 없음) 합산 WAPE={b_summary['WAPE']:.2f}%")

    rows = [
        _report_row("기존 방식", "A", "고정 val(24H1), test=24H2(evaluate_pipeline 기준)",
                    1, [a_val_m["WAPE"]], [a_val_m["RMSE"]], a_test_m),
        _report_row("기존 방식", "B", "53주 walk-forward(재학습 없음)",
                    len(b_val_wape), b_val_wape, b_val_rmse, b_summary),
    ]
    return rows


# ---------------------------------------------------------------------------
# Option 1 / Option 2 (A센터, Expanding Window CV, 진짜 fold별 재학습)
# ---------------------------------------------------------------------------

_a_final_cache = None


def _train_a_final(a_full: pd.DataFrame):
    global _a_final_cache
    if _a_final_cache is not None:
        return _a_final_cache
    print("  [A 최종모델] 2021-01~2023-12 전체 재학습(Option1/Option2 공유)")
    window = cc.slice_by_date(a_full, folds.A_FINAL_TRAIN_START, folds.A_FINAL_TRAIN_END)
    window = cc.apply_boundary_filter(window, folds.A_FINAL_TRAIN_END, horizon_weeks=1)
    window = window[window[TARGET_COL].notna()]
    train_fit, es_val = cc.carve_internal_es_val(window)
    feature_cols = cc.get_feature_cols(train_fit)
    cls_b, reg_b = cc.train_hurdle(train_fit, es_val, feature_cols, label="A_final")

    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cc.save_model_bundle(MODEL_OUT_DIR / "a_final_cv_cls.pkl", cls_b["model"], feature_cols)
    cc.save_model_bundle(MODEL_OUT_DIR / "a_final_cv_reg.pkl", reg_b["model"], feature_cols)

    test_df = cc.slice_by_date(a_full, folds.FINAL_TEST_START, folds.FINAL_TEST_END)
    test_df = test_df[test_df[TARGET_COL].notna()]
    test_m = cc.predict_and_score(test_df, cls_b, reg_b)
    print(f"  [A 최종모델] 2024 전체 Test WAPE={test_m['WAPE']:.2f}%")
    _a_final_cache = test_m
    return test_m


def run_option_a(option_folds: list[dict], option_label: str, cv_method: str,
                  fold_detail_rows: list[dict]) -> dict:
    print("=" * 80)
    print(f"[{option_label}] A센터 {cv_method} — fold별 실제 재학습")
    a_full = cc.load_a_full()

    val_wape, val_rmse = [], []
    for f in option_folds:
        print(f"  -- fold {f['fold']}: train {f['train_start']}~{f['train_end']} "
              f"-> val {f['val_start']}~{f['val_end']}")
        train_window = cc.slice_by_date(a_full, f["train_start"], f["train_end"])
        train_window = cc.apply_boundary_filter(train_window, f["train_end"], horizon_weeks=1)
        train_window = train_window[train_window[TARGET_COL].notna()]
        train_fit, es_val = cc.carve_internal_es_val(train_window)
        feature_cols = cc.get_feature_cols(train_fit)
        cls_b, reg_b = cc.train_hurdle(train_fit, es_val, feature_cols,
                                        label=f"{option_label} fold{f['fold']}")

        val_df = cc.slice_by_date(a_full, f["val_start"], f["val_end"])
        val_df = val_df[val_df[TARGET_COL].notna()]
        m = cc.predict_and_score(val_df, cls_b, reg_b)
        print(f"     val WAPE={m['WAPE']:.2f}% RMSE={m['RMSE']:.2f}")
        val_wape.append(m["WAPE"])
        val_rmse.append(m["RMSE"])
        fold_detail_rows.append({"experiment": option_label, "center": "A", "fold": f["fold"],
                                  "val_start": f["val_start"], "WAPE": m["WAPE"], "RMSE": m["RMSE"]})

    test_m = _train_a_final(a_full)
    return _report_row(option_label, "A", cv_method, len(option_folds), val_wape, val_rmse, test_m)


# ---------------------------------------------------------------------------
# Option 3 (A+B Pooling, 24개월 Rolling Window, B만 Val)
# ---------------------------------------------------------------------------

def run_option3(fold_detail_rows: list[dict]) -> dict:
    print("=" * 80)
    print("[Option3] A+B Pooling + 24개월 Rolling Window — fold별 실제 재학습, Val=B만")
    a_full = cc.load_a_full()
    b_full = cc.load_b_full()
    tuned_reg = cc.load_tuned_params("regressor")
    b_weight = tuned_reg["b_center_weight"] if tuned_reg else None
    print(f"  B sample_weight: {b_weight if b_weight is not None else '(튜닝값 없음, 1.0 균등)'}")

    val_wape, val_rmse = [], []
    for f in folds.OPTION3_FOLDS:
        print(f"  -- fold {f['fold']}: train(A+B) {f['train_start']}~{f['train_end']} "
              f"-> val(B) {f['val_start']}~{f['val_end']}")
        a_window = cc.slice_by_date(a_full, f["train_start"], f["train_end"])
        b_window = cc.slice_by_date(b_full, f["train_start"], f["train_end"])
        pooled = pd.concat([a_window, b_window], ignore_index=True)
        pooled = cc.apply_boundary_filter(pooled, f["train_end"], horizon_weeks=1)
        pooled = pooled[pooled[TARGET_COL].notna()]
        train_fit, es_val = cc.carve_internal_es_val(pooled)
        feature_cols = cc.get_feature_cols(train_fit)
        cls_b, reg_b = cc.train_hurdle(train_fit, es_val, feature_cols,
                                        label=f"option3 fold{f['fold']}", b_weight=b_weight)

        val_df = cc.slice_by_date(b_full, f["val_start"], f["val_end"])
        val_df = val_df[val_df[TARGET_COL].notna()]
        m = cc.predict_and_score(val_df, cls_b, reg_b)
        print(f"     val(B) WAPE={m['WAPE']:.2f}% RMSE={m['RMSE']:.2f} (n={m['n']:,})")
        val_wape.append(m["WAPE"])
        val_rmse.append(m["RMSE"])
        fold_detail_rows.append({"experiment": "option3", "center": "B", "fold": f["fold"],
                                  "val_start": f["val_start"], "WAPE": m["WAPE"], "RMSE": m["RMSE"]})

    print("  [B pooled 최종모델] 2021-01~2023-12 A+B 전량 재학습(rolling 제한 없음)")
    a_final_window = cc.slice_by_date(a_full, folds.AB_POOLED_FINAL_TRAIN_START, folds.AB_POOLED_FINAL_TRAIN_END)
    b_final_window = cc.slice_by_date(b_full, folds.AB_POOLED_FINAL_TRAIN_START, folds.AB_POOLED_FINAL_TRAIN_END)
    pooled_final = pd.concat([a_final_window, b_final_window], ignore_index=True)
    pooled_final = cc.apply_boundary_filter(pooled_final, folds.AB_POOLED_FINAL_TRAIN_END, horizon_weeks=1)
    pooled_final = pooled_final[pooled_final[TARGET_COL].notna()]
    train_fit, es_val = cc.carve_internal_es_val(pooled_final)
    feature_cols = cc.get_feature_cols(train_fit)
    cls_b, reg_b = cc.train_hurdle(train_fit, es_val, feature_cols, label="B_pooled_final", b_weight=b_weight)

    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cc.save_model_bundle(MODEL_OUT_DIR / "b_pooled_final_cls.pkl", cls_b["model"], feature_cols)
    cc.save_model_bundle(MODEL_OUT_DIR / "b_pooled_final_reg.pkl", reg_b["model"], feature_cols)

    b_test_df = cc.slice_by_date(b_full, folds.FINAL_TEST_START, folds.FINAL_TEST_END)
    b_test_df = b_test_df[b_test_df[TARGET_COL].notna()]
    test_m = cc.predict_and_score(b_test_df, cls_b, reg_b)
    print(f"  [B pooled 최종모델] 2024 전체 Test WAPE={test_m['WAPE']:.2f}%")

    return _report_row("옵션3", "B", "A+B Pooling 24개월 Rolling", len(folds.OPTION3_FOLDS),
                        val_wape, val_rmse, test_m)


# ---------------------------------------------------------------------------

def write_reports(rows: list[dict], fold_detail_rows: list[dict]):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_df = pd.DataFrame(rows)
    report_df.to_csv(REPORT_DIR / "cv_comparison_report.csv", index=False, encoding="utf-8-sig")

    md_lines = [
        "| " + " | ".join(report_df.columns) + " |",
        "| " + " | ".join(["---"] * len(report_df.columns)) + " |",
    ]
    for _, r in report_df.iterrows():
        md_lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    md_lines.append("")
    md_lines.append(
        "각주: '기존 방식' A행의 2024 Test는 이 실험 공통 정의(2024년 전체)로 재계산한 값으로, "
        "기존 evaluate_pipeline.py가 보고하는 A_test(2024 H2만)와는 다른 숫자입니다. "
        "'기존 방식' 모델은 early-stopping에 2024 H1(A val)을 이미 참고했으므로 Option1/2와 "
        "완전히 공정한 비교는 아닙니다. Option3 Fold1의 B val(2023-01~06)은 pre 레짐(원래 "
        "프로덕션이 품질 이슈로 제외해온 구간) 데이터를 포함합니다."
    )
    (REPORT_DIR / "cv_comparison_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    if fold_detail_rows:
        pd.DataFrame(fold_detail_rows).to_csv(REPORT_DIR / "fold_detail.csv", index=False, encoding="utf-8-sig")

    print()
    print("=" * 80)
    print("[비교 리포트]")
    print(report_df.to_string(index=False))
    print(f"\n저장 완료 -> {REPORT_DIR}/cv_comparison_report.{{csv,md}}, fold_detail.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="all",
                         choices=["existing", "option1", "option2", "option3", "all"])
    args = parser.parse_args()

    rows: list[dict] = []
    fold_detail_rows: list[dict] = []

    if args.mode in ("existing", "all"):
        rows.extend(run_existing(fold_detail_rows))
    if args.mode in ("option1", "all"):
        rows.append(run_option_a(folds.OPTION1_FOLDS, "옵션1", "6개월 Expanding", fold_detail_rows))
    if args.mode in ("option2", "all"):
        rows.append(run_option_a(folds.OPTION2_FOLDS, "옵션2", "3개월 Expanding", fold_detail_rows))
    if args.mode in ("option3", "all"):
        rows.append(run_option3(fold_detail_rows))

    write_reports(rows, fold_detail_rows)


if __name__ == "__main__":
    main()
