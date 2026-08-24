"""
run_threshold_expansion.py
Hurdle 모델 B센터 개선 2차 실험: threshold 탐색 범위 확장(0.20~0.75, step 0.02).

배경/이력:
    - 1차: run_center_interaction_ablation.py의 Ablation1(상호작용 피처 2종 +
      best_hyperparams.json + Hard 모드)에서 채택된 threshold(0.50)가 탐색 그리드
      상한(0.20~0.50, step 0.05)에 걸려 실제 최적값이 더 클 가능성이 있었음.
    - 2차: 이 스크립트가 그리드를 0.20~0.75(step 0.02)로 넓혀 "WAPE 최소" 기준으로
      단일 threshold(0.74)를 자동 채택해봤으나, threshold를 올릴수록 WAPE는 착시적으로
      좋아지고 Bias(과소추정)는 폭주(-33%대까지)하는 트레이드오프가 드러남. "WAPE 최소"
      단일 지표로 threshold를 자동 결정하는 것 자체가 근거 없는 기준이라는 피드백에 따라,
      이 스크립트는 더 이상 "최적 threshold"를 자동 선택/채택하지 않는다. 대신 threshold
      전 구간(0.20~0.75)에서 A/B 각각의 공식 평가셋(A: A_test 2024-07~12 홀드아웃, B:
      B_walkforward_folds.json 53-fold Pool) 기준 WAPE/Bias(%) 전체 변화 양상을 표로
      남겨, threshold는 사람이 직접 트레이드오프를 보고 정한다.
    - 3차(현재): 이미 center_qty_lag1_inter/center_temp_inter 2종이 프로덕션에 반영되고
      base_model_cls/reg.pkl이 그 피처로 재학습된 상태(2026-08-02). 따라서 이 스크립트가
      "Ablation" 대상으로 스윕하는 모델은 별도 실험 사본이 아니라 현재 배포된 프로덕션
      모델 자체이며, "Before" 비교 기준은 상호작용 피처 반영 전 구버전 모델
      (data/ml/models/_backup_before_center_interaction_20260802/에 백업됨)에서 가져온다.

핵심 설계 (팀 협의 확정):
    1) 분류기/회귀기는 재학습하지 않는다 — 이미 학습되어 저장된 프로덕션/백업 모델을
       그대로 불러와 추론 확률에 대해서만 threshold grid를 스윕한다.
    2) 원래 요청된 3번째 상호작용 피처(center_dayofweek_inter = dayofweek * center_is_B)는
       제외한다 — 이 프로젝트 feature table은 주간(weekly) grain이라 week_st가 전 행에서
       항상 월요일(dayofweek==0 고정)이라 이 피처를 만들면 전 행이 0인 무의미한 상수
       컬럼이 된다(실제 데이터로 확인, 팀 협의로 제외 확정).
    3) MLflow는 이 프로젝트에 설치/연동되어 있지 않다(requirements.txt 미포함, 코드베이스
       전체에 import 없음) — CSV로만 기록한다.
    4) 비교 대상:
         - Baseline(before): 상호작용 피처 반영 전 구버전 모델(백업, threshold 원래값
           고정 1행) + Hard 모드 + 구버전 feature_table_final.parquet(백업)
         - Production(현재, sweep): 상호작용 피처 2종 반영된 현재 프로덕션 모델
           (base_model_cls/reg.pkl) + Hard 모드 + 현재 feature_table_final.parquet,
           threshold=0.20~0.75(step 0.02) 전 구간을 A/B 각각 스윕
    5) 산출물은 data/ml/experiments/center_interaction_ablation/reports/
       threshold_expansion_report.csv에 저장한다(매 threshold마다 A/B 행 포함, "최적값"
       자동 채택 컬럼/로직 없음 — 사람이 그래프/표를 보고 직접 결정).
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(BASE_DIR / "src" / "forecasting" / "archive" / "models_hurdle"))
sys.path.insert(0, str(BASE_DIR / "src" / "forecasting" / "archive" / "feature_engineering"))

from common import (  # noqa: E402
    BASE_CLS_MODEL_PATH,
    BASE_MODEL_PATH,
    CENTER_COL,
    FEATURE_TABLE_PATH,
    TARGET_COL,
    WALKFORWARD_FOLDS_PATH,
    WEEK_COL,
    compute_metrics,
    load_model_bundle,
    predict_base_hurdle,
)

EXP_DIR = BASE_DIR / "data" / "ml" / "experiments" / "center_interaction_ablation"
REPORT_DIR = EXP_DIR / "reports"
NEW_REPORT_PATH = REPORT_DIR / "threshold_expansion_report.csv"

# 구버전(상호작용 피처 반영 전) 프로덕션 아티팩트 백업 — Before 비교 기준
BACKUP_MODEL_DIR = BASE_DIR / "data" / "ml" / "models" / "_backup_before_center_interaction_20260802"
BACKUP_SPLIT_DIR = BASE_DIR / "data" / "ml" / "splits" / "_backup_before_center_interaction_20260802"
BEFORE_CLS_PATH = BACKUP_MODEL_DIR / "base_model_cls.pkl"
BEFORE_REG_PATH = BACKUP_MODEL_DIR / "base_model_reg.pkl"
BEFORE_FEATURE_TABLE_PATH = BACKUP_SPLIT_DIR / "feature_table_final.parquet"

THRESHOLD_GRID = np.round(np.arange(0.20, 0.751, 0.02), 2)

REPORT_COLS = ["model", "threshold", "center", "n", "RMSE", "MAE", "WAPE", "MASE", "Bias(%)"]


def load_center_eval_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A: A_test(2024-07~12 홀드아웃), B: B_walkforward_folds.json 53-fold Pool 구간 합집합.
    evaluate_pipeline.evaluate_a/evaluate_b_walkforward와 동일한 행 집합을 만들되,
    threshold grid 스윕 중 매번 fold를 순회/재필터링하지 않도록 한 번만 계산해 재사용한다
    (스윕 자체는 predict_base_hurdle만 반복 호출 — evaluate_a/evaluate_b_walkforward는
    호출당 fold 표를 전부 print해 28회 반복 시 로그가 과도해지므로 여기서는 쓰지 않음)."""
    a_test = df[(df[CENTER_COL] == "A") & (df["split"] == "test")].copy()
    a_test = a_test[a_test[TARGET_COL].notna()]

    with open(WALKFORWARD_FOLDS_PATH, encoding="utf-8") as f:
        folds = json.load(f)
    b_pool = df[(df[CENTER_COL] == "B") & (df["split"] == "pool")].copy()
    combined_mask = pd.Series(False, index=b_pool.index)
    for (_train_start, _train_end, val_start, val_end) in folds:
        combined_mask |= (b_pool[WEEK_COL] >= val_start) & (b_pool[WEEK_COL] <= val_end)
    b_eval = b_pool[combined_mask]
    b_eval = b_eval[b_eval[TARGET_COL].notna()]

    print(f"  A_test 평가 행수: {len(a_test):,} / B walk-forward(53-fold 합집합) 평가 행수: {len(b_eval):,}")
    return a_test, b_eval


def sweep_thresholds(model_label: str, a_eval: pd.DataFrame, b_eval: pd.DataFrame,
                      cls_bundle: dict, reg_bundle: dict, grid: np.ndarray) -> list[dict]:
    """grid의 모든 threshold에 대해 A_test/B walk-forward 각각의 WAPE/MAE/RMSE/MASE/Bias(%)를
    계산한다. '최적값' 자동 선택 없음 — 전 구간을 표로 남겨 사람이 직접 비교하도록 한다."""
    rows = []
    for center, eval_df in (("A", a_eval), ("B", b_eval)):
        y_true = eval_df[TARGET_COL].to_numpy()
        naive_pred = eval_df["qty"].to_numpy()
        for th in grid:
            y_pred = predict_base_hurdle(eval_df, cls_bundle, reg_bundle, mode="hard", threshold=float(th))
            m = compute_metrics(y_true, y_pred, naive_pred)
            rows.append({
                "model": model_label, "threshold": float(th), "center": center, "n": len(eval_df),
                **{k: round(v, 3) for k, v in m.items()},
            })
        print(f"  [{model_label}] 센터 {center} threshold 스윕 완료({len(grid)}개 값)")
    return rows


def main():
    print("=" * 80)
    print("[1/2] Before: 상호작용 피처 반영 전 구버전 프로덕션 모델(백업) — threshold 원래값 1개만 평가")
    for p in (BEFORE_CLS_PATH, BEFORE_REG_PATH, BEFORE_FEATURE_TABLE_PATH):
        if not p.exists():
            raise FileNotFoundError(f"{p} 없음 — 백업이 없으면 Before 비교 기준을 만들 수 없음")
    before_df = pd.read_parquet(BEFORE_FEATURE_TABLE_PATH)
    before_cls_bundle = load_model_bundle(BEFORE_CLS_PATH)
    before_reg_bundle = load_model_bundle(BEFORE_REG_PATH)
    before_threshold = before_cls_bundle.get("threshold")
    print(f"  Before 모델 threshold: {before_threshold}")
    before_a_eval, before_b_eval = load_center_eval_frames(before_df)
    before_rows = sweep_thresholds(
        "Before (production, pre-interaction-feat)", before_a_eval, before_b_eval,
        before_cls_bundle, before_reg_bundle, np.array([before_threshold]),
    )

    print()
    print("=" * 80)
    print("[2/2] Production(현재): 상호작용 피처 2종 반영된 배포 모델 — threshold 0.20~0.75(step 0.02) 전 구간 스윕")
    prod_df = pd.read_parquet(FEATURE_TABLE_PATH)
    prod_cls_bundle = load_model_bundle(BASE_CLS_MODEL_PATH)
    prod_reg_bundle = load_model_bundle(BASE_MODEL_PATH)
    prod_a_eval, prod_b_eval = load_center_eval_frames(prod_df)
    prod_rows = sweep_thresholds(
        "Production (+center_qty_lag1_inter/center_temp_inter, 2026-08-02)", prod_a_eval, prod_b_eval,
        prod_cls_bundle, prod_reg_bundle, THRESHOLD_GRID,
    )

    report = pd.DataFrame(before_rows + prod_rows)[REPORT_COLS]

    print()
    print("=" * 80)
    print("[Production 모델 threshold 스윕 전체 — WAPE/Bias(%) 변화 양상, 자동 최적값 선택 없음]")
    for center in ("A", "B"):
        sub = report[(report["model"].str.startswith("Production")) & (report["center"] == center)]
        print(f"  --- 센터 {center} ---")
        print(sub[["threshold", "n", "WAPE", "Bias(%)", "RMSE", "MAE", "MASE"]].to_string(index=False))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(NEW_REPORT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[저장 완료] {NEW_REPORT_PATH} (Before 2행 + Production {len(THRESHOLD_GRID) * 2}행 = 총 {len(report)}행)")
    print("[모델 재학습/재저장 없음 — 이미 저장된 Before 백업/Production 배포 모델에 대한 threshold 스윕 평가만 수행]")


if __name__ == "__main__":
    main()
