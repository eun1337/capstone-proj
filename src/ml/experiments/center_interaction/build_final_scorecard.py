"""
build_final_scorecard.py
Hurdle 모델 B센터 개선 최종 제출용 성적표 생성 (P4, P8).

배경: run_threshold_expansion.py의 threshold 스윕(0.20~0.75) 결과 중 3개 후보
(th=0.46 품절방어/0.50 채택안/0.54 오차최적)를 최종 의사결정용으로 정리하는 단계에서
아래 두 가지 이슈가 발견되어 팀 협의로 처리 방침을 확정함(2026-08-02):

    1) Baseline(구버전, 상호작용 피처 반영 전) 모델의 B센터 지표가 평가 방식(공식
       per-fold evaluate_b_walkforward vs 이 실험의 배치 방식)에 따라 다르게 나옴
       (WAPE 89.56% vs 85.13%). 신규(현재 프로덕션) 모델은 두 방식이 완전히 동일한
       값(84.85%)을 줘서 방식 차이의 영향이 없음이 확인됐으나, 구버전 모델만 왜
       차이가 나는지는 근본 원인을 완전히 규명하지 못함(lightgbm 버전/pickle
       호환성 등으로 추정). 팀 결정: 후보군(신규 모델)과 동일 조건(apples-to-apples)
       비교를 위해 Baseline도 배치 방식 수치(85.13%)를 채택하고, 공식 파이프라인
       수치(89.56%)는 각주로 병기한다.
    2) 지금까지 이 세션에서 계산해온 모든 WAPE/RMSE/MAE/MASE는 [SKU × 주] grain
       (행 단위 오차를 그대로 집계)이었는데, 원래 P4 체크리스트는 "[SKU×주] 예측치를
       [센터×주]로 합산(Roll-up)하여 평가"를 요구함. 두 grain은 수치가 크게 다르다
       (SKU×주는 SKU별 과다/과소 오차가 상쇄되지 않아 WAPE가 크게 부풀려짐; Bias(%)는
       선형 집계라 grain과 무관하게 동일). 팀 결정: 최종 성적표에 두 grain을 모두
       병기해 이력을 투명하게 남긴다.

산출물: data/ml/experiments/center_interaction_ablation/reports/final_scorecard.csv
        (+ 콘솔에 사람이 읽기 좋은 표로 출력)
프로덕션 파일은 read-only로만 참조하고 수정하지 않는다.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BASE_DIR / "src" / "ml" / "models"))
sys.path.insert(0, str(BASE_DIR / "src" / "ml" / "feature_engineering"))

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
OUT_PATH = REPORT_DIR / "final_scorecard.csv"

BACKUP_MODEL_DIR = BASE_DIR / "data" / "ml" / "models" / "_backup_before_center_interaction_20260802"
BACKUP_SPLIT_DIR = BASE_DIR / "data" / "ml" / "splits" / "_backup_before_center_interaction_20260802"
BEFORE_CLS_PATH = BACKUP_MODEL_DIR / "base_model_cls.pkl"
BEFORE_REG_PATH = BACKUP_MODEL_DIR / "base_model_reg.pkl"
BEFORE_FEATURE_TABLE_PATH = BACKUP_SPLIT_DIR / "feature_table_final.parquet"

# 각주: 구버전 모델의 공식 per-fold 파이프라인(evaluate_pipeline.evaluate_b_walkforward) 수치.
# 신규 모델은 이 값과 배치 방식이 완전히 일치함(84.854)이 확인됐으나, 구버전 모델만 차이가 남
# (원인 미규명, lightgbm 버전/pickle 호환성 추정) — 팀 결정으로 배치 수치를 채택 대표값으로 사용.
BEFORE_B_OFFICIAL_PERFOLD_WAPE = 89.563
BEFORE_B_OFFICIAL_PERFOLD_BIAS = -7.140
FOOTNOTE = (
    f"Baseline B는 배치 평가 방식 채택(후보군과 apples-to-apples). "
    f"참고: 공식 per-fold 파이프라인(evaluate_pipeline.py) 기준으로는 WAPE={BEFORE_B_OFFICIAL_PERFOLD_WAPE:.2f}%, "
    f"Bias={BEFORE_B_OFFICIAL_PERFOLD_BIAS:+.2f}% (신규 모델은 두 방식이 완전히 일치하나 구버전만 차이 발생, 원인 미규명)"
)

CANDIDATES = [
    ("후보2 (품절방어) [최종 채택]", 0.46),
    ("후보1 (당초 채택안)", 0.50),
    ("후보3 (오차최적)", 0.54),
]


def load_center_eval_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return a_test, b_eval


def eval_one(label: str, threshold: float, eval_df: pd.DataFrame, center: str,
             cls_bundle: dict, reg_bundle: dict, note: str) -> list[dict]:
    """SKU×주 grain과 센터×주 Roll-up grain 두 행을 반환한다."""
    y_pred = predict_base_hurdle(eval_df, cls_bundle, reg_bundle, mode="hard", threshold=threshold)
    y_true = eval_df[TARGET_COL].to_numpy()
    naive = eval_df["qty"].to_numpy()

    m_row = compute_metrics(y_true, y_pred, naive)
    row_sku = {
        "구분": label, "Threshold": threshold, "센터": center, "Grain": "SKU x 주",
        "n": len(eval_df), **{k: round(v, 3) for k, v in m_row.items()}, "비고": note,
    }

    weekly = pd.DataFrame({WEEK_COL: eval_df[WEEK_COL].to_numpy(), "y_true": y_true, "y_pred": y_pred, "naive": naive})
    weekly = weekly.groupby(WEEK_COL, as_index=False).sum(numeric_only=True)
    m_roll = compute_metrics(weekly["y_true"].to_numpy(), weekly["y_pred"].to_numpy(), weekly["naive"].to_numpy())
    row_roll = {
        "구분": label, "Threshold": threshold, "센터": center, "Grain": "센터 x 주(Roll-up)",
        "n": len(weekly), **{k: round(v, 3) for k, v in m_roll.items()}, "비고": note,
    }
    return [row_sku, row_roll]


def main():
    print("=" * 80)
    print("[1/2] Baseline(구버전, 상호작용 피처 반영 전) 평가 — 배치 방식")
    for p in (BEFORE_CLS_PATH, BEFORE_REG_PATH, BEFORE_FEATURE_TABLE_PATH):
        if not p.exists():
            raise FileNotFoundError(f"{p} 없음")
    before_df = pd.read_parquet(BEFORE_FEATURE_TABLE_PATH)
    before_cls = load_model_bundle(BEFORE_CLS_PATH)
    before_reg = load_model_bundle(BEFORE_REG_PATH)
    before_threshold = before_cls.get("threshold")
    before_a_eval, before_b_eval = load_center_eval_frames(before_df)

    rows = []
    rows += eval_one("Baseline", before_threshold, before_a_eval, "A", before_cls, before_reg,
                      "기존 프로덕션 모델 기준점")
    rows += eval_one("Baseline", before_threshold, before_b_eval, "B", before_cls, before_reg, FOOTNOTE)

    print()
    print("=" * 80)
    print("[2/2] 후보 1~3(신규 프로덕션, 상호작용 피처 2종 반영) 평가 — Hard 모드, threshold 0.46/0.50/0.54")
    prod_df = pd.read_parquet(FEATURE_TABLE_PATH)
    prod_cls = load_model_bundle(BASE_CLS_MODEL_PATH)
    prod_reg = load_model_bundle(BASE_MODEL_PATH)
    prod_a_eval, prod_b_eval = load_center_eval_frames(prod_df)

    notes = {
        0.46: (
            "[최종 채택, 2026-08-02] Roll-up(P4 공식) 기준 B WAPE 18.03%->16.55%(-1.48%p) "
            "실개선 + A Bias +0.57%로 거의 무편향. SKU x 주 grain 기준으로는 WAPE가 후보1/3보다 "
            "높아 보이나, 이는 SKU별 오차가 상쇄되지 않는 grain 특성상의 착시로 판단."
        ),
        0.50: (
            "[당초 채택 -> 철회] SKU x 주 grain에서는 우수해 보였으나 Roll-up(P4 공식) 기준 "
            "B WAPE가 Baseline보다 악화(18.03%->18.92%)됨을 확인해 최종안에서 제외."
        ),
        0.54: "전체 오차(WAPE) 최소화 중심(SKU x 주 grain) / Roll-up 기준으로는 4개 구분 중 최하위.",
    }
    for label, th in CANDIDATES:
        rows += eval_one(label, th, prod_a_eval, "A", prod_cls, prod_reg, notes[th])
        rows += eval_one(label, th, prod_b_eval, "B", prod_cls, prod_reg, notes[th])

    report = pd.DataFrame(rows)
    col_order = ["구분", "Threshold", "센터", "Grain", "n", "RMSE", "MAE", "WAPE", "MASE", "Bias(%)", "비고"]
    report = report[col_order]

    print()
    print("=" * 80)
    print("[최종 성적표] SKU x 주 grain")
    print(report[report["Grain"] == "SKU x 주"].drop(columns="Grain").to_string(index=False))
    print()
    print("[최종 성적표] 센터 x 주 Roll-up grain (P4 체크리스트 기준)")
    print(report[report["Grain"] == "센터 x 주(Roll-up)"].drop(columns="Grain").to_string(index=False))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[저장 완료] {OUT_PATH} (총 {len(report)}행 = 4개 구분 x A/B x 2 grain)")
    print(f"[각주] {FOOTNOTE}")


if __name__ == "__main__":
    main()
