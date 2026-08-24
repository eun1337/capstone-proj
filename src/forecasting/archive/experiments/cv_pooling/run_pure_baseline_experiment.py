"""
run_pure_baseline_experiment.py
"기존 방식(프로덕션) vs Pooled 계열"의 공정 비교를 위해, 양쪽 모두 2024 데이터를
어떤 단계(하이퍼파라미터 탐색 + early-stopping)에도 전혀 참조하지 않는 "Pure"
버전으로 다시 학습해 동일 조건에서 비교한다.

배경(중요한 정정): 이전 실험(run_option3_rolling_isolation.py)에서 "Pooled Full
History"/"24M Rolling"은 early-stopping validation만 내부 마지막 4주로 깨끗하게
했을 뿐, LightGBM 하이퍼파라미터 자체(num_leaves/learning_rate/scale_pos_weight_
multiplier/b_center_weight)는 use_pure=True를 켜지 않아 여전히 tune_hyperparameters.py가
A의 2024 H1 val로 탐색한 기존 best_hyperparams.json 값을 그대로 썼다. 즉 "Pooled
계열은 100% 순수"라는 전제는 부정확했다 — 이 스크립트에서 세 모델(기존 방식 재현체 +
Pooled Full + Rolling24) 모두 use_pure=True(hpo_pure.py 산출물, best_hyperparams_pure.json)로
통일해 다시 학습한다.

세 모델:
    1) 기존 방식 Pure Baseline: train_base_model.py와 동일한 학습 레시피(단일 통합
       Hurdle, feature_table_final.parquet의 split=='train' 행 = A 2021~2023 + B
       2023-07~12(post 레짐만) pooled) + common.apply_target_date_boundary_filter로
       동일 경계필터 적용. 유일한 차이는 (a) 하이퍼파라미터를 pure로, (b) early-stop
       val을 A 2024H1 대신 학습구간 마지막 4주 내부 val로 바꾼 것.
    2) A+B Pooled Full History Pure: 2021~2023 전체(B_full, pre+post 포함) pooled,
       use_pure=True.
    3) A+B Pooled 24개월 Rolling Pure: 2022~2023, use_pure=True.

세 모델 모두 동일한 평가셋(B_full_feat.parquet의 2024-01-01~2024-12-31 슬라이스,
n=690,184)에서 채점해 기존 leaky 버전(88.21%/94.32%/93.81%)과 나란히 비교한다.

python run_pure_baseline_experiment.py

산출물: data/ml/cv_experiment/models/{existing_pure,pooled_full_pure,pooled_rolling24_pure}_{cls,reg}.pkl,
        data/ml/cv_experiment/reports/pure_baseline_comparison_report.{csv,md}
프로덕션 파일(base_model_cls/reg.pkl, feature_table_final.parquet 등)은 read-only로만 사용한다.
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as cc  # noqa: E402
import folds  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(BASE_DIR / "src" / "forecasting" / "archive" / "models_hurdle"))
from common import (  # noqa: E402
    BASE_CLS_MODEL_PATH,
    BASE_MODEL_PATH,
    CENTER_COL,
    FEATURE_TABLE_PATH,
    TARGET_COL,
    apply_target_date_boundary_filter,
)

REPORT_DIR = cc.CV_DIR / "reports"
MODEL_OUT_DIR = cc.CV_DIR / "models"

ROLLING_TRAIN_START = "2022-01-01"
ROLLING_TRAIN_END = folds.AB_POOLED_FINAL_TRAIN_END  # "2023-12-31"

ROUND = 3
METRIC_COLS = ["WAPE", "RMSE", "MAE", "MASE", "Bias(%)", "MAPE"]


def _pure_b_weight() -> float | None:
    tuned = cc.load_tuned_params("regressor", use_pure=True)
    return tuned["b_center_weight"] if tuned else None


def train_existing_recipe_pure():
    """train_base_model.py와 동일한 학습 데이터(feature_table_final.parquet의
    split=='train' = A 2021~2023 + B 2023-07~12 post레짐만)로, 하이퍼파라미터와
    early-stop val만 pure로 바꿔 재현한다."""
    print("=== [기존 방식 Pure Baseline] 학습 ===")
    df = pd.read_parquet(FEATURE_TABLE_PATH)
    train_df = df[df["split"] == "train"].copy()
    print(f"  원본 train 행수(A+B, 필터 전): {len(train_df):,} "
          f"(A={int((train_df[CENTER_COL] == 'A').sum()):,}, B={int((train_df[CENTER_COL] == 'B').sum()):,})")

    train_df = apply_target_date_boundary_filter(train_df, horizon_weeks=1)
    train_df = train_df[train_df[TARGET_COL].notna()]

    train_fit, es_val = cc.carve_internal_es_val(train_df)
    print(f"  내부 early-stop val(마지막 4주, A/B 공통): {len(es_val):,}행 / 나머지 학습: {len(train_fit):,}행 "
          f"(참고: 프로덕션은 여기 대신 A 2024H1 val을 씀 — 이게 이번에 없앤 leak)")

    feature_cols = cc.get_feature_cols(train_fit)
    b_weight = _pure_b_weight()
    print(f"  B sample_weight(pure): {b_weight}")

    cls_bundle, reg_bundle = cc.train_hurdle(
        train_fit, es_val, feature_cols, label="existing_recipe_pure", b_weight=b_weight, use_pure=True
    )
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cc.save_model_bundle(MODEL_OUT_DIR / "existing_pure_cls.pkl", cls_bundle["model"], feature_cols)
    cc.save_model_bundle(MODEL_OUT_DIR / "existing_pure_reg.pkl", reg_bundle["model"], feature_cols)
    return cls_bundle, reg_bundle


def _train_pooled_window_pure(a_full: pd.DataFrame, b_full: pd.DataFrame, train_start: str, train_end: str,
                               label: str, out_stub: str):
    print(f"=== [{label}] 학습 (use_pure=True) ===")
    print(f"  학습 구간: {train_start} ~ {train_end}")

    b_weight = _pure_b_weight()
    print(f"  B sample_weight(pure): {b_weight}")

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
        train_fit, es_val, feature_cols, label=label, b_weight=b_weight, use_pure=True
    )
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cc.save_model_bundle(MODEL_OUT_DIR / f"{out_stub}_cls.pkl", cls_bundle["model"], feature_cols)
    cc.save_model_bundle(MODEL_OUT_DIR / f"{out_stub}_reg.pkl", reg_bundle["model"], feature_cols)
    return cls_bundle, reg_bundle


def main():
    b_full = cc.load_b_full()
    test_df = cc.slice_by_date(b_full, folds.FINAL_TEST_START, folds.FINAL_TEST_END)
    test_df = test_df[test_df[TARGET_COL].notna()]
    print(f"[2024 Test] 평가 대상(B만): {len(test_df):,}행 "
          f"({test_df[cc.WEEK_COL].min().date()} ~ {test_df[cc.WEEK_COL].max().date()})")
    print()

    cls_ex, reg_ex = train_existing_recipe_pure()
    existing_pure_m = cc.predict_and_score(test_df, cls_ex, reg_ex)
    print(f"  [기존 방식 Pure] WAPE={existing_pure_m['WAPE']:.2f}% RMSE={existing_pure_m['RMSE']:.2f} "
          f"MAE={existing_pure_m['MAE']:.2f}\n")

    a_full = cc.load_a_full()

    cls_full, reg_full = _train_pooled_window_pure(
        a_full, b_full, folds.AB_POOLED_FINAL_TRAIN_START, folds.AB_POOLED_FINAL_TRAIN_END,
        label="Pooled_Full_History_Pure", out_stub="pooled_full_pure"
    )
    full_pure_m = cc.predict_and_score(test_df, cls_full, reg_full)
    print(f"  [Pooled Full History Pure] WAPE={full_pure_m['WAPE']:.2f}% RMSE={full_pure_m['RMSE']:.2f} "
          f"MAE={full_pure_m['MAE']:.2f}\n")

    cls_roll, reg_roll = _train_pooled_window_pure(
        a_full, b_full, ROLLING_TRAIN_START, ROLLING_TRAIN_END,
        label="Pooled_24M_Rolling_Pure", out_stub="pooled_rolling24_pure"
    )
    rolling_pure_m = cc.predict_and_score(test_df, cls_roll, reg_roll)
    print(f"  [Pooled 24M Rolling Pure] WAPE={rolling_pure_m['WAPE']:.2f}% RMSE={rolling_pure_m['RMSE']:.2f} "
          f"MAE={rolling_pure_m['MAE']:.2f}\n")

    print("=== [참고] 기존 방식(read-only 프로덕션 번들, leaky) 동일 데이터 평가 ===")
    existing_cls = cc.load_model_bundle(BASE_CLS_MODEL_PATH)
    existing_reg = cc.load_model_bundle(BASE_MODEL_PATH)
    existing_leaky_m = cc.predict_and_score(test_df, existing_cls, existing_reg)
    print(f"  [기존 방식(leaky, 프로덕션 실배포)] WAPE={existing_leaky_m['WAPE']:.2f}%")

    rows = [
        {"실험 구분": "기존 방식(leaky, 프로덕션 실배포)", **{k: round(existing_leaky_m[k], ROUND) for k in METRIC_COLS}, "n": existing_leaky_m["n"]},
        {"실험 구분": "기존 방식 재현체(Pure: 동일 데이터, pure HP+es_val)", **{k: round(existing_pure_m[k], ROUND) for k in METRIC_COLS}, "n": existing_pure_m["n"]},
        {"실험 구분": "A+B Pooled Full History(Pure)", **{k: round(full_pure_m[k], ROUND) for k in METRIC_COLS}, "n": full_pure_m["n"]},
        {"실험 구분": "A+B Pooled 24개월 Rolling(Pure)", **{k: round(rolling_pure_m[k], ROUND) for k in METRIC_COLS}, "n": rolling_pure_m["n"]},
    ]
    report_df = pd.DataFrame(rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_DIR / "pure_baseline_comparison_report.csv", index=False, encoding="utf-8-sig")
    md_lines = [
        "| " + " | ".join(report_df.columns) + " |",
        "| " + " | ".join(["---"] * len(report_df.columns)) + " |",
    ]
    for _, r in report_df.iterrows():
        md_lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    md_lines.append("")
    md_lines.append(
        f"각주: 네 모델 모두 동일 평가셋(B_full_feat.parquet 2024-01-01~2024-12-31 슬라이스, n={existing_pure_m['n']:,})에서 "
        "채점했다. '기존 방식(leaky)'만 read-only 프로덕션 번들이고 나머지 셋은 이번에 신규 학습했다. "
        "'기존 방식 재현체(Pure)'는 train_base_model.py와 정확히 같은 학습 데이터(feature_table_final.parquet의 "
        "split=='train': A 2021~2023 + B 2023-07~12 post레짐만, common.apply_target_date_boundary_filter로 동일 "
        "경계필터)를 쓰되 하이퍼파라미터(best_hyperparams_pure.json)와 early-stop val(학습구간 마지막 4주, "
        "A 2024H1 미참조)만 pure로 바꿨다. 'Pooled Full/Rolling(Pure)'도 동일 하이퍼파라미터 소스로 통일해 "
        "이전 실험(leaky 하이퍼파라미터를 쓴 option3_rolling_isolation_report의 94.32%/93.81%)과 대비된다."
    )
    (REPORT_DIR / "pure_baseline_comparison_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print()
    print("=" * 80)
    print("[Pure 조건 통일 후 4-way 비교 — 2024 Test]")
    print(report_df.to_string(index=False))
    print(f"\n저장 완료 -> {REPORT_DIR}/pure_baseline_comparison_report.{{csv,md}}")


if __name__ == "__main__":
    main()
