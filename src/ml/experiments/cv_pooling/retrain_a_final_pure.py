"""
retrain_a_final_pure.py
A 최종모델(a_final_cv_cls/reg.pkl)을 hpo_pure.py가 만든 "2024 완전 미유출"
하이퍼파라미터(best_hyperparams_pure.json)로 재학습한다. 학습 데이터 범위/내부
early-stop val 구성은 이전 세션(run_cv_experiment.py의 _train_a_final)과 완전히
동일 — 유일한 차이는 cc.train_hurdle(..., use_pure=True)로 하이퍼파라미터
소스만 바꾼 것.

data/ml/cv_experiment/models/a_final_cv_{cls,reg}.pkl을 덮어쓴다(이 실험 전용
산출물이라 덮어써도 안전 — 프로덕션 파일 아님).

python retrain_a_final_pure.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as cc  # noqa: E402
import folds  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BASE_DIR / "src" / "ml" / "models"))
from common import TARGET_COL  # noqa: E402

MODEL_OUT_DIR = cc.CV_DIR / "models"


def main():
    a_full = cc.load_a_full()
    window = cc.slice_by_date(a_full, folds.A_FINAL_TRAIN_START, folds.A_FINAL_TRAIN_END)
    window = cc.apply_boundary_filter(window, folds.A_FINAL_TRAIN_END, horizon_weeks=1)
    window = window[window[TARGET_COL].notna()]
    train_fit, es_val = cc.carve_internal_es_val(window)
    feature_cols = cc.get_feature_cols(train_fit)

    print(f"[A 최종모델 재학습 - pure] train={len(train_fit):,}행 / 내부val={len(es_val):,}행")
    cls_b, reg_b = cc.train_hurdle(train_fit, es_val, feature_cols, label="A_final_pure", use_pure=True)

    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cc.save_model_bundle(MODEL_OUT_DIR / "a_final_cv_cls.pkl", cls_b["model"], feature_cols)
    cc.save_model_bundle(MODEL_OUT_DIR / "a_final_cv_reg.pkl", reg_b["model"], feature_cols)

    test_df = cc.slice_by_date(a_full, folds.FINAL_TEST_START, folds.FINAL_TEST_END)
    test_df = test_df[test_df[TARGET_COL].notna()]
    test_m = cc.predict_and_score(test_df, cls_b, reg_b)
    print(f"[A 최종모델 재학습 - pure] 2024 전체 Test WAPE={test_m['WAPE']:.2f}% "
          f"RMSE={test_m['RMSE']:.2f} MAE={test_m['MAE']:.2f} MAPE={test_m['MAPE']:.2f}%")
    print("(참고: 리키지 있던 이전 버전의 2024 Test WAPE는 81.22%였음 — 위 수치와 비교용)")


if __name__ == "__main__":
    main()
