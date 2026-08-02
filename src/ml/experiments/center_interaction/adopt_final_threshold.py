"""
adopt_final_threshold.py
Hurdle 모델 최종 threshold를 0.46(후보2, 품절방어)으로 확정 반영한다.

배경: build_final_scorecard.py로 Baseline/후보1(th=0.50)/후보2(th=0.46)/후보3(th=0.54)를
SKU x 주 grain과 센터 x 주 Roll-up grain(P4 체크리스트 공식 기준) 양쪽으로 비교한 결과,
SKU x 주 grain에서는 threshold를 올릴수록(0.46->0.54) B WAPE가 계속 좋아 보였지만
(오차 상쇄가 없는 grain이라 착시), 실제 재고/발주 단위인 센터 x 주 Roll-up grain에서는
순위가 뒤집혀 th=0.50(당시 채택안)이 오히려 Baseline보다 악화(B WAPE 18.03%->18.92%)됨을
확인했다. th=0.46이 Roll-up 기준 B WAPE를 18.03%->16.55%(-1.48%p)로 실제 개선하고, A센터도
Bias +0.57%로 거의 완벽한 무편향을 보여 최종 채택됨(2026-08-02, 팀 확정).

이 스크립트는 분류기/회귀기를 재학습하지 않는다 — 상호작용 피처/하이퍼파라미터가 이미
채택된 프로덕션 모델(base_model_cls/reg.pkl) 그대로이고, 바뀌는 것은 Hard 모드 판정
threshold 값 하나뿐이다(cls_bundle["threshold"]). 회귀기(base_model_reg.pkl)는 threshold와
무관하므로 건드리지 않는다.
"""

from pathlib import Path
import sys

import joblib

BASE_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BASE_DIR / "src" / "ml" / "models"))

from common import BASE_CLS_MODEL_PATH  # noqa: E402

FINAL_THRESHOLD = 0.46


def main():
    bundle = joblib.load(BASE_CLS_MODEL_PATH)
    old_threshold = bundle.get("threshold")
    bundle["threshold"] = float(FINAL_THRESHOLD)
    bundle["threshold_decision_note"] = (
        "2026-08-02 확정: 센터 x 주 Roll-up(P4 공식 기준) 평가에서 th=0.46이 B WAPE를 "
        "18.03%->16.55%(-1.48%p) 개선하고 A Bias +0.57%로 거의 무편향 — 최종 채택. "
        "SKU x 주 grain 기준으로는 th=0.50/0.54가 더 낮은 WAPE를 보였으나, 이는 SKU별 "
        "과다/과소 오차가 상쇄되지 않는 grain 특성상의 착시로 판단해 채택하지 않음."
    )
    joblib.dump(bundle, BASE_CLS_MODEL_PATH)
    print(f"[threshold 갱신 완료] {old_threshold} -> {FINAL_THRESHOLD}")
    print(f"  저장 위치: {BASE_CLS_MODEL_PATH}")


if __name__ == "__main__":
    main()
