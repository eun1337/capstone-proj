"""
folds.py
A/B 센터 CV 구조 개편 실험(기존 방식 vs Option1/2/3)의 fold 경계 정의.

기존 프로덕션 파이프라인(split_train_val_test.py의 generate_b_walkforward_folds)과
별개로, 이 실험 전용 fold 정의다. 폴드 수가 적어(2/4/2개) 날짜를 자동 생성하는
대신 스펙을 그대로 하드코딩한다 — 월말/윤년 등 날짜 산술로 생성할 때
생길 수 있는 오프바이원 버그보다 명시적 상수가 안전하다는 판단.

Option 1: A센터, 2021~2023년 내 반기(6개월) Expanding Window, 2-fold
Option 2: A센터, 2021~2023년 내 분기(3개월) Expanding Window, 4-fold
Option 3: B센터, A+B Pooling + 최근 24개월 Rolling Window, 2-fold
    (train=A+B pooled 구간, val=B만 — B의 pre 레짐(2023-07 이전)도
    포함하도록 확정함. build_b_full_history.py 참고)

공통: 2024-01-01~2024-12-31은 모든 옵션에서 CV에 전혀 쓰이지 않는
최종 holdout test로 별도 고정(run_cv_experiment.py에서 처리).
"""

OPTION1_FOLDS = [
    {"fold": 1, "train_start": "2021-01-01", "train_end": "2022-12-31",
     "val_start": "2023-01-01", "val_end": "2023-06-30"},
    {"fold": 2, "train_start": "2021-01-01", "train_end": "2023-06-30",
     "val_start": "2023-07-01", "val_end": "2023-12-31"},
]

OPTION2_FOLDS = [
    {"fold": 1, "train_start": "2021-01-01", "train_end": "2022-12-31",
     "val_start": "2023-01-01", "val_end": "2023-03-31"},
    {"fold": 2, "train_start": "2021-01-01", "train_end": "2023-03-31",
     "val_start": "2023-04-01", "val_end": "2023-06-30"},
    {"fold": 3, "train_start": "2021-01-01", "train_end": "2023-06-30",
     "val_start": "2023-07-01", "val_end": "2023-09-30"},
    {"fold": 4, "train_start": "2021-01-01", "train_end": "2023-09-30",
     "val_start": "2023-10-01", "val_end": "2023-12-31"},
]

# train_start는 val_start 기준 24개월 이전으로 고정(rolling window).
# val은 B만, train은 A+B pooled.
OPTION3_FOLDS = [
    {"fold": 1, "train_start": "2021-01-01", "train_end": "2022-12-31",
     "val_start": "2023-01-01", "val_end": "2023-06-30"},
    {"fold": 2, "train_start": "2021-07-01", "train_end": "2023-06-30",
     "val_start": "2023-07-01", "val_end": "2023-12-31"},
]

# 2026-08-06 추가: "A센터 CV를 왜 2023년 안에서만 했는지"를 검증하기 위한 확장 실험.
# Fold1만 train 1년(2021)으로 2022 전체를 검증하고, 이후는 6개월씩 전진(Expanding).
# train 데이터가 짧을수록(Fold1) Val이 불안정해지는지를 Fold2~4(train 2~3년)와 비교하는 게 목적
A_EXPANDING_H2_FOLDS = [
    {"fold": 1, "train_start": "2021-01-01", "train_end": "2021-12-31",
     "val_start": "2022-01-01", "val_end": "2022-12-31"},
    {"fold": 2, "train_start": "2021-01-01", "train_end": "2022-06-30",
     "val_start": "2022-07-01", "val_end": "2022-12-31"},
    {"fold": 3, "train_start": "2021-01-01", "train_end": "2022-12-31",
     "val_start": "2023-01-01", "val_end": "2023-06-30"},
    {"fold": 4, "train_start": "2021-01-01", "train_end": "2023-06-30",
     "val_start": "2023-07-01", "val_end": "2023-12-31"},
]

# 공통 최종 holdout (모든 옵션 공통, CV/튜닝에 전혀 사용하지 않음)
FINAL_TEST_START = "2024-01-01"
FINAL_TEST_END = "2024-12-31"

# Option1/2 최종 재학습 범위 (CV 폴드가 끝난 뒤 전량 데이터로 재학습)
A_FINAL_TRAIN_START = "2021-01-01"
A_FINAL_TRAIN_END = "2023-12-31"

# Option3 최종 재학습 범위 (rolling window로 제한하지 않고 가용 전량 사용)
AB_POOLED_FINAL_TRAIN_START = "2021-01-01"
AB_POOLED_FINAL_TRAIN_END = "2023-12-31"
