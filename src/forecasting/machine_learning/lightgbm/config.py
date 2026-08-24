"""
config.py

LightGBM P13 HPO 설정.
num_leaves와 min_child_samples를 탐색하고,
P10용 FIXED_PARAMS와 P13용 P13_FIXED_PARAMS를 분리해 관리한다.
"""

NUM_LEAVES_CHOICES = (8, 31)
MIN_CHILD_SAMPLES_CHOICES = (100, 1000)

P13_GRID_SEARCH_SPACE = {
    "num_leaves": list(NUM_LEAVES_CHOICES),
    "min_child_samples": list(MIN_CHILD_SAMPLES_CHOICES),
}

# P10/P13 공통 고정 파라미터
FEATURE_FRACTION = 0.4
BAGGING_FRACTION = 0.4
BAGGING_FREQ = 1
LAMBDA_L1 = 0.0
LAMBDA_L2 = 0.0

# P10 calibration용
FIXED_PARAMS = {
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "max_depth": -1,
    "boosting_type": "gbdt",
    "objective": "regression",
    "n_jobs": -1,
    "verbosity": -1,
}

# P13 전용
P13_FIXED_PARAMS = {
    "learning_rate": 0.1,
    "n_estimators": 100,
    "max_depth": -1,
    "boosting_type": "gbdt",
    "objective": "regression",
    "n_jobs": -1,
    "verbosity": -1,
}
