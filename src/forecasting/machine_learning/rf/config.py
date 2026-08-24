"""
config.py

Random Forest P13 HPO 설정.
max_features와 min_samples_leaf를 탐색하고 n_estimators는 100으로 고정한다.
N_ESTIMATORS_CHOICES는 P12 profiling guard 호환을 위해 singleton으로 유지한다.
"""

N_ESTIMATORS = 100  
N_ESTIMATORS_CHOICES = (N_ESTIMATORS,)  # P12 profiling guard 호환
MAX_FEATURES_CHOICES = (0.1, 1 / 3)
MIN_SAMPLES_LEAF_CHOICES = (100, 500)

P13_GRID_SEARCH_SPACE = {
    "max_features": list(MAX_FEATURES_CHOICES),
    "min_samples_leaf": list(MIN_SAMPLES_LEAF_CHOICES),
}

FIXED_PARAMS = {
    "max_depth": None,
    "min_samples_split": 2,
    "bootstrap": True,
    "criterion": "squared_error",
    "n_jobs": -1,
}
