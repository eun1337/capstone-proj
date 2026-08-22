"""
config.py
Random Forest HPO search space와 고정 하이퍼파라미터.
"""

N_ESTIMATORS_CHOICES = (100, 250, 500, 1000)
MAX_FEATURES_CHOICES = (0.1, 0.2, 1 / 3, 0.5, 1.0)
MIN_SAMPLES_LEAF_CHOICES = (1, 5, 10, 100, 500)

FIXED_PARAMS = {
    "max_depth": None,
    "min_samples_split": 2,
    "bootstrap": True,
    "criterion": "squared_error",
    "n_jobs": -1,
}
