"""
config.py
LightGBM Initial Search Space(P10)와 Final Search Space(P13, 현재는 P12 조정 전이라 Initial과
동일)를 정의한다. rf/config.py와 동일하게 discrete choice 축만 두고 Optuna
suggest_categorical로 탐색한다(p10_lgbm.py의 suggest_lgbm_hp가 이 축들을 그대로 읽는다).
P12에서 축 조정이 있으면 이 파일만 갱신하면 된다.

축 이름은 lightgbm.LGBMRegressor(sklearn wrapper)의 정식 파라미터 이름을 그대로 쓴다
(예: colsample_bytree). feature_fraction 같은 native-API alias를 같이 쓰면 LightGBM이
"파라미터가 두 곳에서 지정됨" 경고를 내므로 섞지 않음.
"""

HPO_AXES = ("n_estimators", "learning_rate", "num_leaves", "min_child_samples", "colsample_bytree")

N_ESTIMATORS_CHOICES = (100, 250, 500, 1000)
LEARNING_RATE_CHOICES = (0.01, 0.03, 0.05, 0.1)
NUM_LEAVES_CHOICES = (15, 31, 63, 127)
MIN_CHILD_SAMPLES_CHOICES = (5, 10, 20, 50, 100)
COLSAMPLE_BYTREE_CHOICES = (0.5, 0.7, 0.8, 1.0)

FIXED_PARAMS = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "n_jobs": -1,
    "verbosity": -1,
}

# P10(폭넓은 초기 탐색)/P13(Final Search Space HPO) 공통 Optuna TPE 설정
OPTUNA_N_STARTUP_TRIALS = 10
