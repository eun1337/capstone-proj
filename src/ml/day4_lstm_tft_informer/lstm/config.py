"""
config.py
LSTM 아키텍처 고정값과 smoke test 전용 참고값. hidden_size/learning_rate/batch_size/
weight_decay/lookback은 이미 확정된 HPO 축이며, 아래 SMOKE_* 값은 그 축의 최종 fixed
값이 아니라 functional smoke test 전용 임시값이다.
"""

# 구조 고정값 (HPO 대상 아님)
NUM_LAYERS = 1
DROPOUT = 0.0
OPTIMIZER = "adam"
MAX_EPOCHS = 20

# 확정된 HPO 축 5개 - 탐색 범위/최종값은 이 단계에서 정하지 않는다
HPO_AXES = ("hidden_size", "learning_rate", "batch_size", "weight_decay", "lookback")

# smoke test 전용 참고값 - HPO 최종 fixed 값이 아니며 Search Space 판단에 쓰지 않는다
SMOKE_HIDDEN_SIZE = 16
SMOKE_LEARNING_RATE = 1e-3
SMOKE_BATCH_SIZE = 32
SMOKE_WEIGHT_DECAY = 1e-4
SMOKE_LOOKBACK = 13
SMOKE_EPOCHS = 2
SMOKE_SEED = 42
