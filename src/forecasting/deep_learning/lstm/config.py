"""
config.py

LSTM P13 HPO 설정.

hidden_size만 탐색하며 lookback, learning rate, batch size, weight decay와
모델 구조 파라미터는 고정한다. OOM 발생 시에만 fallback batch size를 사용한다.
"""

# 모델 구조 고정값
NUM_LAYERS = 1
DROPOUT = 0.0
OPTIMIZER = "adam"
MAX_EPOCHS = 20

# P13 고정값
LOOKBACK = 13
LEARNING_RATE = 1e-3
BATCH_SIZE = 1024
WEIGHT_DECAY = 0.0
# OOM/memory allocation 실패 시 1회 fallback
FALLBACK_BATCH_SIZE = 512

# P13 HPO 축
HPO_AXES = ("hidden_size",)
HIDDEN_SIZE_CHOICES = (32, 64)

P13_GRID_SEARCH_SPACE = {"hidden_size": list(HIDDEN_SIZE_CHOICES)}

# functional smoke test 전용
SMOKE_HIDDEN_SIZE = 16
SMOKE_LEARNING_RATE = 1e-3
SMOKE_BATCH_SIZE = 32
SMOKE_WEIGHT_DECAY = 1e-4
SMOKE_LOOKBACK = 13
SMOKE_EPOCHS = 2
SMOKE_SEED = 42
