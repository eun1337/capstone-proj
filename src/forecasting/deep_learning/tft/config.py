"""
config.py

TFT P13 HPO 설정.

hidden_size만 탐색하며 lookback, hidden continuous size, attention head size,
dropout, learning rate, gradient clipping과 모델 구조 파라미터는 고정한다.
EarlyStopping 없이 max_epochs를 모두 학습한 마지막 epoch 모델을 사용한다.
"""

# 모델 구조 및 학습 고정값
LSTM_LAYERS = 1
BATCH_SIZE = 256
OPTIMIZER = "adam"
MAX_EPOCHS = 2

# P13 고정값
LOOKBACK = 13
HIDDEN_CONTINUOUS_SIZE = 8
ATTENTION_HEAD_SIZE = 1
DROPOUT = 0.1
LEARNING_RATE = 1e-3
GRADIENT_CLIP_VAL = 0.1

# P13 HPO 축
HPO_AXES = ("hidden_size",)
HIDDEN_SIZE_CHOICES = (8, 16)

P13_GRID_SEARCH_SPACE = {"hidden_size": list(HIDDEN_SIZE_CHOICES)}

# functional smoke test 전용
SMOKE_HIDDEN_SIZE = 16
SMOKE_HIDDEN_CONTINUOUS_SIZE = 8
SMOKE_ATTENTION_HEAD_SIZE = 1
SMOKE_DROPOUT = 0.1
SMOKE_LEARNING_RATE = 1e-3
SMOKE_GRADIENT_CLIP_VAL = 0.1
SMOKE_LOOKBACK = 13
SMOKE_EPOCHS = 2
SMOKE_SEED = 42
