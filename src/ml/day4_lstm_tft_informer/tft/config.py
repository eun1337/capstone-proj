"""
config.py
TFT 아키텍처 고정값과 smoke test 전용 참고값. hidden_size/hidden_continuous_size/
attention_head_size/dropout/learning_rate/gradient_clip_val/lookback은 이미 확정된
HPO 축이며, 아래 SMOKE_* 값은 그 축의 최종 fixed 값이 아니라 functional smoke test
전용 임시값이다.
"""

# 구조 고정값 (HPO 대상 아님)
LSTM_LAYERS = 2
BATCH_SIZE = 128
OPTIMIZER = "adam"
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-4
EARLY_STOPPING_MODE = "min"
PRUNING = False

# 확정된 HPO 축 7개 - 탐색 범위/최종값은 이 단계에서 정하지 않는다
HPO_AXES = (
    "hidden_size",
    "hidden_continuous_size",
    "attention_head_size",
    "dropout",
    "learning_rate",
    "gradient_clip_val",
    "lookback",
)

# smoke test 전용 참고값 - HPO 최종 fixed 값이 아니며 Search Space 판단에 쓰지 않는다
SMOKE_HIDDEN_SIZE = 16
SMOKE_HIDDEN_CONTINUOUS_SIZE = 8
SMOKE_ATTENTION_HEAD_SIZE = 1
SMOKE_DROPOUT = 0.1
SMOKE_LEARNING_RATE = 1e-3
SMOKE_GRADIENT_CLIP_VAL = 0.1
SMOKE_LOOKBACK = 13
SMOKE_EPOCHS = 2
SMOKE_SEED = 42
