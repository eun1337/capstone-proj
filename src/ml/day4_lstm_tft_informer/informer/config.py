"""
config.py
Informer 아키텍처 고정값과 smoke test 전용 참고값. e_layers/n_heads/lookback은 확정된
HPO 축 3개이며, 아래 SMOKE_* 값은 그 축의 최종 fixed 값이 아니라 functional smoke test
전용 임시값이다.
"""

# 구조 고정값 (HPO 대상 아님)
D_MODEL = 512
D_FF = 2048
D_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
FACTOR = 5
OPTIMIZER = "adam"
MAX_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.0
EARLY_STOPPING_MODE = "min"

# 확정된 HPO 축 3개 - 탐색 범위/최종값은 이 단계에서 정하지 않는다
HPO_AXES = ("e_layers", "n_heads", "lookback")
E_LAYERS_CHOICES = (2, 3, 4, 6)
N_HEADS_CHOICES = (8, 16)
LOOKBACK_CHOICES = (13, 26)

# smoke test 전용 참고값 - HPO 최종 fixed 값이 아니며 Search Space 판단에 쓰지 않는다
SMOKE_E_LAYERS = 2
SMOKE_N_HEADS = 8
SMOKE_LOOKBACK = 13
SMOKE_EPOCHS = 2
SMOKE_SEED = 42


def label_len_for(lookback: int) -> int:
    """decoder label_len = floor(lookback/2). lookback=13 -> 6, lookback=26 -> 13."""
    return lookback // 2


def distilled_encoder_length(lookback: int, e_layers: int) -> int:
    """encoder distilling(Conv1d k=3,pad=1,stride=1 + MaxPool k=3,stride=2,pad=1)을
    e_layers-1번 적용한 뒤 마지막 encoder 길이를 계산한다. L -> floor((L-1)/2)+1을
    반복하며, L=1에서 안정화되어 음수/0으로 떨어지지 않는다(직접 검증됨)."""
    length = lookback
    for _ in range(e_layers - 1):
        length = (length - 1) // 2 + 1
    return length
