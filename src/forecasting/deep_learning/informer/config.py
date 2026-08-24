"""
config.py

Informer P13 HPO 설정.

n_heads만 탐색하며 e_layers, lookback, d_model, d_ff, decoder layers,
dropout, learning rate, batch size와 factor는 고정한다.
EarlyStopping 없이 max_epochs를 모두 학습한 마지막 epoch 모델을 사용한다.
"""

# 모델 구조 및 학습 고정값
D_MODEL = 512
D_FF = 2048
D_LAYERS = 1
DROPOUT = 0.05
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
FACTOR = 5
OPTIMIZER = "adam"
MAX_EPOCHS = 2

# P13 고정값
E_LAYERS = 2
LOOKBACK = 13

# P13 HPO 축
HPO_AXES = ("n_heads",)
N_HEADS_CHOICES = (8, 16)

for _n_heads in N_HEADS_CHOICES:
    if D_MODEL % _n_heads != 0:
        raise RuntimeError(f"d_model={D_MODEL}이 n_heads={_n_heads}로 나누어떨어지지 않음")

P13_GRID_SEARCH_SPACE = {"n_heads": list(N_HEADS_CHOICES)}

# smoke test 순회 로직 호환용 singleton alias
E_LAYERS_CHOICES = (E_LAYERS,)
LOOKBACK_CHOICES = (LOOKBACK,)

# functional smoke test 전용
SMOKE_E_LAYERS = 2
SMOKE_N_HEADS = 8
SMOKE_LOOKBACK = 13
SMOKE_EPOCHS = 2
SMOKE_SEED = 42


def label_len_for(lookback: int) -> int:
    """decoder label_len을 lookback의 절반(floor)으로 계산한다."""
    return lookback // 2


def distilled_encoder_length(lookback: int, e_layers: int) -> int:
    """encoder distilling을 e_layers-1회 적용한 뒤의 sequence length를 계산한다."""
    length = lookback
    for _ in range(e_layers - 1):
        length = (length - 1) // 2 + 1
    return length
