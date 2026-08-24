"""
embedding_utils.py
categorical embedding dimension을 cardinality 기반으로 자동 결정한다. PyTorch
Forecasting의 get_embedding_size(n)과 동일한 규칙이며, LSTM/TFT가 공통으로 쓴다.
"""


def get_embedding_size(n: int, max_size: int = 100) -> int:
    if n > 2:
        return min(max_size, round(1.6 * n ** 0.56))
    return 1
