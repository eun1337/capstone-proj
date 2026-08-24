"""
model.py

Global/pooled LSTM 모델.

time-varying sequence의 마지막 hidden state와 static feature를 결합해
horizon별 log1p 수요를 예측한다. categorical embedding 크기는 train-fit
vocabulary cardinality에 따라 자동 결정한다.
"""

import torch
import torch.nn as nn

from src.forecasting.deep_learning.common.embedding_utils import get_embedding_size


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        n_time_varying: int,
        static_cont_dim: int,
        cat_vocab_sizes: list[int],
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding_sizes = [get_embedding_size(v) for v in cat_vocab_sizes]
        self.embeddings = nn.ModuleList([
            nn.Embedding(v, d) for v, d in zip(cat_vocab_sizes, self.embedding_sizes)
        ])
        self.lstm = nn.LSTM(
            input_size=n_time_varying,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        static_dim = static_cont_dim + sum(self.embedding_sizes)
        self.head = nn.Linear(hidden_size + static_dim, 1)

    def forward(self, time_varying: torch.Tensor, static_cont: torch.Tensor, static_cat: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(time_varying)
        last_hidden = h_n[-1]

        cat_embeds = [emb(static_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        static_repr = torch.cat([static_cont, *cat_embeds], dim=1)

        combined = torch.cat([last_hidden, static_repr], dim=1)
        return self.head(combined).squeeze(-1)
