"""
model.py
Global/pooled LSTM. time-varying(known+observed) 시퀀스를 LSTM으로 인코딩한 마지막 hidden
state를 static(categorical embedding + continuous)과 concat해 단일 Linear로 horizon
스칼라(log1p 예측)를 낸다. categorical embedding dimension은 train-fit vocab size 기준
cardinality 규칙(common/embedding_utils.get_embedding_size)으로 자동 결정하며 별도
HPO 축으로 두지 않는다. Direct h1/h2/h4는 horizon마다 별도로 학습된 모델 인스턴스로
지원한다.
"""

import torch
import torch.nn as nn

from src.ml.day4_lstm_tft_informer.common.embedding_utils import get_embedding_size


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
