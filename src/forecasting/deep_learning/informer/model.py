"""
model.py

Informer 예측 모델.

time-varying sequence를 ProbSparse encoder로 처리하고,
known/value decoder 입력과 cross-attention을 이용해 horizon별 log1p 수요를 예측한다.
static categorical/continuous context는 encoder와 decoder의 모든 timestep에 함께 반영한다.
"""

import torch
import torch.nn as nn

from src.forecasting.deep_learning.common.embedding_utils import get_embedding_size
from src.forecasting.deep_learning.informer.layers import Decoder, Encoder, PositionalEmbedding


class InformerForecaster(nn.Module):
    def __init__(
        self,
        n_time_varying: int,
        n_known: int,
        static_cont_dim: int,
        cat_vocab_sizes: list[int],
        e_layers: int,
        n_heads: int,
        d_model: int = 512,
        d_ff: int = 2048,
        d_layers: int = 2,
        factor: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_sizes = [get_embedding_size(v) for v in cat_vocab_sizes]
        self.embeddings = nn.ModuleList([
            nn.Embedding(v, d) for v, d in zip(cat_vocab_sizes, self.embedding_sizes)
        ])
        static_dim = static_cont_dim + sum(self.embedding_sizes)
        self.static_proj = nn.Linear(static_dim, d_model)

        self.encoder_input_proj = nn.Linear(n_time_varying, d_model)
        self.decoder_value_proj = nn.Linear(1, d_model)
        self.decoder_known_proj = nn.Linear(n_known, d_model)
        self.pos_encoding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.encoder = Encoder(e_layers, d_model, n_heads, d_ff, factor, dropout)
        self.decoder = Decoder(d_layers, d_model, n_heads, d_ff, factor, dropout)
        self.output_layer = nn.Linear(d_model, 1)

        self.e_layers = e_layers
        self.n_heads = n_heads
        self.d_model = d_model

    def _static_context(self, static_cont: torch.Tensor, static_cat: torch.Tensor) -> torch.Tensor:
        cat_embeds = [emb(static_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        static_repr = torch.cat([static_cont, *cat_embeds], dim=1)
        return self.static_proj(static_repr)  # (B, d_model)

    def forward(
        self,
        encoder_input: torch.Tensor,   
        decoder_value: torch.Tensor,   
        decoder_known: torch.Tensor,   
        static_cont: torch.Tensor,     
        static_cat: torch.Tensor,      
    ) -> torch.Tensor:
        static_context = self._static_context(static_cont, static_cat)  

        enc_emb = self.encoder_input_proj(encoder_input)
        enc_emb = enc_emb + self.pos_encoding(enc_emb) + static_context.unsqueeze(1)
        enc_emb = self.dropout(enc_emb)
        enc_out = self.encoder(enc_emb)

        dec_emb = self.decoder_value_proj(decoder_value.unsqueeze(-1)) + self.decoder_known_proj(decoder_known)
        dec_emb = dec_emb + self.pos_encoding(dec_emb) + static_context.unsqueeze(1)
        dec_emb = self.dropout(dec_emb)
        dec_out = self.decoder(dec_emb, enc_out)

        pred_log = self.output_layer(dec_out[:, -1, :]).squeeze(-1)  
        return pred_log
