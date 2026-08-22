"""
model.py
Informer forecaster. encoder는 scaled 21개 time-varying feature 시퀀스를 받아
input projection(Linear 21->d_model) + positional encoding + static context를 더한 뒤
ProbSparse self-attention + distilling encoder layer를 통과한다. decoder는
(label_len개 과거 + 1개 future placeholder) 길이의 decoder_value/decoder_known 토큰을
각각 projection해 더하고 masked ProbSparse self-attention + encoder-decoder
cross-attention을 통과한다. 마지막 decoder timestep(=forecast origin+horizon 위치)만
Linear(d_model, 1)로 사영해 pred_log를 낸다. static context(embedding+continuous)는
project-specific deterministic adaptation으로 encoder/decoder 모든 timestep에
broadcast-add되며 별도 hidden MLP head는 두지 않는다. Direct h1/h2/h4는 horizon마다
별도로 학습된 모델 인스턴스로 지원한다.
"""

import torch
import torch.nn as nn

from src.ml.day4_lstm_tft_informer.common.embedding_utils import get_embedding_size
from src.ml.day4_lstm_tft_informer.informer.layers import Decoder, Encoder, PositionalEmbedding


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
        encoder_input: torch.Tensor,   # (B, lookback, n_time_varying)
        decoder_value: torch.Tensor,   # (B, label_len+1)
        decoder_known: torch.Tensor,   # (B, label_len+1, n_known)
        static_cont: torch.Tensor,     # (B, static_cont_dim)
        static_cat: torch.Tensor,      # (B, n_static_cat)
    ) -> torch.Tensor:
        static_context = self._static_context(static_cont, static_cat)  # (B, d_model)

        enc_emb = self.encoder_input_proj(encoder_input)
        enc_emb = enc_emb + self.pos_encoding(enc_emb) + static_context.unsqueeze(1)
        enc_emb = self.dropout(enc_emb)
        enc_out = self.encoder(enc_emb)

        dec_emb = self.decoder_value_proj(decoder_value.unsqueeze(-1)) + self.decoder_known_proj(decoder_known)
        dec_emb = dec_emb + self.pos_encoding(dec_emb) + static_context.unsqueeze(1)
        dec_emb = self.dropout(dec_emb)
        dec_out = self.decoder(dec_emb, enc_out)

        pred_log = self.output_layer(dec_out[:, -1, :]).squeeze(-1)  # (B,)
        return pred_log
