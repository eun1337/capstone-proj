"""
layers.py

Informer 핵심 attention 및 encoder/decoder layer 구현.

ProbSparse self-attention, encoder distilling, decoder full cross-attention,
multi-head projection과 positional encoding을 제공한다.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    """원논문과 동일한 sinusoidal positional encoding(학습 파라미터 없음)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class ProbAttention(nn.Module):
    """Informer ProbSparse self-attention."""

    def __init__(self, mask_flag: bool, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        self.mask_flag = mask_flag
        self.factor = factor
        self.dropout = nn.Dropout(dropout)

    def _prob_qk(self, q, k, sample_k, n_top):
        # q,k: (B, H, L, D)
        B, H, L_K, D = k.shape
        _, _, L_Q, _ = q.shape

        k_expand = k.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=q.device)
        k_sample = k_expand[:, :, torch.arange(L_Q, device=q.device).unsqueeze(1), index_sample, :]
        q_k_sample = torch.matmul(q.unsqueeze(-2), k_sample.transpose(-2, -1)).squeeze(-2)

        m = q_k_sample.max(-1)[0] - torch.div(q_k_sample.sum(-1), L_K)
        m_top = m.topk(n_top, sorted=False)[1]

        q_reduce = q[
            torch.arange(B, device=q.device)[:, None, None],
            torch.arange(H, device=q.device)[None, :, None],
            m_top, :,
        ]
        q_k = torch.matmul(q_reduce, k.transpose(-2, -1))
        return q_k, m_top

    def _get_initial_context(self, v, l_q):
        B, H, L_V, D = v.shape
        if not self.mask_flag:
            v_mean = v.mean(dim=-2)
            context = v_mean.unsqueeze(-2).expand(B, H, l_q, D).clone()
        else:
            if L_V != l_q:
                raise ValueError("masked ProbAttention은 self-attention(L_V==L_Q)에서만 사용함")
            context = v.cumsum(dim=-2)
        return context

    def _update_context(self, context, v, scores, index, l_q, causal_mask):
        B, H, L_V, D = v.shape
        if self.mask_flag:
            attn_mask = torch.triu(torch.ones(l_q, L_V, dtype=torch.bool, device=v.device), diagonal=1)
            row_mask = attn_mask[index, :]
            scores = scores.masked_fill(row_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        updated = torch.matmul(attn, v)

        context[
            torch.arange(B, device=v.device)[:, None, None],
            torch.arange(H, device=v.device)[None, :, None],
            index, :,
        ] = updated
        return context

    def forward(self, q, k, v):
        # q,k,v: (B, L, H, D) -> (B, H, L, D)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        B, H, L_Q, D = q.shape
        L_K = k.shape[2]

        u_k = min(self.factor * int(math.ceil(math.log(max(L_K, 2)))), L_K)
        u_q = min(self.factor * int(math.ceil(math.log(max(L_Q, 2)))), L_Q)

        scale = 1.0 / math.sqrt(D)
        scores_top, index = self._prob_qk(q, k, sample_k=u_k, n_top=u_q)
        scores_top = scores_top * scale

        context = self._get_initial_context(v, L_Q)
        context = self._update_context(context, v, scores_top, index, L_Q, self.mask_flag)
        return context.transpose(1, 2).contiguous()  # (B, L_Q, H, D)


class FullAttention(nn.Module):
    """decoder cross-attention에 사용하는 scaled dot-product attention."""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # q,k,v: (B, L, H, D)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        d = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        return context.transpose(1, 2).contiguous()  # (B, L_Q, H, D)


class AttentionLayer(nn.Module):
    """attention 연산에 Q/K/V projection을 적용하는 multi-head wrapper."""

    def __init__(self, inner_attention: nn.Module, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model({d_model})이 n_heads({n_heads})로 나누어떨어지지 않음")
        self.inner_attention = inner_attention
        self.n_heads = n_heads
        d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.d_head = d_head

    def forward(self, queries, keys, values):
        B, L_Q, _ = queries.shape
        L_K = keys.shape[1]
        H, D = self.n_heads, self.d_head

        q = self.q_proj(queries).view(B, L_Q, H, D)
        k = self.k_proj(keys).view(B, L_K, H, D)
        v = self.v_proj(values).view(B, L_K, H, D)

        out = self.inner_attention(q, k, v)  # (B, L_Q, H, D)
        out = out.reshape(B, L_Q, H * D)
        return self.out_proj(out)


class ConvLayer(nn.Module):
    """encoder distilling: Conv1d(kernel=3) -> ELU -> MaxPool(stride=2)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) -> conv는 (B, D, L) 필요
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        return x.transpose(1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, factor: int, dropout: float):
        super().__init__()
        self.attention = AttentionLayer(ProbAttention(mask_flag=False, factor=factor, dropout=dropout), d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class Encoder(nn.Module):
    """EncoderLayer 사이에 distilling ConvLayer를 적용하는 Informer encoder."""

    def __init__(self, e_layers: int, d_model: int, n_heads: int, d_ff: int, factor: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, factor, dropout) for _ in range(e_layers)
        ])
        self.conv_layers = nn.ModuleList([ConvLayer(d_model) for _ in range(e_layers - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.conv_layers):
                x = self.conv_layers[i](x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, factor: int, dropout: float):
        super().__init__()
        self.self_attention = AttentionLayer(
            ProbAttention(mask_flag=True, factor=factor, dropout=dropout), d_model, n_heads,
        )
        self.cross_attention = AttentionLayer(FullAttention(dropout=dropout), d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        self_attn_out = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(self_attn_out))
        cross_attn_out = self.cross_attention(x, enc_out, enc_out)
        x = self.norm2(x + self.dropout(cross_attn_out))
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


class Decoder(nn.Module):
    def __init__(self, d_layers: int, d_model: int, n_heads: int, d_ff: int, factor: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, factor, dropout) for _ in range(d_layers)
        ])

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_out)
        return x
