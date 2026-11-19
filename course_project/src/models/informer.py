"""
Mô-đun định nghĩa kiến trúc mô hình Informer dự báo nồng độ PM2.5.
Tích hợp cơ chế ProbSparse Attention O(L log L) và Generative Decoder.
Tuân thủ hợp đồng kiến trúc BaseTimeSeriesModel [B, L, D] -> [B, H, 1].
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class DataEmbedding(nn.Module):
    """Mã hóa giá trị và vị trí thời gian."""

    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class ProbSparseAttention(nn.Module):
    """Cơ chế ProbSparse Attention tối ưu độ phức tạp O(L log L)."""

    def __init__(self, factor=5, attention_dropout=0.1):
        super(ProbSparseAttention, self).__init__()
        self.factor = factor
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries, keys, values = (
            queries.transpose(1, 2),
            keys.transpose(1, 2),
            values.transpose(1, 2),
        )

        # Chọn số lượng Top-K queries theo log(L_K)
        U_part = self.factor * int(np.ceil(np.log(L_K)))
        u = self.factor * int(np.ceil(np.log(L_Q)))
        U_part = min(U_part, L_K)
        u = min(u, L_Q)

        # Tính toán ngẫu nhiên một phần để đo lường Sparsity
        K_expand = keys.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
        index_sample = torch.randint(L_K, (L_Q, U_part), device=queries.device)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            queries.unsqueeze(-2), K_sample.transpose(-2, -1)
        ).squeeze(-2)

        # Tính Sparsity Measurement (M)
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(u, sorted=False)[1]

        # Rút gọn Q
        Q_reduce = queries[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :
        ]
        Q_K = torch.matmul(Q_reduce, keys.transpose(-2, -1))
        Q_K = Q_K / math.sqrt(D)

        # Context mặc định là trung bình của V
        context = (
            values.mean(dim=-2)
            .unsqueeze(-2)
            .expand(B, H, L_Q, values.shape[-1])
            .clone()
        )

        # Cập nhật context cho Top-K Queries
        attn = torch.softmax(Q_K, dim=-1)
        context[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :
        ] = torch.matmul(attn, values).type_as(context)

        return context.transpose(1, 2).contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        self.batch_first = False
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        is_causal: bool = False,
        **kwargs,
    ):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
        )
        out = out.view(B, L, -1)

        # PyTorch TransformerEncoderLayer._sa_block gọi self.self_attn(...)[0]
        # nên bắt buộc phải trả về tuple (tensor_đầu_ra, attn)
        return self.out_projection(out), attn


class BaseTimeSeriesModel(nn.Module):
    def __init__(self, lookback: int, horizon: int, in_features: int):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class InformerModel(BaseTimeSeriesModel):
    """
    Kiến trúc Informer chuẩn tích hợp đầy đủ Encoder-Decoder.
    """

    def __init__(
        self,
        lookback: int = 96,
        horizon: int = 24,
        in_features: int = 15,
        d_model: int = 128,  # Giảm từ 512 -> 128
        n_heads: int = 4,  # Giảm từ 8 -> 4 (mỗi head 32 chiều)
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 256,  # Giảm từ 2048 -> 256
        factor: int = 5,
        dropout: float = 0.15,  # Tăng nhẹ dropout để chống overfit
        activation: str = "gelu",
    ):
        super().__init__(lookback, horizon, in_features)

        # Độ dài gối đầu (Label Length) cho Decoder lấy từ Lookback
        self.label_len = lookback // 2

        # Nhúng (Embedding)
        self.enc_embedding = DataEmbedding(in_features, d_model, dropout)
        self.dec_embedding = DataEmbedding(in_features, d_model, dropout)

        # Encoder (dùng ProbSparse Attention)
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(e_layers)
            ]
        )

        # Cắm ProbSparse vào lớp tự chú ý (Self-Attention) của Encoder
        for layer in self.encoder_layers:
            layer.self_attn = AttentionLayer(
                ProbSparseAttention(factor, dropout), d_model, n_heads
            )

        # Decoder (Generative)
        self.decoder_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(d_layers)
            ]
        )

        # Chiếu về nồng độ PM2.5 (1 biến)
        self.projection = nn.Linear(d_model, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền tiến với Generative Decoder ngầm định.

        Tham số:
            x (torch.Tensor): Đầu vào [B, L, D]
        Trả về:
            torch.Tensor: Đầu ra [B, H, 1]
        """
        B, L, D = x.shape

        enc_out = self.enc_embedding(x)
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)

        dec_inp_token = x[:, -self.label_len :, :]
        dec_inp_zeros = torch.zeros(B, self.horizon, D, device=x.device)
        dec_inp = torch.cat([dec_inp_token, dec_inp_zeros], dim=1)

        dec_out = self.dec_embedding(dec_inp)
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out)

        preds = self.projection(dec_out)
        return preds[:, -self.horizon :, :]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
