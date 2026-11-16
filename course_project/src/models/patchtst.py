"""
PatchTST tối ưu tốc độ cao cho dự báo chuỗi thời gian đa biến PM2.5:
- Vector hóa toàn bộ quá trình tách Patch qua phép unfold (không vòng lặp theo kênh).
- Sử dụng PyTorch 2.x F.scaled_dot_product_attention (FlashAttention / Memory-Efficient SDP).
- Tích hợp RevIN (Reversible Instance Normalization) chống dịch chuyển phân phối theo thời gian.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """Reversible Instance Normalization chuẩn hóa cục bộ từng cửa sổ (B, L, D)."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        target_idx: int = 0,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.target_idx = target_idx
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def normalize(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, L, D)
        mean = x.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()
        x_norm = (x - mean) / stdev
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
        return x_norm, mean, stdev

    def denormalize_target(
        self, y_norm: torch.Tensor, mean: torch.Tensor, stdev: torch.Tensor
    ) -> torch.Tensor:
        # y_norm: (B, H), mean & stdev: (B, 1, D)
        idx = self.target_idx
        if self.affine:
            y_norm = (y_norm - self.affine_bias[:, :, idx]) / (
                self.affine_weight[:, :, idx] + self.eps
            )
        return y_norm * stdev[:, :, idx] + mean[:, :, idx]


class FlashTransformerEncoderLayer(nn.Module):
    """Pre-LN Transformer Encoder Layer sử dụng F.scaled_dot_product_attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.resid_dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B_total, N_patches, d_model)
        B_tot, N, D_m = x.shape

        # 1. Self-Attention với kernel hợp nhất (FlashAttention / Memory-Efficient SDP)
        h = self.norm1(x)
        qkv = self.qkv_proj(h).reshape(B_tot, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_tot, n_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B_tot, N, D_m)
        x = x + self.resid_dropout1(self.out_proj(attn_out))

        # 2. Position-wise Feed-Forward Network
        x = x + self.ffn(self.norm2(x))
        return x


class PatchTST(nn.Module):
    """
    Kiến trúc PatchTST tối ưu cho chuỗi thời gian đa biến/đơn biến.
    Tương thích đầy đủ với build_model() và tùy chọn --freeze-backbone trong fine_tune.py.
    """

    def __init__(
        self,
        lookback: int = 96,
        horizon: int = 24,
        in_features: int = 15,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.25,
        target_idx: int = 0,
        use_revin: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        # Hỗ trợ alias tham số nếu gọi từ các script khác
        lookback = int(kwargs.get("seq_len", lookback))
        horizon = int(kwargs.get("pred_len", horizon))
        in_features = int(kwargs.get("enc_in", kwargs.get("c_in", in_features)))

        self.lookback = lookback
        self.horizon = horizon
        self.in_features = in_features
        self.patch_len = patch_len
        self.stride = stride
        self.target_idx = target_idx
        self.use_revin = use_revin

        if self.use_revin:
            self.revin = RevIN(
                num_features=in_features, affine=True, target_idx=target_idx
            )

        # Tính số lượng patch và độ dài padding cuối chuỗi (nếu cần)
        if lookback < patch_len:
            self.pad_len = patch_len - lookback
        else:
            rem = (lookback - patch_len) % stride
            self.pad_len = (stride - rem) if rem != 0 else 0

        padded_len = lookback + self.pad_len
        self.num_patches = (padded_len - patch_len) // stride + 1

        # Chiếu Patch -> d_model + Positional Embedding học được
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.embed_dropout = nn.Dropout(dropout)

        # Stack các tầng FlashTransformerEncoder
        self.encoder_layers = nn.ModuleList(
            [
                FlashTransformerEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(e_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        # Đầu dự báo (đặt tên chứa 'head' để tương thích --freeze-backbone)
        flat_dim = self.num_patches * d_model
        self.temporal_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(flat_dim, horizon),
        )
        # Trộn thông tin liên kênh siêu nhẹ (D -> 1) kèm kết nối phần dư cho biến mục tiêu
        self.channel_mixer_head = nn.Linear(in_features, 1, bias=False)
        nn.init.zeros_(self.channel_mixer_head.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tham số:
            x: Tensor đầu vào kích thước [B, lookback, in_features].

        Trả về:
            Tensor dự báo PM2.5 kích thước [B, horizon, 1].
        """
        B, L, D = x.shape

        if self.use_revin:
            x, mean, stdev = self.revin.normalize(x)

        # Chuyển sang dạng Channel-Independent: (B, L, D) -> (B * D, L)
        x_ci = x.permute(0, 2, 1).contiguous().reshape(B * D, L)

        # Replication padding ở bước cuối nếu L không chia hết cho stride
        if self.pad_len > 0:
            last_val = x_ci[:, -1:].expand(-1, self.pad_len)
            x_ci = torch.cat([x_ci, last_val], dim=-1)

        # Tách patch vector hóa: (B * D, num_patches, patch_len)
        patches = x_ci.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Nhúng đặc trưng & cộng mã hóa vị trí
        z = self.patch_embedding(patches) + self.pos_embedding
        z = self.embed_dropout(z)

        # Qua các tầng Transformer dùng FlashAttention
        for layer in self.encoder_layers:
            z = layer(z)
        z = self.encoder_norm(z)

        # Trải phẳng theo từng kênh: (B, D, num_patches * d_model)
        z_flat = z.reshape(B, D, -1)

        # Dự báo độc lập theo thời gian cho từng kênh: (B, D, horizon)
        y_per_channel = self.temporal_head(z_flat)

        # Kết hợp biến mục tiêu (kết nối tắt) + đóng góp từ các biến khí tượng
        y_target_base = y_per_channel[:, self.target_idx, :]  # (B, H)
        y_cross_channel = self.channel_mixer_head(
            y_per_channel.permute(0, 2, 1)
        ).squeeze(-1)  # (B, H)
        out = y_target_base + y_cross_channel

        if self.use_revin:
            out = self.revin.denormalize_target(out, mean, stdev)

        return out.unsqueeze(-1)


# Alias đảm bảo tương thích với mọi cách đặt tên trong build_model()
PatchTSTModel = PatchTST
