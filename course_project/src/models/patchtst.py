"""
Mô-đun định nghĩa kiến trúc mô hình PatchTST dự báo nồng độ PM2.5.
Áp dụng cơ chế Channel-Independence và Patching cho Time-Series Forecasting.
Tuân thủ hợp đồng kiến trúc BaseTimeSeriesModel [B, L, D] -> [B, H, 1].
"""

import math
import torch
import torch.nn as nn


class BaseTimeSeriesModel(nn.Module):
    def __init__(self, lookback: int, horizon: int, in_features: int):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PatchTSTModel(BaseTimeSeriesModel):
    """
    Kiến trúc Patch Time Series Transformer (PatchTST).

    Cơ chế hoạt động:
    1. Độc lập kênh: Xử lý độc lập D chuỗi 1D (Channel Independence).
    2. Patching: Chia chuỗi L thành nhiều patch độ dài patch_len.
    3. Transformer Encoder: Khai phá quan hệ chuỗi ngữ nghĩa giữa các patch.
    """

    def __init__(
        self,
        lookback: int = 96,
        horizon: int = 24,
        in_features: int = 15,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 16,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__(lookback, horizon, in_features)

        self.patch_len = patch_len
        self.stride = stride

        # Số lượng patch sinh ra sau khi chia
        self.num_patches = int((lookback - patch_len) / stride) + 1

        # Embedding cho từng Patch và Position Encoding
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, d_model)
        )
        self.dropout_layer = nn.Dropout(dropout)

        # Lõi Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=e_layers
        )

        # Lớp Flatten và chiếu tuyến tính (Prediction Head) cho từng kênh
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(self.num_patches * d_model, horizon)
        )

        # Ánh xạ từ D kênh dự báo về 1 kênh nồng độ PM2.5 duy nhất
        if in_features > 1:
            self.projection_pm25 = nn.Linear(in_features, 1)
        else:
            self.projection_pm25 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền tiến mô hình PatchTST.

        Tham số:
            x (torch.Tensor): Tensor đầu vào kích thước [B, L, D].
        Trả về:
            torch.Tensor: Tensor dự báo kích thước [B, H, 1].
        """
        B, L, D = x.shape

        # 1. Channel Independence: [B, L, D] -> [B, D, L] -> [B*D, 1, L]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B * D, 1, L)

        # 2. Patching (cắt thành các khối): [B*D, num_patches, patch_len]
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride).squeeze(
            1
        )

        # 3. Nhúng dữ liệu và vị trí: [B*D, num_patches, d_model]
        embed = self.patch_embedding(patches) + self.position_embedding
        embed = self.dropout_layer(embed)

        # 4. Truyền qua Transformer Encoder
        enc_out = self.transformer_encoder(embed)

        # 5. Dự báo kết quả tương lai cho TỪNG kênh độc lập: [B*D, H]
        dec_out = self.head(enc_out)

        # 6. Gộp chiều kênh lại: [B*D, H] -> [B, D, H] -> [B, H, D]
        dec_out = dec_out.view(B, D, self.horizon).permute(0, 2, 1).contiguous()

        # 7. Hợp nhất D biến để lấy kênh đích PM2.5: [B, H, 1]
        preds = self.projection_pm25(dec_out)
        if preds.dim() == 2:
            preds = preds.unsqueeze(-1)

        return preds

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
