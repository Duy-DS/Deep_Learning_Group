"""Các kiến trúc MLP PyTorch chuyển đổi từ notebook Keras Nhóm 1."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .base_model import BaseMigratedModel


class FuncApxmMLP(BaseMigratedModel):
    """MLP xấp xỉ mật độ chuẩn hai biến với sigmoid ở cả hai tầng Dense."""

    def __init__(self, hidden_units: int = 16) -> None:
        """Khởi tạo mạng 2 → B → 1 tương ứng ``gauss_mlp(B)``.

        Args:
            hidden_units: Số nút ẩn B; mặc định 16.
        """
        if hidden_units <= 0:
            raise ValueError("hidden_units phải là số nguyên dương.")
        super().__init__("FuncApxmMLP")
        self.fc = nn.Linear(2, hidden_units)
        self.hidden_activation = nn.Sigmoid()
        self.gauss_apxm = nn.Linear(hidden_units, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Đổi ``(N,2)`` thành ``(N,1)`` là xác suất mật độ xấp xỉ."""
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(f"Đầu vào cần shape (Batch, 2), nhận {tuple(x.shape)}.")
        # (N,2) -> (N,B) -> sigmoid -> (N,1) -> sigmoid.
        hidden = self.hidden_activation(self.fc(x))
        return self.output_activation(self.gauss_apxm(hidden))


class MNISTMLP(BaseMigratedModel):
    """MLP phân loại MNIST, trả logits và hỗ trợ lấy activation tầng ``fc2``."""

    def __init__(self, dropout: float = 0.2) -> None:
        """Khởi tạo Flatten → Dense(512) × 2 → Dense(10).

        Args:
            dropout: Xác suất dropout giữa các tầng Dense.
        """
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout phải nằm trong [0, 1).")
        super().__init__("MNISTMLP")
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc3_10ways_softmax = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Làm phẳng ảnh 28×28 và trả logits ``(N,10)`` cho CrossEntropyLoss."""
        if x.ndim == 2 and x.shape[1] == 784:
            flattened = x
        elif (x.ndim == 3 and tuple(x.shape[1:]) == (28, 28)) or (
            x.ndim == 4 and tuple(x.shape[1:]) in {(1, 28, 28), (28, 28, 1)}
        ):
            flattened = self.flatten(x)
        else:
            raise ValueError(
                "MNISTMLP nhận (N,28,28), (N,1,28,28), (N,28,28,1) "
                f"hoặc (N,784); nhận {tuple(x.shape)}."
            )
        # Keras Dense/ReLU/Dropout được ánh xạ từng tầng; đầu ra giữ logits thô.
        hidden1 = self.dropout1(self.relu1(self.fc1(flattened)))
        hidden2 = self.dropout2(self.relu2(self.fc2(hidden1)))
        return self.fc3_10ways_softmax(hidden2)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Trả xác suất 10 lớp; chỉ gọi softmax ở thời điểm suy luận."""
        return F.softmax(self(x), dim=1)

    def count_parameters(self) -> dict[str, int]:
        """Trả bộ đếm chuẩn BaseMigratedModel và tổng tiện dụng cho parity."""
        return super().count_parameters()