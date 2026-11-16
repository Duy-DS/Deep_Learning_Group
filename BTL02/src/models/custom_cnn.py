"""Kiến trúc Custom CNN sâu hơn cho bài toán nhận diện cảm xúc FER2013.

Tuân thủ nghiêm ngặt hiến pháp AGENTS.md:
- Thiết kế 100% hướng đối tượng (OOP) kế thừa từ BaseEmotionModel.
- Đạt chuẩn kiểm chứng cho câu hỏi nghiên cứu RQ1 (Ảnh hưởng của độ sâu mạng CNN):
  1. MediumCNN (3-block): Conv(64) -> Conv(128) -> Conv(256) -> FC(256) -> 7
  2. LargeCNN (4-block): Conv(64) -> Conv(128) -> Conv(256) -> Conv(512) + BatchNorm + Dropout -> FC(512) -> 7
- Cài đặt phương thức get_target_layer() phục vụ phân tích giải thích Grad-CAM.
"""

from typing import Union
import torch
import torch.nn as nn
from src.models.base_model import BaseEmotionModel


class MediumCNN(BaseEmotionModel):
    """Mô hình Custom CNN trung bình 3 khối tích chập (3-block Conv2D)."""

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 1,
        dropout: float = 0.5,
    ) -> None:
        """Khởi tạo các tầng cho MediumCNN.

        Args:
            num_classes: Số lớp cảm xúc đầu ra (mặc định 7).
            in_channels: Số kênh ảnh đầu vào (1 cho Grayscale, 3 cho RGB).
            dropout: Tỷ lệ dropout tại tầng FC (mặc định 0.5).
        """
        super().__init__(num_classes=num_classes, in_channels=in_channels)
        self.dropout_rate = dropout

        # Block 1: in_channels -> 64 | 48x48 -> 24x24
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 64 -> 128 | 24x24 -> 12x12
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 128 -> 256 | 12x12 -> 6x6
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier: 256 * 6 * 6 = 9,216 -> 256 -> 7
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 6 * 6, 256)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền tiến trích xuất đặc trưng và phân loại.

        Args:
            x: Tensor ảnh đầu vào [B, C, 48, 48].

        Returns:
            torch.Tensor: Logits phân loại cảm xúc [B, 7].
        """
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_target_layer(self) -> nn.Module:
        """Trả về tầng Conv2d cuối cùng (self.conv3) phục vụ Grad-CAM."""
        return self.conv3


class LargeCNN(BaseEmotionModel):
    """Mô hình Custom CNN sâu 4 khối tích chập kết hợp Batch Normalization và Dropout."""

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 1,
        dropout: float = 0.5,
    ) -> None:
        """Khởi tạo các tầng cho LargeCNN.

        Args:
            num_classes: Số lớp cảm xúc đầu ra (mặc định 7).
            in_channels: Số kênh ảnh đầu vào (1 cho Grayscale, 3 cho RGB).
            dropout: Tỷ lệ dropout tại tầng FC (mặc định 0.5).
        """
        super().__init__(num_classes=num_classes, in_channels=in_channels)
        self.dropout_rate = dropout

        # Block 1: in_channels -> 64 | 48x48 -> 24x24
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(p=0.2)

        # Block 2: 64 -> 128 | 24x24 -> 12x12
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(p=0.2)

        # Block 3: 128 -> 256 | 12x12 -> 6x6
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout2d(p=0.3)

        # Block 4: 256 -> 512 | 6x6 -> 3x3
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout2d(p=0.3)

        # Classifier: 512 * 3 * 3 = 4,608 -> 512 -> 7
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 3 * 3, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền tiến trích xuất đặc trưng sâu và phân loại.

        Args:
            x: Tensor ảnh đầu vào [B, C, 48, 48].

        Returns:
            torch.Tensor: Logits phân loại cảm xúc [B, 7].
        """
        x = self.drop1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(self.relu3(self.bn3(self.conv3(x)))))
        x = self.drop4(self.pool4(self.relu4(self.bn4(self.conv4(x)))))

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_target_layer(self) -> nn.Module:
        """Trả về tầng Conv2d cuối cùng (self.conv4) phục vụ Grad-CAM."""
        return self.conv4
