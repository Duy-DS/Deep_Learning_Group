"""Mô hình Baseline CNN hai khối cho bài toán nhận diện cảm xúc FER2013."""

import torch
import torch.nn as nn
from src.models.base_model import BaseEmotionModel
from src.models.custom_cnn import ActivationFactory


class SmallCNN(BaseEmotionModel):
    """Mô hình Baseline 2-block Convolutional Neural Network."""

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 1,
        dropout: float = 0.5,
        activation: str = "relu",
    ) -> None:
        """Khởi tạo các tầng tích chập và tầng phân loại của SmallCNN.

        Args:
            num_classes: Số lượng lớp cảm xúc đầu ra (mặc định 7).
            in_channels: Số lượng kênh ảnh đầu vào (1 cho Grayscale, 3 cho RGB).
            dropout: Tỷ lệ dropout tại tầng Fully Connected (mặc định 0.5).
            activation: Tên hàm kích hoạt lấy từ cấu hình ablation.
        """
        super().__init__(num_classes=num_classes, in_channels=in_channels)
        self.dropout_rate = dropout

        # Hai khối dùng số kênh trùng với thiết lập RQ1: 64 rồi 128.
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu1 = ActivationFactory.create(activation)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = ActivationFactory.create(activation)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 12 * 12, 128)
        self.relu_fc = ActivationFactory.create(activation)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(128, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền tiến qua các tầng mạng SmallCNN.

        Args:
            x (torch.Tensor): Batch ảnh đầu vào [Batch_Size, In_Channels, 48, 48].

        Returns:
            torch.Tensor: Logits phân loại cảm xúc [Batch_Size, 7].
        """
        # Truyền tuần tự qua hai khối tích chập rồi bộ phân loại.
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_target_layer(self) -> nn.Module:
        """Trả về tầng Conv2d cuối cùng (self.conv2) phục vụ Grad-CAM."""
        return self.conv2
