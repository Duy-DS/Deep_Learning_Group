"""Mô hình Baseline CNN (SmallCNN 2-block) cho bài toán nhận diện cảm xúc FER2013.

Kiến trúc SmallCNN:
- Block 1: Conv2d(in_channels -> 32, 3x3, pad=1) + ReLU + MaxPool2d(2x2)
- Block 2: Conv2d(32 -> 64, 3x3, pad=1) + ReLU + MaxPool2d(2x2)
- Classifier: Flatten + Linear(64*12*12 -> 128) + ReLU + Dropout(p) + Linear(128 -> num_classes)
"""

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

        # TODO 1: Khai báo 2 block tích chập (conv1, relu1, pool1 và conv2, relu2, pool2)
        # Viết code ở đây:
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3,padding = 1)

        self.relu1 = ActivationFactory.create(activation)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)


        self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.relu2 = ActivationFactory.create(activation)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        # TODO 2: Khai báo tầng phân loại Fully Connected (flatten, fc1, relu_fc, dropout, fc2)
        # Viết code ở đây:

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
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
        # TODO 3: Cho x đi tuần tự qua Block 1 -> Block 2 -> Classifier -> return x
        # Viết code ở đây:
        # 1. Đi qua Block 1: x -> conv1 -> relu1 -> pool1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # 2. Đi qua Block 2: x -> conv2 -> relu2 -> pool2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # 3. Trải phẳng và đi qua Classifier: flatten -> fc1 -> relu -> dropout -> fc2
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x # trả về tensor [B, 7]
        

    def get_target_layer(self) -> nn.Module:
        """Trả về tầng Conv2d cuối cùng (self.conv2) phục vụ Grad-CAM."""
        # TODO 4: return tầng conv2
        # Viết code ở đây:

        return self.conv2
