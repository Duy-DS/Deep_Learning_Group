"""Các kiến trúc CNN Nhóm 2 được chuyển đổi từ notebook Keras legacy."""

from __future__ import annotations

import torch
from torch import nn

from .base_model import BaseMigratedModel


class MNISTCNN(BaseMigratedModel):
    """CNN phân loại MNIST, giữ nguyên Flatten và dense layers của Keras."""

    def __init__(self, dropout: float = 0.2) -> None:
        """Khởi tạo ba block tích chập và đầu phân loại 10 lớp."""
        super().__init__("MNISTCNN")
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv_dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv_dropout2 = nn.Dropout(dropout)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # Tên pool3 giữ đúng tên MaxPooling2D legacy để hook trích feature.
        self.pool3 = nn.MaxPool2d(2)
        self.conv_dropout3 = nn.Dropout(dropout)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc_dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 512)
        self.fc_dropout2 = nn.Dropout(dropout)
        self.fc3_10ways_softmax = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trả logits; các lớp tích chập nhận ảnh NCHW (B,1,28,28).

        Kích thước không gian lần lượt 28→14→7→3 sau mỗi MaxPool2d.
        Pool3 có shape (B,256,3,3), được Flatten thành 2304 đặc trưng như Keras.
        """
        self.validate_input_shape(x, expected_channels=1)
        x = self.conv_dropout1(self.pool1(self.relu(self.conv1(x))))
        x = self.conv_dropout2(self.pool2(self.relu(self.conv2(x))))
        x = self.conv_dropout3(self.pool3(self.relu(self.conv3(x))))
        x = self.flatten(x)
        x = self.fc_dropout1(self.relu(self.fc1(x)))
        x = self.fc_dropout2(self.relu(self.fc2(x)))
        return self.fc3_10ways_softmax(x)


class HumanGenderCNN(BaseMigratedModel):
    """CNN WIKI Gender gồm đúng năm block và classifier Keras gốc."""

    def __init__(self, dropout: float = 0.2) -> None:
        """Tạo 13 convolution same, năm pooling và ba Dense."""
        super().__init__("HumanGenderCNN")
        channels = (1, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128, 256, 256, 256)
        conv_names = (
            "conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv2_3",
            "conv3_1", "conv3_2", "conv3_3", "conv4_1", "conv4_2",
            "conv4_3", "conv5_1", "conv5_2", "conv5_3",
        )
        self.convolutions = nn.ModuleDict({
            name: nn.Conv2d(channels[i], channels[i + 1], 3, padding=1)
            for i, name in enumerate(conv_names)
        })
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.pool5 = nn.MaxPool2d(2)
        self.conv_dropout1 = nn.Dropout(dropout)
        self.conv_dropout2 = nn.Dropout(dropout)
        self.conv_dropout3 = nn.Dropout(dropout)
        self.conv_dropout4 = nn.Dropout(dropout)
        self.conv_dropout5 = nn.Dropout(dropout)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(256 * 2 * 2, 1024)
        self.fc_dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc_dropout2 = nn.Dropout(dropout)
        self.fc3_2ways_softmax = nn.Linear(1024, 2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền (B,1,64,64) qua năm block; đầu ra là hai logits."""
        self.validate_input_shape(x, expected_channels=1)
        block_specs = (
            (("conv1_1", "conv1_2"), self.pool1, self.conv_dropout1),
            (("conv2_1", "conv2_2", "conv2_3"), self.pool2, self.conv_dropout2),
            (("conv3_1", "conv3_2", "conv3_3"), self.pool3, self.conv_dropout3),
            (("conv4_1", "conv4_2", "conv4_3"), self.pool4, self.conv_dropout4),
            (("conv5_1", "conv5_2", "conv5_3"), self.pool5, self.conv_dropout5),
        )
        for names, pool, dropout in block_specs:
            for name in names:
                x = self.relu(self.convolutions[name](x))
            x = dropout(pool(x))
        # 64→32→16→8→4→2; 256×2×2=1024 đầu vào Dense.
        x = self.flatten(x)
        x = self.fc_dropout1(self.relu(self.fc1(x)))
        x = self.fc_dropout2(self.relu(self.fc2(x)))
        return self.fc3_2ways_softmax(x)


class EmotionCNN(BaseMigratedModel):
    """CNN FER2013 nguyên bản, bao gồm Dense 1024 gây overfitting baseline."""

    def __init__(self, dropout_block: float = 0.25, dropout_dense: float = 0.5) -> None:
        """Khởi tạo convolution valid/same theo đúng thứ tự notebook."""
        super().__init__("EmotionCNN")
        specs = (
            ("conv1_1", 1, 64, 1), ("conv1_2", 64, 64, 0),
            ("conv2_1", 64, 128, 1), ("conv2_2", 128, 128, 0),
            ("conv2_3", 128, 218, 1), ("conv2_4", 218, 218, 0),
        )
        self.convolutions = nn.ModuleDict({
            name: nn.Conv2d(in_channels, out_channels, 3, padding=padding)
            for name, in_channels, out_channels, padding in specs
        })
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv_dropout1 = nn.Dropout(dropout_block)
        self.conv_dropout2 = nn.Dropout(dropout_block)
        self.flatten = nn.Flatten(start_dim=1)
        # 48→46→23→21→19→9; flatten = 218×9×9 = 17,658.
        self.fc1 = nn.Linear(218 * 9 * 9, 1024)
        self.fc_dropout1 = nn.Dropout(dropout_dense)
        self.fc3_7ways_softmax = nn.Linear(1024, 7)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Chuyển (B,1,48,48) thành logits (B,7), không áp softmax trước loss."""
        self.validate_input_shape(x, expected_channels=1)
        for name in ("conv1_1", "conv1_2"):
            x = self.relu(self.convolutions[name](x))
        x = self.conv_dropout1(self.pool1(x))
        for name in ("conv2_1", "conv2_2", "conv2_3", "conv2_4"):
            x = self.relu(self.convolutions[name](x))
        x = self.conv_dropout2(self.pool2(x))
        x = self.flatten(x)
        x = self.fc_dropout1(self.relu(self.fc1(x)))
        return self.fc3_7ways_softmax(x)