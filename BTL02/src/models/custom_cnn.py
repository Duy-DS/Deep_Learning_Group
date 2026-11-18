"""Kiến trúc Custom CNN có thể cấu hình cho bài toán nhận diện FER2013.

Các lớp giữ nguyên tên tầng Conv/BatchNorm để tương thích checkpoint cũ,
đồng thời cho phép thay đổi activation, số kênh và dropout từ cấu hình YAML.
"""

from typing import List, Optional, Sequence, Type

import torch
import torch.nn as nn

from src.models.base_model import BaseEmotionModel


class ActivationFactory:
    """Tạo module kích hoạt PyTorch từ tên cấu hình đã được cho phép."""

    _ACTIVATIONS = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "elu": nn.ELU,
    }

    @classmethod
    def create(cls, name: str) -> nn.Module:
        """Tạo activation tương ứng hoặc báo lỗi nếu tên không hợp lệ.

        Args:
            name: Tên activation: relu, leaky_relu, gelu hoặc elu.

        Returns:
            nn.Module: Module kích hoạt mới.
        """
        normalized_name = name.strip().lower()
        activation_type: Optional[Type[nn.Module]] = cls._ACTIVATIONS.get(normalized_name)
        if activation_type is None:
            allowed = ", ".join(cls._ACTIVATIONS)
            raise ValueError(f"Activation '{name}' không hợp lệ. Chọn một trong: {allowed}.")
        if activation_type is nn.ReLU:
            return activation_type(inplace=True)
        if activation_type is nn.LeakyReLU:
            return activation_type(negative_slope=0.01, inplace=True)
        return activation_type()


class _CustomCNNBase(BaseEmotionModel):
    """Lớp cơ sở dùng chung để khởi tạo các tầng cho CNN nhiều khối."""

    def _validate_channels(self, channels: Sequence[int], expected_blocks: int) -> List[int]:
        """Kiểm tra danh sách số kênh theo độ sâu kiến trúc."""
        channel_list = list(channels)
        if len(channel_list) != expected_blocks or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in channel_list
        ):
            raise ValueError(f"channels phải chứa {expected_blocks} số nguyên dương.")
        return channel_list

    def _set_layer(self, name: str, layer: nn.Module) -> None:
        """Đăng ký một tầng theo tên ổn định để giữ tương thích checkpoint."""
        setattr(self, name, layer)

    def _build_blocks(
        self,
        channels: Sequence[int],
        activation: str,
        batch_norm: bool,
        spatial_dropout: Sequence[float],
    ) -> None:
        """Khởi tạo các khối Conv, activation, pooling và dropout theo cấu hình."""
        in_channels = self.in_channels
        for index, out_channels in enumerate(channels, start=1):
            self._set_layer(
                f"conv{index}",
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            )
            if batch_norm:
                self._set_layer(f"bn{index}", nn.BatchNorm2d(out_channels))
            self._set_layer(f"activation{index}", ActivationFactory.create(activation))
            self._set_layer(f"pool{index}", nn.MaxPool2d(kernel_size=2, stride=2))
            if spatial_dropout[index - 1] > 0:
                self._set_layer(f"drop{index}", nn.Dropout2d(p=spatial_dropout[index - 1]))
            in_channels = out_channels

    def _forward_blocks(self, x: torch.Tensor, block_count: int, batch_norm: bool) -> torch.Tensor:
        """Lan truyền tensor tuần tự qua các khối tích chập."""
        for index in range(1, block_count + 1):
            x = getattr(self, f"conv{index}")(x)
            if batch_norm:
                x = getattr(self, f"bn{index}")(x)
            x = getattr(self, f"activation{index}")(x)
            x = getattr(self, f"pool{index}")(x)
            drop = getattr(self, f"drop{index}", None)
            if drop is not None:
                x = drop(x)
        return x

    def _build_classifier(
        self,
        last_channels: int,
        block_count: int,
        hidden_features: int,
        num_classes: int,
        dropout_rate: float,
        batch_norm: bool,
    ) -> None:
        """Khởi tạo bộ phân loại với kích thước đặc trưng suy ra từ độ sâu."""
        spatial_size = 48 // (2 ** block_count)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(last_channels * spatial_size * spatial_size, hidden_features)
        if batch_norm:
            self.bn_fc = nn.BatchNorm1d(hidden_features)
        self.activation_fc = ActivationFactory.create(self.activation_name)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_features, num_classes)

    def _forward_classifier(self, x: torch.Tensor, batch_norm: bool) -> torch.Tensor:
        """Lan truyền qua các tầng fully-connected và trả về logits."""
        x = self.flatten(x)
        x = self.fc1(x)
        if batch_norm:
            x = self.bn_fc(x)
        x = self.activation_fc(x)
        x = self.dropout(x)
        return self.fc2(x)


class MediumCNN(_CustomCNNBase):
    """Mô hình CNN ba khối, mặc định có số kênh 64 → 128 → 256."""

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 1,
        dropout: float = 0.5,
        activation: str = "relu",
        channels: Sequence[int] = (64, 128, 256),
        dropout_rate: Optional[float] = None,
    ) -> None:
        """Khởi tạo MediumCNN theo tham số cấu hình.

        Args:
            num_classes: Số lớp cảm xúc đầu ra.
            in_channels: Số kênh ảnh đầu vào.
            dropout: Tỷ lệ dropout tương thích cấu hình phiên bản trước.
            activation: Tên hàm kích hoạt cần dùng.
            channels: Số kênh đầu ra của ba khối tích chập.
            dropout_rate: Tỷ lệ dropout mới; nếu bỏ trống dùng ``dropout``.
        """
        super().__init__(num_classes=num_classes, in_channels=in_channels)
        self.channels = self._validate_channels(channels, expected_blocks=3)
        self.activation_name = activation
        self.dropout_rate = dropout if dropout_rate is None else float(dropout_rate)
        ActivationFactory.create(activation)  # Xác thực trước khi cấp phát các tầng.
        self._build_blocks(self.channels, activation, False, (0.0, 0.0, 0.0))
        self._build_classifier(
            self.channels[-1], 3, self.channels[-1], num_classes, self.dropout_rate, False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền ảnh qua ba khối và bộ phân loại để sinh logits."""
        x = self._forward_blocks(x, block_count=3, batch_norm=False)
        return self._forward_classifier(x, batch_norm=False)

    def get_target_layer(self) -> nn.Module:
        """Trả về tầng tích chập cuối phục vụ Grad-CAM."""
        return self.conv3


class LargeCNN(_CustomCNNBase):
    """Mô hình CNN bốn khối có BatchNorm2d và Dropout2d."""

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 1,
        dropout: float = 0.5,
        activation: str = "relu",
        channels: Sequence[int] = (64, 128, 256, 512),
        dropout_rate: Optional[float] = None,
    ) -> None:
        """Khởi tạo LargeCNN theo tham số cấu hình.

        Args:
            num_classes: Số lớp cảm xúc đầu ra.
            in_channels: Số kênh ảnh đầu vào.
            dropout: Tỷ lệ dropout tương thích cấu hình phiên bản trước.
            activation: Tên hàm kích hoạt cần dùng.
            channels: Số kênh đầu ra của bốn khối tích chập.
            dropout_rate: Tỷ lệ dropout mới; nếu bỏ trống dùng ``dropout``.
        """
        super().__init__(num_classes=num_classes, in_channels=in_channels)
        self.channels = self._validate_channels(channels, expected_blocks=4)
        self.activation_name = activation
        self.dropout_rate = dropout if dropout_rate is None else float(dropout_rate)
        ActivationFactory.create(activation)
        self._build_blocks(self.channels, activation, True, (0.2, 0.2, 0.3, 0.3))
        self._build_classifier(
            self.channels[-1], 4, self.channels[-1], num_classes, self.dropout_rate, True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền ảnh qua bốn khối có chuẩn hóa và bộ phân loại."""
        x = self._forward_blocks(x, block_count=4, batch_norm=True)
        return self._forward_classifier(x, batch_norm=True)

    def get_target_layer(self) -> nn.Module:
        """Trả về tầng tích chập cuối phục vụ Grad-CAM."""
        return self.conv4