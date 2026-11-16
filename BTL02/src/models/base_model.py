"""Lớp cơ sở trừu tượng (Abstract Base Class) cho tất cả các mô hình nhận diện cảm xúc.

Mọi kiến trúc mô hình (Baseline CNN, Custom CNN, Pretrained SOTA) trong dự án
bắt buộc phải kế thừa lớp này theo quy định tại AGENTS.md (Mục II.5).
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseEmotionModel(nn.Module, ABC):
    """Lớp cơ sở trừu tượng cho các kiến trúc mạng học sâu phân loại cảm xúc FER2013."""

    def __init__(self, num_classes: int = 7, in_channels: int = 1) -> None:
        """Khởi tạo thông số kích thước đầu vào và đầu ra của mô hình.

        Args:
            num_classes: Số lượng lớp cảm xúc đầu ra (mặc định 7 theo FER2013).
            in_channels: Số kênh ảnh đầu vào (1 cho Grayscale 48x48, 3 cho RGB).
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền tiến (Forward pass) trích xuất đặc trưng và xuất logits phân loại.

        Args:
            x (torch.Tensor): Tensor ảnh đầu vào kích thước [Batch_Size, In_Channels, Height, Width].

        Returns:
            torch.Tensor: Tensor logits chưa qua Softmax kích thước [Batch_Size, num_classes].
        """
        raise NotImplementedError("Các lớp mô hình con bắt buộc phải định nghĩa phương thức forward().")

    @abstractmethod
    def get_target_layer(self) -> nn.Module:
        """Trả về tầng tích chập (Conv2d) cuối cùng của mạng phục vụ trực quan hóa Grad-CAM.

        Returns:
            nn.Module: Tầng Conv2d mục tiêu để hook gradient và feature maps.
        """
        raise NotImplementedError("Các lớp mô hình con bắt buộc phải chỉ định tầng Conv cuối cho Grad-CAM.")

