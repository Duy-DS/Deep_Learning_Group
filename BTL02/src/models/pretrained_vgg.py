"""Mô hình VGG16 ImageNet tinh chỉnh cho bài toán FER2013."""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights

from src.models.base_model import BaseEmotionModel


class PretrainedVGG16(BaseEmotionModel):
    """VGG16 nhận ảnh RGB và phân loại thành các lớp cảm xúc FER2013."""

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5,
        hidden_dim: int = 512,
    ) -> None:
        """Khởi tạo VGG16 với đầu phân loại mới.

        Args:
            num_classes: Số lớp đầu ra.
            pretrained: Dùng trọng số VGG16 ImageNet V1 nếu là ``True``.
            freeze_backbone: Đóng băng feature extractor lúc khởi tạo.
            dropout_rate: Tỷ lệ dropout trong đầu phân loại.
            hidden_dim: Số nơ-ron ở tầng ẩn của đầu phân loại.
        """
        super().__init__(num_classes=num_classes, in_channels=3)
        if num_classes <= 0 or hidden_dim <= 0:
            raise ValueError("num_classes và hidden_dim phải là số nguyên dương.")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate phải nằm trong [0, 1).")

        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        self.pretrained = pretrained
        self.backbone = models.vgg16(weights=weights)
        self.backbone.classifier = self._build_classifier_head(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )
        if freeze_backbone:
            self.freeze_backbone()

    def _build_classifier_head(
        self, num_classes: int, hidden_dim: int, dropout_rate: float
    ) -> nn.Sequential:
        """Tạo đầu phân loại mới và khởi tạo trọng số phù hợp."""
        classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )
        nn.init.kaiming_normal_(classifier[0].weight, nonlinearity="relu")
        nn.init.zeros_(classifier[0].bias)
        nn.init.xavier_uniform_(classifier[3].weight)
        nn.init.zeros_(classifier[3].bias)
        return classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tạo logits lớp cảm xúc từ batch ảnh RGB."""
        return self.backbone(x)

    def get_target_layer(self) -> nn.Module:
        """Trả về convolution cuối của VGG16 để hỗ trợ Grad-CAM."""
        for layer in reversed(self.backbone.features):
            if isinstance(layer, nn.Conv2d):
                return layer
        raise RuntimeError("Không tìm thấy tầng Conv2d trong VGG16 features.")

    def freeze_backbone(self) -> None:
        """Đóng băng toàn bộ backbone features, giữ classifier có thể huấn luyện."""
        for parameter in self.backbone.features.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.classifier.parameters():
            parameter.requires_grad = True

    def unfreeze_backbone(self, from_block: Optional[int] = None) -> None:
        """Mở khóa backbone toàn bộ hoặc từ một trong năm block tích chập.

        Args:
            from_block: Block VGG (1–5) bắt đầu mở khóa đến cuối; ``None`` mở hết.
        """
        if from_block is None:
            for parameter in self.backbone.features.parameters():
                parameter.requires_grad = True
        else:
            if (
                isinstance(from_block, bool)
                or not isinstance(from_block, int)
                or not 1 <= from_block <= 5
            ):
                raise ValueError("from_block phải là số nguyên từ 1 đến 5 hoặc None.")
            block = 1
            for layer in self.backbone.features:
                if block >= from_block:
                    for parameter in layer.parameters():
                        parameter.requires_grad = True
                if isinstance(layer, nn.MaxPool2d):
                    block += 1
        for parameter in self.backbone.classifier.parameters():
            parameter.requires_grad = True

    def get_parameter_summary(self) -> Dict[str, int]:
        """Tóm tắt số tham số toàn mô hình, đang học và đang đóng băng."""
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": total - trainable,
        }