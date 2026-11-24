"""Mô hình EfficientNet-B0 ImageNet tinh chỉnh cho FER2013."""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

from src.models.base_model import BaseEmotionModel


class PretrainedEfficientNetB0(BaseEmotionModel):
    """EfficientNet-B0 RGB với classifier 7 lớp và điều khiển fine-tuning."""

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.2,
        hidden_dim: Optional[int] = None,
    ) -> None:
        """Khởi tạo EfficientNet-B0 và thay đầu phân loại.

        Args:
            num_classes: Số lớp cảm xúc đầu ra.
            pretrained: Dùng trọng số ImageNet V1 nếu đúng.
            freeze_backbone: Đóng băng features lúc khởi tạo.
            dropout_rate: Tỷ lệ dropout classifier.
            hidden_dim: Tầng ẩn tùy chọn; ``None`` giữ head gọn một tầng Linear.
        """
        super().__init__(num_classes=num_classes, in_channels=3)
        if isinstance(num_classes, bool) or num_classes <= 0:
            raise ValueError("num_classes phải là số nguyên dương.")
        if hidden_dim is not None and (
            isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0
        ):
            raise ValueError("hidden_dim phải là số nguyên dương hoặc None.")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate phải nằm trong [0, 1).")

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.pretrained = pretrained
        self.backbone = models.efficientnet_b0(weights=weights)
        self.backbone.classifier = self._build_classifier_head(num_classes, dropout_rate, hidden_dim)
        if freeze_backbone:
            self.freeze_backbone()

    def _build_classifier_head(
        self, num_classes: int, dropout_rate: float, hidden_dim: Optional[int]
    ) -> nn.Sequential:
        """Tạo classifier 1280 đặc trưng, có hoặc không có tầng ẩn."""
        if hidden_dim is None:
            classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(1280, num_classes),
            )
            linear_layers = (classifier[1],)
        else:
            classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(1280, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, num_classes),
            )
            linear_layers = (classifier[1], classifier[4])
        for index, layer in enumerate(linear_layers):
            if index == 0 and hidden_dim is not None:
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            else:
                nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        return classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền batch ảnh RGB và trả logits cảm xúc."""
        return self.backbone(x)

    def get_target_layer(self) -> nn.Module:
        """Trả về convolution cuối cùng trong features phục vụ Grad-CAM."""
        for layer in reversed(self.backbone.features):
            for module in reversed(list(layer.modules())):
                if isinstance(module, nn.Conv2d):
                    return module
        raise RuntimeError("Không tìm thấy Conv2d trong EfficientNet-B0 features.")

    def freeze_backbone(self) -> None:
        """Đóng băng feature extractor và giữ classifier trainable."""
        for parameter in self.backbone.features.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.classifier.parameters():
            parameter.requires_grad = True

    def unfreeze_backbone(self, from_block: Optional[int] = None) -> None:
        """Mở toàn features hoặc mở từ block chỉ định (index 0–8) đến cuối.

        Args:
            from_block: Chỉ số block trong ``features``; ``None`` mở toàn bộ.
        """
        if from_block is None:
            modules = self.backbone.features
        else:
            if isinstance(from_block, bool) or not isinstance(from_block, int) or not 0 <= from_block <= 8:
                raise ValueError("from_block phải là số nguyên từ 0 đến 8 hoặc None.")
            modules = list(self.backbone.features)[from_block:]
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
        for parameter in self.backbone.classifier.parameters():
            parameter.requires_grad = True

    def get_parameter_summary(self) -> Dict[str, int]:
        """Trả về tổng tham số, tham số trainable và tham số đóng băng."""
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"total_params": total, "trainable_params": trainable, "frozen_params": total - trainable}