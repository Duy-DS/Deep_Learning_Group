"""Mô hình ResNet50 ImageNet tinh chỉnh cho nhận diện cảm xúc FER2013."""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from src.models.base_model import BaseEmotionModel


class PretrainedResNet50(BaseEmotionModel):
    """ResNet50 RGB với đầu phân loại FER2013 và chính sách fine-tuning linh hoạt."""

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5,
        hidden_dim: Optional[int] = None,
    ) -> None:
        """Khởi tạo ResNet50 và thay tầng fully-connected cuối.

        Args:
            num_classes: Số lớp cảm xúc đầu ra.
            pretrained: Dùng trọng số ImageNet V2 nếu đúng, nếu sai khởi tạo ngẫu nhiên.
            freeze_backbone: Đóng băng extractor trừ đầu phân loại khi khởi tạo.
            dropout_rate: Tỷ lệ dropout trong classifier.
            hidden_dim: Số nơ-ron tầng ẩn; ``None`` dùng head tuyến tính chuẩn.
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

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.pretrained = pretrained
        self.backbone = models.resnet50(weights=weights)
        self.backbone.fc = self._build_classifier_head(num_classes, dropout_rate, hidden_dim)
        if freeze_backbone:
            self.freeze_backbone()

    def _build_classifier_head(
        self, num_classes: int, dropout_rate: float, hidden_dim: Optional[int]
    ) -> nn.Module:
        """Tạo classifier tuyến tính hoặc classifier có tầng ẩn."""
        if hidden_dim is None:
            classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(2048, num_classes),
            )
            linear_layers = (classifier[1],)
        else:
            classifier = nn.Sequential(
                nn.Linear(2048, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, num_classes),
            )
            linear_layers = (classifier[0], classifier[3])
        for index, layer in enumerate(linear_layers):
            if index == 0 and hidden_dim is not None:
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            else:
                nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        return classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền batch ảnh RGB và trả logits bảy lớp."""
        return self.backbone(x)

    def get_target_layer(self) -> nn.Module:
        """Trả về convolution cuối layer4 phục vụ Grad-CAM."""
        for layer in reversed(list(self.backbone.layer4.modules())):
            if isinstance(layer, nn.Conv2d):
                return layer
        raise RuntimeError("Không tìm thấy Conv2d trong ResNet50 layer4.")

    def freeze_backbone(self) -> None:
        """Đóng băng stem và bốn residual stages, giữ fc trainable."""
        for name in ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4"):
            for parameter in getattr(self.backbone, name).parameters():
                parameter.requires_grad = False
        for parameter in self.backbone.fc.parameters():
            parameter.requires_grad = True

    def unfreeze_backbone(self, from_layer: Optional[int] = None) -> None:
        """Mở toàn backbone hoặc mở từ residual stage chỉ định đến layer4.

        Args:
            from_layer: Stage từ 1 đến 4; ``None`` mở cả stem và mọi stage.
        """
        if from_layer is None:
            modules = (self.backbone.conv1, self.backbone.bn1, self.backbone.layer1,
                       self.backbone.layer2, self.backbone.layer3, self.backbone.layer4)
        else:
            if isinstance(from_layer, bool) or not isinstance(from_layer, int) or not 1 <= from_layer <= 4:
                raise ValueError("from_layer phải là số nguyên từ 1 đến 4 hoặc None.")
            modules = tuple(getattr(self.backbone, f"layer{index}") for index in range(from_layer, 5))
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
        for parameter in self.backbone.fc.parameters():
            parameter.requires_grad = True

    def get_parameter_summary(self) -> Dict[str, int]:
        """Trả về tổng tham số, tham số trainable và tham số đóng băng."""
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"total_params": total, "trainable_params": trainable, "frozen_params": total - trainable}