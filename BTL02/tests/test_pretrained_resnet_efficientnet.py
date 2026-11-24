"""Kiểm thử ResNet50 và EfficientNet-B0 pretrained cho FER2013."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn
import yaml
from torchvision import models as torchvision_models
from torchvision.models import EfficientNet_B0_Weights, ResNet50_Weights

from src.data.transforms_sota import SOTATransformBuilder
from src.models import PretrainedEfficientNetB0, PretrainedResNet50
from src.train.trainer import Trainer

torch.set_num_threads(1)


class TestPretrainedResNetEfficientNet:
    """Xác minh shape, weights, freeze policy và luồng huấn luyện hai kiến trúc."""

    @pytest.mark.parametrize("model_type", [PretrainedResNet50, PretrainedEfficientNetB0])
    @pytest.mark.parametrize("image_size", [128, 224])
    def test_forward_shapes_and_finite_outputs(self, model_type, image_size: int) -> None:
        """Cả hai mạng nhận hai kích thước SOTA và trả logits hữu hạn [B,7]."""
        model = model_type(pretrained=False).eval()
        with torch.no_grad():
            output = model(torch.randn(2, 3, image_size, image_size))
        assert output.shape == (2, 7)
        assert torch.isfinite(output).all()

    def test_imagenet_weight_enums_are_available(self) -> None:
        """Các enum ImageNet đúng phiên bản được khai báo trong torchvision."""
        assert ResNet50_Weights.IMAGENET1K_V2 is not None
        assert EfficientNet_B0_Weights.IMAGENET1K_V1 is not None

    @pytest.mark.parametrize(
        "model_type, module_path, weights_enum",
        [
            (PretrainedResNet50, "src.models.pretrained_resnet", ResNet50_Weights.IMAGENET1K_V2),
            (
                PretrainedEfficientNetB0,
                "src.models.pretrained_efficientnet",
                EfficientNet_B0_Weights.IMAGENET1K_V1,
            ),
        ],
    )
    def test_pretrained_flag_selects_expected_weights(
        self, model_type, module_path: str, weights_enum
    ) -> None:
        """True chọn enum ImageNet, false truyền weights=None cho torchvision."""
        with patch(f"{module_path}.models") as model_factory:
            factory = model_factory.resnet50 if model_type is PretrainedResNet50 else model_factory.efficientnet_b0
            if model_type is PretrainedResNet50:
                factory.side_effect = lambda **kwargs: torchvision_models.resnet50(weights=None)
            else:
                factory.side_effect = lambda **kwargs: torchvision_models.efficientnet_b0(weights=None)
            model_type(pretrained=True)
            assert factory.call_args.kwargs["weights"] is weights_enum
            model_type(pretrained=False)
            assert factory.call_args.kwargs["weights"] is None

    def test_resnet_freeze_partial_and_full_unfreeze(self) -> None:
        """ResNet giữ fc trainable, mở layer4 riêng hoặc mở toàn bộ backbone."""
        model = PretrainedResNet50(pretrained=False)
        model.freeze_backbone()
        assert all(not p.requires_grad for name in ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")
                   for p in getattr(model.backbone, name).parameters())
        assert all(p.requires_grad for p in model.backbone.fc.parameters())
        frozen = model.get_parameter_summary()
        assert frozen["trainable_params"] < frozen["total_params"]

        model.unfreeze_backbone(from_layer=4)
        assert all(not p.requires_grad for name in ("conv1", "bn1", "layer1", "layer2", "layer3")
                   for p in getattr(model.backbone, name).parameters())
        assert all(p.requires_grad for p in model.backbone.layer4.parameters())
        partial = model.get_parameter_summary()
        assert frozen["trainable_params"] < partial["trainable_params"] < partial["total_params"]

        model.unfreeze_backbone()
        summary = model.get_parameter_summary()
        assert summary["trainable_params"] == summary["total_params"]
        assert summary["frozen_params"] == 0
        assert isinstance(model.get_target_layer(), nn.Conv2d)

    def test_efficientnet_freeze_partial_and_full_unfreeze(self) -> None:
        """EfficientNet mở classifier khi freeze, block cuối rồi toàn features."""
        model = PretrainedEfficientNetB0(pretrained=False)
        model.freeze_backbone()
        assert all(not p.requires_grad for p in model.backbone.features.parameters())
        assert all(p.requires_grad for p in model.backbone.classifier.parameters())
        frozen = model.get_parameter_summary()
        assert frozen["trainable_params"] < frozen["total_params"]

        model.unfreeze_backbone(from_block=7)
        features = list(model.backbone.features)
        assert all(not p.requires_grad for block in features[:7] for p in block.parameters())
        assert all(p.requires_grad for block in features[7:] for p in block.parameters())
        partial = model.get_parameter_summary()
        assert frozen["trainable_params"] < partial["trainable_params"] < partial["total_params"]

        model.unfreeze_backbone()
        summary = model.get_parameter_summary()
        assert summary["trainable_params"] == summary["total_params"]
        assert summary["frozen_params"] == 0
        assert isinstance(model.get_target_layer(), nn.Conv2d)

    @pytest.mark.parametrize(
        "config_name, expected_type",
        [
            ("exp05_resnet50_finetune.yaml", PretrainedResNet50),
            ("exp06_efficientnet_b0_finetune.yaml", PretrainedEfficientNetB0),
        ],
    )
    def test_trainer_yaml_and_sota_forward_backward(
        self, config_name: str, expected_type
    ) -> None:
        """Trainer nạp YAML SOTA và từng model chạy transform, loss, backward."""
        config_path = Path(__file__).parents[1] / "experiments/sota" / config_name
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        assert config["seed"] == 42
        assert config["image_size"] == 128
        assert config["in_channels"] == 3
        assert config["normalization"] == "imagenet"
        assert config["transform_mode"] == "sota"

        # Mô hình test dùng weights=None để tránh tải tài nguyên mạng.
        config["pretrained"] = False
        with patch("src.train.trainer.get_dataloaders", return_value=([], [], [])):
            trainer = Trainer(config)
        assert isinstance(trainer.model, expected_type)
        transform = SOTATransformBuilder(image_size=128, use_augmentation=False)
        image = Image.fromarray(np.zeros((48, 48), dtype=np.uint8), mode="L")
        batch = transform.build_eval_transform()(image).unsqueeze(0).repeat(2, 1, 1, 1)
        batch = batch.to(trainer.device)
        labels = torch.tensor([0, 1], device=trainer.device)
        logits = trainer.model(batch)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        assert logits.shape == (2, 7)
        assert torch.isfinite(loss)
        assert any(parameter.grad is not None for parameter in trainer.model.parameters()
                   if parameter.requires_grad)