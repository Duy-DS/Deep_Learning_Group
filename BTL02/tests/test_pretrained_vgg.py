"""Kiểm thử mô hình VGG16 pretrained và tích hợp pipeline SOTA."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn
import yaml
from torchvision.models import VGG16_Weights

from src.data.transforms_sota import SOTATransformBuilder
from src.models import PretrainedVGG16
from src.train.trainer import Trainer

torch.set_num_threads(1)


class TestPretrainedVGG16:
    """Kiểm tra đầu ra, weights và chính sách fine-tuning của VGG16."""

    @pytest.mark.parametrize("image_size", [128, 224])
    def test_forward_output_shape_and_finite_values(self, image_size: int) -> None:
        """Hai độ phân giải ảnh SOTA đều tạo logits hữu hạn cho bảy lớp."""
        model = PretrainedVGG16(pretrained=False).eval()
        with torch.no_grad():
            output = model(torch.randn(2, 3, image_size, image_size))
        assert output.shape == (2, 7)
        assert torch.isfinite(output).all()

    def test_pretrained_weights_enum_and_loading_choice(self) -> None:
        """Enum ImageNet V1 hợp lệ và được truyền đúng tùy chọn khởi tạo."""
        assert VGG16_Weights.IMAGENET1K_V1 is not None
        with patch("src.models.pretrained_vgg.models.vgg16", wraps=__import__(
            "torchvision.models", fromlist=["vgg16"]
        ).vgg16) as factory:
            model = PretrainedVGG16(pretrained=False)
        assert model.pretrained is False
        assert factory.call_args.kwargs["weights"] is None

    def test_freeze_unfreeze_and_parameter_summary(self) -> None:
        """Freeze chỉ khóa features; unfreeze mở toàn mạng và summary chính xác."""
        model = PretrainedVGG16(pretrained=False)
        total = sum(parameter.numel() for parameter in model.parameters())
        model.freeze_backbone()
        assert all(not p.requires_grad for p in model.backbone.features.parameters())
        assert all(p.requires_grad for p in model.backbone.classifier.parameters())
        frozen_summary = model.get_parameter_summary()
        assert frozen_summary["total_params"] == total
        assert frozen_summary["trainable_params"] < total
        assert frozen_summary["frozen_params"] == total - frozen_summary["trainable_params"]

        model.unfreeze_backbone()
        summary = model.get_parameter_summary()
        assert all(p.requires_grad for p in model.parameters())
        assert summary["trainable_params"] == summary["total_params"] == total
        assert isinstance(model.get_target_layer(), nn.Conv2d)

    def test_partial_unfreeze_opens_selected_later_blocks(self) -> None:
        """Mở khóa từ block ba giữ hai block đầu đóng băng."""
        model = PretrainedVGG16(pretrained=False)
        model.freeze_backbone()
        model.unfreeze_backbone(from_block=3)
        conv_block = 1
        flags = []
        for layer in model.backbone.features:
            if isinstance(layer, nn.Conv2d):
                flags.append((conv_block, layer.weight.requires_grad))
            if isinstance(layer, nn.MaxPool2d):
                conv_block += 1
        assert all(not enabled for block, enabled in flags if block < 3)
        assert all(enabled for block, enabled in flags if block >= 3)

    def test_pretrained_true_requests_imagenet_weights(self) -> None:
        """Chế độ pretrained yêu cầu đúng bộ trọng số ImageNet V1."""
        initialized = PretrainedVGG16(pretrained=False).backbone
        with patch("src.models.pretrained_vgg.models.vgg16") as factory:
            factory.return_value = initialized
            PretrainedVGG16(pretrained=True)
        assert factory.call_args.kwargs["weights"] is VGG16_Weights.IMAGENET1K_V1

    def test_sota_transform_forward_backward_and_exp04_config(self) -> None:
        """Cấu hình EXP-04, biến đổi ảnh RGB và một bước backward tương thích."""
        config_path = Path(__file__).parents[1] / "experiments/sota/exp04_vgg16_finetune.yaml"
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        assert config["exp_group"] == "EXP-04"
        assert config["rq_tag"] == "RQ2_TransferLearning"
        assert config["transform_mode"] == "sota"
        assert config["image_size"] == 128

        builder = SOTATransformBuilder(image_size=128, use_augmentation=False)
        image = Image.fromarray(np.zeros((48, 48), dtype=np.uint8), mode="L")
        batch = builder.build_eval_transform()(image).unsqueeze(0).repeat(2, 1, 1, 1)
        model = PretrainedVGG16(pretrained=False, freeze_backbone=True)
        logits = model(batch)
        loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1]))
        loss.backward()
        assert logits.shape == (2, 7)
        assert all(p.grad is None for p in model.backbone.features.parameters())
        assert any(p.grad is not None for p in model.backbone.classifier.parameters())

    def test_trainer_builds_vgg_and_selects_sota_loader(self) -> None:
        """Trainer dùng đúng model VGG và truyền transform SOTA từ YAML."""
        config = {
            "architecture": "PretrainedVGG16",
            "pretrained": False,
            "freeze_backbone": True,
            "image_size": 128,
            "in_channels": 3,
            "normalization": "imagenet",
            "transform_mode": "sota",
            "use_augmentation": True,
            "optimizer": "adamw",
            "scheduler": "ReduceLROnPlateau",
            "scheduler_params": {"mode": "min", "patience": 2},
        }
        with patch("src.train.trainer.get_dataloaders", return_value=([], [], [])) as loader:
            trainer = Trainer(config)
        assert isinstance(trainer.model, PretrainedVGG16)
        assert trainer.model.pretrained is False
        assert loader.call_args.kwargs["transform_mode"] == "sota"
        assert loader.call_args.kwargs["use_augmentation"] is True
        optimizer_parameters = {
            id(parameter)
            for group in trainer.optimizer.param_groups
            for parameter in group["params"]
        }
        assert all(id(parameter) in optimizer_parameters for parameter in trainer.model.parameters())
        trainer.model.unfreeze_backbone()
        assert all(
            id(parameter) in optimizer_parameters
            for parameter in trainer.model.backbone.features.parameters()
        )
        assert isinstance(trainer.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_trainer_keeps_legacy_cnn_and_custom_transform_defaults(self) -> None:
        """Việc thêm SOTA không đổi model hay pipeline mặc định của SmallCNN."""
        from src.models import SmallCNN

        with patch("src.train.trainer.get_dataloaders", return_value=([], [], [])) as loader:
            trainer = Trainer({"architecture": "SmallCNN"})
        assert isinstance(trainer.model, SmallCNN)
        assert loader.call_args.kwargs["transform_mode"] == "custom"