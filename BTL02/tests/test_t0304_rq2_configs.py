"""Kiểm thử ma trận cấu hình và tương thích input size của RQ2 T0304."""

from pathlib import Path
import ast
import json
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from src.data.transforms_sota import SOTATransformBuilder
from src.models import PretrainedEfficientNetB0, PretrainedResNet50, PretrainedVGG16
from src.train.trainer import Trainer

torch.set_num_threads(1)


class TestT0304RQ2Configurations:
    """Xác nhận recipe được kiểm soát và smoke test từng YAML qua Trainer."""

    _CONFIGS = (
        "exp04_vgg16_finetune.yaml",
        "exp05_resnet50_finetune.yaml",
        "exp06_efficientnet_b0_finetune.yaml",
        "exp04b_vgg16_scratch_128.yaml",
        "exp05b_resnet50_scratch_128.yaml",
        "exp06b_efficientnet_b0_scratch_128.yaml",
        "exp04c_vgg16_pretrained_48.yaml",
        "exp05c_resnet50_pretrained_48.yaml",
        "exp06c_efficientnet_b0_pretrained_48.yaml",
    )
    _ROOT = Path(__file__).resolve().parents[1]

    def _load_configs(self):
        """Đọc chín cấu hình RQ2 từ thư mục experiments/sota."""
        return {
            name: yaml.safe_load((self._ROOT / "experiments" / "sota" / name).read_text(encoding="utf-8"))
            for name in self._CONFIGS
        }

    def test_all_nine_configs_have_fair_controlled_recipe(self) -> None:
        """Mọi config giữ split, seed, augment, optimizer và lịch học đồng nhất."""
        configs = self._load_configs()
        assert len(configs) == 9
        controlled_fields = (
            "seed", "in_channels", "num_classes", "normalization", "transform_mode",
            "use_augmentation", "aug_config", "augmentation", "split_counts", "optimizer",
            "lr", "weight_decay", "scheduler", "scheduler_params", "batch_size", "epochs",
            "patience", "num_workers", "dropout", "dropout_rate",
        )
        for field in controlled_fields:
            assert len({str(config[field]) for config in configs.values()}) == 1, field
        assert configs[self._CONFIGS[0]]["seed"] == 42
        assert configs[self._CONFIGS[0]]["split_counts"] == {
            "Training": 28709, "PublicTest": 3589, "PrivateTest": 3589
        }
        assert {config["image_size"] for config in configs.values()} == {48, 128}
        assert {config["pretrained"] for config in configs.values()} == {True, False}
        run_names = {
            Trainer._format_run_name(
                exp_id=config["exp_id"],
                architecture=config["architecture"],
                optimizer=config["optimizer"],
                activation=config.get("activation", "relu"),
                pretrained=config["pretrained"],
                image_size=config["image_size"],
            )
            for config in configs.values()
        }
        assert len(run_names) == 9

    @pytest.mark.parametrize(
        "config_name, expected_type, image_size",
        [
            ("exp04_vgg16_finetune.yaml", PretrainedVGG16, 128),
            ("exp05_resnet50_finetune.yaml", PretrainedResNet50, 128),
            ("exp06_efficientnet_b0_finetune.yaml", PretrainedEfficientNetB0, 128),
            ("exp04b_vgg16_scratch_128.yaml", PretrainedVGG16, 128),
            ("exp05b_resnet50_scratch_128.yaml", PretrainedResNet50, 128),
            ("exp06b_efficientnet_b0_scratch_128.yaml", PretrainedEfficientNetB0, 128),
            ("exp04c_vgg16_pretrained_48.yaml", PretrainedVGG16, 48),
            ("exp05c_resnet50_pretrained_48.yaml", PretrainedResNet50, 48),
            ("exp06c_efficientnet_b0_pretrained_48.yaml", PretrainedEfficientNetB0, 48),
        ],
    )
    def test_each_yaml_builds_and_runs_forward_backward(
        self, config_name: str, expected_type, image_size: int
    ) -> None:
        """Mỗi YAML được nạp qua Trainer và chạy một bước loss/backward offline."""
        config = yaml.safe_load(
            (self._ROOT / "experiments" / "sota" / config_name).read_text(encoding="utf-8")
        )
        assert config["image_size"] == image_size
        requested_pretrained = config["pretrained"]
        # Smoke test không tải weights mạng; việc chọn enum được kiểm tra riêng ở T0302/T0303.
        config["pretrained"] = False
        device = torch.device("cpu")
        config["freeze_backbone"] = True
        with patch("src.train.trainer.get_device", return_value=device), patch(
            "src.train.trainer.get_dataloaders", return_value=([], [], [])
        ):
            trainer = Trainer(config)
        assert isinstance(trainer.model, expected_type)
        assert trainer.pretrained is False
        assert requested_pretrained in (True, False)

        image = Image.fromarray(np.zeros((48, 48), dtype=np.uint8), mode="L")
        transform = SOTATransformBuilder(image_size=image_size, use_augmentation=False)
        batch = transform.build_eval_transform()(image).unsqueeze(0).to(trainer.device)
        logits = trainer.model(batch)
        loss = torch.nn.CrossEntropyLoss()(logits, torch.tensor([0], device=trainer.device))
        loss.backward()
        assert logits.shape == (1, 7)
        assert torch.isfinite(logits).all()
        assert any(parameter.grad is not None for parameter in trainer.model.parameters()
                   if parameter.requires_grad)
        del trainer, logits, loss, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @pytest.mark.parametrize(
        "model_type", [PretrainedVGG16, PretrainedResNet50, PretrainedEfficientNetB0]
    )
    def test_all_sota_models_accept_rgb_48_input(self, model_type) -> None:
        """48×48 RGB smoke forward xác minh tensor shape của cả ba backbone."""
        image = Image.fromarray(np.zeros((48, 48), dtype=np.uint8), mode="L")
        batch = SOTATransformBuilder(image_size=48, use_augmentation=False).build_eval_transform()(image)
        model = model_type(pretrained=False).eval()
        with torch.no_grad():
            logits = model(batch.unsqueeze(0).repeat(2, 1, 1, 1))
        assert logits.shape == (2, 7)
        assert torch.isfinite(logits).all()

    def test_colab_notebook_is_valid_and_runs_all_nine_configs(self) -> None:
        """Notebook hợp lệ về JSON/cú pháp và liệt kê đủ ma trận YAML."""
        notebook = json.loads((self._ROOT / "notebooks" / "run_t0304_colab.ipynb").read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        for index, cell in enumerate(code_cells, start=1):
            ast.parse("".join(cell["source"]), filename=f"t0304-colab-cell-{index}")
        sources = "\n".join("".join(cell["source"]) for cell in code_cells)
        for config_name in self._CONFIGS:
            assert config_name in sources
        assert "t0304_rq2_transfer_vs_scratch_24col.csv" in sources
        assert "MLflowLogger.CSV_COLUMNS" in sources
        assert "params.experiment_id" not in sources
        assert "isin(set(manifest['run_name']))" in sources