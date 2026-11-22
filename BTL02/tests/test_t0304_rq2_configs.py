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
        assert notebook["nbformat"] == 4
        assert len(notebook["cells"]) == 8
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert len(code_cells) == 8
        for index, cell in enumerate(code_cells, start=1):
            ast.parse("".join(cell["source"]), filename=f"t0304-colab-cell-{index}")
        sources = "\n".join("".join(cell["source"]) for cell in code_cells)
        for config_name in self._CONFIGS:
            assert config_name in sources
        assert "t0304_rq2_transfer_vs_scratch_24col.csv" in sources
        assert "MLflowLogger.CSV_COLUMNS" in sources
        assert "params.experiment_id" not in sources
        assert "sanitize_to_python_scalars" in sources
        assert "LOCAL_MLRUNS_DIR = '/content/mlruns_t0304'" in sources
        assert "LOCAL_CKPT_DIR = '/content/checkpoints_t0304'" in sources
        assert "sync_and_export_results" in sources
        assert "subprocess.Popen" in sources
        assert "Path('outputs/logs/t0304_all_runs_export.csv')" in sources
        assert "stable_age_seconds=drive_sync_stability_seconds" in sources
        assert "sync_and_export_results(sync_checkpoints=False, drive_sync_stability_seconds=5)" in sources
        cell_six = "".join(code_cells[5]["source"])
        assert "with sync_lock:" in cell_six
        assert "os.replace(drive_csv_tmp, drive_csv)" in cell_six
        assert "result[MLflowLogger.CSV_COLUMNS]" in cell_six
        assert "raw_export.unlink(missing_ok=True)" in cell_six
        cell_seven = "".join(code_cells[6]["source"])
        assert "sync_errors = []" in cell_seven
        assert "sync_and_export_results(sync_checkpoints=(return_code == 0))" in cell_seven
        assert "sync_and_export_results(sync_checkpoints=False, drive_sync_stability_seconds=5)" in cell_seven
        assert "finally:" in cell_seven

    def test_notebook_manifest_sanitizer_handles_numpy_2_values(self) -> None:
        """Hàm sanitizer trong notebook xử lý scalar/array NumPy trước DataFrame."""
        notebook = json.loads((self._ROOT / "notebooks" / "run_t0304_colab.ipynb").read_text(encoding="utf-8"))
        namespace = {"np": np}
        code = "".join(notebook["cells"][4]["source"])
        start = code.index("def sanitize_to_python_scalars")
        end = code.index("raw_configs = []")
        exec(code[start:end], namespace)
        sanitized = namespace["sanitize_to_python_scalars"](
            {
                "numpy_int": np.int64(42),
                "numpy_float": np.float64(0.25),
                "numpy_bool": np.bool_(True),
                "scalar_array": np.array(7),
                "array": np.array([1, 2]),
                "list": [np.int64(3)],
                "dict": {"value": np.float64(2.5)},
                "none": None,
            }
        )
        frame = __import__("pandas").DataFrame([sanitized])
        assert type(sanitized["numpy_int"]) is int
        assert type(sanitized["numpy_float"]) is float
        assert type(sanitized["numpy_bool"]) is bool
        assert sanitized["scalar_array"] == 7
        assert isinstance(sanitized["array"], str)
        assert frame.loc[0, "numpy_int"] == 42

    def test_notebook_run_names_match_trainer_and_drive_sync_is_atomic(self, tmp_path) -> None:
        """Run resume dùng đúng tên Trainer; Drive copy dùng replace và bỏ file đang đổi."""
        import os
        import shutil
        import time

        notebook = json.loads((self._ROOT / "notebooks" / "run_t0304_colab.ipynb").read_text(encoding="utf-8"))
        cell_five = "".join(notebook["cells"][4]["source"])
        config_source = cell_five[cell_five.index("CONFIG_FILES = [") : cell_five.index("from mlflow.tracking import MlflowClient")]
        config_source = config_source.replace("manifest = pd.DataFrame([sanitize_to_python_scalars(record) for record in raw_configs])", "manifest = None")
        config_source = config_source.replace("manifest = manifest.astype({'pretrained': bool, 'image_size': int, 'seed': int, 'batch_size': int, 'epochs': int, 'lr': float})", "manifest = None")
        config_source = config_source.replace("manifest['status'] = ['DONE' if name in finished_names else 'PENDING' for name in manifest['run_name']]", "manifest = None")
        namespace = {"Path": Path, "yaml": yaml}
        exec(config_source, namespace)
        for config in namespace["raw_configs"]:
            expected = Trainer._format_run_name(
                exp_id=config["exp_id"],
                architecture=config["architecture"],
                optimizer=config["optimizer"].lower(),
                pretrained=config["pretrained"],
                image_size=config["image_size"],
            )
            assert config["run_name"] == expected

        cell_two = "".join(notebook["cells"][1]["source"])
        sync_source = cell_two[cell_two.index("def sync_tree_new_files") : cell_two.index("# Resume:")]
        sync_namespace = {"os": os, "Path": Path, "shutil": shutil, "time": time}
        exec(sync_source, sync_namespace)
        source, destination = tmp_path / "source", tmp_path / "destination"
        (source / "nested").mkdir(parents=True)
        (source / "nested" / "run.meta").write_text("finished", encoding="utf-8")
        (source / "unfinished.part").write_text("partial", encoding="utf-8")
        sync_namespace["sync_tree_new_files"](source, destination)
        assert (destination / "nested" / "run.meta").read_text(encoding="utf-8") == "finished"
        assert not (destination / "unfinished.part").exists()