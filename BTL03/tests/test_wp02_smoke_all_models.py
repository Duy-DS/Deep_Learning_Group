"""Kiểm thử tích hợp parity và huấn luyện một epoch mười cấu hình WP02."""

from __future__ import annotations

import ast
import csv
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from scripts.wp02_smoke import PROJECT_ROOT, WP02_CASES, WP02SmokeHarness
from src.datasets import WikiGenderDataset
from src.train import ProjectTrainingCLI


def test_wp02_parameter_parity_all_ten_cases() -> None:
    """Mười cấu hình khớp baseline; EXP-01 dùng B=16 nên có 65 params."""
    harness = WP02SmokeHarness()
    for case in WP02_CASES:
        torch.manual_seed(42)
        model = harness.model(case)
        assert model.count_parameters()["total_params"] == case["params"], case["key"]


@pytest.mark.parametrize("case", WP02_CASES, ids=[case["key"] for case in WP02_CASES])
def test_wp02_one_epoch_smoke(case: dict, tmp_path: Path) -> None:
    """Huấn luyện/validation một epoch và cô lập mọi log/artifact trong tmp_path."""
    torch.set_num_threads(1)
    repository_master = PROJECT_ROOT / "experiments/csv_logs/master_experiment_runs.csv"
    original_master = repository_master.read_bytes()
    result = WP02SmokeHarness().run_case(case, tmp_path)
    assert result["status"] == "PASS"
    assert result["delta"] == 0
    assert len(result["history"]["train_loss"]) == 1
    assert result["logger_status"].startswith(("PASS", "BLOCKED:"))
    assert repository_master.read_bytes() == original_master
    assert all(
        tmp_path.resolve() in path.resolve().parents
        for path in tmp_path.rglob("*") if path.is_file()
    )
    if case["group"] in {"04_training_utilities", "05_others"}:
        assert Path(result["checkpoint"]).is_file()


def test_wp02_configs_use_seed42_names_and_relative_paths() -> None:
    """Mười cấu hình có model_name, seed cố định và không hardcode máy cá nhân."""
    config_names = (
        "mlp_func_apxm.yaml", "mlp_mnist.yaml", "cnn_mnist.yaml", "cnn_gender.yaml",
        "cnn_emotion.yaml", "cnn_lstm_mnist.yaml", "utilities_gender.yaml",
        "utilities_emotion.yaml", "segnet.yaml", "fcn_obj_detection.yaml",
    )
    for filename in config_names:
        config = yaml.safe_load((PROJECT_ROOT / "configs" / filename).read_text(encoding="utf-8"))
        assert config["seed"] == 42
        assert config.get("model_name")
        assert config.get("optimizer")
        assert config.get("criterion")
        assert config.get("epochs")
        assert config.get("lr", config.get("learning_rate")) > 0
        for key, value in config.items():
            if isinstance(value, str) and key.endswith(("root", "path", "images", "labels", "csv", "figure", "filepath")):
                assert not Path(value).is_absolute(), f"{filename}: {key}={value}"


def test_wp02_wiki_rotation_and_nchw_contract() -> None:
    """WIKI dùng rot90(k=3) trước NHWC→NCHW, đúng vị trí pixel dự kiến."""
    images = np.zeros((1, 64, 64, 1), dtype=np.uint8)
    images[0, 2, 5, 0] = 255
    dataset = WikiGenderDataset(images, np.array([1]), apply_legacy_rotation=True, normalization="auto")
    assert dataset.images.shape == (1, 1, 64, 64)
    assert torch.nonzero(dataset.images[0, 0], as_tuple=False).tolist() == [[5, 61]]
    assert next(iter(DataLoader(dataset, batch_size=1)))[0].shape == (1, 1, 64, 64)


def test_wp02_source_has_no_tensorflow_or_keras_imports() -> None:
    """AST audit bảo đảm runtime source thuần PyTorch."""
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            assert not any(
                module == prefix or module.startswith(prefix + ".")
                for module in modules for prefix in ("tensorflow", "keras")
            ), path


def test_wp02_unified_cli_smoke_dispatches_to_temp_path_and_restores_config(tmp_path: Path, monkeypatch) -> None:
    """CLI config path chạy một epoch synthetic, không sửa YAML/master chính."""
    import src.train as train_module

    original_config = PROJECT_ROOT / "configs/cnn_lstm_mnist.yaml"
    original_bytes = original_config.read_bytes()
    original_master = (PROJECT_ROOT / "experiments/csv_logs/master_experiment_runs.csv").read_bytes()
    monkeypatch.setattr(train_module.tempfile, "tempdir", str(tmp_path))
    result = ProjectTrainingCLI().run(
        "configs/cnn_lstm_mnist.yaml", seed=42, epochs=1, smoke_test=True, member_alias="Duy-DS"
    )
    assert result["status"] == "PASS"
    assert result["logger_status"].startswith("BLOCKED:")
    assert original_config.read_bytes() == original_bytes
    assert (PROJECT_ROOT / "experiments/csv_logs/master_experiment_runs.csv").read_bytes() == original_master


def test_wp02_source_has_zero_keras_imports() -> None:
    """Source ứng dụng chạy không import TensorFlow/Keras; baseline helper bị loại trừ."""
    forbidden_roots = (PROJECT_ROOT / "src", PROJECT_ROOT / "app")
    for root in forbidden_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                module = node.module if isinstance(node, ast.ImportFrom) else None
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                else:
                    names = [module or ""]
                assert not any(
                    name == prefix or name.startswith(prefix + ".")
                    for name in names for prefix in ("tensorflow", "keras")
                ), path