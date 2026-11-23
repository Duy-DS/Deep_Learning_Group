"""Kiểm thử cấu hình và dữ liệu smoke cho runner T0209."""

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.run_group4_utilities import Group4UtilitiesRunner
from src.engine import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def test_group4_yaml_matches_legacy_and_documents_fer_extensions():
    """YAML giữ hyperparameters legacy và minh bạch extensions FER callbacks."""
    root = Path(__file__).resolve().parents[1] / "configs"
    gender = __import__("yaml").safe_load((root / "utilities_gender.yaml").read_text(encoding="utf-8"))
    emotion = __import__("yaml").safe_load((root / "utilities_emotion.yaml").read_text(encoding="utf-8"))
    assert (gender["learning_rate"], gender["batch_size"], gender["epochs"]) == (0.001, 128, 100000)
    assert gender["early_stopping"]["patience"] == 5
    assert gender["reduce_lr"]["factor"] == 0.1
    assert gender["reduce_lr"]["patience"] == 5
    assert gender["reduce_lr"]["min_lr"] == 0.0001
    assert (emotion["learning_rate"], emotion["batch_size"], emotion["epochs"]) == (0.0001, 128, 100)
    assert emotion["early_stopping"]["patience"] == 5
    assert emotion["augmentation"]["rotation_range_degrees"] == 45
    assert emotion["checkpoint"]["save_best_only"] is True


@pytest.mark.parametrize("task, expected_shape", [("gender", (1, 64, 64)), ("emotion", (1, 48, 48))])
def test_group4_smoke_datasets_preserve_contract_and_callbacks(tmp_path, task, expected_shape):
    """Dữ liệu giả có đúng shape/class, và factories tạo bộ callbacks đầy đủ."""
    runner = Group4UtilitiesRunner.__new__(Group4UtilitiesRunner)
    runner.project_root = tmp_path
    runner.member_alias = "SmokeTestMember"
    runner.configs = {
        "gender": {
            "seed": 42, "learning_rate": 0.001, "model_name": "HumanGenderCNN",
            "model_group": "04_training_utilities", "batch_size": 4, "epochs": 2,
            "num_workers": 0, "dropout": 0.2, "normalization": "auto",
            "apply_legacy_rotation": True, "data_root": "data/raw/wiki_gender",
            "train_images": "train_x.npy", "train_labels": "train_y.npy",
            "validation_images": "validation_x.npy", "validation_labels": "validation_y.npy",
            "early_stopping": {"monitor":"val_loss","min_delta":0,"patience":5,"mode":"min","restore_best_weights":True},
            "reduce_lr": {"monitor":"val_loss","factor":0.1,"patience":5,"min_delta":0.0001,"cooldown":0,"min_lr":0.0001,"mode":"min"},
            "checkpoint": {"monitor":"val_loss","mode":"min","save_best_only":True,"save_weights_only":False,"sync_to_drive":False,"filepath":"outputs/checkpoints/04_training_utilities/human_best.pt"},
            "figure_path":"outputs/figures/gender.png","csv_root":"experiments/csv_logs","tensorboard_root":"experiments/tensorboard/Group4_Utilities",
        },
        "emotion": {
            "seed": 42, "learning_rate": 0.0001, "model_name": "EmotionCNN",
            "model_group": "04_training_utilities", "batch_size": 4, "epochs": 2,
            "num_workers": 0, "dropout_block": 0.25, "dropout_dense": 0.5,
            "normalization":"preserve","data_root":"data/raw/fer2013",
            "train_csv":"train.csv","validation_csv":"validation.csv",
            "early_stopping": {"monitor":"val_loss","min_delta":0,"patience":5,"mode":"min","restore_best_weights":True},
            "reduce_lr": {"monitor":"val_loss","factor":0.1,"patience":5,"min_delta":0.0001,"cooldown":0,"min_lr":1e-6,"mode":"min"},
            "checkpoint": {"monitor":"val_loss","mode":"min","save_best_only":True,"save_weights_only":False,"sync_to_drive":False,"filepath":"outputs/checkpoints/04_training_utilities/emotion_best.pt"},
            "figure_path":"outputs/figures/emotion.png","csv_root":"experiments/csv_logs","tensorboard_root":"experiments/tensorboard/Group4_Utilities",
        },
    }
    data, validation = runner._load_datasets(task, runner.configs[task], smoke_test=True)
    image, label = data[0]
    assert tuple(image.shape) == expected_shape
    assert image.dtype == torch.float32
    assert label.dtype == torch.long
    val_image, _ = validation[0]
    assert tuple(val_image.shape) == expected_shape

    config = runner.configs[task]
    callbacks = runner._build_callbacks(
        task, runner.configs[task], torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=0.1), smoke_test=True
    )
    assert isinstance(callbacks[0], ModelCheckpoint)
    assert isinstance(callbacks[1], ReduceLROnPlateau)
    assert isinstance(callbacks[2], EarlyStopping)


def test_fer_augmentation_is_reproducible_and_validation_is_unwrapped():
    """FER train augmentation tái lập; validation giữ nguyên pixels/dataset."""
    from scripts.run_group2_cnn import LegacyFERAugmentationDataset
    from src.datasets import FEREmotionDataset

    images = np.zeros((2, 48, 48), dtype=np.float32)
    images[:, 12:36, 18:30] = 1.0
    labels = np.array([0, 1])
    base = FEREmotionDataset(images, labels, normalization="preserve")
    first = LegacyFERAugmentationDataset(base, seed=42)
    second = LegacyFERAugmentationDataset(base, seed=42)
    for index in range(len(base)):
        assert torch.equal(first[index][0], second[index][0])
    assert torch.equal(base[0][0], torch.from_numpy(images[0]).unsqueeze(0))


def test_summarywriter_import_failure_rolls_back_empty_run_artifacts(tmp_path):
    """Logger hỏng trước epoch không để run directory/epoch CSV rỗng như run thật."""
    from unittest.mock import patch

    from src.logger import ExperimentLogger

    def fail_import(name, *args, **kwargs):
        if name == "torch.utils.tensorboard":
            raise ImportError("simulated protobuf mismatch")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    with patch("builtins.__import__", side_effect=fail_import):
        with pytest.raises(RuntimeError, match="SummaryWriter"):
            ExperimentLogger(
                model_group="Group4_Utilities",
                model_name="SmokeModel",
                exp_type="SmokeTest_constructor_failure",
                member_alias="TestMember",
                project_root=tmp_path,
                csv_root="experiments/csv_logs",
                tensorboard_root="experiments/tensorboard",
            )
    csv_root = tmp_path / "experiments/csv_logs"
    run_dirs = list((tmp_path / "experiments/tensorboard").rglob("SmokeTest_constructor_failure*"))
    empty_epoch_logs = list(csv_root.glob("SmokeTest_constructor_failure*_epochs.csv"))
    assert run_dirs == []
    assert empty_epoch_logs == []
    master = csv_root / "master_experiment_runs.csv"
    assert master.is_file()
    assert len(master.read_text(encoding="utf-8-sig").splitlines()) == 1
