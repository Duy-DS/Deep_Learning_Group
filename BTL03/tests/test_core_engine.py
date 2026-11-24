"""Kiểm thử seed, model base, trainer và tích hợp logger/checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from src.engine import BaseTrainer
from src.logger import CheckpointManager, ExperimentLogger
from src.models import BaseMigratedModel
from src.utils import get_torch_generator, seed_everything, seed_worker


class TinyConvClassifier(BaseMigratedModel):
    """Mạng nhỏ Conv2d + Linear dùng để kiểm chứng determinism CPU."""

    def __init__(self) -> None:
        """Tạo một convolution, pooling và classifier 2 lớp."""
        super().__init__("TinyConvClassifier")
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tính logits từ ảnh NCHW một kênh."""
        self.validate_input_shape(x, expected_channels=1)
        return self.classifier(self.features(x).flatten(1))


def _make_loader(seed: int) -> DataLoader:
    """Tạo tập giả ổn định và DataLoader có generator riêng."""
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(24, 1, 8, 8, generator=generator)
    labels = (images.mean(dim=(1, 2, 3)) > 0).long()
    dataset = TensorDataset(images, labels)
    return DataLoader(
        dataset,
        batch_size=6,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=get_torch_generator(seed),
    )


def _run_reproducible_training(tmp_path: Path, run_name: str) -> tuple[dict, list[float], dict]:
    """Tạo model mới, train ba epoch và trả state/history/metadata."""
    seed_everything(42)
    model = TinyConvClassifier()
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = SGD(model.parameters(), lr=0.05)
    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        metric_fn=lambda logits, targets: (logits.argmax(1) == targets).float().mean(),
        model_group="Group0_SmokeTest",
        model_name="TinyConvClassifier",
        seed=42,
        config={"run": run_name, "batch_size": 6},
    )
    logger = ExperimentLogger(
        model_group="Group0_SmokeTest",
        model_name=f"TinyConvClassifier_{run_name}",
        exp_type=f"SmokeTest_{run_name}",
        seed=42,
        optimizer="SGD",
        learning_rate=0.05,
        batch_size=6,
        epochs=3,
        member_alias="CoreEngineTest",
        project_root=tmp_path,
    )
    manager = CheckpointManager(project_root=tmp_path)
    try:
        history = trainer.fit(
            _make_loader(42),
            _make_loader(43),
            epochs=3,
            logger=logger,
            checkpoint_manager=manager,
        )
    finally:
        logger.close()
    final = {name: value.detach().clone() for name, value in model.state_dict().items()}
    checkpoint = CheckpointManager.load(
        tmp_path / "outputs" / "checkpoints" / "Group0_SmokeTest" / "TinyConvClassifier_best.pth",
        map_location="cpu",
    )
    return {"initial": initial, "final": final}, history["train_loss"], checkpoint


def test_seed_everything_repeats_initialization_training_and_history(tmp_path):
    """Cùng seed tạo cùng trọng số, lịch sử loss và trọng số sau 3 epoch."""
    first, first_losses, first_checkpoint = _run_reproducible_training(tmp_path, "run_a")
    second, second_losses, second_checkpoint = _run_reproducible_training(tmp_path, "run_b")
    assert first["initial"].keys() == second["initial"].keys()
    assert all(torch.equal(first["initial"][key], second["initial"][key]) for key in first["initial"])
    assert first_losses == pytest.approx(second_losses, abs=0.0, rel=0.0)
    assert all(torch.equal(first["final"][key], second["final"][key]) for key in first["final"])
    assert first_checkpoint["seed"] == second_checkpoint["seed"] == 42


def test_seed_helpers_control_python_numpy_torch_and_dataloader():
    """Seed helper đồng bộ Python/NumPy/Torch và generator DataLoader."""
    seed_everything(123)
    py_first = __import__("random").random()
    np_first = np.random.rand()
    torch_first = torch.rand(3)
    seed_everything(123)
    assert __import__("random").random() == py_first
    assert np.random.rand() == np_first
    assert torch.equal(torch.rand(3), torch_first)
    assert get_torch_generator(9).initial_seed() == 9
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_count_parameters_shape_validation_summary_and_hooks(capsys):
    """Base model thống kê đúng, phát hiện NHWC và trích feature qua hook."""
    model = TinyConvClassifier()
    counts = model.count_parameters()
    assert counts == {"total_params": 50, "trainable_params": 50, "non_trainable_params": 0}
    valid = torch.randn(4, 1, 8, 8)
    model.validate_input_shape(valid, expected_channels=1)
    with pytest.raises(ValueError, match="NHWC"):
        model.validate_input_shape(torch.randn(4, 8, 8, 1))
    with pytest.raises(ValueError, match="channel"):
        model.validate_input_shape(torch.randn(4, 3, 8, 8), expected_channels=1)

    model.register_feature_hooks(["features.0"])
    output, features = model.extract_features(valid)
    assert output.shape == (4, 2)
    assert features["features.0"].shape == (4, 4, 6, 6)
    model.clear_hooks()
    _, cleared = model.extract_features(valid)
    assert cleared == {}

    summary = model.summary((1, 8, 8))
    assert "Total params: 50" in summary
    assert "features.0" in summary
    assert "Total params: 50" in capsys.readouterr().out


def test_validation_does_not_update_parameters_and_callback_can_stop(tmp_path):
    """Validation không đổi weights; callback có thể kết thúc fit sớm."""
    seed_everything(42)
    model = TinyConvClassifier()
    optimizer = SGD(model.parameters(), lr=0.02)
    trainer = BaseTrainer(model, optimizer, nn.CrossEntropyLoss(), device=torch.device("cpu"))
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    trainer.validate_one_epoch(_make_loader(43))
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())

    class StopAfterOneEpoch:
        """Callback thử nghiệm yêu cầu dừng sau epoch đầu."""

        def on_epoch_end(self, trainer, epoch, logs):
            return epoch == 1

    history = trainer.fit(_make_loader(42), _make_loader(43), epochs=4, callbacks=[StopAfterOneEpoch()])
    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1


def test_fit_requires_real_member_alias_when_logger_is_attached():
    """Trainer không ghi log nếu logger thiếu Member_Alias thật."""
    model = TinyConvClassifier()
    trainer = BaseTrainer(
        model,
        SGD(model.parameters(), lr=0.01),
        nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    )

    class LoggerWithoutAlias:
        member_alias = ""

    with pytest.raises(ValueError, match="member_alias thật"):
        trainer.fit(_make_loader(42), _make_loader(43), epochs=1, logger=LoggerWithoutAlias())
