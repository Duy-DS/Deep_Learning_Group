"""Unit test callbacks, checkpoint metadata và tích hợp lifecycle BaseTrainer."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from src.engine import (
    BaseCallback,
    BaseTrainer,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)


class TinyClassifier(nn.Module):
    """Mạng tuyến tính nhỏ để kiểm thử callback không cần dữ liệu ảnh."""

    def __init__(self) -> None:
        """Tạo bộ phân loại hai lớp trên vector hai chiều."""
        super().__init__()
        self.layer = nn.Linear(2, 2)
        self.model_name = "TinyClassifier"

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Trả logits hai lớp."""
        return self.layer(inputs)


class FakeTrainer:
    """Trainer tối thiểu để gọi trực tiếp callback monitor/checkpoint."""

    def __init__(self, model: nn.Module, optimizer: SGD, project_root: Path) -> None:
        """Cung cấp đúng thuộc tính được callbacks sử dụng."""
        self.model = model
        self.optimizer = optimizer
        self.model_name = "TinyClassifier"
        self.seed = 42
        self.config = {"project_root": str(project_root), "seed": 42}
        self.stop_training = False
        self.callbacks = []


def test_early_stopping_waits_patience_and_restores_best_weights():
    """Sequence [1,.8,.85,.86] dừng epoch 4 và khôi phục epoch 2."""
    model = nn.Linear(1, 1, bias=False)
    callback = EarlyStopping(patience=2, restore_best_weights=True, verbose=False)
    trainer = FakeTrainer(model, SGD(model.parameters(), lr=0.1), Path.cwd())
    callback.set_trainer(trainer)
    values = [1.0, 0.8, 0.85, 0.86, 0.87]
    best_state = None
    stopped_at = None
    for epoch, value in enumerate(values, start=1):
        with torch.no_grad():
            model.weight.fill_(float(epoch))
        if epoch == 2:
            best_state = model.weight.detach().clone()
        if callback.on_epoch_end(epoch, {"val_loss": value}):
            stopped_at = epoch
            break
    assert stopped_at == 4
    assert callback.stopped_epoch == 4
    assert callback.best_epoch == 2
    assert torch.equal(model.weight.detach(), best_state)
    assert trainer.stop_training is True


def test_reduce_lr_on_plateau_reduces_every_group_and_respects_floor():
    """Plateau đủ patience giảm 1e-3→1e-4 rồi dừng tại min_lr=1e-5."""
    model = nn.Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=1e-3)
    callback = ReduceLROnPlateau(
        optimizer, factor=0.1, patience=2, min_lr=1e-5, verbose=False
    )
    for epoch, value in enumerate([1.0, 1.1, 1.2], start=1):
        callback.on_epoch_end(epoch, {"val_loss": value})
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    for epoch, value in enumerate([1.3, 1.4], start=4):
        callback.on_epoch_end(epoch, {"val_loss": value})
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)
    callback.on_epoch_end(5, {"val_loss": 1.0})
    assert optimizer.param_groups[0]["lr"] >= 1e-5


def test_reduce_lr_on_plateau_uses_all_param_groups_and_cooldown():
    """Mọi group nhận cùng factor; cooldown không đếm plateau trong thời gian chờ."""
    model = nn.Linear(2, 2)
    optimizer = SGD([
        {"params": [model.weight], "lr": 1e-3},
        {"params": [model.bias], "lr": 2e-3},
    ])
    callback = ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2, cooldown=2, min_lr=1e-5, verbose=False
    )
    for epoch, value in enumerate([1.0, 1.1, 1.2], start=1):
        callback.on_epoch_end(epoch, {"val_loss": value})
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([5e-4, 1e-3])
    assert callback.cooldown_counter == 2
    callback.on_epoch_end(4, {"val_loss": 1.3})
    callback.on_epoch_end(5, {"val_loss": 1.4})
    assert callback.lr_changes == [(3, 1e-3, 5e-4), (3, 2e-3, 1e-3)]


def test_model_checkpoint_only_overwrites_on_improvement_and_round_trips(tmp_path):
    """Checkpoint `.pt` đổi epoch/weights chỉ khi val_loss phá kỷ lục."""
    model = nn.Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.01)
    trainer = FakeTrainer(model, optimizer, tmp_path)
    checkpoint_path = tmp_path / "outputs/checkpoints/04_training_utilities/tiny_best.pt"
    callback = ModelCheckpoint(
        checkpoint_path, save_best_only=True, sync_to_drive=False, verbose=False
    )
    callback.set_trainer(trainer)
    for epoch, value, weight in [(1, 1.0, 1.0), (2, 1.2, 2.0), (3, 0.8, 3.0)]:
        with torch.no_grad():
            model.weight.fill_(weight)
        callback.on_epoch_end(epoch, {"val_loss": value})
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert callback.saved_epochs == [1, 3]
    assert callback.best_epoch == 3
    assert payload["epoch"] == 3
    assert payload["best_metric"] == pytest.approx(0.8)
    assert payload["seed"] == 42
    assert payload["monitor"] == "val_loss"
    assert payload["best_score"] == pytest.approx(0.8)
    restored = nn.Linear(2, 2)
    restored.load_state_dict(payload["model_state_dict"])
    assert torch.equal(restored.weight, model.weight)


def test_trainer_lifecycle_logs_and_scheduler_lr_order(tmp_path):
    """Trainer phát đủ lifecycle, log scalars và vẫn hỗ trợ legacy callbacks."""
    model = TinyClassifier()
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = BaseTrainer(
        model, optimizer, nn.CrossEntropyLoss(), device=torch.device("cpu"),
        metric_fn=lambda logits, labels: (logits.argmax(1) == labels).float().mean(),
        config={"project_root": str(tmp_path)},
    )
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]])
    targets = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(features, targets), batch_size=2, shuffle=False)

    class Lifecycle(BaseCallback):
        """Thu lifecycle và đối chiếu keys logs của epoch."""

        def __init__(self) -> None:
            super().__init__()
            self.events = []

        def on_train_begin(self, logs=None):
            self.events.append(("train_begin", set(logs)))

        def on_epoch_begin(self, epoch, logs=None):
            self.events.append(("epoch_begin", epoch))

        def on_epoch_end(self, epoch, logs=None):
            assert {"train_loss", "val_loss", "train_metric", "val_metric", "lr", "model", "optimizer"} <= set(logs)
            self.events.append(("epoch_end", epoch))
            return False

        def on_train_end(self, logs=None):
            self.events.append(("train_end", logs["best_epoch"]))

    class LegacyStop:
        """Giao diện cũ nhận trainer, epoch và logs theo vị trí."""

        def on_epoch_end(self, old_trainer, epoch, logs):
            assert old_trainer is trainer and "val_loss" in logs
            return epoch == 2

    lifecycle = Lifecycle()
    history = trainer.fit(loader, loader, 5, callbacks=[lifecycle, LegacyStop()])
    assert len(history["train_loss"]) == 2
    assert lifecycle.events[0][0] == "train_begin"
    expected_best = min(
        range(1, 3), key=lambda epoch: history["val_loss"][epoch - 1]
    )
    assert lifecycle.events[-1] == ("train_end", expected_best)


def test_early_stopping_restores_weights_selected_by_matching_checkpoint_monitor(tmp_path):
    """EarlyStopping khôi phục đúng state của ModelCheckpoint cùng monitor."""
    model = TinyClassifier()
    optimizer = SGD(model.parameters(), lr=0.05)
    trainer = FakeTrainer(model, optimizer, tmp_path)
    checkpoint_path = tmp_path / "outputs/checkpoints/04_training_utilities/tiny_best.pt"
    checkpoint = ModelCheckpoint(
        checkpoint_path, sync_to_drive=False, verbose=False
    )
    early = EarlyStopping(patience=1, restore_best_weights=True, verbose=False)
    trainer.callbacks = [checkpoint, early]
    checkpoint.set_trainer(trainer)
    early.set_trainer(trainer)
    for epoch, value, weight in [(1, 0.5, 1.0), (2, 0.7, 2.0)]:
        with torch.no_grad():
            model.layer.weight.fill_(weight)
        logs = {"val_loss": value}
        checkpoint.on_epoch_end(epoch, logs)
        early.on_epoch_end(epoch, logs)
    assert early.stopped_epoch == 2
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["epoch"] == 1
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(value, payload["model_state_dict"][name])


def test_trainer_final_checkpoint_best_snapshot_does_not_mutate_live_model(tmp_path):
    """Trainer ghi best checkpoint snapshot nhưng giữ LR và model restore riêng."""
    model = TinyClassifier()
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)
    trainer = BaseTrainer(
        model, optimizer, nn.CrossEntropyLoss(), device=torch.device("cpu"),
        model_group="04_training_utilities", model_name="TinyClassifier", seed=42,
        config={"project_root": str(tmp_path)},
    )
    loader = DataLoader(
        TensorDataset(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]),
            torch.tensor([0, 1, 0, 1]),
        ), batch_size=2,
    )
    path = tmp_path / "outputs/checkpoints/04_training_utilities/best.pt"
    callback = ModelCheckpoint(path, sync_to_drive=False, verbose=False)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    trainer.fit(loader, loader, epochs=1, callbacks=[callback])
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert callback.best_epoch == payload["epoch"] == 1
    assert all(torch.equal(payload["model_state_dict"][key], value) for key, value in model.state_dict().items())
    # Mạng đã train nên state khác khởi tạo; save-best snapshot không khôi phục/mutate thêm.
    assert any(not torch.equal(before[key], value) for key, value in model.state_dict().items())


def test_trainer_does_not_finalize_logger_when_callback_raises():
    """Lỗi callback giữ exception gốc và không tạo master summary run giả."""
    model = TinyClassifier()
    optimizer = SGD(model.parameters(), lr=0.05)
    trainer = BaseTrainer(model, optimizer, nn.CrossEntropyLoss(), device=torch.device("cpu"))
    loader = DataLoader(
        TensorDataset(torch.ones(2, 2), torch.tensor([0, 1])), batch_size=2
    )

    class BrokenCallback(BaseCallback):
        """Callback lỗi sau epoch đầu để mô phỏng training failure."""

        def on_epoch_end(self, epoch, logs=None):
            raise RuntimeError("callback failure")

    class LoggerProbe:
        """Logger probe kiểm tra finish_run không gọi trên exception."""

        member_alias = "TestMember"

        def __init__(self):
            self.finish_called = False
            self.close_called = False

        def log_epoch(self, **kwargs):
            pass

        def finish_run(self, **kwargs):
            self.finish_called = True

        def close(self):
            self.close_called = True

    logger = LoggerProbe()
    with pytest.raises(RuntimeError, match="callback failure"):
        trainer.fit(loader, loader, epochs=2, logger=logger, callbacks=[BrokenCallback()])
    assert logger.finish_called is False
    assert len(trainer.history["train_loss"]) == 1


def test_checkpoint_callback_writes_exact_path_and_drive_mirror(tmp_path):
    """CheckpointManager ghi đúng filepath cấu hình và mirror nếu mount có thật."""
    model = TinyClassifier()
    optimizer = SGD(model.parameters(), lr=0.02)
    drive_root = tmp_path / "drive" / "MyDrive" / "BTL03"
    drive_root.mkdir(parents=True)
    trainer = FakeTrainer(model, optimizer, tmp_path)
    path = tmp_path / "outputs/checkpoints/04_training_utilities/gender_best.pt"
    callback = ModelCheckpoint(
        path, sync_to_drive=True, drive_root=drive_root, verbose=False
    )
    callback.set_trainer(trainer)
    callback.on_epoch_end(1, {"val_loss": 0.7})
    assert path.is_file()
    assert (drive_root / "checkpoints" / "04_training_utilities" / path.name).is_file()


def test_callback_min_delta_mode_and_missing_metric_validation():
    """Monitor hỗ trợ mode=max/min, min_delta và bỏ qua giá trị không hữu hạn."""
    model = nn.Linear(1, 1)
    callback = EarlyStopping(
        monitor="val_accuracy", mode="auto", min_delta=0.05, patience=2, verbose=False
    )
    callback.set_trainer(FakeTrainer(model, SGD(model.parameters(), lr=0.1), Path.cwd()))
    assert callback.on_epoch_end(1, {"val_accuracy": 0.5}) is False
    assert callback.on_epoch_end(2, {"val_accuracy": 0.53}) is False
    assert callback.best_score == pytest.approx(0.5)
    assert callback.on_epoch_end(3, {"val_accuracy": float("nan")}) is False
    assert callback.on_epoch_end(4, {"val_accuracy": 0.4}) is True