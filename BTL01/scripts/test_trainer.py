"""Kiểm thử hồi quy T0303 và chạy smoke test huấn luyện/xác thực 10 epoch.

Chạy từ thư mục gốc dự án: python -m scripts.test_trainer
Không mở file CSV test. Các kiểm thử metadata split nhân tạo chỉ dùng TensorDataset.
"""

from __future__ import annotations

import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
import yaml

from src.data_generator import GaussianDataset
from src.models import FunctionApproximatorMLP
from src.trainer import EarlyStopping, Trainer


ROOT = Path(__file__).resolve().parents[1]


class TrainerTests(unittest.TestCase):
    """Kiểm thử cập nhật trọng số, cô lập tập validation, dừng sớm và lưu/tải checkpoint."""

    def setUp(self) -> None:
        """Cố định seed RNG và cấp phát thư mục checkpoint tạm tự động dọn dẹp.

        Trả về:
            None; mỗi test case nhận một mô hình mới và DataLoader xác định.
        """
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.use_deterministic_algorithms(True)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.model = nn.Linear(2, 1)
        self.loader = DataLoader(
            TensorDataset(torch.ones(5, 2), torch.zeros(5, 1)), batch_size=2
        )
        self.trainer = Trainer(
            self.model, torch.optim.Adam(self.model.parameters(), lr=0.01),
            checkpoint_dir=Path(self.temporary.name),
        )

    def test_training_updates_weights(self) -> None:
        """Xác minh backprop cập nhật trọng số và giữ mô hình ở chế độ training."""
        before = self.model.weight.detach().clone()
        loss = self.trainer.train_one_epoch(self.loader)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, self.model.weight))
        self.assertTrue(self.model.training)

    def test_validation_is_no_grad_and_weighted(self) -> None:
        """Kiểm tra đánh giá validation không tính grad và tính trung bình đúng với batch cuối không đều."""
        with torch.no_grad():
            self.model.weight.zero_()
            self.model.bias.zero_()
        targets = torch.tensor([[1.], [2.], [3.], [4.], [5.]])
        loader = DataLoader(
            TensorDataset(torch.zeros(5, 2), targets), batch_size=2
        )
        before = copy.deepcopy(self.model.state_dict())
        modes: list[tuple[bool, bool]] = []

        def observe(module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            """Ghi nhận chế độ model và trạng thái grad khi forward trên validation.

            Tham số:
                module: Mô hình hồi quy đang theo dõi.
                args: Các Tensor đầu vào, phần tử đầu tiên có kích thước (B, 2).
            """
            modes.append((module.training, torch.is_grad_enabled()))

        handle = self.model.register_forward_pre_hook(observe)
        self.addCleanup(handle.remove)
        self.assertAlmostEqual(self.trainer.validate(loader), 11.0, places=6)
        self.assertEqual(modes, [(False, False)] * 3)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[key]))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_drop_last_uses_processed_count(self) -> None:
        """Kiểm tra mẫu số tính trung bình dựa trên số mẫu thực tế xử lý, không phải toàn bộ dataset."""
        loader = DataLoader(self.loader.dataset, batch_size=2, drop_last=True)
        self.assertAlmostEqual(
            self.trainer.validate(loader), self.trainer.validate(self.loader)
        )

    def test_empty_loaders_rejected(self) -> None:
        """Từ chối loader rỗng thay vì gây lỗi chia cho 0."""
        empty = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, 1)))
        for method in [self.trainer.validate, self.trainer.train_one_epoch]:
            with self.assertRaises(ValueError):
                method(empty)

    def test_invalid_shapes_and_nan_rejected(self) -> None:
        """Từ chối nhãn sai chiều gây broadcasting ngoài ý muốn và đầu vào NaN trước khi cập nhật."""
        for targets in [torch.ones(5), torch.full((5, 1), float("nan"))]:
            loader = DataLoader(TensorDataset(torch.ones(5, 2), targets))
            with self.assertRaises(ValueError):
                self.trainer.train_one_epoch(loader)

    def test_test_split_rejected_through_wrappers(self) -> None:
        """Từ chối metadata tập test ngay cả khi bọc qua các lớp Dataset wrappers."""
        dataset = self.loader.dataset
        dataset.split = "test"
        wrappers = [dataset, Subset(dataset, [0]), ConcatDataset([dataset])]
        for wrapper in wrappers:
            loader = DataLoader(wrapper)
            with self.assertRaises(ValueError):
                self.trainer.validate(loader)
            with self.assertRaises(ValueError):
                self.trainer.train_one_epoch(loader)

    def test_strict_checkpoint_independent_of_min_delta(self) -> None:
        """Lưu checkpoint khi có cải thiện nhỏ nghiêm ngặt ngay cả khi dừng sớm kích hoạt."""
        self.trainer.patience = 2
        self.trainer.min_delta = 0.1
        with patch.object(self.trainer, "train_one_epoch", return_value=1.0):
            with patch.object(
                self.trainer, "validate", side_effect=[1.0, 0.99, 0.98, 0.97]
            ):
                history = self.trainer.fit(
                    self.loader, self.loader, epochs=10, verbose=False
                )
        self.assertEqual(len(history["val_loss"]), 3)
        self.assertTrue(self.trainer.stopped_early)
        checkpoint = torch.load(
            self.trainer.checkpoint_path, weights_only=True
        )
        self.assertEqual(checkpoint["epoch"], 3)
        self.assertEqual(checkpoint["best_val_loss"], 0.98)

    def test_restore_best_model_and_optimizer(self) -> None:
        """Khôi phục mô hình tại epoch tốt nhất trước đó, bao gồm cả biến đếm bước của Adam."""
        with patch.object(
            self.trainer, "validate", side_effect=[0.1, 0.2, 0.3]
        ):
            self.trainer.fit(self.loader, self.loader, epochs=3, verbose=False)
        checkpoint = torch.load(
            self.trainer.checkpoint_path, weights_only=True
        )
        self.assertEqual(checkpoint["epoch"], 1)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(
                value, checkpoint["model_state_dict"][key]
            ))
        for state in self.trainer.optimizer.state.values():
            self.assertEqual(state["step"].item(), 3)
        self.assertEqual(len(self.trainer.history["lr"]), 3)

    def test_fit_reset_and_checkpoint_ties(self) -> None:
        """Mỗi lần gọi fit sẽ reset lại lịch sử; nếu validation loss hòa nhau giữ lại epoch đầu tiên."""
        for _ in range(2):
            with patch.object(self.trainer, "validate", return_value=0.1):
                history = self.trainer.fit(
                    self.loader, self.loader, epochs=2, verbose=False
                )
            self.assertEqual(len(history["lr"]), 2)
            self.assertEqual(self.trainer.best_epoch, 1)

    def test_invalid_arguments_and_reduction(self) -> None:
        """Từ chối số epoch không hợp lệ, đường dẫn traversal và hàm loss không phải mean reduction."""
        for epochs in [0, -1, True]:
            with self.assertRaises(ValueError):
                self.trainer.fit(self.loader, self.loader, epochs=epochs)
        for name in ["../bad.pt", "folder/bad.pt", "bad.pth"]:
            with self.assertRaises(ValueError):
                self.trainer.fit(
                    self.loader, self.loader, checkpoint_name=name
                )
        with self.assertRaises(ValueError):
            Trainer(self.model, criterion=nn.MSELoss(reduction="sum"))
        for patience, delta in [(0, 0.0), (1, -1.0), (1, float("nan"))]:
            with self.assertRaises(ValueError):
                EarlyStopping(patience, delta)

    @patch("src.trainer.Path.replace", side_effect=[PermissionError("locked"), None])
    def test_checkpoint_replace_retries_transient_permission_error(self, replace) -> None:
        """Checkpoint writes retry a transient Windows file-lock failure."""
        self.trainer.best_val_loss = 0.5
        self.trainer.save_checkpoint(1)
        self.assertEqual(replace.call_count, 2)

    def test_legacy_config_keywords(self) -> None:
        """Duy trì khởi tạo từ config và các bí danh phương thức trước đây."""
        trainer = Trainer(
            self.model, train_config=self.trainer.config,
            device=torch.device("cpu"),
            checkpoint_path=Path(self.temporary.name) / "legacy.pt",
        )
        trainer.fit(self.loader, self.loader, epochs=1, verbose=False)
        self.assertTrue(trainer.checkpoint_path.exists())
        self.assertAlmostEqual(
            trainer.evaluate(self.loader), trainer.validate(self.loader)
        )

    def test_csv_smoke_ten_epochs(self) -> None:
        """Huấn luyện MLP của dự án trên file CSV train/val và lưu trữ artifacts."""
        with (ROOT / "configs/baseline.yaml").open(
            encoding="utf-8"
        ) as stream:
            cfg = yaml.safe_load(stream)
        train = GaussianDataset(cfg["data"]["train_full_path"])
        val = GaussianDataset(cfg["data"]["val_path"])
        batch_size = cfg["training_baseline"]["batch_size"]
        train_loader = DataLoader(
            train, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(42),
        )
        val_loader = DataLoader(val, batch_size=batch_size)
        model = FunctionApproximatorMLP(cfg["model_baseline"], cfg["data"])
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg["training_baseline"]["learning_rate"]
        )
        trainer = Trainer(model, optimizer, device=torch.device("cpu"))
        initial_val = trainer.validate(val_loader)
        history = trainer.fit(
            train_loader, val_loader, epochs=10,
            checkpoint_name="T0303_smoke_best.pt",
        )
        self.assertTrue(all(np.isfinite(v).all() for v in history.values()))
        self.assertLess(history["train_loss"][-1], history["train_loss"][0])
        self.assertLess(history["val_loss"][-1], history["val_loss"][0])
        self.assertLess(trainer.best_val_loss, initial_val)
        checkpoint = torch.load(
            trainer.checkpoint_path, map_location="cpu", weights_only=True
        )
        clone = FunctionApproximatorMLP(cfg["model_baseline"], cfg["data"])
        clone.load_state_dict(checkpoint["model_state_dict"])
        clone.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(
                model(val.features), clone(val.features)
            ))
        self.assertAlmostEqual(
            trainer.validate(val_loader), min(history["val_loss"]), places=8
        )
        table = ROOT / "outputs/tables/T0303_smoke_history.csv"
        table.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"epoch": range(1, 11), **history}).to_csv(
            table, index=False
        )
        print(f"SMOKE initial_val={initial_val:.10f} "
              f"best_val={trainer.best_val_loss:.10f} "
              f"best_epoch={trainer.best_epoch} rows={len(train)}/{len(val)}")
        print(f"CHECKPOINT bytes={trainer.checkpoint_path.stat().st_size}")

    def test_stopping_reset_and_disabled(self) -> None:
        """Kiểm tra reset bộ đếm patience và trường hợp vô hiệu hóa dừng sớm sẽ chạy hết epoch."""
        stopping = EarlyStopping(2, 0.1)
        self.assertTrue(stopping(1.0))
        self.assertFalse(stopping(0.95))
        self.assertTrue(stopping(0.8))
        self.assertEqual(stopping.counter, 0)
        self.trainer.use_early_stopping = False
        self.trainer.patience = 1
        with patch.object(self.trainer, "validate", return_value=1.0):
            history = self.trainer.fit(
                self.loader, self.loader, epochs=3, verbose=False
            )
        self.assertEqual(len(history["val_loss"]), 3)
        self.assertFalse(self.trainer.stopped_early)

    def test_nonfinite_validation_aborts(self) -> None:
        """Dừng tiến trình khi gặp loss không hữu hạn thay vì âm thầm khôi phục file cũ."""
        with patch.object(self.trainer, "validate", return_value=float("nan")):
            with self.assertRaises(ValueError):
                self.trainer.fit(
                    self.loader, self.loader, epochs=1, verbose=False
                )
        self.assertFalse(self.trainer.checkpoint_path.exists())

    def test_sgd_and_csv_metadata(self) -> None:
        """Kiểm tra cấu hình SGD và ngăn chặn gán nhãn sai tập val thành train."""
        config = copy.deepcopy(self.trainer.config)
        config["optimizer"] = "sgd"
        trainer = Trainer(
            self.model, train_config=config,
            checkpoint_dir=Path(self.temporary.name),
        )
        self.assertIsInstance(trainer.optimizer, torch.optim.SGD)
        self.assertTrue(np.isfinite(trainer.train_one_epoch(self.loader)))
        with self.assertRaises(ValueError):
            GaussianDataset("data/val.csv", split="train")


if __name__ == "__main__":
    unittest.main(verbosity=2)
