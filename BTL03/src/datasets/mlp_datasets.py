"""Bộ dữ liệu tổng hợp Gaussian cho các thí nghiệm MLP."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


class BivariateNormalDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lưới mật độ chuẩn hai biến, có thể lấy mẫu train tái lập bằng seed.

    Tập train mặc định tái tạo lưới 32×32 của notebook Keras; tập test tái tạo
    lưới 10×10. Mỗi phần tử trả về ``x: (2,)`` và ``y: (1,)`` dạng float32.
    """

    def __init__(
        self,
        split: Literal["train", "test"] = "train",
        sample_count: int = 800,
        seed: int = 42,
        train_grid_size: int = 32,
        test_grid_size: int = 10,
    ) -> None:
        """Tạo lưới Gaussian và chọn mẫu train không hoàn lại.

        Args:
            split: ``train`` lấy mẫu ngẫu nhiên; ``test`` giữ toàn bộ lưới.
            sample_count: Số điểm train cần lấy, mặc định 800.
            seed: Seed riêng cho phép chọn chỉ số train.
            train_grid_size: Số tọa độ mỗi trục của lưới train.
            test_grid_size: Số tọa độ mỗi trục của lưới test.
        """
        super().__init__()
        if split not in {"train", "test"}:
            raise ValueError("split phải là 'train' hoặc 'test'.")
        if sample_count <= 0 or train_grid_size <= 0 or test_grid_size <= 0:
            raise ValueError("sample_count và kích thước lưới phải dương.")
        self.split = split
        self.seed = int(seed)
        self.grid_size = train_grid_size if split == "train" else test_grid_size
        coordinates, targets = self._make_grid(self.grid_size)
        self.full_inputs = torch.from_numpy(coordinates.copy()).to(torch.float32)
        self.full_targets = torch.from_numpy(targets.copy()).to(torch.float32).reshape(-1, 1)
        if split == "train":
            if sample_count > 1000:
                raise ValueError("sample_count tối đa 1000 theo pool lấy mẫu legacy.")
            # RandomState tái tạo chính xác API np.random.choice(range(...)) legacy.
            indices = np.random.RandomState(self.seed).choice(
                range(1, 1001), size=sample_count, replace=False
            )
            if int(indices.max()) >= len(coordinates):
                raise ValueError(
                    "Chỉ số lấy mẫu legacy vượt số điểm lưới; không thể ánh xạ an toàn."
                )
            coordinates, targets = coordinates[indices], targets[indices]
        self.inputs = torch.from_numpy(coordinates.copy()).to(torch.float32)
        self.targets = torch.from_numpy(targets.copy()).to(torch.float32).reshape(-1, 1)

    @staticmethod
    def _make_grid(grid_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Tạo lưới ``[-3,3]`` và mật độ Gaussian chuẩn độc lập hai trục."""
        grid_step = complex(0, grid_size)
        x1, x2 = np.mgrid[-3.0:3.0:grid_step, -3.0:3.0:grid_step]
        coordinates = np.column_stack((x1.ravel(), x2.ravel()))
        # mu=[0,0], Sigma=diag([1,1]); det(Sigma)=1, Sigma^{-1}=I.
        density = np.exp(-0.5 * np.square(coordinates).sum(axis=1)) / (2.0 * np.pi)
        return coordinates.astype(np.float32), density.astype(np.float32)

    def __len__(self) -> int:
        """Trả số điểm của lưới hoặc số mẫu train đã chọn."""
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Trả tọa độ hai chiều và mật độ mục tiêu của một điểm."""
        return self.inputs[index], self.targets[index]