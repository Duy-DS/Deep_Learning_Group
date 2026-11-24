"""Bộ dữ liệu hồi quy tổng chữ số trên chuỗi MNIST zero-padded."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .vision_datasets import MNISTDataset


class SequentialMNISTDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sinh cố định chuỗi tối đa ``seq_len`` ảnh và tổng nhãn chữ số.

    Mỗi mẫu có từ 1 đến ``seq_len`` ảnh được lấy có hoàn lại từ split MNIST;
    ảnh được đặt liên tiếp từ đầu chuỗi, các khung còn lại là số không.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        seq_len: int = 8,
        seed: int = 42,
        is_train: bool = True,
    ) -> None:
        """Khởi tạo kế hoạch chuỗi tái lập, không lưu bản sao 5D toàn tập.

        Args:
            images: Ảnh MNIST thô dạng ``(N,28,28)`` hoặc ``(N,28,28,1)``.
            labels: Nhãn chữ số dạng ``(N,)`` hoặc one-hot mười lớp.
            seq_len: Số khung tối đa trong một chuỗi.
            seed: Hạt giống cho kế hoạch sinh chuỗi.
            is_train: Chọn nhánh seed độc lập cho train/test.

        Raises:
            ValueError: Nếu kích thước, nhãn hoặc độ dài chuỗi không hợp lệ.
        """
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len phải là số nguyên dương.")
        image_array = np.asarray(images)
        if image_array.ndim == 4 and image_array.shape[-1] == 1:
            image_array = image_array[..., 0]
        if image_array.ndim != 3 or image_array.shape[1:] != (28, 28):
            raise ValueError(f"MNIST cần shape (N,28,28), nhận {image_array.shape}.")
        label_array = np.asarray(labels)
        if label_array.ndim == 2:
            if label_array.shape != (len(image_array), 10):
                raise ValueError("Nhãn one-hot phải có shape (N,10).")
            label_array = label_array.argmax(axis=1)
        if label_array.ndim != 1 or len(label_array) != len(image_array):
            raise ValueError("Số nhãn phải khớp số ảnh MNIST.")
        if len(image_array) == 0 or np.any(label_array < 0) or np.any(label_array > 9):
            raise ValueError("MNIST phải có mẫu và nhãn nguyên trong miền 0..9.")

        # Dùng Dataset vision hiện có để bảo toàn chuẩn hóa /255 và layout CHW.
        self.mnist = MNISTDataset(image_array, label_array)
        self.seq_len = int(seq_len)
        self.seed = int(seed)
        self.is_train = bool(is_train)
        split_seed = np.random.SeedSequence([self.seed, int(not self.is_train)])
        rng = np.random.default_rng(split_seed)

        sample_count = len(self.mnist)
        # Giữ thứ tự RNG của notebook: lấy count rồi choice trước mẫu kế tiếp.
        self.frame_counts = np.empty(sample_count, dtype=np.int64)
        self.source_indices = np.full((sample_count, self.seq_len), -1, dtype=np.int64)
        self.targets = np.zeros(sample_count, dtype=np.float32)
        source_labels = self.mnist.labels.numpy()
        # Mỗi chuỗi lấy mẫu có hoàn lại như np.random.choice mặc định legacy.
        for row in range(sample_count):
            # ceil(U*T) có xác suất cực nhỏ bằng 0 khi U==0; ràng buộc DoD là 1..T.
            frame_count = max(1, int(np.ceil(rng.random() * self.seq_len)))
            self.frame_counts[row] = frame_count
            selected = rng.choice(sample_count, size=int(frame_count), replace=True)
            self.source_indices[row, :frame_count] = selected
            self.targets[row] = float(source_labels[selected].sum())

    def __len__(self) -> int:
        """Trả số chuỗi, bằng số ảnh trong split MNIST nguồn."""
        return len(self.frame_counts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Trả ảnh ``(T,1,28,28)`` float32 và tổng nhãn ``(1,)`` float32."""
        sequence = torch.zeros((self.seq_len, 1, 28, 28), dtype=torch.float32)
        frame_count = int(self.frame_counts[index])
        selected = self.source_indices[index, :frame_count]
        sequence[:frame_count] = self.mnist.images[torch.from_numpy(selected)]
        target = torch.tensor([self.targets[index]], dtype=torch.float32)
        return sequence, target

    def metadata(self, index: int) -> dict[str, int | bool]:
        """Trả số khung thật, tổng nhãn, seed và split để kiểm tra provenance."""
        return {
            "frame_count": int(self.frame_counts[index]),
            "target_sum": int(self.targets[index]),
            "seed": self.seed,
            "is_train": self.is_train,
        }