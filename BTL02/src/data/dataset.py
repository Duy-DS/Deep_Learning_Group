"""Module quản lý dữ liệu FER2013 với PyTorch Dataset và DataLoader.

Tuân thủ nghiêm ngặt hiến pháp AGENTS.md:
- Thiết kế 100% hướng đối tượng (OOP) với Docstring Tiếng Việt đầy đủ.
- Phân chia cố định 3 tập: Training (28.709), PublicTest (3.589), PrivateTest (3.589).
- Đảm bảo tính tái lập (Reproducibility) với seed=42 được truyền vào generator của DataLoader.
- Hỗ trợ tính toán Class Weights phục vụ xử lý mất cân bằng lớp ở WP4 (Focal Loss / Weighted Loss).
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import get_transforms


# Ánh xạ tên phân tập chuẩn của FER2013
SPLIT_MAPPING: Dict[str, str] = {
    "train": "Training",
    "training": "Training",
    "val": "PublicTest",
    "validation": "PublicTest",
    "publictest": "PublicTest",
    "test": "PrivateTest",
    "testing": "PrivateTest",
    "privatetest": "PrivateTest",
}

# Số lượng mẫu kỳ vọng cho từng phân tập theo đặc tả FER2013
EXPECTED_SPLIT_COUNTS: Dict[str, int] = {
    "Training": 28709,
    "PublicTest": 3589,
    "PrivateTest": 3589,
}


class FER2013Dataset(Dataset):
    """Lớp PyTorch Dataset cho tập dữ liệu biểu cảm khuôn mặt FER2013.

    Dữ liệu được nạp từ file CSV thô gồm 35.887 mẫu, chuyển đổi chuỗi pixel thành
    ma trận numpy uint8 (48x48) trong bộ nhớ RAM để tối ưu tốc độ đọc khi huấn luyện.
    """

    def __init__(
        self,
        csv_path: Union[str, Path] = "data/raw/fer2013.csv",
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """Khởi tạo dataset cho phân tập được chỉ định.

        Args:
            csv_path: Đường dẫn tới file fer2013.csv.
            split: Tên phân tập ('train', 'val', 'test' hoặc 'Training', 'PublicTest', 'PrivateTest').
            transform: Callable biến đổi ảnh (torchvision transforms).
            target_transform: Callable biến đổi nhãn nếu có.

        Raises:
            FileNotFoundError: Nếu không tìm thấy file CSV tại csv_path.
            ValueError: Nếu tên phân tập không hợp lệ.
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu FER2013 tại: {self.csv_path.resolve()}")

        split_key = split.strip().lower()
        if split_key not in SPLIT_MAPPING:
            raise ValueError(
                f"Phân tập '{split}' không hợp lệ! Vui lòng chọn trong: {list(SPLIT_MAPPING.keys())}"
            )

        self.split_name = SPLIT_MAPPING[split_key]
        self.transform = transform
        self.target_transform = target_transform

        # Đọc dữ liệu và lọc theo cột Usage
        self.images, self.labels = self._load_data()

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Đọc và trích xuất dữ liệu ảnh từ CSV thành mảng NumPy.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - images: Mảng uint8 kích thước (N, 48, 48)
                - labels: Mảng int64 kích thước (N,)
        """
        df = pd.read_csv(self.csv_path)

        # Lọc theo phân tập Usage
        subset_df = df[df["Usage"] == self.split_name].reset_index(drop=True)
        num_samples = len(subset_df)

        expected_count = EXPECTED_SPLIT_COUNTS.get(self.split_name)
        if expected_count is not None and num_samples != expected_count:
            raise ValueError(
                f"Phân tập {self.split_name} có {num_samples} mẫu, khác với số lượng chuẩn {expected_count}!"
            )

        # Tối ưu bộ nhớ: chuyển toàn bộ chuỗi pixel thành mảng (N, 48, 48) uint8
        pixels_list = [
            np.fromstring(p, dtype=np.uint8, sep=" ").reshape(48, 48)
            for p in subset_df["pixels"]
        ]
        images = np.stack(pixels_list, axis=0)
        labels = subset_df["emotion"].to_numpy(dtype=np.int64)

        return images, labels

    def __len__(self) -> int:
        """Trả về tổng số mẫu trong phân tập."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lấy một mẫu dữ liệu theo chỉ số.

        Args:
            idx: Vị trí của mẫu dữ liệu.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - image: Tensor ảnh đã qua transform (kích thước [C, H, W]).
                - label: Tensor nhãn cảm xúc kiểu long (giá trị từ 0 đến 6).
        """
        img_array = self.images[idx]
        label = self.labels[idx]

        # Chuyển đổi sang PIL Image dạng Grayscale (Mode 'L')
        img = Image.fromarray(img_array, mode="L")

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            # Fallback nếu không truyền transform: chuyển thành tensor [1, 48, 48] trong khoảng [0, 1]
            img_tensor = torch.from_numpy(img_array).unsqueeze(0).float() / 255.0

        label_tensor = torch.tensor(label, dtype=torch.long)
        if self.target_transform is not None:
            label_tensor = self.target_transform(label_tensor)

        return img_tensor, label_tensor

    def get_class_counts(self) -> Dict[int, int]:
        """Thống kê số lượng mẫu của từng lớp cảm xúc trong phân tập hiện tại.

        Returns:
            Dict[int, int]: Từ điển ánh xạ nhãn lớp (0-6) -> số lượng mẫu.
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        counts_dict = {int(cls): 0 for cls in range(7)}
        for cls, count in zip(unique, counts):
            counts_dict[int(cls)] = int(count)
        return counts_dict

    def get_class_weights(self, method: str = "balanced") -> torch.Tensor:
        """Tính toán trọng số các lớp (Class Weights) cho hàm mất mát để xử lý mất cân bằng lớp.

        Công thức balanced:
            w_c = N / (n_classes * N_c)

        Args:
            method: Phương pháp tính trọng số ("balanced" hoặc "inverse").

        Returns:
            torch.Tensor: Tensor 1D chứa 7 trọng số ứng với 7 lớp cảm xúc.
        """
        counts_dict = self.get_class_counts()
        total_samples = len(self.labels)
        n_classes = 7

        weights = np.zeros(n_classes, dtype=np.float32)
        for cls in range(n_classes):
            c_count = counts_dict.get(cls, 1)
            if c_count == 0:
                c_count = 1

            if method == "balanced":
                weights[cls] = total_samples / (n_classes * c_count)
            elif method == "inverse":
                weights[cls] = total_samples / c_count
            else:
                raise ValueError(f"Không hỗ trợ phương pháp tính trọng số: '{method}'")

        # Chuẩn hóa để trọng số trung bình bằng 1.0
        weights = weights / np.mean(weights)
        return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(
    csv_path: Union[str, Path] = "data/raw/fer2013.csv",
    batch_size: int = 64,
    num_workers: int = 0,
    image_size: int = 48,
    in_channels: int = 1,
    augmentation: str = "None",
    seed: int = 42,
    pin_memory: bool = False,
    normalization: str = "fer2013",
    transform_mode: str = "custom",
    use_augmentation: bool = True,
    aug_config: Optional[Dict[str, Any]] = None,
    augmentation_mode: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Hàm kiến tạo và trả về bộ 3 DataLoaders (Training, Validation/PublicTest, Test/PrivateTest).

    Args:
        csv_path: Đường dẫn file CSV thô fer2013.
        batch_size: Kích thước batch cho mỗi lần lặp.
        num_workers: Số luồng xử lý song song khi nạp dữ liệu.
        image_size: Kích thước ảnh cạnh vuông (48, 128, 224).
        in_channels: Số lượng kênh ảnh (1 cho Grayscale, 3 cho RGB).
        augmentation: Chiến lược tăng cường dữ liệu cho tập Train ("None", "Standard", "Advanced").
        seed: Random seed cố định tính ngẫu nhiên của DataLoader shuffle.
        pin_memory: Bật pin_memory nếu huấn luyện trên GPU để tăng tốc chuyển đổi CPU -> CUDA.
        normalization: Chuẩn hóa ``fer2013`` cho pipeline cũ hoặc ``imagenet`` khi mode SOTA.
        transform_mode: ``custom`` giữ pipeline WP2; ``sota`` dùng ảnh RGB ImageNet đa kích thước.
        use_augmentation: Chỉ bật augmentation cho transform của tập Training ở mode SOTA.
        aug_config: Các mức augmentation tùy chọn dành riêng cho tập Training SOTA.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    # 1. Chọn pipeline theo mode; giữ mặc định cũ để không ảnh hưởng WP2.
    mode = transform_mode.strip().lower()
    if mode == "custom":
        train_augmentation = augmentation
        if augmentation_mode is not None:
            train_augmentation = {"mode": augmentation_mode}
            if aug_config:
                train_augmentation.update(aug_config)
        train_transform, eval_transform = get_transforms(
            image_size=image_size,
            in_channels=in_channels,
            augmentation=train_augmentation,
            normalization=normalization,
        )
    elif mode == "sota":
        from src.data.transforms_sota import SOTATransformBuilder

        builder = SOTATransformBuilder(
            image_size=image_size,
            use_augmentation=use_augmentation,
            aug_config={**(aug_config or {}), **({"mode": augmentation_mode} if augmentation_mode else {})},
        )
        train_transform = builder.build_train_transform()
        eval_transform = builder.build_eval_transform()
    else:
        raise ValueError("transform_mode phải là 'custom' hoặc 'sota'.")

    # 2. Khởi tạo 3 phân tập Dataset
    train_dataset = FER2013Dataset(
        csv_path=csv_path,
        split="train",
        transform=train_transform,
    )
    val_dataset = FER2013Dataset(
        csv_path=csv_path,
        split="val",
        transform=eval_transform,
    )
    test_dataset = FER2013Dataset(
        csv_path=csv_path,
        split="test",
        transform=eval_transform,
    )

    # 3. Đảm bảo tính tái lập với seed của Generator
    generator = torch.Generator()
    generator.manual_seed(seed)

    # 4. Khởi tạo DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        drop_last=False,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

