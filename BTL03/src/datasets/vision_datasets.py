"""Dataset ảnh MNIST, WIKI Gender và FER2013 theo hợp đồng NCHW."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class _ImageClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lớp cơ sở chuyển mảng ảnh sang tensor float32 NCHW và nhãn long."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        image_size: int,
        num_classes: int,
        normalization: str,
    ) -> None:
        """Khởi tạo dataset sau khi xác thực ảnh và nhãn.

        Args:
            images: Ảnh NHW hoặc NHWC; giá trị pixel nằm trong [0, 255]
                hoặc [0, 1].
            labels: Nhãn số nguyên N hoặc nhãn one-hot có shape (N, K).
            image_size: Chiều cao và chiều rộng ảnh vuông.
            num_classes: Số lớp phân loại.
            normalization: ``unit`` giữ nguyên [0, 1], ``byte`` chia 255,
                ``auto`` chỉ chia 255 khi max ảnh lớn hơn 1.
        """
        self.images = self._prepare_images(images, image_size, normalization)
        self.labels = self._prepare_labels(labels, len(self.images), num_classes)

    @staticmethod
    def _prepare_images(
        images: np.ndarray, image_size: int, normalization: str
    ) -> torch.Tensor:
        """Đưa ảnh NHW/NHWC thành float32 NCHW, không làm mờ sai schema."""
        array = np.asarray(images)
        if array.ndim == 3:
            array = array[..., np.newaxis]
        if array.ndim != 4:
            raise ValueError(f"Ảnh cần dạng NHW hoặc NHWC; nhận được {array.shape}.")
        if array.shape[1:3] != (image_size, image_size):
            raise ValueError(
                f"Kích thước ảnh phải là {image_size}x{image_size}; "
                f"nhận được {array.shape[1:3]}."
            )
        if array.shape[3] not in (1, 3, 4):
            raise ValueError(f"Số kênh NHWC không hợp lệ: {array.shape[3]}.")

        # Bản sao sở hữu bộ nhớ để hỗ trợ np.load(mmap_mode="r") an toàn.
        tensor = torch.from_numpy(np.array(array, copy=True, order="C")).to(torch.float32)
        minimum = float(tensor.min()) if tensor.numel() else 0.0
        maximum = float(tensor.max()) if tensor.numel() else 0.0
        if minimum < 0:
            raise ValueError("Giá trị pixel âm không được hỗ trợ.")
        if normalization == "byte":
            if maximum > 255:
                raise ValueError("Pixel byte phải nằm trong [0, 255].")
            tensor = tensor / 255.0
        elif normalization == "unit":
            if maximum > 1:
                raise ValueError("Dữ liệu khai báo unit phải nằm trong [0, 1].")
        elif normalization == "auto":
            if maximum > 255:
                raise ValueError("Pixel phải nằm trong [0, 255].")
            if maximum > 1:
                tensor = tensor / 255.0
        else:
            raise ValueError("normalization phải là 'auto', 'unit' hoặc 'byte'.")

        # (N, H, W, C) -> (N, C, H, W), tương đương permute(0, 3, 1, 2).
        return tensor.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _prepare_labels(
        labels: np.ndarray, sample_count: int, num_classes: int
    ) -> torch.Tensor:
        """Đổi nhãn số nguyên hoặc one-hot thành chỉ số lớp torch.long."""
        array = np.asarray(labels)
        if array.ndim == 2:
            if array.shape != (sample_count, num_classes):
                raise ValueError(
                    f"Nhãn one-hot cần shape {(sample_count, num_classes)}; "
                    f"nhận được {array.shape}."
                )
            array = array.argmax(axis=1)
        if array.ndim != 1 or len(array) != sample_count:
            raise ValueError(f"Nhãn cần shape ({sample_count},), nhận được {array.shape}.")
        if not np.issubdtype(array.dtype, np.integer):
            if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
                raise ValueError("Nhãn phân loại phải là số nguyên hoặc one-hot hợp lệ.")
        labels_tensor = torch.as_tensor(array, dtype=torch.long)
        if labels_tensor.numel() and (
            int(labels_tensor.min()) < 0 or int(labels_tensor.max()) >= num_classes
        ):
            raise ValueError(f"Nhãn phải nằm trong [0, {num_classes - 1}].")
        return labels_tensor

    def __len__(self) -> int:
        """Trả về số ảnh trong tập dữ liệu."""
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Trả về ảnh float32 (C,H,W) và nhãn long vô hướng."""
        return self.images[index], self.labels[index]


class MNISTDataset(_ImageClassificationDataset):
    """Dataset MNIST 28×28 với chia giá trị pixel 255 như legacy Keras."""

    def __init__(self, images: np.ndarray | str | Path, labels: np.ndarray) -> None:
        """Khởi tạo MNIST; ảnh có thể là mảng hoặc tệp `.npy`."""
        image_array = np.load(images, mmap_mode="r") if isinstance(images, (str, Path)) else images
        super().__init__(image_array, labels, 28, 10, "byte")


class WikiGenderDataset(_ImageClassificationDataset):
    """Dataset WIKI Gender bảo toàn phép xoay gốc trước chuyển trục."""

    def __init__(
        self,
        images: np.ndarray | str | Path,
        labels: np.ndarray | str | Path,
        apply_legacy_rotation: bool = True,
        normalization: str = "auto",
    ) -> None:
        """Khởi tạo WIKI Gender từ NHW/NHWC hoặc tệp NumPy.

        Args:
            images: Mảng ảnh NHW/NHWC hoặc đường dẫn `.npy`.
            labels: Mảng nhãn hoặc đường dẫn `.npy`; hỗ trợ one-hot 2 lớp.
            apply_legacy_rotation: Áp dụng `np.rot90(k=3, axes=(1, 2))`
                đúng như notebook legacy trên trục không gian batch ảnh.
            normalization: `auto` giữ ảnh [0,1], chia 255 nếu nguồn [0,255].
        """
        image_array = np.load(images, mmap_mode="r") if isinstance(images, (str, Path)) else images
        label_array = np.load(labels, mmap_mode="r") if isinstance(labels, (str, Path)) else labels
        image_array = np.asarray(image_array)
        if image_array.ndim == 3:
            image_array = image_array[..., np.newaxis]
        if image_array.ndim != 4:
            raise ValueError(f"Ảnh WIKI cần NHW hoặc NHWC; nhận được {image_array.shape}.")
        if apply_legacy_rotation:
            # (N,64,64,C) -> xoay 270° ngược chiều kim đồng hồ trên (H,W).
            image_array = np.rot90(image_array, k=3, axes=(1, 2))
        super().__init__(image_array, label_array, 64, 2, normalization)


class FEREmotionDataset(_ImageClassificationDataset):
    """Dataset FER2013 48×48 với bảy lớp theo thứ tự legacy Keras."""

    def __init__(
        self,
        images: np.ndarray | str | Path,
        labels: np.ndarray | str | Path | None = None,
        normalization: str = "auto",
    ) -> None:
        """Khởi tạo FER từ mảng, cặp `.npy` hoặc CSV legacy one-hot."""
        if isinstance(images, (str, Path)) and Path(images).suffix.lower() == ".csv":
            if labels is not None:
                raise ValueError("Với CSV FER legacy, không truyền riêng labels.")
            image_array, label_array = load_fer_onehot_csv(images)
        else:
            if labels is None:
                raise ValueError("Nguồn FER dạng mảng/NPY phải đi kèm labels.")
            image_array = np.load(images, mmap_mode="r") if isinstance(images, (str, Path)) else images
            label_array = np.load(labels, mmap_mode="r") if isinstance(labels, (str, Path)) else labels
        super().__init__(image_array, label_array, 48, 7, normalization)


def load_fer_onehot_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Đọc CSV legacy FER gồm 2304 pixel rồi đến 7 nhãn one-hot.

    Giá trị pixel trong notebook legacy không bị chia 255. Hàm trả ảnh NHW;
    Dataset sẽ giữ nguyên giá trị nếu chúng đã nằm trong [0,1], nếu là byte
    [0,255] thì `normalization='auto'` mới chia 255.
    """
    rows = np.loadtxt(Path(path), delimiter=",", dtype=np.float32, ndmin=2)
    if rows.shape[1] != 2311:
        raise ValueError(
            f"FER legacy cần 2311 cột (2304 pixel + 7 nhãn); "
            f"nhận được {rows.shape[1]} cột ở {path}."
        )
    images = rows[:, :2304].reshape(-1, 48, 48)
    labels = rows[:, 2304:]
    if not np.isfinite(rows).all():
        raise ValueError(f"CSV FER chứa NaN/Inf: {path}.")
    return images, labels