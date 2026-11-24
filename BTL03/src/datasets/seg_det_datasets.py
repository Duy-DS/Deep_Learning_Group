"""Dataset PyTorch cho phân vùng SegNet và score-map phát hiện vật thể."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SemanticSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Nạp cặp ảnh depth/mask và trả ảnh NCHW cùng mask class-index.

    Dataset chấp nhận mảng NHW/NHWC, hai tệp NumPy, hoặc thư mục ``depth``
    với thư mục mask song song ``labels``. Màu mask legacy được ánh xạ sang
    bảy lớp theo đúng thứ tự notebook SegNet gốc.
    """

    LEGACY_COLORS = (255, 43, 85, 128, 170, 212, 0)

    def __init__(
        self,
        images: np.ndarray | str | Path | None = None,
        masks: np.ndarray | str | Path | None = None,
        *,
        image_size: int = 96,
        num_classes: int = 7,
        synthetic_fallback: bool = False,
        synthetic_samples: int = 8,
        seed: int = 42,
        normalize: str = "preserve",
    ) -> None:
        """Khởi tạo dữ liệu segmentation từ mảng, file hoặc thư mục.

        Args:
            images: Mảng ảnh NHW/NHWC, file NPY, hoặc thư mục ``depth``.
            masks: Mảng/file mask tương ứng; có thể bỏ qua với thư mục depth.
            image_size: Chiều cao/rộng ảnh vuông chuẩn của SegNet.
            num_classes: Số lớp mask; legacy dùng bảy lớp.
            synthetic_fallback: Tạo dữ liệu giả xác định nếu thiếu dữ liệu.
            synthetic_samples: Số mẫu tạo trong chế độ giả lập.
            seed: Seed riêng cho dữ liệu giả.
            normalize: ``preserve`` giữ pixel 0..255; ``unit`` chia 255.
        """
        if image_size <= 0 or num_classes <= 1:
            raise ValueError("image_size phải dương và num_classes phải lớn hơn 1.")
        if normalize not in {"preserve", "unit"}:
            raise ValueError("normalize phải là 'preserve' hoặc 'unit'.")
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.normalize = normalize

        try:
            image_array, mask_array = self._load_arrays(images, masks)
            self.images = self._prepare_images(image_array)
            self.masks = self._prepare_masks(mask_array)
        except (FileNotFoundError, OSError, ValueError) as error:
            if not synthetic_fallback:
                raise
            if synthetic_samples < 2:
                raise ValueError("synthetic_samples phải >= 2 để chia train/validation.") from error
            generator = np.random.default_rng(seed)
            self.images = torch.from_numpy(
                generator.integers(0, 256, (synthetic_samples, 1, image_size, image_size), dtype=np.uint8)
                .astype(np.float32)
            )
            self.masks = torch.from_numpy(
                generator.integers(0, num_classes, (synthetic_samples, image_size, image_size), dtype=np.int64)
            )

    @staticmethod
    def _load_npy(value: np.ndarray | str | Path) -> np.ndarray:
        """Nạp mảng hoặc tệp NumPy mà không giữ view mmap readonly."""
        return np.load(value, mmap_mode="r") if isinstance(value, (str, Path)) else np.asarray(value)

    @classmethod
    def _load_arrays(
        cls, images: np.ndarray | str | Path | None, masks: np.ndarray | str | Path | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Đọc dữ liệu dạng mảng/tệp NPY hoặc cặp thư mục depth-labels."""
        if images is None:
            raise FileNotFoundError("Chưa cấu hình nguồn ảnh segmentation.")
        if isinstance(images, (str, Path)) and Path(images).is_dir():
            image_root = Path(images)
            mask_root = Path(masks) if isinstance(masks, (str, Path)) else Path(
                str(image_root).replace("depth", "labels")
            )
            image_paths = sorted(path for path in image_root.iterdir() if path.is_file())
            mask_paths = [mask_root / path.name for path in image_paths]
            if not image_paths or not all(path.is_file() for path in mask_paths):
                raise FileNotFoundError(f"Thiếu ảnh hoặc mask tương ứng trong {image_root} / {mask_root}.")
            image_values = [np.asarray(Image.open(path)) for path in image_paths]
            mask_values = [np.asarray(Image.open(path)) for path in mask_paths]
            return np.stack(image_values), np.stack(mask_values)
        if masks is None:
            raise ValueError("Nguồn ảnh mảng/NPY phải đi kèm mask.")
        return cls._load_npy(images), cls._load_npy(masks)

    def _prepare_images(self, values: np.ndarray) -> torch.Tensor:
        """Đổi NHW/NHWC thành float32 NCHW một kênh và resize nearest."""
        array = np.asarray(values)
        if array.ndim == 3:
            array = array[..., None]
        if array.ndim != 4:
            raise ValueError(f"Ảnh phải có dạng NHW/NHWC; nhận {array.shape}.")
        if array.shape[-1] not in (1, 3, 4) and array.shape[1] == 1:
            array = np.moveaxis(array, 1, -1)
        if array.shape[-1] != 1:
            raise ValueError("SegNet legacy yêu cầu ảnh depth grayscale một kênh.")
        tensor = torch.from_numpy(np.array(array, copy=True, order="C")).float().permute(0, 3, 1, 2)
        if tensor.shape[-2:] != (self.image_size, self.image_size):
            tensor = torch.nn.functional.interpolate(tensor, (self.image_size, self.image_size), mode="nearest")
        return tensor.contiguous() / 255.0 if self.normalize == "unit" else tensor.contiguous()

    def _prepare_masks(self, values: np.ndarray) -> torch.Tensor:
        """Đổi class-index, one-hot hoặc màu legacy sang (N,H,W) torch.long."""
        array = np.asarray(values)
        if array.ndim == 4:
            if array.shape[-1] == self.num_classes:
                array = array.argmax(axis=-1)
            elif array.shape[-1] == 1:
                array = array[..., 0]
            else:
                raise ValueError(f"Mask NHWC không hợp lệ: {array.shape}.")
        if array.ndim != 3:
            raise ValueError(f"Mask cần dạng NHW hoặc NHWC; nhận {array.shape}.")
        if array.shape[-2:] != (self.image_size, self.image_size):
            tensor = torch.from_numpy(np.array(array, copy=True, order="C")).float().unsqueeze(1)
            array = torch.nn.functional.interpolate(
                tensor, (self.image_size, self.image_size), mode="nearest"
            ).squeeze(1).numpy()
        if not np.issubdtype(array.dtype, np.integer):
            if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
                raise ValueError("Mask class-index phải chứa số nguyên hữu hạn.")
        labels = np.array(array, dtype=np.int64, copy=True)
        # Mask legacy là ảnh grayscale với palette màu cố định, chưa phải chỉ số lớp.
        if labels.size and (labels.min() < 0 or labels.max() >= self.num_classes):
            mapped = np.full(labels.shape, -1, dtype=np.int64)
            for class_index, color in enumerate(self.LEGACY_COLORS[: self.num_classes]):
                mapped[labels == color] = class_index
            if np.any(mapped < 0):
                unknown = np.unique(labels[mapped < 0])[:8].tolist()
                raise ValueError(f"Mask có màu/lớp không nhận diện được: {unknown}.")
            labels = mapped
        return torch.from_numpy(labels).long().contiguous()

    def __len__(self) -> int:
        """Trả số cặp ảnh-mask."""
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Trả ảnh (1,H,W) float32 và mask (H,W) long cho CrossEntropyLoss."""
        return self.images[index], self.masks[index]


class FCNObjectDetectionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset xe/không-xe, chuẩn hóa RGB và target score legacy ±1."""

    def __init__(
        self,
        images: np.ndarray | str | Path | None = None,
        labels: np.ndarray | str | Path | None = None,
        *,
        image_size: int = 64,
        synthetic_fallback: bool = False,
        synthetic_samples: int = 8,
        seed: int = 42,
    ) -> None:
        """Khởi tạo từ mảng/Numpy hoặc thư mục vehicle/non-vehicle.

        Args:
            images: NHWC/NCHW RGB, NPY hoặc gốc ceps; có thể vắng khi fallback.
            labels: Class 0/1 hoặc score -1/+1; với thư mục tự suy ra từ tên.
            image_size: Kích thước ảnh huấn luyện vuông, mặc định 64.
            synthetic_fallback: Dùng ảnh giả có nhãn ±1 nếu thiếu nguồn thật.
            synthetic_samples: Số mẫu giả xác định.
            seed: Seed riêng cho synthetic data.
        """
        if image_size < 16 or image_size % 8:
            raise ValueError("image_size phải >=16 và chia hết cho 8.")
        self.image_size = int(image_size)
        try:
            image_array, label_array = self._load_data(images, labels)
            self.images = self._prepare_images(image_array)
            self.labels = self._prepare_labels(label_array, len(self.images))
        except (FileNotFoundError, OSError, ValueError) as error:
            if not synthetic_fallback:
                raise
            if synthetic_samples < 2:
                raise ValueError("synthetic_samples phải >= 2 để chia train/validation.") from error
            generator = np.random.default_rng(seed)
            array = generator.integers(0, 256, (synthetic_samples, image_size, image_size, 3), dtype=np.uint8)
            self.images = self._prepare_images(array)
            self.labels = torch.tensor(
                [1.0 if index % 2 else -1.0 for index in range(synthetic_samples)], dtype=torch.float32
            )

    @staticmethod
    def _load_data(
        images: np.ndarray | str | Path | None, labels: np.ndarray | str | Path | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Đọc các thư mục legacy vehicle/non-vehicles hoặc cặp mảng/NPY."""
        if images is None:
            raise FileNotFoundError("Chưa cấu hình nguồn ảnh detection.")
        if isinstance(images, (str, Path)) and Path(images).is_dir():
            root = Path(images)
            positive = sorted((root / "vehicles").glob("*/*.png"))
            negative = sorted((root / "non-vehicles").glob("*/*.png"))
            paths = positive + negative
            if not paths:
                raise FileNotFoundError(f"Không thấy vehicles/non-vehicles trong {root}.")
            return np.stack([np.asarray(Image.open(path).convert("RGB")) for path in paths]), np.array(
                [1.0] * len(positive) + [-1.0] * len(negative), dtype=np.float32
            )
        if labels is None:
            raise ValueError("Nguồn ảnh mảng/NPY phải đi kèm nhãn.")
        image_array = np.load(images, mmap_mode="r") if isinstance(images, (str, Path)) else np.asarray(images)
        label_array = np.load(labels, mmap_mode="r") if isinstance(labels, (str, Path)) else np.asarray(labels)
        return image_array, label_array

    def _prepare_images(self, values: np.ndarray) -> torch.Tensor:
        """Chuẩn hóa byte RGB thành NCHW float32 theo x/127.5 - 1.0."""
        array = np.asarray(values)
        if array.ndim != 4:
            raise ValueError(f"Ảnh cần là batch 4D NHWC/NCHW; nhận {array.shape}.")
        if array.shape[1] == 3 and array.shape[-1] != 3:
            tensor = torch.from_numpy(np.array(array, copy=True, order="C")).float()
        elif array.shape[-1] == 3:
            tensor = torch.from_numpy(np.array(array, copy=True, order="C")).float().permute(0, 3, 1, 2)
        else:
            raise ValueError("FCN detector yêu cầu đúng ba kênh RGB.")
        if tensor.numel() and (float(tensor.min()) < 0 or float(tensor.max()) > 255):
            raise ValueError("Ảnh nguồn phải có pixel trong [0,255] trước chuẩn hóa.")
        if tensor.shape[-2:] != (self.image_size, self.image_size):
            tensor = torch.nn.functional.interpolate(
                tensor, (self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
        return (tensor.contiguous() / 127.5 - 1.0).contiguous()

    @staticmethod
    def _prepare_labels(values: np.ndarray, sample_count: int) -> torch.Tensor:
        """Chuẩn hóa nhãn class 0/1 hoặc score ±1 thành target float32."""
        array = np.asarray(values).reshape(-1)
        if len(array) != sample_count or not np.isfinite(array).all():
            raise ValueError("Số lượng nhãn phải khớp ảnh và không chứa NaN/Inf.")
        unique = set(float(value) for value in np.unique(array))
        if unique.issubset({0.0, 1.0}):
            array = np.where(array == 1, 1.0, -1.0)
        elif not unique.issubset({-1.0, 1.0}):
            raise ValueError("Nhãn FCN legacy phải là class {0,1} hoặc score {-1,+1}.")
        return torch.from_numpy(np.array(array, dtype=np.float32, copy=True))

    def __len__(self) -> int:
        """Trả số ảnh huấn luyện."""
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Trả ảnh RGB (3,H,W) đã chuẩn hóa và score scalar float32."""
        # Giữ target cùng kích thước patch-score (1,1,1), tránh broadcast B×B
        # khi MSELoss so với đầu ra FCN (B,1,1,1).
        return self.images[index], self.labels[index].reshape(1, 1, 1)
