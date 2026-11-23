"""Pipeline đa kích thước và chuẩn hóa ImageNet cho mô hình Transfer Learning."""

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ToPILImage:
    """Chuyển ảnh PIL, NumPy hoặc Tensor về PIL để dùng torchvision transforms."""

    def __call__(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        """Chuẩn hóa kiểu đầu vào thành ảnh PIL.

        Args:
            image: Ảnh xám hoặc RGB dạng PIL, NumPy, hay Tensor.

        Returns:
            Image.Image: Ảnh PIL đã chuyển đổi.

        Raises:
            TypeError: Kiểu ảnh đầu vào không được hỗ trợ.
        """
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, np.ndarray):
            return transforms.ToPILImage()(image)
        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 3:
                raise ValueError("Tensor ảnh phải có dạng [H,W], [1,H,W] hoặc [C,H,W].")
            return transforms.ToPILImage()(tensor)
        raise TypeError("Ảnh đầu vào phải là PIL.Image, numpy.ndarray hoặc torch.Tensor.")


class SOTATransformBuilder:
    """Kiến tạo train/eval transforms cho đầu vào ImageNet 3 kênh đa kích thước."""

    _AUG_DEFAULTS = {
        "horizontal_flip_p": 0.5,
        "rotation_degrees": 10.0,
        "affine_degrees": 0.0,
        "translate": (0.05, 0.05),
        "color_jitter": 0.05,
        "random_erasing_p": 0.0,
    }

    def __init__(
        self,
        image_size: int = 224,
        use_augmentation: bool = True,
        aug_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Khởi tạo cấu hình tiền xử lý và kiểm tra tham số.

        Args:
            image_size: Cạnh ảnh đầu ra; hỗ trợ 128, 224 hoặc số nguyên >=48.
            use_augmentation: Chỉ áp dụng biến đổi ngẫu nhiên cho tập train.
            aug_config: Tùy chọn các mức flip/rotation/affine/jitter/erasing.

        Raises:
            ValueError: Kích thước hoặc mức augmentation không hợp lệ.
        """
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 48:
            raise ValueError("image_size phải là số nguyên >= 48 (khuyến nghị 128 hoặc 224).")
        self.image_size = image_size
        self.use_augmentation = bool(use_augmentation)
        self.aug_config = {**self._AUG_DEFAULTS, **(aug_config or {})}
        self._validate_augmentation()

    def _validate_augmentation(self) -> None:
        """Kiểm tra xác suất và miền giá trị augmentation cấu hình."""
        for key in ("horizontal_flip_p", "random_erasing_p"):
            value = float(self.aug_config[key])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} phải nằm trong [0, 1].")
        for key in ("rotation_degrees", "affine_degrees", "color_jitter"):
            if float(self.aug_config[key]) < 0.0:
                raise ValueError(f"{key} không được âm.")
        translate = self.aug_config["translate"]
        if not isinstance(translate, Sequence) or len(translate) != 2:
            raise ValueError("translate phải là tuple/list gồm hai tỷ lệ.")
        if any(float(value) < 0.0 or float(value) > 1.0 for value in translate):
            raise ValueError("Các thành phần translate phải nằm trong [0, 1].")

    def _base_transforms(self) -> list:
        """Tạo phần deterministic chung: PIL, RGB, bilinear resize."""
        return [
            ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=InterpolationMode.BILINEAR,
            ),
        ]

    def build_train_transform(self) -> transforms.Compose:
        """Tạo pipeline train; augmentation ngẫu nhiên chỉ xuất hiện ở đây."""
        pipeline = self._base_transforms()
        if self.use_augmentation:
            mode = str(self.aug_config.get("mode", "advanced")).strip().lower()
            if mode not in {"none", "basic", "advanced"}:
                raise ValueError("augmentation.mode phải là 'none', 'basic' hoặc 'advanced'.")
            if mode in {"basic", "advanced"} and self.aug_config["horizontal_flip_p"] > 0:
                pipeline.append(
                    transforms.RandomHorizontalFlip(p=float(self.aug_config["horizontal_flip_p"]))
                )
            if mode in {"basic", "advanced"} and self.aug_config["rotation_degrees"] > 0:
                pipeline.append(
                    transforms.RandomRotation(degrees=float(self.aug_config["rotation_degrees"]))
                )
            if mode in {"basic", "advanced"} and (
                self.aug_config["affine_degrees"] > 0 or any(self.aug_config["translate"])
            ):
                pipeline.append(
                    transforms.RandomAffine(
                        degrees=float(self.aug_config["affine_degrees"]),
                        translate=tuple(float(v) for v in self.aug_config["translate"]),
                    )
                )
            jitter = float(self.aug_config["color_jitter"]) if mode == "advanced" else 0.0
            if jitter > 0:
                pipeline.append(
                    transforms.ColorJitter(brightness=jitter, contrast=jitter)
                )
        pipeline.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        mode = str(self.aug_config.get("mode", "advanced")).strip().lower()
        erase_p = float(self.aug_config["random_erasing_p"])
        if self.use_augmentation and mode == "advanced" and erase_p > 0:
            pipeline.append(
                transforms.RandomErasing(
                    p=erase_p,
                    scale=tuple(self.aug_config.get("random_erasing_scale", (0.02, 0.15))),
                    ratio=tuple(self.aug_config.get("random_erasing_ratio", (0.3, 3.3))),
                    value=self.aug_config.get("random_erasing_value", "random"),
                )
            )
        return transforms.Compose(pipeline)

    def build_eval_transform(self) -> transforms.Compose:
        """Tạo pipeline deterministic cho PublicTest và PrivateTest, không augmentation."""
        return transforms.Compose(
            self._base_transforms()
            + [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )