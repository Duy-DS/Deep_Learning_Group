"""Module định nghĩa các phép biến đổi (Transforms) và tăng cường dữ liệu (Data Augmentation) cho FER2013.

Tuân thủ nghiêm ngặt nguyên tắc hiến pháp AGENTS.md:
- Tham số chuẩn hóa Mean và Std được tính toán độc lập trên tập Training (T0105).
- TUYỆT ĐỐI KHÔNG áp dụng Data Augmentation lên tập Validation (PublicTest) và Test (PrivateTest).
- Hỗ trợ ảnh xám 1 kênh (mặc định) và ảnh 3 kênh cho các mô hình Transfer Learning.
"""

from typing import Tuple, Union
import torchvision.transforms as transforms

# Tham số chuẩn hóa được tính từ EDA (Task T0105) trên 28.709 ảnh tập Training
FER2013_MEAN: Tuple[float, ...] = (0.5077,)
FER2013_STD: Tuple[float, ...] = (0.2550,)

FER2013_MEAN_RGB: Tuple[float, ...] = (0.5077, 0.5077, 0.5077)
FER2013_STD_RGB: Tuple[float, ...] = (0.2550, 0.2550, 0.2550)


class FER2013TransformBuilder:
    """Lớp builder kiến tạo pipeline biến đổi cho tập huấn luyện và tập kiểm định/đánh giá."""

    def __init__(
        self,
        image_size: int = 48,
        in_channels: int = 1,
        mean: Union[Tuple[float, ...], None] = None,
        std: Union[Tuple[float, ...], None] = None,
        normalization: str = "fer2013",
    ) -> None:
        """Khởi tạo builder với các thông số cấu hình cơ bản.

        Args:
            image_size: Kích thước cạnh ảnh đầu ra (mặc định 48, hoặc 128, 224).
            in_channels: Số kênh ảnh đầu vào (1 cho Grayscale, 3 cho RGB).
            mean: Tuple giá trị trung bình pixel. Nếu None, dùng giá trị mặc định của FER2013.
            normalization: Chọn ``fer2013`` hoặc ``imagenet`` cho pipeline RGB.
            std: Tuple độ lệch chuẩn pixel. Nếu None, dùng giá trị mặc định của FER2013.
        """
        if in_channels not in (1, 3):
            raise ValueError(f"in_channels phải là 1 hoặc 3, nhận giá trị: {in_channels}")

        self.image_size = image_size
        self.in_channels = in_channels

        normalization_name = normalization.strip().lower()
        if normalization_name not in ("fer2013", "imagenet"):
            raise ValueError("normalization phải là 'fer2013' hoặc 'imagenet'.")
        if normalization_name == "imagenet" and in_channels != 3:
            raise ValueError("ImageNet normalization chỉ áp dụng cho ảnh RGB 3 kênh.")

        if normalization_name == "imagenet":
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)
        elif in_channels == 1:
            self.mean = FER2013_MEAN
            self.std = FER2013_STD
        else:
            self.mean = FER2013_MEAN_RGB
            self.std = FER2013_STD_RGB
        if mean is not None:
            self.mean = mean
        if std is not None:
            self.std = std

    def build_train_transforms(self, augmentation: str = "None") -> transforms.Compose:
        """Tạo chuỗi biến đổi dữ liệu cho tập huấn luyện (Training Set).

        Args:
            augmentation: Chiến lược tăng cường dữ liệu ("None", "Standard", "Advanced").

        Returns:
            transforms.Compose: Chuỗi các phép biến đổi ảnh PyTorch.
        """
        transform_list = []

        # 1. Chuyển đổi số kênh nếu mô hình yêu cầu 3 kênh (RGB)
        if self.in_channels == 3:
            transform_list.append(transforms.Grayscale(num_output_channels=3))

        # 2. Thay đổi kích thước ảnh nếu kích thước khác 48x48
        if self.image_size != 48:
            transform_list.append(transforms.Resize((self.image_size, self.image_size)))

        # 3. Áp dụng các kỹ thuật Data Augmentation trên tập Train
        aug_lower = augmentation.lower()
        if aug_lower in ("standard", "basic", "flip_crop"):
            transform_list.extend([
                transforms.RandomCrop(self.image_size, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ])
        elif aug_lower in ("advanced", "strong"):
            transform_list.extend([
                transforms.RandomCrop(self.image_size, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ])
        elif aug_lower not in ("none", "null", ""):
            raise ValueError(f"Không hỗ trợ chiến lược augmentation: '{augmentation}'")

        # 4. Chuyển đổi sang Tensor và Chuẩn hóa (Normalize)
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        return transforms.Compose(transform_list)

    def build_eval_transforms(self) -> transforms.Compose:
        """Tạo chuỗi biến đổi dữ liệu cho tập kiểm định (PublicTest) và kiểm thử (PrivateTest).

        TUYỆT ĐỐI KHÔNG áp dụng bất kỳ phép tăng cường ngẫu nhiên nào.

        Returns:
            transforms.Compose: Chuỗi các phép biến đổi deterministic cho tập đánh giá.
        """
        transform_list = []

        # 1. Chuyển đổi số kênh nếu mô hình yêu cầu 3 kênh (RGB)
        if self.in_channels == 3:
            transform_list.append(transforms.Grayscale(num_output_channels=3))

        # 2. Thay đổi kích thước ảnh nếu kích thước khác 48x48
        if self.image_size != 48:
            transform_list.append(transforms.Resize((self.image_size, self.image_size)))

        # 3. Chuyển đổi sang Tensor và Chuẩn hóa (Normalize)
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        return transforms.Compose(transform_list)


def get_transforms(
    image_size: int = 48,
    in_channels: int = 1,
    augmentation: str = "None",
    mean: Union[Tuple[float, ...], None] = None,
    std: Union[Tuple[float, ...], None] = None,
    normalization: str = "fer2013",
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Hàm tiện ích khởi tạo nhanh cặp transforms (train_transform, eval_transform).

    Args:
        image_size: Kích thước ảnh cạnh vuông (48, 128, 224).
        in_channels: Số lượng kênh đầu vào (1 hoặc 3).
        augmentation: Kiểu tăng cường dữ liệu cho tập train.
        mean: Bộ giá trị trung bình pixel.
        std: Bộ giá trị độ lệch chuẩn pixel.
        normalization: Chuẩn hóa ``fer2013`` hoặc ``imagenet`` (chỉ cho RGB).

    Returns:
        Tuple[transforms.Compose, transforms.Compose]: (train_transform, eval_transform)
    """
    builder = FER2013TransformBuilder(
        image_size=image_size,
        in_channels=in_channels,
        mean=mean,
        std=std,
        normalization=normalization,
    )
    train_transform = builder.build_train_transforms(augmentation=augmentation)
    eval_transform = builder.build_eval_transforms()
    return train_transform, eval_transform

