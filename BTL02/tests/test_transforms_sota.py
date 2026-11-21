"""Kiểm thử pipeline đa kích thước và chuẩn hóa ImageNet cho WP3."""

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from src.data.transforms_sota import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SOTATransformBuilder,
    ToPILImage,
)


@pytest.mark.parametrize("image_size", [128, 224])
def test_sota_transform_shapes_and_types(image_size):
    """Ảnh FER2013 48×48 được đổi thành tensor RGB đúng độ phân giải."""
    image = Image.fromarray(np.full((48, 48), 127, dtype=np.uint8), mode="L")
    builder = SOTATransformBuilder(image_size=image_size, use_augmentation=False)
    output = builder.build_eval_transform()(image)
    assert output.shape == (3, image_size, image_size)
    assert output.dtype == torch.float32


def test_eval_transform_repeats_gray_channel_then_applies_imagenet_formula():
    """Ba kênh trước Normalize bằng nhau và từng kênh đúng công thức ImageNet."""
    pixel = 127
    image = Image.fromarray(np.full((48, 48), pixel, dtype=np.uint8), mode="L")
    builder = SOTATransformBuilder(image_size=128, use_augmentation=False)

    before_normalize = transforms.Compose(
        builder._base_transforms() + [transforms.ToTensor()]
    )(image)
    assert torch.equal(before_normalize[0], before_normalize[1])
    assert torch.equal(before_normalize[1], before_normalize[2])

    output = builder.build_eval_transform()(image)
    raw_value = pixel / 255.0
    for channel, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        expected = (raw_value - mean) / std
        assert torch.allclose(output[channel], torch.full_like(output[channel], expected))


def test_eval_transform_is_deterministic_and_has_no_random_ops():
    """Public/Private eval pipeline chỉ có biến đổi xác định, tuyệt đối không random."""
    image = Image.fromarray(np.arange(48 * 48, dtype=np.uint8).reshape(48, 48), mode="L")
    builder = SOTATransformBuilder(image_size=128, use_augmentation=True)
    evaluation = builder.build_eval_transform()
    random_types = (
        transforms.RandomCrop,
        transforms.RandomHorizontalFlip,
        transforms.RandomRotation,
        transforms.RandomAffine,
        transforms.ColorJitter,
        transforms.RandomErasing,
    )
    assert not any(isinstance(operation, random_types) for operation in evaluation.transforms)
    first = evaluation(image)
    second = evaluation(image)
    assert torch.equal(first, second)


def test_train_transform_can_generate_random_augmentation_variants():
    """Train pipeline có phép ngẫu nhiên và có thể tạo hai biến thể khác nhau."""
    image = Image.fromarray(np.arange(48 * 48, dtype=np.uint8).reshape(48, 48), mode="L")
    builder = SOTATransformBuilder(
        image_size=128,
        use_augmentation=True,
        aug_config={
            "horizontal_flip_p": 0.5,
            "rotation_degrees": 25.0,
            "affine_degrees": 0.0,
            "translate": (0.0, 0.0),
            "color_jitter": 0.0,
            "random_erasing_p": 0.0,
        },
    )
    training = builder.build_train_transform()
    assert any(isinstance(operation, transforms.RandomHorizontalFlip) for operation in training.transforms)
    torch.manual_seed(17)
    first = training(image)
    torch.manual_seed(29)
    second = training(image)
    assert not torch.equal(first, second)


@pytest.mark.parametrize("input_kind", ["pil", "numpy", "tensor"])
def test_input_adapter_accepts_pil_numpy_and_tensor(input_kind):
    """Adapter đầu vào hỗ trợ PIL, NumPy và Tensor xám."""
    array = np.full((48, 48), 100, dtype=np.uint8)
    if input_kind == "pil":
        image = Image.fromarray(array, mode="L")
    elif input_kind == "numpy":
        image = array
    else:
        image = torch.from_numpy(array.copy())
    converted = ToPILImage()(image)
    assert converted.mode == "L"
    builder = SOTATransformBuilder(image_size=224, use_augmentation=False)
    result = builder.build_eval_transform()(image)
    assert result.shape == (3, 224, 224)


@pytest.mark.parametrize("invalid_size", [47, 0, -1, 128.0, True])
def test_sota_builder_rejects_invalid_image_size(invalid_size):
    """Kích thước không nguyên hoặc nhỏ hơn 48 bị từ chối."""
    with pytest.raises(ValueError, match="image_size"):
        SOTATransformBuilder(image_size=invalid_size)


def test_sota_builder_rejects_invalid_augmentation_probability():
    """Xác suất augmentation phải nằm trong đoạn đóng [0,1]."""
    with pytest.raises(ValueError, match="horizontal_flip_p"):
        SOTATransformBuilder(aug_config={"horizontal_flip_p": 1.5})