"""Kiểm thử shape, kiểu dữ liệu và phép xoay Dataset vision BTL03."""

import gzip
import struct

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.download_data import DataBootstrapper
from src.datasets import FEREmotionDataset, MNISTDataset, WikiGenderDataset
from src.datasets.vision_datasets import load_fer_onehot_csv


def test_datasets_emit_nchw_float_and_long_labels(tmp_path):
    """Ba Dataset phải xuất NCHW float32 và nhãn long qua DataLoader."""
    rng = np.random.default_rng(42)
    mnist = MNISTDataset(
        rng.integers(0, 256, (6, 28, 28), dtype=np.uint8), np.arange(6) % 10
    )
    wiki = WikiGenderDataset(
        rng.integers(0, 256, (6, 64, 64, 1), dtype=np.uint8),
        np.eye(2, dtype=np.float32)[np.arange(6) % 2],
    )
    fer = FEREmotionDataset(
        rng.integers(0, 256, (6, 48, 48), dtype=np.uint8),
        np.eye(7, dtype=np.float32)[np.arange(6) % 7],
    )

    for dataset, shape, classes in [
        (mnist, (1, 28, 28), 10),
        (wiki, (1, 64, 64), 2),
        (fer, (1, 48, 48), 7),
    ]:
        images, labels = next(iter(DataLoader(dataset, batch_size=3)))
        assert tuple(images.shape) == (3, *shape)
        assert images.dtype == torch.float32
        assert labels.dtype == torch.long
        assert 0 <= float(images.min()) <= float(images.max()) <= 1
        assert int(labels.min()) >= 0 and int(labels.max()) < classes


def test_wiki_rotation_matches_legacy_axes_and_direction():
    """WIKI xoay đúng k=3 trên axes=(1,2), trước chuyển NCHW."""
    image = np.zeros((1, 64, 64, 1), dtype=np.uint8)
    image[0, 2, 5, 0] = 255
    labels = np.array([1])
    # Dataset thực hiện đúng phép xoay rồi chuẩn hóa uint8 về [0, 1].
    expected = np.rot90(image, k=3, axes=(1, 2)).astype(np.float32) / 255.0
    dataset = WikiGenderDataset(image, labels)
    actual_nhwc = dataset.images.permute(0, 2, 3, 1).numpy()
    assert np.array_equal(actual_nhwc, expected)
    assert actual_nhwc[0, 5, 61, 0] == 1.0


def test_wiki_can_disable_rotation():
    """Có thể tắt xoay khi nguồn đã được xoay trước đó."""
    images = np.zeros((1, 64, 64, 1), dtype=np.uint8)
    images[0, 2, 5, 0] = 255
    dataset = WikiGenderDataset(images, np.array([0]), apply_legacy_rotation=False)
    assert dataset.images[0, 0, 2, 5].item() == 1.0


def test_fer_legacy_csv_has_2304_pixels_then_seven_onehot_labels(tmp_path):
    """Bộ đọc FER xác thực đúng số cột và giữ cấu trúc nhãn legacy."""
    pixels = np.arange(2 * 2304, dtype=np.float32).reshape(2, 2304) % 256
    labels = np.eye(7, dtype=np.float32)[[2, 6]]
    path = tmp_path / "fer.csv"
    np.savetxt(path, np.concatenate([pixels, labels], axis=1), delimiter=",")
    images, loaded_labels = load_fer_onehot_csv(path)
    assert images.shape == (2, 48, 48)
    assert loaded_labels.shape == (2, 7)
    assert loaded_labels.argmax(1).tolist() == [2, 6]
    dataset = FEREmotionDataset(images, loaded_labels, normalization="auto")
    assert dataset.images.max().item() == pytest.approx(1.0)


def test_wrong_image_size_and_label_range_are_rejected():
    """Schema sai phải báo lỗi thay vì âm thầm đổi kích thước/nhãn."""
    with pytest.raises(ValueError, match="Kích thước ảnh"):
        MNISTDataset(np.zeros((2, 27, 28), dtype=np.uint8), np.array([0, 1]))
    with pytest.raises(ValueError, match="Nhãn phải nằm"):
        MNISTDataset(np.zeros((2, 28, 28), dtype=np.uint8), np.array([0, 10]))


def test_idx_readers_validate_headers_and_shapes(tmp_path):
    """Đọc IDX ảnh/nhãn MNIST bằng parser chuẩn thư viện hệ thống."""
    image_path = tmp_path / "images.gz"
    label_path = tmp_path / "labels.gz"
    with gzip.open(image_path, "wb") as stream:
        stream.write(struct.pack(">IIII", 2051, 2, 28, 28))
        stream.write(bytes(range(256)) * 6 + bytes(range(32)))
    with gzip.open(label_path, "wb") as stream:
        stream.write(struct.pack(">II", 2049, 2))
        stream.write(bytes([3, 9]))
    images = DataBootstrapper.read_idx_images(image_path)
    labels = DataBootstrapper.read_idx_labels(label_path)
    assert images.shape == (2, 28, 28) and images.dtype == np.uint8
    assert labels.tolist() == [3, 9]