"""Unit tests cho module FER2013Dataset, DataLoaders và Transforms."""

import pytest
import torch
from pathlib import Path

from src.data.dataset import FER2013Dataset, get_dataloaders
from src.data.transforms import get_transforms, FER2013_MEAN, FER2013_STD

CSV_PATH = None


@pytest.fixture(scope="module")
def check_csv_exists(tmp_path_factory):
    """Sinh FER2013 giả lập ngoài repo, không mở hoặc sửa dữ liệu thật."""
    global CSV_PATH
    CSV_PATH = tmp_path_factory.mktemp("fer2013_synthetic") / "fer2013.csv"
    pixels = " ".join(["127"] * 2304)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as stream:
        stream.write("emotion,pixels,Usage\n")
        for split, size in (("Training", 28709), ("PublicTest", 3589), ("PrivateTest", 3589)):
            for index in range(size):
                label = index % 7 if index % 10 == 0 else 3
                stream.write(f"{label},{pixels},{split}\n")


def test_split_counts_integrity(check_csv_exists):
    """Kiểm tra tính toàn vẹn và phân chia mẫu đúng chuẩn 28.709 / 3.589 / 3.589."""
    train_ds = FER2013Dataset(csv_path=CSV_PATH, split="train")
    val_ds = FER2013Dataset(csv_path=CSV_PATH, split="val")
    test_ds = FER2013Dataset(csv_path=CSV_PATH, split="test")

    assert len(train_ds) == 28709, f"Kỳ vọng 28709 mẫu train, nhận được: {len(train_ds)}"
    assert len(val_ds) == 3589, f"Kỳ vọng 3589 mẫu val, nhận được: {len(val_ds)}"
    assert len(test_ds) == 3589, f"Kỳ vọng 3589 mẫu test, nhận được: {len(test_ds)}"
    assert len(train_ds) + len(val_ds) + len(test_ds) == 35887


def test_item_shapes_and_types(check_csv_exists):
    """Kiểm tra định dạng tensor ảnh [1, 48, 48] và nhãn int64 trong khoảng [0, 6]."""
    train_transform, eval_transform = get_transforms(image_size=48, in_channels=1)
    ds = FER2013Dataset(csv_path=CSV_PATH, split="train", transform=train_transform)

    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert img.shape == torch.Size([1, 48, 48])
    assert img.dtype == torch.float32
    assert label.dtype == torch.long
    assert 0 <= label.item() <= 6


def test_class_weights_balanced(check_csv_exists):
    """Kiểm tra tính toán class weights phản ánh sự mất cân bằng giữa Disgust và Happy."""
    ds = FER2013Dataset(csv_path=CSV_PATH, split="train")
    weights = ds.get_class_weights(method="balanced")

    assert len(weights) == 7
    assert isinstance(weights, torch.Tensor)
    # Lớp Disgust (ID 1) ít mẫu nhất nên trọng số phải cao hơn lớp Happy (ID 3)
    assert weights[1] > weights[3]
    print(f"\nClass weights (balanced): {weights.tolist()}")


def test_dataloader_batch_dimensions(check_csv_exists):
    """Kiểm tra DataLoader sinh ra batch đúng kích thước [B, 1, 48, 48] và [B]."""
    batch_size = 32
    train_loader, val_loader, test_loader = get_dataloaders(
        csv_path=CSV_PATH,
        batch_size=batch_size,
        num_workers=0,
        image_size=48,
        in_channels=1,
        augmentation="Standard",
        seed=42,
    )

    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == torch.Size([batch_size, 1, 48, 48])
    assert batch_y.shape == torch.Size([batch_size])

    val_x, val_y = next(iter(val_loader))
    assert val_x.shape == torch.Size([batch_size, 1, 48, 48])
    assert val_y.shape == torch.Size([batch_size])


def test_rgb_and_resize_mode(check_csv_exists):
    """Kiểm tra chế độ 3 kênh RGB và kích thước mở rộng (ví dụ 128x128 cho Transfer Learning)."""
    batch_size = 16
    train_loader, _, _ = get_dataloaders(
        csv_path=CSV_PATH,
        batch_size=batch_size,
        num_workers=0,
        image_size=128,
        in_channels=3,
        augmentation="None",
        seed=42,
    )

    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == torch.Size([batch_size, 3, 128, 128])
    assert batch_y.shape == torch.Size([batch_size])


def test_reproducibility_dataloader_shuffle(check_csv_exists):
    """Kiểm tra tính tái lập khi thiết lập seed=42 trên DataLoader train."""
    loader_1, _, _ = get_dataloaders(
        csv_path=CSV_PATH,
        batch_size=16,
        num_workers=0,
        seed=42,
    )
    batch1_x, batch1_y = next(iter(loader_1))

    loader_2, _, _ = get_dataloaders(
        csv_path=CSV_PATH,
        batch_size=16,
        num_workers=0,
        seed=42,
    )
    batch2_x, batch2_y = next(iter(loader_2))

    assert torch.equal(batch1_y, batch2_y), "Labels giữa 2 lần chạy seed=42 phải giống nhau tuyệt đối!"


def test_imagenet_normalization_for_rgb_pipeline():
    """Kiểm tra pipeline RGB tùy chọn dùng mean/std ImageNet cho WP3."""
    _, eval_transform = get_transforms(
        image_size=128,
        in_channels=3,
        normalization="imagenet",
    )
    normalize = next(transform for transform in eval_transform.transforms if type(transform).__name__ == "Normalize")
    assert tuple(normalize.mean) == (0.485, 0.456, 0.406)
    assert tuple(normalize.std) == (0.229, 0.224, 0.225)

