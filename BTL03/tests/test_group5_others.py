"""Kiểm thử Dataset, parity tham số và heatmap cho nhóm 5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets import FCNObjectDetectionDataset, SemanticSegmentationDataset
from src.models import FCNObjectDetector, SegNet
from scripts.run_group5_others import Group5OthersRunner


def test_segmentation_dataset_converts_legacy_palette_and_one_hot():
    """Ảnh depth về NCHW, màu/one-hot mask thành class-index long."""
    images = np.arange(2 * 96 * 96, dtype=np.uint8).reshape(2, 96, 96)
    colors = np.array(SemanticSegmentationDataset.LEGACY_COLORS, dtype=np.uint8)
    masks = np.tile(colors[np.arange(96)[:, None] % 7], (1, 96))[None].repeat(2, axis=0)
    dataset = SemanticSegmentationDataset(images, masks, image_size=96)
    image, mask = dataset[0]
    assert image.shape == (1, 96, 96) and image.dtype == torch.float32
    assert mask.shape == (96, 96) and mask.dtype == torch.long
    assert set(mask.unique().tolist()) == set(range(7))
    batched = next(iter(DataLoader(dataset, batch_size=2)))
    assert batched[1].shape == (2, 96, 96)

    one_hot = np.eye(7, dtype=np.uint8)[np.zeros((2, 96, 96), dtype=np.int64)]
    _, one_hot_mask = SemanticSegmentationDataset(images, one_hot)[0]
    assert one_hot_mask.shape == (96, 96)
    assert torch.all(one_hot_mask == 0)


def test_group5_datasets_use_correct_normalization_and_synthetic_fallback():
    """FCN RGB dùng công thức byte-range; cả hai dataset có fallback xác định."""
    rgb = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    rgb[1] = 255
    detector_data = FCNObjectDetectionDataset(rgb, np.array([0, 1]))
    first, target = detector_data[0]
    second, positive_target = detector_data[1]
    assert first.shape == (3, 64, 64) and first.dtype == torch.float32
    assert torch.all(first == -1) and torch.all(second == 1)
    assert target.shape == (1, 1, 1) and target.item() == -1
    assert positive_target.item() == 1

    seg_a = SemanticSegmentationDataset(synthetic_fallback=True, synthetic_samples=4)
    seg_b = SemanticSegmentationDataset(synthetic_fallback=True, synthetic_samples=4)
    assert torch.equal(seg_a.images, seg_b.images)
    assert seg_a[0][1].dtype == torch.long
    fcn = FCNObjectDetectionDataset(synthetic_fallback=True, synthetic_samples=4)
    assert fcn[0][0].min() >= -1 and fcn[0][0].max() <= 1


def test_group5_models_match_legacy_parameter_counts_and_shapes():
    """SegNet/FCN totals match stored Keras model.summary and output contracts."""
    torch.set_num_threads(1)
    segnet = SegNet().eval()
    assert segnet.count_parameters() == {
        "total_params": 29_458_211,
        "trainable_params": 29_442_325,
        "non_trainable_params": 15_886,
    }
    with torch.inference_mode():
        logits = segnet(torch.zeros(1, 1, 96, 96))
    assert logits.shape == (1, 7, 96, 96)
    assert torch.isfinite(torch.nn.functional.cross_entropy(logits, torch.zeros(1, 96, 96, dtype=torch.long)))

    detector = FCNObjectDetector().eval()
    assert detector.count_parameters()["total_params"] == 83_367
    with torch.inference_mode():
        patch = detector(torch.zeros(2, 3, 64, 64))
        large = detector(torch.zeros(1, 3, 160, 192))
        heatmap, boxes = detector.detect_objects(torch.zeros(1, 3, 160, 192), threshold=0.0)
    assert patch.shape == (2, 1, 1, 1)
    assert large.shape == (1, 1, 13, 17)
    assert heatmap.shape == large.shape
    assert isinstance(boxes, list)


def test_group5_metrics_and_yaml_configuration():
    """Metrics pixel/IoU và YAML giữ hyperparameters legacy cơ bản."""
    config_root = Path(__file__).resolve().parents[1] / "configs"
    seg = yaml.safe_load((config_root / "segnet.yaml").read_text(encoding="utf-8"))
    det = yaml.safe_load((config_root / "fcn_obj_detection.yaml").read_text(encoding="utf-8"))
    assert (seg["num_classes"], seg["image_size"], seg["learning_rate"], seg["batch_size"], seg["epochs"]) == (7, 96, 1e-4, 8, 15)
    assert (det["image_size"], det["learning_rate"], det["batch_size"], det["epochs"]) == (64, 0.001, 128, 5)
    masks = torch.tensor([[[0, 1], [1, 0]]])
    logits = torch.nn.functional.one_hot(masks, 2).permute(0, 3, 1, 2).float() * 10
    assert Group5OthersRunner.pixel_accuracy(logits, masks).item() == 1.0
    assert Group5OthersRunner.mean_iou(logits, masks, 2).item() == 1.0


def test_missing_segmentation_data_errors_without_fallback():
    """Không âm thầm đổi thiếu dữ liệu thật thành synthetic nếu không bật cờ."""
    with pytest.raises((FileNotFoundError, ValueError)):
        SemanticSegmentationDataset("missing-depth", "missing-labels")
