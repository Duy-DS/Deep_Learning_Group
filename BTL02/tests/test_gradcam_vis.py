"""Kiểm thử Grad-CAM và trích xuất tích chập trên CNN PyTorch nhỏ/sâu."""

import torch
import torch.nn as nn

from src.evaluate.gradcam_vis import GradCAMVisualizer


class SmallCNN(nn.Module):
    """CNN nhỏ dùng kiểm thử CPU độc lập dữ liệu FER2013."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 7)

    def forward(self, image):
        image = torch.relu(self.conv1(image))
        image = torch.relu(self.conv2(image))
        return self.fc(self.pool(image).flatten(1))


class DeepCNN(nn.Module):
    """CNN sâu hơn để kiểm tra chọn tầng cuối tổng quát."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 8, 3, padding=1),
            nn.ReLU(), nn.Conv2d(8, 12, 3, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(12, 7)

    def forward(self, image):
        return self.fc(self.pool(self.features(image)).flatten(1))


def test_gradcam_heatmap_shape_range_and_overlay(tmp_path):
    model = SmallCNN()
    visualizer = GradCAMVisualizer(model, tmp_path, device=torch.device("cpu"))
    image = torch.rand(1, 48, 48)
    heatmap = visualizer.compute_heatmap(image, class_index=2)
    assert heatmap.shape == (48, 48)
    assert torch.isfinite(heatmap).all()
    assert 0.0 <= float(heatmap.min()) <= float(heatmap.max()) <= 1.0
    assert visualizer.save_overlay(image, heatmap).is_file()


def test_extract_filters_and_feature_maps_for_shallow_and_deep_models(tmp_path):
    for model in (SmallCNN(), DeepCNN()):
        visualizer = GradCAMVisualizer(model, tmp_path, device=torch.device("cpu"))
        maps = visualizer.extract_feature_maps(torch.rand(2, 1, 32, 32))
        assert maps
        assert all(value.shape[0] == 2 and value.ndim == 4 for value in maps.values())
        assert visualizer.plot_filters(max_filters=4).is_file()
        assert visualizer.plot_feature_maps(maps, max_channels=2).is_file()
        assert visualizer.compute_heatmap(torch.rand(1, 1, 32, 32)).shape == (32, 32)