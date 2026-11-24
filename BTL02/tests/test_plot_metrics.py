"""Kiểm thử offline cho trực quan hóa metrics FER2013."""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate.plot_metrics import MetricsVisualizer


def test_synthetic_history_and_plot(tmp_path):
    visualizer = MetricsVisualizer(tmp_path)
    history = visualizer.synthetic_history(epochs=4)
    path = visualizer.plot_training_curves(history)
    assert path.is_file()
    assert path.stat().st_size > 0


def test_summary_csv_is_not_misread_as_epoch_history(tmp_path):
    path = tmp_path / "summary.csv"
    pd.DataFrame({"metrics.train_loss": [0.5], "metrics.train_acc": [80]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="không có lịch sử theo epoch"):
        MetricsVisualizer.load_history(path)


def test_confusion_matrix_and_classification_report(tmp_path):
    visualizer = MetricsVisualizer(tmp_path, class_names=("Angry", "Disgust", "Fear"))
    targets = torch.tensor([0, 0, 1, 2, 2])
    predictions = torch.tensor([0, 1, 1, 2, 0])
    result = visualizer.build_report(targets, predictions)
    matrix = result["confusion_matrix"]
    assert matrix.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 1]]
    assert result["classification_report"]["macro avg"]["support"] == 5
    paths = visualizer.save_report(result, "PublicTest")
    assert all(path.is_file() for path in paths.values())
    assert (tmp_path / "confusion_PublicTest_counts.png").is_file()
    assert (tmp_path / "confusion_PublicTest_normalized.png").is_file()


def test_inference_is_eval_only_and_supports_all_classes(tmp_path):
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 3))
    data = torch.randn(6, 1, 2, 2)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    model.train()
    visualizer = MetricsVisualizer(tmp_path, class_names=("A", "B", "C"), device=torch.device("cpu"))
    targets, predictions = visualizer.infer(model, DataLoader(TensorDataset(data, labels), batch_size=2))
    assert not model.training
    assert targets.shape == predictions.shape == (6,)
    assert np.array_equal(targets.numpy(), labels.numpy())