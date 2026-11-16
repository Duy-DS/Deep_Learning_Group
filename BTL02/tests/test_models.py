"""Unit tests cho các kiến trúc mô hình: SmallCNN, MediumCNN, LargeCNN."""

import pytest
import torch
import torch.nn as nn

from src.models import SmallCNN, MediumCNN, LargeCNN
from src.train.utils import count_parameters


def test_small_cnn_architecture():
    """Kiểm tra SmallCNN 2-block: Output shape [B, 7], số tham số ~1.20M và target layer."""
    model = SmallCNN(num_classes=7, in_channels=1, dropout=0.5)
    dummy_input = torch.randn(2, 1, 48, 48)
    output = model(dummy_input)

    assert output.shape == torch.Size([2, 7])
    assert count_parameters(model) == 1199495
    assert isinstance(model.get_target_layer(), nn.Conv2d)
    assert model.get_target_layer().out_channels == 64


def test_medium_cnn_architecture():
    """Kiểm tra MediumCNN 3-block: Output shape [B, 7], số tham số ~2.73M và target layer."""
    model = MediumCNN(num_classes=7, in_channels=1, dropout=0.5)
    dummy_input = torch.randn(2, 1, 48, 48)
    output = model(dummy_input)

    assert output.shape == torch.Size([2, 7])
    assert count_parameters(model) == 2731015
    assert isinstance(model.get_target_layer(), nn.Conv2d)
    assert model.get_target_layer().out_channels == 256


def test_large_cnn_architecture():
    """Kiểm tra LargeCNN 4-block + BatchNorm + Dropout: Output shape [B, 7], số tham số ~3.92M."""
    model = LargeCNN(num_classes=7, in_channels=1, dropout=0.5)
    dummy_input = torch.randn(2, 1, 48, 48)
    output = model(dummy_input)

    assert output.shape == torch.Size([2, 7])
    assert count_parameters(model) == 3916167
    assert isinstance(model.get_target_layer(), nn.Conv2d)
    assert model.get_target_layer().out_channels == 512
