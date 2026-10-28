"""Các module cốt lõi cho bài toán xấp xỉ hàm Gaussian trong dự án BTL01."""

from src.data_generator import GaussianDataGenerator, GaussianDataset
from src.models import FunctionApproximatorMLP
from src.trainer import Trainer
from src.visualizer import Visualizer

__all__ = [
    "GaussianDataGenerator", "GaussianDataset", "FunctionApproximatorMLP",
    "Trainer", "Visualizer",
]
