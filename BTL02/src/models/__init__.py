"""Model architectures: Baseline, Custom CNN, and Pretrained models."""

from src.models.base_model import BaseEmotionModel
from src.models.baseline import SmallCNN

__all__ = ["BaseEmotionModel", "SmallCNN"]