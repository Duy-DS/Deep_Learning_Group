"""Gói dữ liệu FER2013 với nạp module lười để CLI tải chỉ cần thư viện chuẩn."""

__all__ = [
    "FER2013Dataset",
    "get_dataloaders",
    "get_transforms",
    "FER2013TransformBuilder",
    "FER2013_MEAN",
    "FER2013_STD",
    "SOTATransformBuilder",
    "ToPILImage",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]


def __getattr__(name: str):
    """Nạp lớp dữ liệu hoặc transforms khi mã gọi cần đến chúng."""
    if name in ("FER2013Dataset", "get_dataloaders"):
        from src.data import dataset

        return getattr(dataset, name)
    if name in ("get_transforms", "FER2013TransformBuilder", "FER2013_MEAN", "FER2013_STD"):
        from src.data import transforms

        return getattr(transforms, name)
    if name in ("SOTATransformBuilder", "ToPILImage", "IMAGENET_MEAN", "IMAGENET_STD"):
        from src.data import transforms_sota

        return getattr(transforms_sota, name)
    raise AttributeError(f"module {__name__!r} không có thuộc tính {name!r}")
