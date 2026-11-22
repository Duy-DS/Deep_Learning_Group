"""Điểm vào CLI mẫu và export trainer lõi của BTL03."""

import argparse

from .engine import BaseTrainer, TrainerCallback

__all__ = ["BaseTrainer", "TrainerCallback", "main"]


def main() -> None:
    """Hiển thị lệnh trợ giúp; training pipeline cụ thể được bổ sung theo task."""
    import sys

    # Hỗ trợ Windows console có code page không biểu diễn đủ tiếng Việt.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Core trainer BTL03 đã sẵn sàng; chọn model/config khi pipeline được nối."
    )
    parser.add_argument(
        "--config",
        help="Đường dẫn YAML cấu hình pipeline; hiện chưa có runner model hoàn chỉnh.",
    )
    parser.add_argument(
        "--help-core",
        action="store_true",
        help="Hiển thị trạng thái các API core trainer.",
    )
    args = parser.parse_args()
    if args.config:
        parser.error("Core trainer đã sẵn sàng nhưng pipeline model/config chưa được nối ở task này.")
    if args.help_core:
        print("BaseTrainer, BaseMigratedModel, seed_everything đã sẵn sàng.")


if __name__ == "__main__":
    main()