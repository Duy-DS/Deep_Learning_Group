"""Khởi tạo các biểu đồ Thực nghiệm C từ bảng dữ liệu tổng hợp đã lưu trữ."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualizer import Visualizer


def main() -> None:
    """Đọc bảng CSV tổng hợp Thực nghiệm C và xuất các biểu đồ theo yêu cầu."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    summary = pd.read_csv(args.output_root / "tables/summary_exp_C.csv")
    visualizer = Visualizer(args.output_root / "figures")
    visualizer.plot_depth_convergence_time(
        summary, "exp_C_depth_convergence_time.png"
    )
    visualizer.plot_factor_error(
        summary,
        "depth",
        "test_mse",
        "exp_C_error_vs_depth.png",
        "Hidden depth C",
    )
    print("Generated Experiment C figures")


if __name__ == "__main__":
    main()
