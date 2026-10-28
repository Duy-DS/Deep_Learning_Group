"""Các hàm tiện ích trực quan hóa dữ liệu và biểu đồ khảo sát phân bố Gaussian."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_generator import GaussianDataGenerator


class Visualizer:
    """Tạo biểu đồ và bảng kết quả chất lượng cao cho báo cáo (T0203).

    Tham số:
        figures_dir: Thư mục dùng để lưu trữ các đồ thị sinh ra.
        dpi: Độ phân giải của đồ thị xuất ra (quy định dự án tối thiểu 300).
        figure_format: Định dạng ảnh xuất ra của Matplotlib (ví dụ: ``png``).

    Ngoại lệ:
        ValueError: Nếu ``dpi`` nhỏ hơn 300.
    """

    def __init__(
        self,
        figures_dir: Path,
        dpi: int = 300,
        figure_format: str = "png",
    ) -> None:
        """Khởi tạo visualizer và đảm bảo thư mục lưu trữ tồn tại."""
        if dpi < 300:
            raise ValueError("dpi phải đạt tối thiểu 300.")
        self.figures_dir = figures_dir
        self.dpi = dpi
        self.figure_format = figure_format
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def plot_data_distribution(
        self,
        splits: Mapping[str, pd.DataFrame],
        output_name: str = "data_distribution_3d.png",
    ) -> Path:
        """Vẽ biểu đồ phân tán 3D thể hiện phân bố dữ liệu train/val/test đã sinh.

        Tham số:
            splits: Ánh xạ tên tập dữ liệu sang DataFrame chứa các cột ``x1``, ``x2``, và ``y``.
            output_name: Tên file lưu trong ``figures_dir``.

        Trả về:
            Đường dẫn Path tới file ảnh PNG đã sinh.
        """
        output_path = self.figures_dir / output_name
        fig = plt.figure(figsize=(8.0, 6.0))
        axis = fig.add_subplot(111, projection="3d")
        colors = {
            "train_full": "#1f77b4",
            "val": "#ff7f0e",
            "test": "#2ca02c",
        }
        labels = {
            "train_full": "Train full",
            "val": "Validation",
            "test": "Test",
        }
        for split_name, frame in splits.items():
            axis.scatter(
                frame["x1"],
                frame["x2"],
                frame["y"],
                s=9,
                alpha=0.58,
                color=colors.get(split_name),
                label=labels.get(split_name, split_name),
                depthshade=True,
            )
        axis.set_title("Phân bố dữ liệu Gaussian 2D đã sinh")
        axis.set_xlabel("x1")
        axis.set_ylabel("x2")
        axis.set_zlabel("Mật độ xác suất y")
        axis.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_true_surface(
        self,
        generator: GaussianDataGenerator,
        grid_size: tuple[int, int] = (101, 101),
        output_name: str = "true_gaussian_surface_3d.png",
    ) -> Path:
        """Vẽ bề mặt 3D thực tế của hàm mật độ xác suất Gaussian trên toàn bộ miền xác định.

        Tham số:
            generator: Bộ sinh dữ liệu Gaussian đã cấu hình.
            grid_size: Số lượng điểm lưới cho ``x1`` và ``x2``.
            output_name: Tên file lưu trong ``figures_dir``.

        Trả về:
            Đường dẫn Path tới file ảnh PNG đã sinh.
        """
        output_path = self.figures_dir / output_name
        x1 = np.linspace(
            generator.x1_range[0],
            generator.x1_range[1],
            grid_size[0],
        )
        x2 = np.linspace(
            generator.x2_range[0],
            generator.x2_range[1],
            grid_size[1],
        )
        mesh_x1, mesh_x2 = np.meshgrid(x1, x2)
        grid = np.column_stack((mesh_x1.ravel(), mesh_x2.ravel()))
        density = generator.compute_density(grid).reshape(mesh_x1.shape)

        fig = plt.figure(figsize=(8.0, 6.0))
        axis = fig.add_subplot(111, projection="3d")
        surface = axis.plot_surface(
            mesh_x1,
            mesh_x2,
            density,
            cmap="viridis",
            linewidth=0.0,
            antialiased=True,
            alpha=0.92,
        )
        axis.set_title("Bề mặt hàm mật độ xác suất Gaussian 2D thực tế")
        axis.set_xlabel("x1")
        axis.set_ylabel("x2")
        axis.set_zlabel("Mật độ xác suất y")
        fig.colorbar(surface, ax=axis, shrink=0.62, label="Mật độ xác suất y")
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def build_summary_table(
        self,
        splits: Mapping[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Xây dựng bảng thống kê mô tả cho các tọa độ đầu vào và nhãn y.

        Tham số:
            splits: Ánh xạ tên tập dữ liệu sang DataFrame chứa các cột ``x1``, ``x2``, và ``y``.

        Trả về:
            DataFrame thống kê với các cột: split, feature, count, mean, std, min, và max.
        """
        rows: list[dict[str, float | int | str]] = []
        for split_name, frame in splits.items():
            for column in ["x1", "x2", "y"]:
                series = frame[column]
                rows.append(
                    {
                        "split": split_name,
                        "feature": column,
                        "count": int(series.count()),
                        "mean": float(series.mean()),
                        "std": float(series.std(ddof=1)),
                        "min": float(series.min()),
                        "max": float(series.max()),
                    }
                )
        return pd.DataFrame(rows)
