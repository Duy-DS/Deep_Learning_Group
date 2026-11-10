"""Các hàm tiện ích trực quan hóa dữ liệu và biểu đồ khảo sát phân bố Gaussian."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

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

    def plot_factor_error(
        self,
        summary: pd.DataFrame,
        factor_column: str,
        metric: str,
        output_name: str,
        factor_label: str,
    ) -> Path:
        """Plot mean metric against an experiment factor with sample std bars."""
        mean_column = f"{metric}_mean"
        std_column = f"{metric}_std"
        required = {factor_column, mean_column, std_column}
        missing = required.difference(summary.columns)
        if missing:
            raise ValueError(f"Thiếu cột biểu đồ: {sorted(missing)}")
        output_path = self.figures_dir / output_name
        fig, axis = plt.subplots(figsize=(8.0, 5.0))
        axis.errorbar(
            summary[factor_column],
            summary[mean_column],
            yerr=summary[std_column],
            marker="o",
            capsize=4,
            label=f"Mean {metric.upper()} ± sample std",
        )
        axis.set_title(f"{metric.upper()} theo {factor_label}")
        axis.set_xlabel(factor_label)
        axis.set_ylabel(metric.upper())
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_train_validation_gap(
        self,
        summary: pd.DataFrame,
        factor_column: str,
        output_name: str,
        factor_label: str,
    ) -> Path:
        """Plot validation-minus-training MSE to diagnose capacity gaps."""
        required = {factor_column, "train_mse_mean", "val_mse_mean"}
        missing = required.difference(summary.columns)
        if missing:
            raise ValueError(f"Thiếu cột biểu đồ: {sorted(missing)}")
        output_path = self.figures_dir / output_name
        gap = summary["val_mse_mean"] - summary["train_mse_mean"]
        fig, axis = plt.subplots(figsize=(8.0, 5.0))
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.plot(summary[factor_column], gap, marker="o", label="Validation MSE - Train MSE")
        axis.set_title(f"Khoảng cách train-validation theo {factor_label}")
        axis.set_xlabel(factor_label)
        axis.set_ylabel("MSE gap")
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_depth_convergence_time(
        self, summary: pd.DataFrame, output_name: str
    ) -> Path:
        """Plot depth error, best epoch, and training time in labeled panels."""
        required = {
            "depth",
            "test_mse_mean",
            "test_mse_std",
            "training_time_seconds_mean",
            "training_time_seconds_std",
            "best_epoch_mean",
        }
        missing = required.difference(summary.columns)
        if missing:
            raise ValueError(f"Thiếu cột biểu đồ: {sorted(missing)}")
        output_path = self.figures_dir / output_name
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5))
        depths = summary["depth"]
        axes[0].errorbar(
            depths,
            summary["test_mse_mean"],
            yerr=summary["test_mse_std"],
            marker="o",
            capsize=4,
            label="Test MSE ± sample std",
        )
        axes[0].set_title("Test error")
        axes[0].set_xlabel("Hidden depth C")
        axes[0].set_ylabel("Test MSE")
        axes[0].legend()
        axes[1].plot(depths, summary["best_epoch_mean"], marker="o", label="Best epoch")
        axes[1].set_title("Convergence")
        axes[1].set_xlabel("Hidden depth C")
        axes[1].set_ylabel("Epoch")
        axes[1].legend()
        axes[2].errorbar(
            depths,
            summary["training_time_seconds_mean"],
            yerr=summary["training_time_seconds_std"],
            marker="o",
            capsize=4,
            label="Training time ± sample std",
        )
        axes[2].set_title("Training cost")
        axes[2].set_xlabel("Hidden depth C")
        axes[2].set_ylabel("Seconds")
        axes[2].legend()
        fig.suptitle("Độ sâu, hội tụ và thời gian huấn luyện")
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_surface_comparison(
        self,
        generator: GaussianDataGenerator,
        model: torch.nn.Module,
        device: torch.device | str = "cpu",
        grid_size: tuple[int, int] = (101, 101),
        output_name: str = "best_model_surface_comparison_3d.png",
    ) -> Path:
        """Vẽ đối sánh 2 bề mặt 3D cạnh nhau: Hàm Gaussian thực tế vs Bề mặt xấp xỉ của mô hình MLP tốt nhất.

        Tham số:
            generator: Bộ sinh dữ liệu Gaussian đã cấu hình.
            model: Mô hình PyTorch (FunctionApproximatorMLP) đã huấn luyện.
            device: Thiết bị tính toán (CPU hoặc CUDA/MPS).
            grid_size: Số lượng điểm lưới cho (x1, x2). Mặc định là (101, 101).
            output_name: Tên file ảnh lưu trong thư mục figures_dir.

        Trả về:
            Đường dẫn Path tới file ảnh PNG chất lượng cao (>= 300 DPI).
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
        grid_points = np.column_stack((mesh_x1.ravel(), mesh_x2.ravel()))

        # Mật độ thực tế từ công thức giải tích
        y_true = generator.compute_density(grid_points).reshape(mesh_x1.shape)

        # Mật độ dự đoán từ mô hình MLP
        dev = torch.device(device)
        model.eval()
        with torch.no_grad():
            tensor_input = torch.tensor(
                grid_points, dtype=torch.float32, device=dev
            )
            y_pred_tensor = model(tensor_input)
            y_pred = y_pred_tensor.cpu().numpy().reshape(mesh_x1.shape)

        fig = plt.figure(figsize=(15.0, 6.5))

        # Panel 1: Bề mặt Gaussian thực tế
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        surf1 = ax1.plot_surface(
            mesh_x1,
            mesh_x2,
            y_true,
            cmap="viridis",
            linewidth=0.0,
            antialiased=True,
            alpha=0.92,
        )
        ax1.set_title("Hàm Gaussian 2D Thực tế y", fontsize=12, fontweight="bold")
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax1.set_zlabel("Mật độ y")
        fig.colorbar(surf1, ax=ax1, shrink=0.55, aspect=10, label="Mật độ y")

        # Panel 2: Bề mặt xấp xỉ của mô hình MLP
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        surf2 = ax2.plot_surface(
            mesh_x1,
            mesh_x2,
            y_pred,
            cmap="plasma",
            linewidth=0.0,
            antialiased=True,
            alpha=0.92,
        )
        ax2.set_title(
            "Mặt cong Xấp xỉ ŷ (Mô hình MLP tốt nhất)",
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_xlabel("x1")
        ax2.set_ylabel("x2")
        ax2.set_zlabel("Mật độ ŷ")
        fig.colorbar(surf2, ax=ax2, shrink=0.55, aspect=10, label="Dự đoán ŷ")

        fig.suptitle(
            "Đối sánh Bề mặt Hàm Mật độ Xác suất 3D: Thực tế vs Mô hình MLP",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_parity_scatter(
        self,
        y_true: np.ndarray | list[float],
        y_pred: np.ndarray | list[float],
        output_name: str = "best_model_parity_plot.png",
        title: str = "Tương quan giữa Nhãn thực tế y và Dự đoán ŷ (Tập Test)",
    ) -> Path:
        """Vẽ đồ thị tương quan (Parity / Scatter Plot) giữa nhãn thực tế y và dự đoán ŷ.

        Tham số:
            y_true: Mảng 1 chiều chứa nhãn thực tế.
            y_pred: Mảng 1 chiều chứa giá trị dự đoán từ mô hình.
            output_name: Tên file ảnh xuất ra.
            title: Tiêu đề biểu đồ.

        Trả về:
            Đường dẫn Path tới file ảnh PNG.
        """
        y_t = np.asarray(y_true, dtype=np.float64).ravel()
        y_p = np.asarray(y_pred, dtype=np.float64).ravel()
        output_path = self.figures_dir / output_name

        mse = float(np.mean((y_t - y_p) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_t - y_p)))
        # Hệ số xác định R^2
        ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
        ss_res = float(np.sum((y_t - y_p) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

        fig, axis = plt.subplots(figsize=(7.5, 6.0))
        axis.scatter(
            y_t,
            y_p,
            alpha=0.75,
            color="#1f77b4",
            edgecolors="k",
            linewidths=0.5,
            s=35,
            label="Mẫu kiểm tra (Test samples)",
        )

        # Đường chéo lý tưởng y = x
        min_val = min(float(y_t.min()), float(y_p.min()))
        max_val = max(float(y_t.max()), float(y_p.max()))
        margin = 0.05 * (max_val - min_val) if max_val > min_val else 0.01
        ref_line = np.linspace(min_val - margin, max_val + margin, 100)
        axis.plot(
            ref_line,
            ref_line,
            color="#d62728",
            linestyle="--",
            linewidth=1.8,
            label="Đường chuẩn lý tưởng (y = ŷ)",
        )

        # Hộp thông tin chỉ số đo đạc
        metrics_text = f"MSE: {mse:.4e}\nRMSE: {rmse:.4e}\nMAE: {mae:.4e}\nR²: {r2:.4f}"
        axis.text(
            0.05,
            0.95,
            metrics_text,
            transform=axis.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5),
        )

        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_xlabel("Nhãn thực tế y", fontsize=11)
        axis.set_ylabel("Giá trị dự đoán ŷ", fontsize=11)
        axis.legend(loc="lower right")
        axis.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_residual_surface(
        self,
        generator: GaussianDataGenerator,
        model: torch.nn.Module,
        device: torch.device | str = "cpu",
        grid_size: tuple[int, int] = (101, 101),
        output_name: str = "best_model_residual_surface.png",
    ) -> Path:
        """Vẽ phân bố sai số dư |y - ŷ| gồm bề mặt 3D và bản đồ nhiệt contour 2D.

        Tham số:
            generator: Bộ sinh dữ liệu Gaussian đã cấu hình.
            model: Mô hình PyTorch đã huấn luyện.
            device: Thiết bị tính toán (CPU/CUDA/MPS).
            grid_size: Số điểm lưới cho x1, x2.
            output_name: Tên file ảnh lưu.

        Trả về:
            Đường dẫn Path tới file ảnh PNG chất lượng cao.
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
        grid_points = np.column_stack((mesh_x1.ravel(), mesh_x2.ravel()))

        y_true = generator.compute_density(grid_points).reshape(mesh_x1.shape)

        dev = torch.device(device)
        model.eval()
        with torch.no_grad():
            tensor_input = torch.tensor(
                grid_points, dtype=torch.float32, device=dev
            )
            y_pred = model(tensor_input).cpu().numpy().reshape(mesh_x1.shape)

        abs_error = np.abs(y_true - y_pred)

        fig = plt.figure(figsize=(15.0, 6.0))

        # Panel 1: Bề mặt sai số tuyệt đối 3D
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        surf = ax1.plot_surface(
            mesh_x1,
            mesh_x2,
            abs_error,
            cmap="inferno",
            linewidth=0.0,
            antialiased=True,
            alpha=0.92,
        )
        ax1.set_title(
            "Bề mặt Sai số Tuyệt đối |y - ŷ| (3D Residual Surface)",
            fontsize=11,
            fontweight="bold",
        )
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax1.set_zlabel("|y - ŷ|")
        fig.colorbar(surf, ax=ax1, shrink=0.55, aspect=10, label="Sai số tuyệt đối")

        # Panel 2: Bản đồ nhiệt 2D Contour
        ax2 = fig.add_subplot(1, 2, 2)
        contour = ax2.contourf(
            mesh_x1,
            mesh_x2,
            abs_error,
            levels=20,
            cmap="inferno",
        )
        ax2.contour(
            mesh_x1,
            mesh_x2,
            abs_error,
            levels=10,
            colors="white",
            linewidths=0.5,
        )
        ax2.set_title(
            "Bản đồ Nhiệt Sai số Tuyệt đối theo Không gian Đầu vào (x1, x2)",
            fontsize=11,
            fontweight="bold",
        )
        ax2.set_xlabel("x1")
        ax2.set_ylabel("x2")
        fig.colorbar(contour, ax=ax2, shrink=0.8, label="Sai số |y - ŷ|")

        fig.suptitle(
            "Khảo sát Phân bố Sai số Xấp xỉ của Mô hình MLP Tối ưu",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_overall_comparison(
        self,
        summary_overall: pd.DataFrame,
        output_name: str = "overall_comparison_metrics.png",
    ) -> Path:
        """Vẽ biểu đồ đối sánh tổng hợp giữa các kịch bản thực nghiệm A, B, C và Baseline.

        Tham số:
            summary_overall: DataFrame chứa các cột: 'config_name', 'test_mse_mean', 'test_rmse_mean', 'test_mae_mean'.
            output_name: Tên file ảnh lưu.

        Trả về:
            Đường dẫn Path tới file ảnh PNG.
        """
        output_path = self.figures_dir / output_name
        required = {
            "config_name",
            "test_mse_mean",
            "test_rmse_mean",
            "test_mae_mean",
        }
        missing = required.difference(summary_overall.columns)
        if missing:
            raise ValueError(f"Thiếu cột biểu đồ tổng hợp: {sorted(missing)}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.5))
        configs = summary_overall["config_name"]
        x = np.arange(len(configs))

        # Subplot 1: Test MSE
        bars1 = ax1.bar(
            x,
            summary_overall["test_mse_mean"],
            color="#2ca02c",
            width=0.45,
            alpha=0.85,
            edgecolor="black",
        )
        ax1.set_title(
            "So sánh Sai số Test MSE giữa các Cấu hình Tiêu biểu",
            fontsize=11,
            fontweight="bold",
        )
        ax1.set_xlabel("Cấu hình mô hình")
        ax1.set_ylabel("Test MSE (thấp hơn là tốt hơn)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, rotation=15, ha="right")
        ax1.grid(axis="y", linestyle=":", alpha=0.6)
        for bar in bars1:
            yval = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval * 1.02,
                f"{yval:.2e}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Subplot 2: Test RMSE & MAE
        width = 0.35
        ax2.bar(
            x - width / 2,
            summary_overall["test_rmse_mean"],
            width,
            label="Test RMSE",
            color="#1f77b4",
            edgecolor="black",
        )
        ax2.bar(
            x + width / 2,
            summary_overall["test_mae_mean"],
            width,
            label="Test MAE",
            color="#ff7f0e",
            edgecolor="black",
        )
        ax2.set_title(
            "So sánh Test RMSE và Test MAE",
            fontsize=11,
            fontweight="bold",
        )
        ax2.set_xlabel("Cấu hình mô hình")
        ax2.set_ylabel("Sai số")
        ax2.set_xticks(x)
        ax2.set_xticklabels(configs, rotation=15, ha="right")
        ax2.legend()
        ax2.grid(axis="y", linestyle=":", alpha=0.6)

        fig.suptitle(
            "Tổng hợp Đối sánh Hiệu năng Thực nghiệm (Baseline vs Best A/B/C)",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

