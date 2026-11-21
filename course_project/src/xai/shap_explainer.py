"""Module tính toán và phân tích SHAP Global Feature Importance cho mô hình chuỗi thời gian.

Tuân thủ quy chuẩn thiết kế OOP 100%, Docstring Tiếng Việt và PEP 8.
Mã nhiệm vụ: BPM25_T0401 thuộc gói công việc WP04 (XAI Integration).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import yaml


class _HorizonModel(nn.Module):
    """Bọc mô hình PyTorch để trích xuất dự báo tại một bước horizon cụ thể."""

    def __init__(self, model: nn.Module, target_horizon_idx: int = 0) -> None:
        """Khởi tạo wrapper cho mô hình theo chỉ số horizon.

        Args:
            model: Mô hình PyTorch nhận tensor [B, L, D] và trả dự báo đa bước.
            target_horizon_idx: Chỉ số bước dự báo cần giải thích (0 đến H-1).
        """
        super().__init__()
        self.model = model
        self.target_horizon_idx = target_horizon_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Chuyển tiếp qua mô hình và trích xuất đúng horizon chỉ định.

        Args:
            x: Tensor đầu vào có kích thước [B, L, D].

        Returns:
            Tensor dự báo tại horizon đích có kích thước [B, 1].
        """
        preds = self.model(x)
        if preds.ndim == 2:
            return preds[:, self.target_horizon_idx : self.target_horizon_idx + 1]
        elif preds.ndim == 3:
            return preds[:, self.target_horizon_idx, :]
        else:
            raise ValueError(f"Dự báo mô hình có số chiều không hợp lệ: {preds.shape}")


class SHAPExplainer:
    """Bộ giải thích SHAP toàn cục và cục bộ cho mô hình chuỗi thời gian."""

    def __init__(
        self,
        model: nn.Module,
        background_data: torch.Tensor,
        feature_names: List[str],
        device: str = "cpu",
    ) -> None:
        """Khởi tạo bộ giải thích SHAP với mô hình và dữ liệu nền.

        Args:
            model: Mô hình PyTorch đã được huấn luyện.
            background_data: Tensor dữ liệu nền tham chiếu [N, L, D] lấy từ tập Train.
            feature_names: Danh sách tên của D đặc trưng đầu vào.
            device: Thiết bị thực thi ('cpu' hoặc 'cuda').
        """
        self.device = device
        self.model = model.to(device).eval()
        self.background_data = background_data.to(device)
        self.feature_names = feature_names

    def compute_shap_values(
        self, test_samples: torch.Tensor, target_horizon_idx: int = 0
    ) -> np.ndarray:
        """Tính toán giá trị SHAP cho tập mẫu kiểm tra.

        Args:
            test_samples: Tensor dữ liệu kiểm tra [N, L, D].
            target_horizon_idx: Bước thời gian dự báo cần giải thích.

        Returns:
            Mảng numpy chứa giá trị SHAP có kích thước [N, L, D].
        """
        wrapped_model = _HorizonModel(self.model, target_horizon_idx)
        explainer = shap.GradientExplainer(wrapped_model, self.background_data)

        samples = test_samples.to(self.device)
        shap_values = explainer.shap_values(samples)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
            shap_values = np.squeeze(shap_values, axis=-1)

        return shap_values

    def get_global_feature_importance(
        self, shap_values: np.ndarray
    ) -> Dict[str, float]:
        """Tính mean(|SHAP|) cho từng biến và sắp xếp theo thứ tự giảm dần.

        Args:
            shap_values: Mảng giá trị SHAP có kích thước [N, L, D].

        Returns:
            Từ điển ánh xạ từ tên biến đến giá trị độ quan trọng, sắp xếp giảm dần.
        """
        if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
            shap_values = np.squeeze(shap_values, axis=-1)

        importance = np.mean(np.abs(shap_values), axis=(0, 1))

        feat_dict = {
            self.feature_names[i]: float(importance[i])
            for i in range(len(self.feature_names))
        }

        return dict(sorted(feat_dict.items(), key=lambda item: item[1], reverse=True))

    def export_for_dashboard(
        self, shap_values: np.ndarray, top_k: int = 10
    ) -> Dict[str, Any]:
        """Xuất dữ liệu chuẩn định dạng JSON phục vụ hiển thị trên Dashboard (WP05).

        Args:
            shap_values: Mảng SHAP [N, L, D].
            top_k: Số lượng đặc trưng quan trọng nhất cần lấy.

        Returns:
            Từ điển chứa bảng xếp hạng và các thông tin trực quan hóa.
        """
        importance_dict = self.get_global_feature_importance(shap_values)
        top_features = list(importance_dict.items())[:top_k]

        return {
            "num_samples": int(shap_values.shape[0]),
            "lookback": int(shap_values.shape[1]),
            "num_features": int(shap_values.shape[2]),
            "global_feature_importance": importance_dict,
            "top_features": dict(top_features),
        }

    def plot_feature_importance(
        self,
        shap_values: np.ndarray,
        save_path: Optional[str] = None,
        top_k: int = 15,
        title: str = "SHAP Global Feature Importance (Beijing PM2.5)",
    ) -> plt.Figure:
        """Vẽ biểu đồ thanh ngang (Horizontal Bar Chart) độ quan trọng toàn cục.

        Args:
            shap_values: Mảng SHAP có kích thước [N, L, D].
            save_path: Đường dẫn file PNG để lưu biểu đồ (nếu có).
            top_k: Số lượng biến dẫn đầu cần vẽ.
            title: Tiêu đề biểu đồ.

        Returns:
            Đối tượng Figure của Matplotlib.
        """
        importance_dict = self.get_global_feature_importance(shap_values)
        sorted_feats = list(importance_dict.keys())[:top_k]
        sorted_scores = list(importance_dict.values())[:top_k]

        sorted_feats.reverse()
        sorted_scores.reverse()

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        bars = ax.barh(sorted_feats, sorted_scores, color="#2563eb", edgecolor="#1e40af", alpha=0.85)

        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + (max(sorted_scores) * 0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}",
                va="center",
                ha="left",
                fontsize=9,
                color="#1e293b",
            )

        ax.set_xlabel("Độ quan trọng trung bình: mean(|SHAP value|)", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"[INFO] Đã lưu biểu đồ Xếp hạng tại: {save_path}")

        return fig

    def plot_temporal_profile(
        self,
        shap_values: np.ndarray,
        save_path: Optional[str] = None,
        top_k: int = 6,
        title: str = "Độ ảnh hưởng của 96 giờ quá khứ theo từng biến (Temporal SHAP Profile)",
    ) -> plt.Figure:
        """Vẽ biểu đồ đường thể hiện mức độ quan trọng qua 96 bước thời gian quan sát.

        Args:
            shap_values: Mảng SHAP có kích thước [N, L=96, D].
            save_path: Đường dẫn file PNG để lưu biểu đồ.
            top_k: Số biến hàng đầu cần vẽ đường diễn biến thời gian.
            title: Tiêu đề biểu đồ.

        Returns:
            Đối tượng Figure của Matplotlib.
        """
        if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
            shap_values = np.squeeze(shap_values, axis=-1)

        # Tính mean(|SHAP|) theo từng bước thời gian: [L, D]
        temporal_imp = np.mean(np.abs(shap_values), axis=0)
        lookback = temporal_imp.shape[0]
        hours = np.arange(-lookback, 0)

        importance_dict = self.get_global_feature_importance(shap_values)
        top_feats = list(importance_dict.keys())[:top_k]

        fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
        colors = ["#dc2626", "#2563eb", "#16a34a", "#d97706", "#9333ea", "#0891b2"]

        for idx, feat in enumerate(top_feats):
            feat_idx = self.feature_names.index(feat)
            curve = temporal_imp[:, feat_idx]
            color = colors[idx % len(colors)]
            ax.plot(hours, curve, label=feat, color=color, linewidth=2, alpha=0.9)

        ax.set_xlabel("Thời gian quá khứ (từ t-96 đến t-1 giờ)", fontsize=11, fontweight="bold")
        ax.set_ylabel("mean(|SHAP value|)", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.axvline(x=-24, color="gray", linestyle=":", label="Mốc 24h trước")
        ax.axvline(x=-12, color="orange", linestyle=":", label="Mốc 12h trước")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", frameon=True)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"[INFO] Đã lưu biểu đồ Temporal Profile tại: {save_path}")

        return fig

    def plot_domain_groups(
        self,
        shap_values: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Tỷ trọng đóng góp theo nhóm đặc trưng (Domain Groups Breakdown)",
    ) -> plt.Figure:
        """Vẽ biểu đồ tròn Donut phân rã đóng góp theo 4 nhóm biến khoa học.

        Args:
            shap_values: Mảng SHAP có kích thước [N, L, D].
            save_path: Đường dẫn file PNG để lưu biểu đồ.
            title: Tiêu đề biểu đồ.

        Returns:
            Đối tượng Figure của Matplotlib.
        """
        importance_dict = self.get_global_feature_importance(shap_values)

        groups = {
            "Bụi tự hồi quy\n(PM2.5, PM10)": 0.0,
            "Khí phát thải\n(SO2, NO2, CO, O3)": 0.0,
            "Khí tượng & Gió\n(TEMP, DEW, PRES, WD...)": 0.0,
            "Chu kỳ thời gian\n(Giờ & Tháng sin/cos)": 0.0,
        }

        for feat, score in importance_dict.items():
            feat_lower = feat.lower()
            if feat_lower in ["pm2.5", "pm10"]:
                groups["Bụi tự hồi quy\n(PM2.5, PM10)"] += score
            elif feat_lower in ["so2", "no2", "co", "o3"]:
                groups["Khí phát thải\n(SO2, NO2, CO, O3)"] += score
            elif any(k in feat_lower for k in ["hour_", "month_"]):
                groups["Chu kỳ thời gian\n(Giờ & Tháng sin/cos)"] += score
            else:
                groups["Khí tượng & Gió\n(TEMP, DEW, PRES, WD...)"] += score

        labels = list(groups.keys())
        values = list(groups.values())
        colors = ["#ef4444", "#f59e0b", "#3b82f6", "#10b981"]

        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            pctdistance=0.75,
            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        )

        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(10)
            autotext.set_fontweight("bold")

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"[INFO] Đã lưu biểu đồ Nhóm đặc trưng tại: {save_path}")

        return fig

    def plot_beeswarm_summary(
        self,
        shap_values: np.ndarray,
        test_samples: torch.Tensor,
        save_path: Optional[str] = None,
        top_k: int = 12,
    ) -> None:
        """Vẽ biểu đồ Beeswarm thể hiện chiều hướng tác động (dương/âm) của các biến.

        Args:
            shap_values: Mảng SHAP có kích thước [N, L, D].
            test_samples: Tensor dữ liệu kiểm tra [N, L, D].
            save_path: Đường dẫn file PNG để lưu biểu đồ.
            top_k: Số lượng biến cần hiển thị.
        """
        if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
            shap_values = np.squeeze(shap_values, axis=-1)

        # Rút gọn qua chiều thời gian để đưa vào shap.summary_plot [N, D]
        mean_shap = np.mean(shap_values, axis=1)
        mean_x = np.mean(test_samples.cpu().numpy(), axis=1)

        plt.figure(figsize=(10, 6), dpi=150)
        shap.summary_plot(
            mean_shap,
            mean_x,
            feature_names=self.feature_names,
            max_display=top_k,
            show=False,
        )
        plt.title("SHAP Beeswarm: Chiều hướng ảnh hưởng của các biến hàng đầu", fontsize=12, fontweight="bold", pad=10)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"[INFO] Đã lưu biểu đồ Beeswarm tại: {save_path}")


def load_model_and_checkpoint(
    checkpoint_path: str, device: str = "cpu"
) -> Tuple[nn.Module, List[str], Dict[str, Any]]:
    """Nạp trọng số mô hình từ checkpoint và tự động áp dụng ResidualAnchorWrapper.

    Args:
        checkpoint_path: Đường dẫn tệp .pt của checkpoint.
        device: Thiết bị thực thi ('cpu' hoặc 'cuda').

    Returns:
        Tuple gồm (mô hình đã load weights, danh sách feature_cols, checkpoint dict).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt.get("model_name", "patchtst").lower()
    lookback = ckpt.get("lookback", 96)
    horizon = ckpt.get("horizon", 24)
    in_features = ckpt.get("in_features", 31)
    feature_cols = ckpt.get("feature_cols", [])

    if "patchtst" in model_name:
        from src.models.patchtst import PatchTSTModel

        base_model = PatchTSTModel(
            lookback=lookback,
            horizon=horizon,
            in_features=in_features,
            dropout=ckpt.get("dropout", 0.25),
        )
    elif "lstm" in model_name:
        from src.models.lstm import LSTMModel

        base_model = LSTMModel(
            lookback=lookback,
            horizon=horizon,
            in_features=in_features,
            dropout=ckpt.get("dropout", 0.2),
        )
    elif "dlinear" in model_name:
        from src.models.dlinear import DLinearModel

        base_model = DLinearModel(
            lookback=lookback,
            horizon=horizon,
            in_features=in_features,
        )
    elif "informer" in model_name:
        from src.models.informer import InformerModel

        base_model = InformerModel(
            lookback=lookback,
            horizon=horizon,
            in_features=in_features,
        )
    else:
        raise ValueError(f"Không hỗ trợ mô hình: {model_name}")

    if "res_scale" in ckpt.get("model_state_dict", {}):
        from training_loop import ResidualAnchorWrapper

        model = ResidualAnchorWrapper(
            base_model, horizon=horizon, in_features=in_features
        )
    else:
        model = base_model

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, feature_cols, ckpt


def sample_windows_from_csv(
    csv_path: str,
    feature_cols: List[str],
    target_col: str = "pm2.5",
    lookback: int = 96,
    horizon: int = 24,
    num_samples: int = 100,
    stride: int = 24,
    seed: int = 42,
) -> torch.Tensor:
    """Tạo cửa sổ trượt từ CSV và lấy mẫu ngẫu nhiên có cố định random seed.

    Args:
        csv_path: Đường dẫn tệp CSV tiền xử lý.
        feature_cols: Danh sách các cột đặc trưng cần lấy.
        target_col: Cột mục tiêu dự báo.
        lookback: Chiều dài chuỗi quan sát quá khứ L.
        horizon: Bước thời gian dự báo H.
        num_samples: Số lượng mẫu cần rút trích.
        stride: Bước trượt cửa sổ.
        seed: Random seed cố định.

    Returns:
        Tensor dữ liệu kích thước [num_samples, lookback, len(feature_cols)].
    """
    from src.data_pipeline.dataset import make_sliding_windows

    df = pd.read_csv(csv_path)
    X, _ = make_sliding_windows(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        lookback=lookback,
        horizon=horizon,
        stride=stride,
    )

    if len(X) == 0:
        raise ValueError(f"Không tạo được cửa sổ trượt nào từ file: {csv_path}")

    if len(X) > num_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(X), size=num_samples, replace=False)
        X = X[indices]

    return torch.from_numpy(X).float()


def run_t0401_pipeline(
    config_path: str = "configs/default.yaml",
    num_bg: int = 100,
    num_eval: int = 100,
    checkpoint_path: Optional[str] = None,
    output_dir: str = "outputs/reports",
) -> Dict[str, Any]:
    """Chạy toàn bộ pipeline nghiệm thu T0401 trên dữ liệu và checkpoint thật."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = (
        checkpoint_path
        or "outputs/checkpoints/global/patchtst_multivariate_H24_seed42_pretrained.pt"
    )

    print(f"[INFO] 1. Nạp mô hình từ checkpoint: {ckpt_path} (Thiết bị: {device})")
    model, feature_cols, ckpt = load_model_and_checkpoint(ckpt_path, device=device)

    data_dir = cfg.get("paths", {}).get("data_dir", "data/preprocess")
    train_path = f"{data_dir}/train.csv"
    test_path = f"{data_dir}/test.csv"

    lookback = ckpt.get("lookback", 96)
    horizon = ckpt.get("horizon", 24)

    print(f"[INFO] 2. Lấy mẫu dữ liệu nền ({num_bg} mẫu) từ: {train_path}")
    bg_tensor = sample_windows_from_csv(
        train_path,
        feature_cols=feature_cols,
        lookback=lookback,
        horizon=horizon,
        num_samples=num_bg,
        stride=48,
        seed=42,
    )

    print(f"[INFO] 3. Lấy mẫu dữ liệu đánh giá ({num_eval} mẫu) từ: {test_path}")
    test_tensor = sample_windows_from_csv(
        test_path,
        feature_cols=feature_cols,
        lookback=lookback,
        horizon=horizon,
        num_samples=num_eval,
        stride=24,
        seed=42,
    )

    print("[INFO] 4. Khởi tạo SHAPExplainer và tính toán giá trị SHAP...")
    explainer = SHAPExplainer(model, bg_tensor, feature_cols, device=device)
    shap_vals = explainer.compute_shap_values(test_tensor, target_horizon_idx=0)
    print(f"[SUCCESS] Tính toán SHAP thành công, kích thước: {shap_vals.shape}")

    print("[INFO] 5. Xếp hạng độ quan trọng đặc trưng toàn cục mean(|SHAP|):")
    importance_scores = explainer.get_global_feature_importance(shap_vals)
    for rank, (feat, score) in enumerate(list(importance_scores.items())[:10], start=1):
        print(f"   Top {rank:2d}: {feat:<12} | mean(|SHAP|) = {score:.6f}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_file = out_path / "shap_global_importance.json"
    dashboard_data = explainer.export_for_dashboard(shap_vals, top_k=15)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Đã lưu kết quả JSON tại: {json_file}")

    print("[INFO] 6. Xuất bộ 4 biểu đồ trực quan hóa chi tiết:")
    # 1. Bar chart xếp hạng
    chart_1 = str(out_path / "shap_global_importance.png")
    explainer.plot_feature_importance(shap_vals, save_path=chart_1, top_k=15)

    # 2. Line chart theo 96 giờ quá khứ
    chart_2 = str(out_path / "shap_temporal_profile.png")
    explainer.plot_temporal_profile(shap_vals, save_path=chart_2, top_k=6)

    # 3. Donut chart phân rã theo nhóm biến
    chart_3 = str(out_path / "shap_domain_groups.png")
    explainer.plot_domain_groups(shap_vals, save_path=chart_3)

    # 4. Beeswarm summary plot
    chart_4 = str(out_path / "shap_beeswarm_summary.png")
    explainer.plot_beeswarm_summary(shap_vals, test_tensor, save_path=chart_4, top_k=12)

    return dashboard_data


def _run_smoke_test() -> None:
    """Chạy kiểm thử nhanh trên dummy data để đảm bảo tính sẵn sàng của module."""
    torch.manual_seed(42)
    dummy_names = ["PM2.5", "TEMP", "DEW", "RAIN", "WSPM"]

    class DummyModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.mean(dim=1, keepdim=True).repeat(1, 24, 1)[:, :, :1]

    model = DummyModel()
    bg_data = torch.randn(20, 96, 5)
    test_data = torch.randn(5, 96, 5)

    explainer = SHAPExplainer(model, bg_data, dummy_names)
    vals = explainer.compute_shap_values(test_data, target_horizon_idx=0)
    scores = explainer.get_global_feature_importance(vals)

    assert vals.shape == (5, 96, 5), f"Shape không hợp lệ: {vals.shape}"
    assert len(scores) == 5, f"Số biến không khớp: {len(scores)}"
    print("[SUCCESS] Smoke test PASS 100%! Tensor shape:", vals.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy SHAP Global Feature Importance (BPM25_T0401)")
    parser.add_argument("--smoke-test", action="store_true", help="Chạy smoke test trên dummy data")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Đường dẫn file cấu hình")
    parser.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn checkpoint cần giải thích")
    parser.add_argument("--num-bg", type=int, default=50, help="Số mẫu nền (background)")
    parser.add_argument("--num-eval", type=int, default=50, help="Số mẫu đánh giá (eval)")
    parser.add_argument("--output-dir", type=str, default="outputs/reports", help="Thư mục xuất báo cáo")

    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
    else:
        run_t0401_pipeline(
            config_path=args.config,
            num_bg=args.num_bg,
            num_eval=args.num_eval,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
        )
