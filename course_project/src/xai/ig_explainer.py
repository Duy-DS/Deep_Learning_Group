"""Module tính toán và phân tích Integrated Gradients Local Attribution cho chuỗi thời gian.

Tuân thủ quy chuẩn thiết kế OOP 100%, Docstring Tiếng Việt và PEP 8.
Mã nhiệm vụ: BPM25_T0402 thuộc gói công việc WP04 (XAI Integration).
Hỗ trợ giải thích cục bộ tại các đỉnh ô nhiễm (Smog Spikes) để trả lời RQ3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from captum.attr import IntegratedGradients


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
        return preds


class IGExplainer:
    """Bộ giải thích Integrated Gradients cho mô hình dự báo chuỗi thời gian."""

    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str],
        device: str = "cpu",
    ) -> None:
        """Khởi tạo bộ giải thích Integrated Gradients với mô hình PyTorch.

        Args:
            model: Mô hình PyTorch đã được huấn luyện.
            feature_names: Danh sách tên của D đặc trưng đầu vào.
            device: Thiết bị thực thi ('cpu' hoặc 'cuda').
        """
        self.device = device
        self.model = model.to(device).eval()
        self.feature_names = feature_names

    def attribute(
        self,
        inputs: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        target_horizon_idx: int = 0,
        n_steps: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tính toán ma trận attribution của Integrated Gradients và sai số Riemann delta.

        Args:
            inputs: Tensor dữ liệu kiểm tra [N, L, D].
            baseline: Tensor điểm mốc tham chiếu [N, L, D]. Nếu None, dùng tensor zeros.
            target_horizon_idx: Bước thời gian dự báo cần giải thích (mặc định 0).
            n_steps: Số bước xấp xỉ tích phân Riemann (mặc định 50).

        Returns:
            Tuple (attributions [N, L, D], convergence_deltas [N]).
        """
        wrapped_model = _HorizonModel(self.model, target_horizon_idx)
        ig = IntegratedGradients(wrapped_model)

        inputs_tensor = inputs.to(self.device).requires_grad_(True)
        if baseline is None:
            baseline_tensor = torch.zeros_like(inputs_tensor)
        else:
            baseline_tensor = baseline.to(self.device)

        # Tính toán attribution và sai số hội tụ delta
        attributions, delta = ig.attribute(
            inputs_tensor,
            baselines=baseline_tensor,
            target=0,
            n_steps=n_steps,
            return_convergence_delta=True,
        )

        attrs_np = attributions.detach().cpu().numpy()
        delta_np = delta.detach().cpu().numpy()

        return attrs_np, delta_np

    def get_feature_attributions(
        self, attributions: np.ndarray
    ) -> Dict[str, float]:
        """Tính độ lớn trung bình mean(|attribution|) của từng biến qua toàn bộ mẫu và thời gian.

        Args:
            attributions: Mảng attribution có kích thước [N, L, D].

        Returns:
            Từ điển ánh xạ tên biến đến độ quan trọng trung bình, sắp xếp giảm dần.
        """
        importance = np.mean(np.abs(attributions), axis=(0, 1))
        feat_dict = {
            self.feature_names[i]: float(importance[i])
            for i in range(len(self.feature_names))
        }
        return dict(sorted(feat_dict.items(), key=lambda item: item[1], reverse=True))

    def export_for_dashboard(
        self, attributions: np.ndarray, deltas: np.ndarray, top_k: int = 10
    ) -> Dict[str, Any]:
        """Xuất kết quả định dạng JSON phục vụ tích hợp giao diện Streamlit (WP05).

        Args:
            attributions: Mảng attribution [N, L, D].
            deltas: Mảng sai số delta [N].
            top_k: Số đặc trưng quan trọng nhất cần lấy.

        Returns:
            Từ điển JSON-compatible tóm tắt attribution.
        """
        importance = self.get_feature_attributions(attributions)
        return {
            "num_samples": int(attributions.shape[0]),
            "lookback": int(attributions.shape[1]),
            "num_features": int(attributions.shape[2]),
            "mean_convergence_delta": float(np.mean(np.abs(deltas))),
            "top_attributions": dict(list(importance.items())[:top_k]),
            "all_attributions": importance,
        }

    def plot_spike_heatmap(
        self,
        attributions: np.ndarray,
        sample_idx: int = 0,
        top_k: int = 15,
        save_path: Optional[str] = None,
        title: str = "Ma trận nhiệt Integrated Gradients tại đỉnh ô nhiễm (Smog Spike)",
    ) -> plt.Figure:
        """Vẽ biểu đồ nhiệt Heatmap [Lookback 96h x Top Features] cho một mẫu đỉnh ô nhiễm.

        Args:
            attributions: Mảng attribution [N, L=96, D].
            sample_idx: Chỉ số mẫu đỉnh ô nhiễm cần vẽ.
            top_k: Số lượng biến dẫn đầu cần đưa vào biểu đồ nhiệt.
            save_path: Đường dẫn lưu ảnh PNG.
            title: Tiêu đề biểu đồ.

        Returns:
            Figure của Matplotlib.
        """
        sample_attr = attributions[sample_idx]  # [96, D]
        feat_importance = np.mean(np.abs(sample_attr), axis=0)
        top_indices = np.argsort(-feat_importance)[:top_k]

        selected_attr = sample_attr[:, top_indices].T  # [top_k, 96]
        selected_names = [self.feature_names[i] for i in top_indices]

        hours = np.arange(-selected_attr.shape[1], 0)

        fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
        vmax = np.percentile(np.abs(selected_attr), 98)
        sns.heatmap(
            selected_attr,
            cmap="RdBu_r",
            center=0,
            vmax=vmax,
            vmin=-vmax,
            yticklabels=selected_names,
            xticklabels=12,
            cbar_kws={"label": "Attribution Score (Đỏ: Thúc đẩy PM2.5 / Xanh: Kìm hãm)"},
            ax=ax,
        )

        ax.set_xticks(np.arange(0, 96, 12))
        ax.set_xticklabels([f"t{h}h" for h in hours[::12]], rotation=0)
        ax.set_xlabel("Thời gian quá khứ (từ t-96 đến t-1 giờ)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Đặc trưng đầu vào", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"[INFO] Đã lưu Heatmap tại: {save_path}")

        return fig

    def plot_spike_vs_normal_comparison(
        self,
        spike_attrs: np.ndarray,
        normal_attrs: np.ndarray,
        top_k: int = 10,
        save_path: Optional[str] = None,
        title: str = "So sánh đóng góp đặc trưng: Đỉnh ô nhiễm (Smog Spikes) vs Ngày bình thường",
    ) -> plt.Figure:
        """Vẽ biểu đồ cột so sánh attribution giữa ngày ô nhiễm nặng và ngày bình thường (RQ3).

        Args:
            spike_attrs: Attribution mảng tại các đỉnh ô nhiễm [N_spike, L, D].
            normal_attrs: Attribution mảng tại ngày bình thường [N_normal, L, D].
            top_k: Số lượng biến cần so sánh.
            save_path: Đường dẫn lưu file ảnh PNG.
            title: Tiêu đề biểu đồ.

        Returns:
            Figure của Matplotlib.
        """
        spike_imp = self.get_feature_attributions(spike_attrs)
        normal_imp = self.get_feature_attributions(normal_attrs)

        top_feats = list(spike_imp.keys())[:top_k]
        spike_scores = [spike_imp[f] for f in top_feats]
        normal_scores = [normal_imp.get(f, 0.0) for f in top_feats]

        x = np.arange(len(top_feats))
        width = 0.38

        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        rects1 = ax.bar(x - width / 2, spike_scores, width, label="Đỉnh ô nhiễm (PM2.5 > 150)", color="#dc2626", alpha=0.9)
        rects2 = ax.bar(x + width / 2, normal_scores, width, label="Ngày bình thường (PM2.5 ≤ 75)", color="#2563eb", alpha=0.85)

        ax.set_ylabel("mean(|Attribution|)", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(top_feats, rotation=35, ha="right", fontsize=10)
        ax.legend(frameon=True, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"[INFO] Đã lưu biểu đồ So sánh Spikes vs Normal tại: {save_path}")

        return fig

    def plot_temporal_attribution_curve(
        self,
        attributions: np.ndarray,
        top_k: int = 6,
        save_path: Optional[str] = None,
        title: str = "Diễn biến mức độ ảnh hưởng của các biến tiến gần tới đỉnh ô nhiễm",
    ) -> plt.Figure:
        """Vẽ biểu đồ đường biểu diễn sự tăng vọt attribution của các biến trong 96h quá khứ.

        Args:
            attributions: Mảng attribution [N, L=96, D].
            top_k: Số lượng biến hàng đầu cần vẽ.
            save_path: Đường dẫn lưu ảnh PNG.
            title: Tiêu đề biểu đồ.

        Returns:
            Figure của Matplotlib.
        """
        temp_profile = np.mean(np.abs(attributions), axis=0)  # [96, D]
        hours = np.arange(-temp_profile.shape[0], 0)

        importance_dict = self.get_feature_attributions(attributions)
        top_feats = list(importance_dict.keys())[:top_k]

        fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
        colors = ["#dc2626", "#ea580c", "#2563eb", "#059669", "#7c3aed", "#0891b2"]

        for idx, feat in enumerate(top_feats):
            feat_idx = self.feature_names.index(feat)
            curve = temp_profile[:, feat_idx]
            ax.plot(hours, curve, label=feat, color=colors[idx % len(colors)], linewidth=2)

        ax.set_xlabel("Thời gian quá khứ (giờ)", fontsize=11, fontweight="bold")
        ax.set_ylabel("mean(|Attribution|)", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.axvline(x=-24, color="gray", linestyle=":", label="Mốc 24h trước đỉnh")
        ax.axvline(x=-6, color="red", linestyle="--", label="Cận đỉnh (6h)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", frameon=True)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"[INFO] Đã lưu biểu đồ Đường diễn biến thời gian tại: {save_path}")

        return fig

    def plot_wind_attribution_breakdown(
        self,
        attributions: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Xếp hạng đóng góp của 16 hướng gió tại các đỉnh ô nhiễm (Phục vụ RQ3)",
    ) -> plt.Figure:
        """Vẽ biểu đồ đóng góp của toàn bộ 16 hướng gió trong các đợt bùng phát ô nhiễm.

        Args:
            attributions: Mảng attribution [N, L, D].
            save_path: Đường dẫn lưu ảnh PNG.
            title: Tiêu đề biểu đồ.

        Returns:
            Figure của Matplotlib.
        """
        importance_dict = self.get_feature_attributions(attributions)
        wind_feats = [f for f in self.feature_names if f.startswith("wd_")]

        wind_scores = {f: importance_dict.get(f, 0.0) for f in wind_feats}
        sorted_wind = dict(sorted(wind_scores.items(), key=lambda x: x[1], reverse=True))

        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        names = [k.replace("wd_", "") for k in sorted_wind.keys()]
        scores = list(sorted_wind.values())

        # Gió hướng Nam/Đông Nam (mang khói bụi) tô màu đỏ, hướng Bắc/Tây Bắc tô màu xanh
        bar_colors = [
            "#dc2626" if any(w in n for w in ["S", "SE", "SW", "E"]) else "#2563eb"
            for n in names
        ]

        bars = ax.bar(names, scores, color=bar_colors, alpha=0.85, edgecolor="#1e293b")
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + (max(scores) * 0.01),
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )

        ax.set_ylabel("mean(|Attribution|)", fontsize=11, fontweight="bold")
        ax.set_xlabel("16 Hướng gió (Đỏ: Hướng Nam/Đông/Tây Nam — Xanh: Hướng Bắc/Tây Bắc)", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"[INFO] Đã lưu biểu đồ Hướng gió tại: {save_path}")

        return fig


def load_model_and_checkpoint(
    checkpoint_path: str, device: str = "cpu"
) -> Tuple[nn.Module, List[str], Dict[str, Any]]:
    """Nạp trọng số mô hình từ checkpoint và tự động áp dụng ResidualAnchorWrapper."""
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


def extract_spikes_and_normal_windows(
    csv_path: str,
    feature_cols: List[str],
    target_col: str = "pm2.5",
    spike_threshold: float = 150.0,
    normal_threshold: float = 75.0,
    lookback: int = 96,
    horizon: int = 24,
    stride: int = 12,
    num_samples: int = 40,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trích xuất tập mẫu đỉnh ô nhiễm (Spikes) và ngày bình thường (Normal) từ tệp CSV."""
    from src.data_pipeline.dataset import make_sliding_windows

    df = pd.read_csv(csv_path)
    X, y = make_sliding_windows(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        lookback=lookback,
        horizon=horizon,
        stride=stride,
    )

    spikes_mask = y[:, 0] > spike_threshold
    normal_mask = y[:, 0] <= normal_threshold

    X_spikes = X[spikes_mask]
    X_normal = X[normal_mask]

    rng = np.random.default_rng(seed)

    if len(X_spikes) > num_samples:
        idx_spikes = rng.choice(len(X_spikes), size=num_samples, replace=False)
        X_spikes = X_spikes[idx_spikes]

    if len(X_normal) > num_samples:
        idx_normal = rng.choice(len(X_normal), size=num_samples, replace=False)
        X_normal = X_normal[idx_normal]

    return torch.from_numpy(X_spikes).float(), torch.from_numpy(X_normal).float()


def run_t0402_pipeline(
    config_path: str = "configs/default.yaml",
    checkpoint_path: Optional[str] = None,
    output_dir: str = "outputs/reports",
    n_steps: int = 40,
    num_samples: int = 30,
) -> Dict[str, Any]:
    """Chạy toàn bộ quy trình nghiệm thu T0402 trên dữ liệu thực tế."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = (
        checkpoint_path
        or "outputs/checkpoints/global/patchtst_multivariate_H24_seed42_pretrained.pt"
    )

    print(f"[INFO] 1. Nạp mô hình từ checkpoint: {ckpt_path} (Thiết bị: {device})")
    model, feature_cols, ckpt = load_model_and_checkpoint(ckpt_path, device=device)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg.get("paths", {}).get("data_dir", "data/preprocess")
    test_path = f"{data_dir}/test.csv"
    lookback = ckpt.get("lookback", 96)
    horizon = ckpt.get("horizon", 24)

    print(f"[INFO] 2. Trích xuất đỉnh ô nhiễm (PM2.5 > 150) và ngày bình thường từ: {test_path}")
    spikes_tensor, normal_tensor = extract_spikes_and_normal_windows(
        test_path,
        feature_cols=feature_cols,
        lookback=lookback,
        horizon=horizon,
        num_samples=num_samples,
        stride=12,
        seed=42,
    )
    print(f"[INFO] Đã trích xuất {len(spikes_tensor)} mẫu đỉnh ô nhiễm và {len(normal_tensor)} mẫu ngày bình thường.")

    print(f"[INFO] 3. Tính toán Integrated Gradients với n_steps={n_steps}...")
    explainer = IGExplainer(model, feature_cols, device=device)

    spike_attrs, spike_deltas = explainer.attribute(spikes_tensor, n_steps=n_steps)
    normal_attrs, normal_deltas = explainer.attribute(normal_tensor, n_steps=n_steps)

    print(f"[SUCCESS] Tính IG hoàn tất! Sai số Riemann delta trung bình: {np.mean(np.abs(spike_deltas)):.6f}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Xuất JSON
    json_file = out_path / "ig_smog_spikes_attribution.json"
    dashboard_data = explainer.export_for_dashboard(spike_attrs, spike_deltas, top_k=15)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Đã lưu kết quả JSON tại: {json_file}")

    print("[INFO] 4. Xuất bộ 4 biểu đồ trực quan hóa chuyên sâu cho T0402:")
    # 1. Heatmap ma trận 96h x Features
    chart_1 = str(out_path / "ig_smog_spike_heatmap.png")
    explainer.plot_spike_heatmap(spike_attrs, sample_idx=0, top_k=15, save_path=chart_1)

    # 2. So sánh Spikes vs Normal (RQ3)
    chart_2 = str(out_path / "ig_spike_vs_normal_comparison.png")
    explainer.plot_spike_vs_normal_comparison(spike_attrs, normal_attrs, top_k=12, save_path=chart_2)

    # 3. Temporal attribution curve tiến về đỉnh
    chart_3 = str(out_path / "ig_temporal_attribution_curve.png")
    explainer.plot_temporal_attribution_curve(spike_attrs, top_k=6, save_path=chart_3)

    # 4. Phân rã 16 hướng gió tại đỉnh ô nhiễm
    chart_4 = str(out_path / "ig_wind_directions_breakdown.png")
    explainer.plot_wind_attribution_breakdown(spike_attrs, save_path=chart_4)

    return dashboard_data


def _run_smoke_test() -> None:
    """Chạy kiểm thử nhanh trên dummy data để đảm bảo tính sẵn sàng của module."""
    torch.manual_seed(42)
    dummy_names = ["PM2.5", "TEMP", "DEW", "RAIN", "WSPM"]

    class DummyModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.mean(dim=1, keepdim=True).repeat(1, 24, 1)[:, :, :1]

    model = DummyModel()
    explainer = IGExplainer(model, dummy_names)

    test_sample = torch.randn(2, 96, 5)
    attrs, delta = explainer.attribute(test_sample, n_steps=20)

    assert attrs.shape == (2, 96, 5), f"Shape không hợp lệ: {attrs.shape}"
    assert delta.shape == (2,), f"Delta shape không hợp lệ: {delta.shape}"
    print("[SUCCESS] Smoke test PASS 100%! Attribution shape:", attrs.shape, "Delta:", delta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Integrated Gradients Local Attribution (BPM25_T0402)")
    parser.add_argument("--smoke-test", action="store_true", help="Chạy smoke test trên dummy data")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Đường dẫn file cấu hình")
    parser.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn checkpoint cần giải thích")
    parser.add_argument("--n-steps", type=int, default=40, help="Số bước xấp xỉ tích phân Riemann")
    parser.add_argument("--num-samples", type=int, default=30, help="Số mẫu đỉnh ô nhiễm cần trích xuất")
    parser.add_argument("--output-dir", type=str, default="outputs/reports", help="Thư mục xuất báo cáo")

    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
    else:
        run_t0402_pipeline(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            n_steps=args.n_steps,
            num_samples=args.num_samples,
        )
