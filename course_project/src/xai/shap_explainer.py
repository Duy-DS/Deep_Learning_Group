"""Cung cấp bộ giải thích SHAP cho mô hình dự báo chuỗi thời gian PyTorch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import shap
import torch
import torch.nn as nn
import yaml
from pydantic import BaseModel, ConfigDict, Field


class SHAPGlobalExplanation(BaseModel):
    """Hợp đồng dữ liệu giải thích SHAP toàn cục đã chuẩn hóa."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    feature_names: List[str] = Field(description="Tên biến theo đúng thứ tự tensor đầu vào.")
    shap_values: List[List[List[float]]] = Field(
        description="Giá trị SHAP có kích thước [số_mẫu, lookback, số_biến]."
    )
    global_feature_importance: Dict[str, float] = Field(
        description="Độ quan trọng mean(|SHAP|) của từng biến, sắp xếp giảm dần."
    )
    top_positive_features: List[str] = Field(
        description="Các biến có attribution có dấu trung bình lớn nhất."
    )
    top_negative_features: List[str] = Field(
        description="Các biến có attribution có dấu trung bình nhỏ nhất."
    )


class _HorizonModel(nn.Module):
    """Bộ chuyển đổi đầu ra đa bước thành một đầu ra vô hướng theo horizon."""

    def __init__(self, model: nn.Module, target_horizon_idx: int) -> None:
        """Khởi tạo bộ chuyển đổi cho mô hình và horizon được chọn.

        Args:
            model: Mô hình nhận tensor [B, L, D] và trả tensor dự báo.
            target_horizon_idx: Chỉ số bước dự báo cần giải thích.
        """
        super().__init__()
        self.model = model
        self.target_horizon_idx = target_horizon_idx

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Lấy đầu ra của horizon đã chọn.

        Args:
            inputs: Tensor đầu vào có kích thước [B, L, D].

        Returns:
            Tensor dự báo có kích thước [B, 1].

        Raises:
            ValueError: Nếu đầu ra mô hình không có dạng [B, H, 1].
        """
        predictions = self.model(inputs)
        if predictions.ndim != 3 or predictions.shape[-1] != 1:
            raise ValueError(
                "Mô hình phải trả tensor có kích thước [B, H, 1]; "
                f"nhận được {tuple(predictions.shape)}."
            )
        if self.target_horizon_idx >= predictions.shape[1]:
            raise ValueError(
                f"Chỉ số horizon {self.target_horizon_idx} vượt quá "
                f"horizon đầu ra {predictions.shape[1]}."
            )
        return predictions[:, self.target_horizon_idx, :]


class SHAPExplainer:
    """Tính attribution SHAP cục bộ và độ quan trọng biến toàn cục.

    Dữ liệu nền phải được lấy từ tập Train; không dùng dữ liệu Validation/Test
    để chọn hoặc xây dựng dữ liệu nền nhằm tránh rò rỉ thông tin.
    """

    def __init__(
        self,
        model: nn.Module,
        background_data: torch.Tensor,
        feature_names: List[str],
        device: str = "cpu",
        lookback: Optional[int] = None,
        nsamples: int = 100,
    ) -> None:
        """Khởi tạo bộ giải thích và kiểm tra hợp đồng tensor.

        Args:
            model: Mô hình PyTorch nhận [B, 96, D], trả [B, H, 1].
            background_data: Tensor tham chiếu [N, 96, D] chỉ lấy từ tập Train.
            feature_names: Tên D biến theo đúng thứ tự chiều cuối.
            device: Thiết bị thực thi, ví dụ ``"cpu"`` hoặc ``"cuda"``.
            lookback: Độ dài chuỗi đầu vào; mặc định lấy từ dữ liệu nền.
            nsamples: Số mẫu xấp xỉ SHAP dùng cho mỗi lần giải thích.

        Raises:
            TypeError: Nếu mô hình, tensor hoặc tên biến không đúng kiểu.
            ValueError: Nếu dữ liệu nền rỗng, sai chiều hoặc cấu hình không hợp lệ.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("model phải là một thể hiện của torch.nn.Module.")
        if not isinstance(background_data, torch.Tensor):
            raise TypeError("background_data phải là torch.Tensor.")
        if background_data.ndim != 3 or background_data.shape[0] == 0:
            raise ValueError("background_data phải có dạng [N>0, L, D].")
        if not feature_names or len(feature_names) != background_data.shape[2]:
            raise ValueError("Số feature_names phải khớp chiều D của dữ liệu nền.")
        if lookback is not None and lookback != background_data.shape[1]:
            raise ValueError("lookback cấu hình phải khớp chiều L của dữ liệu nền.")
        if nsamples <= 0:
            raise ValueError("nsamples phải là số nguyên dương.")
        if any(not isinstance(name, str) or not name for name in feature_names):
            raise ValueError("Mỗi tên biến phải là chuỗi không rỗng.")
        if not torch.isfinite(background_data).all():
            raise ValueError("background_data chứa giá trị NaN hoặc vô hạn.")

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.background_data = background_data.detach().to(self.device)
        self.feature_names = list(feature_names)
        self.lookback = int(lookback or background_data.shape[1])
        self.nsamples = int(nsamples)
        self._last_target_horizon_idx = 0

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        background_data: torch.Tensor,
        config_path: str | Path,
        feature_names: Optional[List[str]] = None,
        device: Optional[str] = None,
    ) -> "SHAPExplainer":
        """Khởi tạo bộ giải thích từ tệp YAML cấu hình dự án.

        Đọc cấu hình chuẩn duy nhất ``default.yaml``; tên biến luôn lấy từ
        cấu hình nếu không truyền riêng.

        Args:
            model: Mô hình PyTorch đã nạp trọng số.
            background_data: Tensor [N, L, D] lấy duy nhất từ tập Train.
            config_path: Đường dẫn tệp YAML cấu hình.
            feature_names: Tên biến theo thứ tự tensor; ưu tiên hơn YAML.
            device: Thiết bị tùy chọn; nếu bỏ trống sẽ lấy từ cấu hình.

        Returns:
            Một ``SHAPExplainer`` có tham số khớp cấu hình.

        Raises:
            FileNotFoundError: Nếu không tìm thấy tệp YAML.
            ValueError: Nếu cấu hình thiếu lookback hoặc tên biến không hợp lệ.
        """
        path = Path(config_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy tệp cấu hình: {path}")
        with path.open("r", encoding="utf-8") as config_file:
            config: Mapping[str, Any] = yaml.safe_load(config_file) or {}

        data_config = config.get("data", {})
        project_config = config.get("project", {})
        xai_config = config.get("xai", {}).get("shap", {})
        if "lookback" not in data_config:
            raise ValueError("Cấu hình data.lookback không tồn tại.")
        configured_names = feature_names or data_config.get(
            "canonical_multivariate_cols", data_config.get("feature_names")
        )
        if not configured_names:
            raise ValueError("Cấu hình thiếu danh sách feature_names.")

        configured_device = device or project_config.get("device", "cpu")
        if configured_device == "auto":
            configured_device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(
            model=model,
            background_data=background_data,
            feature_names=list(configured_names),
            device=str(configured_device),
            lookback=int(data_config["lookback"]),
            nsamples=int(xai_config.get("nsamples", 100)),
        )

    def compute_shap_values(
        self, test_samples: torch.Tensor, target_horizon_idx: int = 0
    ) -> np.ndarray:
        """Tính giá trị SHAP cho một horizon của chuỗi đa biến.

        Args:
            test_samples: Tensor cần giải thích có dạng [B, 96, D].
            target_horizon_idx: Chỉ số horizon đầu ra cần giải thích.

        Returns:
            Mảng NumPy float32 có dạng [B, 96, D].

        Raises:
            TypeError: Nếu test_samples không phải tensor PyTorch.
            ValueError: Nếu kích thước đầu vào sai hoặc chỉ số horizon âm.
        """
        if not isinstance(test_samples, torch.Tensor):
            raise TypeError("test_samples phải là torch.Tensor.")
        if test_samples.ndim != 3:
            raise ValueError("test_samples phải có dạng [B, L, D].")
        if test_samples.shape[0] == 0 or test_samples.shape[1:] != self.background_data.shape[1:]:
            raise ValueError(
                "Kích thước test_samples phải là [B>0, L, D] và khớp dữ liệu nền."
            )
        if target_horizon_idx < 0:
            raise ValueError("target_horizon_idx không được âm.")
        if not torch.isfinite(test_samples).all():
            raise ValueError("test_samples chứa giá trị NaN hoặc vô hạn.")

        self._last_target_horizon_idx = target_horizon_idx
        wrapped_model = _HorizonModel(self.model, target_horizon_idx).to(self.device)
        wrapped_model.eval()
        explainer = shap.GradientExplainer(wrapped_model, self.background_data)
        raw_values: Any = explainer.shap_values(
            test_samples.detach().to(self.device), nsamples=self.nsamples
        )

        # Chuẩn hóa định dạng danh sách một đầu ra giữa các phiên bản SHAP.
        if isinstance(raw_values, (list, tuple)):
            if len(raw_values) != 1:
                raise ValueError("SHAP trả số đầu ra không khớp một horizon đã chọn.")
            raw_values = raw_values[0]
        values = np.asarray(raw_values, dtype=np.float32)
        expected_shape = tuple(test_samples.shape)
        if values.shape == (*expected_shape, 1):
            values = values[..., 0]
        if values.shape != expected_shape:
            raise ValueError(f"SHAP trả shape {values.shape}, cần {expected_shape}.")
        if not np.isfinite(values).all():
            raise ValueError("SHAP trả về giá trị NaN hoặc vô hạn.")
        return values

    def get_global_feature_importance(
        self, shap_values: np.ndarray
    ) -> Dict[str, float]:
        """Tổng hợp mean(|SHAP|) theo mẫu và toàn bộ bước thời gian.

        Args:
            shap_values: Mảng attribution có dạng [B, L, D].

        Returns:
            Từ điển tên biến đến mean(|SHAP|), được xếp giảm dần.

        Raises:
            ValueError: Nếu mảng không hữu hạn hoặc không khớp số biến.
        """
        values = np.asarray(shap_values)
        if values.ndim != 3 or values.shape[-1] != len(self.feature_names):
            raise ValueError("shap_values phải có dạng [B, L, D] với D khớp tên biến.")
        if values.shape[0] == 0 or values.shape[1] == 0 or not np.isfinite(values).all():
            raise ValueError("shap_values phải không rỗng và chỉ chứa giá trị hữu hạn.")
        importance = np.mean(np.abs(values), axis=(0, 1))
        order = np.argsort(-importance, kind="stable")
        return {self.feature_names[index]: float(importance[index]) for index in order}

    def export_for_dashboard(
        self, shap_values: np.ndarray, top_k: int = 5
    ) -> Dict[str, Any]:
        """Tạo dữ liệu JSON-compatible cho biểu đồ XAI trên bảng điều khiển.

        Args:
            shap_values: Mảng attribution [B, L, D] do bộ giải thích tính toán.
            top_k: Số biến dẫn đầu được liệt kê cho mỗi chiều dấu.

        Returns:
            Từ điển chuẩn hóa từ hợp đồng ``SHAPGlobalExplanation``.

        Raises:
            ValueError: Nếu top_k không dương hoặc attribution không hợp lệ.
        """
        if top_k <= 0:
            raise ValueError("top_k phải là số nguyên dương.")
        values = np.asarray(shap_values, dtype=np.float64)
        importance = self.get_global_feature_importance(values)
        signed_means = np.mean(values, axis=(0, 1))
        positive_order = np.argsort(-signed_means, kind="stable")
        negative_order = np.argsort(signed_means, kind="stable")
        result = SHAPGlobalExplanation(
            feature_names=self.feature_names,
            shap_values=values.astype(np.float32).tolist(),
            global_feature_importance=importance,
            top_positive_features=[
                self.feature_names[index] for index in positive_order[:top_k]
            ],
            top_negative_features=[
                self.feature_names[index] for index in negative_order[:top_k]
            ],
        )
        return result.model_dump()


class DummyTimeSeriesModel(nn.Module):
    """Mô hình chuỗi thời gian nhỏ để kiểm thử độc lập bộ giải thích."""

    def __init__(self, input_features: int, horizon: int = 24) -> None:
        """Khởi tạo lớp chiếu tuyến tính cho dữ liệu chuỗi.

        Args:
            input_features: Số biến đầu vào.
            horizon: Số bước dự báo.
        """
        super().__init__()
        self.projection = nn.Linear(input_features, horizon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Dự báo [B, H, 1] từ chuỗi đầu vào [B, 96, D].

        Args:
            inputs: Tensor đầu vào ba chiều [B, L, D].

        Returns:
            Tensor dự báo ba chiều [B, H, 1].
        """
        return self.projection(inputs.mean(dim=1)).unsqueeze(-1)


def _run_smoke_test() -> None:
    """Chạy kiểm thử tái lập được bằng mô hình và dữ liệu giả lập."""
    torch.manual_seed(42)
    np.random.seed(42)
    feature_names = [
        "PM2.5", "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "wd_NE", "wd_NW",
        "wd_SE", "wd_SW", "wd_cv", "hour_sin", "hour_cos", "month_sin", "month_cos",
    ]
    model = DummyTimeSeriesModel(input_features=len(feature_names))
    background_data = torch.randn(8, 96, len(feature_names))
    test_samples = torch.randn(4, 96, len(feature_names))
    explainer = SHAPExplainer(model, background_data, feature_names)
    values = explainer.compute_shap_values(test_samples, target_horizon_idx=0)
    importance = explainer.get_global_feature_importance(values)
    dashboard_data = explainer.export_for_dashboard(values)

    assert values.shape == (4, 96, len(feature_names)), values.shape
    assert len(importance) == len(feature_names)
    assert list(importance.values()) == sorted(importance.values(), reverse=True)
    assert dashboard_data["feature_names"] == feature_names
    print("[INFO] Seed cố định: 42.")
    print(f"[INFO] Tensor nền: {tuple(background_data.shape)}; mẫu kiểm tra: {tuple(test_samples.shape)}.")
    print(f"[INFO] Tensor SHAP: {values.shape}; số biến: {len(importance)}.")
    print(f"[INFO] 5 biến quan trọng nhất: {list(importance.items())[:5]}.")
    print("[SUCCESS] SHAP GradientExplainer và hợp đồng bảng điều khiển: PASS.")


if __name__ == "__main__":
    _run_smoke_test()
