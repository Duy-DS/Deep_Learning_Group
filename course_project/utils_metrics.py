"""
Mô-đun tính toán và tổng hợp độ đo đánh giá mô hình dự báo PM2.5.

Mô-đun này cung cấp class MetricsCalculator để tính MSE và MAE trên
tensor PyTorch, cùng hàm tổng hợp kết quả multi-seed để xuất báo cáo
mean ± std theo chuẩn ExperimentMetricsContract của dự án DL-PM25-BJ.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import numpy as np
import torch
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Hợp đồng dữ liệu (Pydantic v2 — tuân thủ AGENTS.md)
# ---------------------------------------------------------------------------

class ExperimentMetricsContract(BaseModel):
    """
    Hợp đồng dữ liệu đầu ra bắt buộc cho mọi kịch bản đánh giá mô hình
    và Ablation Study trong dự án DL-PM25-BJ.

    Mọi kết quả MSE/MAE từ WP02, WP03 phải đóng gói qua schema này trước
    khi ghi vào docs/experiences/ hoặc bàn giao cho WP05 (Dashboard).
    """

    experiment_id: str = Field(..., description="Mã thực nghiệm, ví dụ: EXP01")
    model_name: Literal["LSTM", "DLinear", "Informer", "Informer_NoProbSparse", "PatchTST"] = Field(
        ..., description="Tên kiến trúc mô hình"
    )
    station_id: str = Field(
        ..., description="Tên trạm quan trắc (hoặc 'ALL_36_STATIONS')"
    )
    horizon: Literal[24, 48, 96, 192] = Field(
        ..., description="Khung dự báo tương lai (H giờ)"
    )
    seeds: List[int] = Field(
        default=[42, 1024, 2026],
        min_length=3,
        description="Danh sách 3 random seeds đã dùng"
    )
    mse_per_seed: List[float] = Field(
        ..., description="MSE tương ứng với từng seed"
    )
    mae_per_seed: List[float] = Field(
        ..., description="MAE tương ứng với từng seed"
    )
    mse_mean: float = Field(..., ge=0.0, description="MSE trung bình qua 3 seeds")
    mse_std: float = Field(..., ge=0.0, description="Độ lệch chuẩn MSE qua 3 seeds")
    mae_mean: float = Field(..., ge=0.0, description="MAE trung bình qua 3 seeds")
    mae_std: float = Field(..., ge=0.0, description="Độ lệch chuẩn MAE qua 3 seeds")
    checkpoint_path: str = Field(
        ..., description="Đường dẫn lưu trọng số .pth tốt nhất"
    )
    extra_notes: Optional[str] = Field(
        default=None, description="Ghi chú kỹ thuật bổ sung (nếu có)"
    )


# ---------------------------------------------------------------------------
# Lớp tính toán độ đo chính
# ---------------------------------------------------------------------------

class MetricsCalculator:
    """
    Bộ tính toán và tổng hợp độ đo đánh giá mô hình dự báo PM2.5.

    Hỗ trợ:
    - Tính MSE (Mean Squared Error) và MAE (Mean Absolute Error) trên
      tensor PyTorch kích thước [B, H, 1] theo chuẩn dự án DL-PM25-BJ.
    - Tổng hợp kết quả nhiều lần chạy (multi-seed) thành mean ± std.
    - Xuất báo cáo theo hợp đồng ExperimentMetricsContract (Pydantic v2).

    Ví dụ sử dụng:
        >>> calc = MetricsCalculator()
        >>> metrics = calc.compute(y_pred=pred_tensor, y_true=true_tensor)
        >>> contract = calc.summarize_multi_seed(
        ...     experiment_id="EXP01",
        ...     model_name="LSTM",
        ...     station_id="ALL_36_STATIONS",
        ...     horizon=24,
        ...     mse_per_seed=[0.12, 0.13, 0.11],
        ...     mae_per_seed=[0.25, 0.27, 0.24],
        ...     checkpoint_path="checkpoints/lstm_h24_best.pth",
        ... )
    """

    def compute(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Tính MSE và MAE giữa tensor dự báo và tensor thực tế.

        Tham số:
            y_pred (torch.Tensor): Tensor dự báo kích thước [B, H, 1].
            y_true (torch.Tensor): Tensor giá trị thực kích thước [B, H, 1].

        Trả về:
            Dict[str, float]: Từ điển chứa 'mse' và 'mae' (kiểu float).

        Ngoại lệ:
            ValueError: Nếu kích thước y_pred và y_true không khớp nhau.
        """
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Tensor shape mismatch: y_pred={y_pred.shape}, "
                f"y_true={y_true.shape}. Expected shape [B, H, 1]."
            )

        diff = y_pred - y_true
        mse = torch.mean(diff ** 2).item()
        mae = torch.mean(torch.abs(diff)).item()

        return {"mse": mse, "mae": mae}

    def compute_numpy(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> Dict[str, float]:
        """
        Tính MSE và MAE từ mảng NumPy (tiện dụng khi không dùng GPU).

        Tham số:
            y_pred (np.ndarray): Mảng dự báo shape [B, H] hoặc [B, H, 1].
            y_true (np.ndarray): Mảng thực tế shape [B, H] hoặc [B, H, 1].

        Trả về:
            Dict[str, float]: Từ điển chứa 'mse' và 'mae'.
        """
        pred_tensor = torch.from_numpy(y_pred.astype(np.float32))
        true_tensor = torch.from_numpy(y_true.astype(np.float32))

        if pred_tensor.dim() == 2:
            pred_tensor = pred_tensor.unsqueeze(-1)
            true_tensor = true_tensor.unsqueeze(-1)

        return self.compute(pred_tensor, true_tensor)

    def summarize_multi_seed(
        self,
        experiment_id: str,
        model_name: Literal["LSTM", "DLinear", "Informer", "Informer_NoProbSparse", "PatchTST"],
        station_id: str,
        horizon: Literal[24, 48, 96, 192],
        mse_per_seed: List[float],
        mae_per_seed: List[float],
        checkpoint_path: str,
        seeds: List[int] = None,
        extra_notes: Optional[str] = None,
    ) -> ExperimentMetricsContract:
        """
        Tổng hợp kết quả từ nhiều lần chạy (multi-seed) thành mean ± std.

        Tham số:
            experiment_id (str): Mã thực nghiệm.
            model_name (str): Tên kiến trúc.
            station_id (str): Tên trạm quan trắc.
            horizon (int): Khung dự báo H.
            mse_per_seed (List[float]): Danh sách MSE từ từng seed.
            mae_per_seed (List[float]): Danh sách MAE từ từng seed.
            checkpoint_path (str): Đường dẫn trọng số .pth tốt nhất.
            seeds (List[int], optional): Danh sách seeds đã dùng.
            extra_notes (str, optional): Ghi chú bổ sung.

        Trả về:
            ExperimentMetricsContract: Hợp đồng kết quả Pydantic v2.
        """
        if seeds is None:
            seeds = [42, 1024, 2026]

        if len(mse_per_seed) < 3:
            raise ValueError(f"At least 3 MSE values required, got: {len(mse_per_seed)}.")
        if len(mae_per_seed) < 3:
            raise ValueError(f"At least 3 MAE values required, got: {len(mae_per_seed)}.")

        mse_array = np.array(mse_per_seed, dtype=np.float64)
        mae_array = np.array(mae_per_seed, dtype=np.float64)

        contract = ExperimentMetricsContract(
            experiment_id=experiment_id,
            model_name=model_name,
            station_id=station_id,
            horizon=horizon,
            seeds=seeds,
            mse_per_seed=mse_per_seed,
            mae_per_seed=mae_per_seed,
            mse_mean=float(np.mean(mse_array)),
            mse_std=float(np.std(mse_array, ddof=0)),
            mae_mean=float(np.mean(mae_array)),
            mae_std=float(np.std(mae_array, ddof=0)),
            checkpoint_path=checkpoint_path,
            extra_notes=extra_notes,
        )

        return contract

    def format_report_row(self, contract: ExperimentMetricsContract) -> str:
        """
        Định dạng kết quả thực nghiệm thành dòng bảng Markdown để ghi vào docs/.
        """
        return (
            f"| {contract.model_name} | H={contract.horizon} | "
            f"{contract.mse_mean:.4f} ± {contract.mse_std:.4f} | "
            f"{contract.mae_mean:.4f} ± {contract.mae_std:.4f} |"
        )

    def print_summary(self, contract: ExperimentMetricsContract) -> None:
        """
        In tóm tắt kết quả thực nghiệm ra terminal theo định dạng chuẩn.
        """
        print(f"[INFO] Model       : {contract.model_name}")
        print(f"[INFO] Station     : {contract.station_id}")
        print(f"[INFO] Horizon (H) : {contract.horizon} hours")
        print(f"[INFO] Seeds       : {contract.seeds}")
        print(f"[INFO] MSE         : {contract.mse_mean:.6f} ± {contract.mse_std:.6f}")
        print(f"[INFO] MAE         : {contract.mae_mean:.6f} ± {contract.mae_std:.6f}")
        print(f"[INFO] Checkpoint  : {contract.checkpoint_path}")


# ---------------------------------------------------------------------------
# Smoke test nội bộ
# ---------------------------------------------------------------------------

def _run_smoke_test() -> None:
    """
    Kiểm tra nhanh tính đúng đắn của MetricsCalculator bằng tensor tổng hợp.
    """
    torch.manual_seed(42)
    calc = MetricsCalculator()

    print("[INFO] Bắt đầu smoke test MetricsCalculator...")

    # Test 1: Shape [4, 24, 1]
    batch_size, horizon = 4, 24
    y_true = torch.ones(batch_size, horizon, 1)
    y_pred = torch.zeros(batch_size, horizon, 1)
    res1 = calc.compute(y_pred, y_true)
    assert abs(res1["mse"] - 1.0) < 1e-6, f"MSE error: {res1['mse']}"
    assert abs(res1["mae"] - 1.0) < 1e-6, f"MAE error: {res1['mae']}"
    print(f"[INFO] Test 1 (MSE=1.0, MAE=1.0): MSE={res1['mse']:.6f}, MAE={res1['mae']:.6f} → PASS")

    # Test 2: Tensor ngẫu nhiên với shape [8, 96, 1]
    y_true2 = torch.randn(8, 96, 1)
    y_pred2 = y_true2 + 0.1
    res2 = calc.compute(y_pred2, y_true2)
    assert abs(res2["mse"] - 0.01) < 1e-5, f"MSE error: {res2['mse']}"
    assert abs(res2["mae"] - 0.1) < 1e-5, f"MAE error: {res2['mae']}"
    print(f"[INFO] Test 2 (diff +0.1, MSE≈0.01, MAE≈0.1): MSE={res2['mse']:.6f}, MAE={res2['mae']:.6f} → PASS")

    # Test 3: Multi-seed aggregation
    contract = calc.summarize_multi_seed(
        experiment_id="EXP_SMOKE",
        model_name="LSTM",
        station_id="ALL_36_STATIONS",
        horizon=24,
        mse_per_seed=[0.100, 0.110, 0.090],
        mae_per_seed=[0.250, 0.260, 0.240],
        checkpoint_path="checkpoints/lstm_h24_seed42_best.pth",
    )
    assert abs(contract.mse_mean - 0.1) < 1e-9, f"MSE mean error: {contract.mse_mean}"
    assert abs(contract.mae_mean - 0.25) < 1e-9, f"MAE mean error: {contract.mae_mean}"
    print(f"[INFO] Test 3 (ExperimentMetricsContract): MSE={contract.mse_mean:.4f}±{contract.mse_std:.4f}, MAE={contract.mae_mean:.4f}±{contract.mae_std:.4f} → PASS")

    # Test 4: Pydantic schema validation
    contract_dict = contract.model_dump()
    assert "mse_mean" in contract_dict
    assert "mae_mean" in contract_dict
    print("[INFO] Test 4 (Pydantic schema validation): PASS")

    table_row = calc.format_report_row(contract)
    print(f"[INFO] Sample Markdown row: {table_row}")
    calc.print_summary(contract)

    print("[SUCCESS] Toàn bộ smoke test MetricsCalculator: PASS. Sẵn sàng tích hợp vào WP02 và WP03.")


if __name__ == "__main__":
    _run_smoke_test()
