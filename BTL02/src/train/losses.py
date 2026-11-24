"""Hàm mất mát PyTorch cho phân loại FER2013 mất cân bằng lớp."""

from collections.abc import Sequence
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    train_labels: Union[Sequence[int], torch.Tensor],
    num_classes: int = 7,
    method: str = "inverse_freq",
) -> torch.Tensor:
    """Tính trọng số lớp chỉ từ nhãn của tập huấn luyện.

    Với ``inverse_freq``, áp dụng ``N / (K * N_c)`` rồi chuẩn hóa trung bình
    trọng số bằng 1. ``effective_num`` dùng số mẫu hiệu dụng với beta=0.9999.

    Args:
        train_labels: Nhãn Training; không truyền nhãn Validation/Test.
        num_classes: Số lớp cảm xúc.
        method: ``inverse_freq`` hoặc ``effective_num``.

    Returns:
        Tensor float32 một chiều có ``num_classes`` phần tử, trung bình bằng 1.

    Raises:
        ValueError: Nhãn rỗng, ngoài miền, thiếu lớp hoặc phương pháp không hỗ trợ.
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes phải là số nguyên dương.")
    labels = torch.as_tensor(train_labels, dtype=torch.long).detach().cpu().reshape(-1)
    if labels.numel() == 0:
        raise ValueError("train_labels không được rỗng.")
    if bool(((labels < 0) | (labels >= num_classes)).any()):
        raise ValueError("Nhãn Training nằm ngoài miền [0, num_classes).")
    counts = torch.bincount(labels, minlength=num_classes).to(dtype=torch.float64)
    if bool((counts == 0).any()):
        raise ValueError("Mọi lớp cần có ít nhất một mẫu trong Training để tính trọng số.")

    normalized_method = method.strip().lower()
    if normalized_method == "inverse_freq":
        weights = labels.numel() / (num_classes * counts)
    elif normalized_method == "effective_num":
        beta = 0.9999
        weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
    else:
        raise ValueError("method phải là 'inverse_freq' hoặc 'effective_num'.")
    weights = weights / weights.mean()
    return weights.to(dtype=torch.float32)


class FocalLoss(nn.Module):
    """Focal Loss multiclass, làm giảm đóng góp của mẫu đã phân loại dễ."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[torch.Tensor, float]] = None,
        reduction: str = "mean",
    ) -> None:
        """Khởi tạo Focal Loss và đăng ký alpha để tự chuyển theo device.

        Args:
            gamma: Số mũ điều tiết đóng góp của mẫu dễ, thường bằng 2.
            alpha: Hệ số vô hướng hoặc vector trọng số theo lớp.
            reduction: ``none``, ``mean`` hoặc ``sum``.

        Raises:
            ValueError: Tham số gamma/reduction không hợp lệ.
        """
        super().__init__()
        if not torch.isfinite(torch.tensor(float(gamma))) or gamma < 0:
            raise ValueError("gamma phải không âm.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction phải là 'none', 'mean' hoặc 'sum'.")
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is None:
            alpha_tensor = torch.empty(0, dtype=torch.float32)
            self.alpha_is_scalar = False
        elif isinstance(alpha, torch.Tensor):
            alpha_tensor = alpha.detach().to(dtype=torch.float32).clone()
            if alpha_tensor.ndim != 1 or not torch.isfinite(alpha_tensor).all() or (alpha_tensor < 0).any():
                raise ValueError("alpha dạng tensor phải là vector hữu hạn, không âm.")
            self.alpha_is_scalar = False
        else:
            if not torch.isfinite(torch.tensor(float(alpha))) or float(alpha) < 0:
                raise ValueError("alpha vô hướng phải không âm.")
            alpha_tensor = torch.tensor([float(alpha)], dtype=torch.float32)
            self.alpha_is_scalar = True
        self.register_buffer("alpha", alpha_tensor)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Tính loss ổn định số học từ logits chưa qua Softmax.

        Args:
            logits: Tensor logits có shape ``[N, C, ...]``.
            targets: Nhãn nguyên có shape ``[N, ...]``.

        Returns:
            Loss scalar hoặc tensor theo reduction.
        """
        targets = targets.long()
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        # Từ log-probability CE tính pt ổn định, tránh log(softmax) thủ công.
        pt = torch.exp(-ce_loss).clamp(min=0.0, max=1.0)
        loss = (1.0 - pt).clamp_min(0.0).pow(self.gamma) * ce_loss
        if self.alpha.numel():
            if self.alpha_is_scalar:
                loss = loss * self.alpha[0]
            else:
                if self.alpha.numel() != logits.shape[1]:
                    raise ValueError("Số alpha phải bằng số lớp trong logits.")
                alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
                loss = loss * alpha[targets]
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.mean()
        return loss


class LossFactory:
    """Tạo criterion từ cấu hình YAML, chỉ dùng thống kê nhãn Training."""

    @staticmethod
    def create(
        config: Optional[Dict[str, Any]] = None,
        train_labels: Optional[Union[Sequence[int], torch.Tensor]] = None,
        num_classes: int = 7,
        device: Optional[Union[str, torch.device]] = None,
    ) -> nn.Module:
        """Khởi tạo CrossEntropy, Weighted CE hoặc Focal Loss.

        Args:
            config: Nội dung của khối YAML ``loss``.
            train_labels: Nhãn Training duy nhất dùng tính class weights.
            num_classes: Số lớp đầu ra.
            device: Thiết bị đích của criterion.

        Returns:
            nn.Module: Hàm mất mát đã chuyển đến device yêu cầu.

        Raises:
            ValueError: Cấu hình loss không hợp lệ hoặc thiếu nhãn Training.
        """
        options = dict(config or {})
        name = str(options.get("name", "cross_entropy")).strip().lower()
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        if name == "cross_entropy":
            criterion: nn.Module = nn.CrossEntropyLoss()
        elif name == "weighted_ce":
            if train_labels is None:
                raise ValueError("weighted_ce yêu cầu nhãn Training để tính class weights.")
            weights = compute_class_weights(
                train_labels,
                num_classes=num_classes,
                method=str(options.get("weight_method", "inverse_freq")),
            )
            criterion = nn.CrossEntropyLoss(weight=weights)
        elif name == "focal":
            alpha: Optional[Union[torch.Tensor, float]] = options.get("alpha")
            if options.get("use_class_weights", False):
                if train_labels is None:
                    raise ValueError("Focal Loss có class weights cần nhãn Training.")
                alpha = compute_class_weights(
                    train_labels,
                    num_classes=num_classes,
                    method=str(options.get("weight_method", "inverse_freq")),
                )
            criterion = FocalLoss(
                gamma=float(options.get("gamma", 2.0)),
                alpha=alpha,
                reduction=str(options.get("reduction", "mean")),
            )
        else:
            raise ValueError(f"Loss '{name}' không hỗ trợ; chọn cross_entropy, weighted_ce hoặc focal.")
        return criterion.to(target_device)