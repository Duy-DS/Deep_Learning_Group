import numpy as np
import torch
from sklearn.metrics import root_mean_squared_error


def compute_rmse(y_true, y_pred) -> float:
    """Tính Root Mean Squared Error tiêu chuẩn."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    return float(root_mean_squared_error(y_true, y_pred))


def compute_rmse_log(y_true, y_pred, is_already_log: bool = True) -> float:
    """
    Tính RMSE giữa log(y_true) và log(y_pred).
    Nếu is_already_log=True, coi như đầu vào đã ở dạng log1p.
    Nếu is_already_log=False, sẽ tính log1p trước khi đo RMSE.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    if not is_already_log:
        y_true = np.log1p(np.clip(y_true, 0, None))
        y_pred = np.log1p(np.clip(y_pred, 0, None))

    return float(root_mean_squared_error(y_true, y_pred))
