import csv
from pathlib import Path
import torch


def regression_metrics(y_true, y_pred):
    """Tính MSE, MAE và RMSE cho bài toán hồi quy."""
    y_true = torch.as_tensor(y_true, dtype=torch.float32).reshape(-1)
    y_pred = torch.as_tensor(y_pred, dtype=torch.float32).reshape(-1)

    if y_true.numel() == 0 or y_true.shape != y_pred.shape:
        raise ValueError("y_true và y_pred phải có cùng số phần tử và không được rỗng.")

    error = y_pred - y_true
    mse = torch.mean(error ** 2).item()
    mae = torch.mean(torch.abs(error)).item()
    return {"MSE": mse, "MAE": mae, "RMSE": mse ** 0.5}


def append_csv_log(path, row, fieldnames=None):
    """Ghi một dòng log CSV; tự tạo header nếu file chưa tồn tại."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    if fieldnames is None:
        fieldnames = list(row.keys())
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
