import torch
from torch.utils.data import Dataset


class HousePriceDataset(Dataset):
    """
    PyTorch Dataset cho tập dữ liệu Ames Housing Prices.
    Hỗ trợ cả trường hợp có target (train/valid) và không có target (test inference).
    """
    def __init__(self, X, y=None):
        if not isinstance(X, torch.Tensor):
            self.X = torch.tensor(X.values if hasattr(X, "values") else X, dtype=torch.float32)
        else:
            self.X = X.float()

        self.y = None
        if y is not None:
            if not isinstance(y, torch.Tensor):
                self.y = torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.float32)
            else:
                self.y = y.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]
