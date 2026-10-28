from pathlib import Path
import copy
import csv
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from metrics import regression_metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
CKPT = ROOT / "checkpoints"
OUT.mkdir(exist_ok=True)
CKPT.mkdir(exist_ok=True)

# Nhóm A: chỉ thay đổi số mẫu train
SAMPLE_SIZES = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
SEEDS = [17, 42, 2026]  # >= 3 seeds theo khuyến nghị đề bài
N_VALIDATION = 500
N_TEST = 100
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.01
HIDDEN_NODES = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tham số Gaussian theo đề bài mới
MU = np.array([0.0, 0.0], dtype=np.float64)
COV = np.array([[0.3, -0.5], [-0.5, 2.0]], dtype=np.float64)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gaussian_2d(x):
    """Mật độ Gaussian 2D: mu=[0,0], C=[[0.3,-0.5],[-0.5,2]]."""
    x = np.asarray(x, dtype=np.float64)
    delta = x - MU
    det_cov = np.linalg.det(COV)
    inv_cov = np.linalg.inv(COV)
    exponent = -0.5 * np.sum((delta @ inv_cov) * delta, axis=1)
    return (1.0 / (2.0 * np.pi * np.sqrt(det_cov))) * np.exp(exponent)


def generate_dataset(n, seed):
    """Lấy x độc lập, đều trên [-2,2] x [-4,4], y=f(x)."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2.0, 2.0, n)
    x2 = rng.uniform(-4.0, 4.0, n)
    x = np.column_stack((x1, x2)).astype(np.float32)
    y = gaussian_2d(x).astype(np.float32).reshape(-1, 1)
    return torch.from_numpy(x), torch.from_numpy(y)


class MLP(nn.Module):
    """MLP một tầng ẩn, 128 nơ-ron; sigmoid theo cấu trúc báo cáo mẫu."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, HIDDEN_NODES),
            nn.Sigmoid(),
            nn.Linear(HIDDEN_NODES, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(x.to(DEVICE)).cpu()


def evaluate(model, x, y):
    y_pred = predict(model, x)
    return regression_metrics(y, y_pred)


def train_one_run(size, seed, x_pool, y_pool, x_val, y_val, x_test, y_test):
    set_seed(seed)

    # Lấy tập con có seed cố định từ cùng một training pool 2000 mẫu.
    rng = np.random.default_rng(seed + size)
    indices = rng.permutation(len(x_pool))[:size]
    x_train, y_train = x_pool[indices], y_pool[indices]

    model = MLP().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1000 + size),
    )

    best_val_mse = float("inf")
    best_state = None
    best_epoch = 0
    epoch_rows = []
    start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        train_mse = evaluate(model, x_train, y_train)["MSE"]
        val_metrics = evaluate(model, x_val, y_val)
        val_mse = val_metrics["MSE"]
        epoch_rows.append({
            "seed": seed, "train_size": size, "epoch": epoch,
            "train_MSE": train_mse, "val_MSE": val_mse,
        })

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    elapsed = time.perf_counter() - start
    model.load_state_dict(best_state)

    # Chỉ đánh giá test sau khi đã chọn checkpoint bằng validation.
    train_metrics = evaluate(model, x_train, y_train)
    val_metrics = evaluate(model, x_val, y_val)
    test_metrics = evaluate(model, x_test, y_test)

    ckpt_path = CKPT / f"mlp_A{size}_seed{seed}.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "train_size": size,
        "seed": seed,
        "best_epoch": best_epoch,
        "hidden_nodes": HIDDEN_NODES,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    }, ckpt_path)

    row = {
        "train_size": size, "seed": seed, "best_epoch": best_epoch,
        "training_seconds": elapsed,
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"validation_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "checkpoint": str(ckpt_path.relative_to(ROOT)),
    }
    return row, epoch_rows


def main():
    # Dữ liệu cố định để so sánh công bằng giữa các quy mô.
    x_pool, y_pool = generate_dataset(2000, seed=100)
    x_val, y_val = generate_dataset(N_VALIDATION, seed=200)
    x_test, y_test = generate_dataset(N_TEST, seed=300)

    all_runs, all_epochs = [], []
    for size in SAMPLE_SIZES:
        for seed in SEEDS:
            print(f"Running Group A | samples={size} | seed={seed}")
            row, epoch_rows = train_one_run(
                size, seed, x_pool, y_pool, x_val, y_val, x_test, y_test
            )
            all_runs.append(row)
            all_epochs.extend(epoch_rows)

    runs = pd.DataFrame(all_runs)
    epochs = pd.DataFrame(all_epochs)
    runs.to_csv(OUT / "group_a_runs.csv", index=False, encoding="utf-8-sig")
    epochs.to_csv(OUT / "group_a_epoch_log.csv", index=False, encoding="utf-8-sig")

    metric_cols = [
        "train_MSE", "validation_MSE", "test_MSE",
        "test_MAE", "test_RMSE", "training_seconds"
    ]
    summary = runs.groupby("train_size")[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUT / "group_a_summary.csv", index=False, encoding="utf-8-sig")

    # Đồ thị sai số theo quy mô: trung bình ± độ lệch chuẩn qua 3 seeds.
    fig, ax = plt.subplots(figsize=(9, 5))
    for metric, label in [
        ("train_MSE", "Train MSE"),
        ("validation_MSE", "Validation MSE"),
        ("test_MSE", "Test MSE"),
    ]:
        means = runs.groupby("train_size")[metric].mean()
        stds = runs.groupby("train_size")[metric].std().fillna(0)
        ax.plot(means.index, means.values, marker="o", label=label)
        ax.fill_between(means.index, means.values - stds.values,
                        means.values + stds.values, alpha=0.15)
    ax.set(title="Group A: Regression error vs. training-set size",
           xlabel="Number of training samples", ylabel="MSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "group_a_error_by_size.png", dpi=160)
    plt.close(fig)

    # Đồ thị hội tụ: trung bình train/validation MSE theo epoch cho từng quy mô.
    epoch_mean = epochs.groupby(["train_size", "epoch"])[["train_MSE", "val_MSE"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    for size in SAMPLE_SIZES:
        part = epoch_mean[epoch_mean["train_size"] == size]
        ax.plot(part["epoch"], part["val_MSE"], label=f"Val n={size}")
    ax.set(title="Group A: Validation convergence by epoch",
           xlabel="Epoch", ylabel="Validation MSE")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "group_a_convergence.png", dpi=160)
    plt.close(fig)

    print("\nFinished.")
    print(f"Device: {DEVICE}")
    print(f"Run-level results: {OUT / 'group_a_runs.csv'}")
    print(f"Summary (mean/std): {OUT / 'group_a_summary.csv'}")
    print(f"Epoch logs: {OUT / 'group_a_epoch_log.csv'}")
    print(f"Plots: {OUT / 'group_a_error_by_size.png'}, {OUT / 'group_a_convergence.png'}")


if __name__ == "__main__":
    main()
