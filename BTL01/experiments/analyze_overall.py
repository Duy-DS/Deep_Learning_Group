"""Xây dựng các artifacts đối sánh tổng thể A/B/C cho T0701 và T0702."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import torch
import yaml

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_generator import GaussianDataGenerator, GaussianDataset
from src.models import FunctionApproximatorMLP
from src.visualizer import Visualizer


EXPERIMENT_FACTORS = {
    "exp_A": "train_size",
    "exp_B": "width",
    "exp_C": "depth",
}


def load_yaml(path: Path) -> dict[str, Any]:
    """Đọc file YAML với cơ chế phân tích cú pháp an toàn."""
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve_device(config: dict[str, Any]) -> torch.device:
    """Xác định thiết bị tính toán PyTorch được cấu hình."""
    requested = config["hardware"]["device"]
    if requested != "auto":
        return torch.device(requested)
    for name in config["hardware"]["device_priority"]:
        if name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if name == "cpu":
            return torch.device("cpu")
    raise RuntimeError("No available PyTorch device found.")


def read_csv(path: Path) -> pd.DataFrame:
    """Đọc file CSV artifact bắt buộc kèm thông báo lỗi chi tiết."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def choose_best_runs(tables_dir: Path) -> pd.DataFrame:
    """Chọn một lần chạy tốt nhất cho mỗi thực nghiệm chỉ dựa trên validation MSE."""
    rows: list[dict[str, Any]] = []
    # Baseline được đại diện bởi lần chạy tiêu chuẩn A=1000, width=10.
    runs_a = read_csv(tables_dir / "runs_exp_A.csv")
    baseline_runs = runs_a[
        (runs_a["train_size"] == 1000) & (runs_a["status"] == "completed")
    ]
    if not baseline_runs.empty:
        baseline_best = baseline_runs.sort_values(["val_mse", "run_id"]).iloc[0]
        rows.append(
            {
                "experiment": "baseline",
                "run_id": baseline_best["run_id"],
                "selection_metric": "baseline",
                "selection_value": float(baseline_best["val_mse"]),
                "factor": "baseline",
                "factor_value": "standard",
                "config_name": "Baseline [10]",
                "seed": int(baseline_best["seed"]),
                "hidden_dims": str(baseline_best.get("hidden_dims", "[10]")),
                "train_size": baseline_best.get("train_size", 1000),
                "parameter_count": baseline_best.get("parameter_count", 41),
                "train_mse": float(baseline_best["train_mse"]),
                "val_mse": float(baseline_best["val_mse"]),
                "test_mse": float(baseline_best["test_mse"]),
                "test_rmse": float(baseline_best["test_rmse"]),
                "test_mae": float(baseline_best["test_mae"]),
                "training_time_seconds": float(baseline_best["training_time_seconds"]),
                "best_epoch": float(baseline_best["best_epoch"]),
            }
        )

    for experiment, factor_column in EXPERIMENT_FACTORS.items():
        runs = read_csv(tables_dir / f"runs_{experiment}.csv")
        completed = runs[runs["status"] == "completed"].copy()
        if completed.empty:
            raise ValueError(f"No completed runs available for {experiment}.")
        completed["val_mse"] = completed["val_mse"].astype(float)
        best = completed.sort_values(["val_mse", "run_id"]).iloc[0]
        factor_value = best.get(factor_column)
        rows.append(
            {
                "experiment": experiment,
                "run_id": best["run_id"],
                "selection_metric": "val_mse",
                "selection_value": float(best["val_mse"]),
                "factor": factor_column,
                "factor_value": factor_value,
                "config_name": f"{experiment}:{factor_column}={factor_value}",
                "seed": int(best["seed"]),
                "hidden_dims": str(best.get("hidden_dims", "")),
                "train_size": best.get("train_size", ""),
                "parameter_count": best.get("parameter_count", ""),
                "train_mse": float(best["train_mse"]),
                "val_mse": float(best["val_mse"]),
                "test_mse": float(best["test_mse"]),
                "test_rmse": float(best["test_rmse"]),
                "test_mae": float(best["test_mae"]),
                "training_time_seconds": float(best["training_time_seconds"]),
                "best_epoch": float(best["best_epoch"]),
            }
        )
    return pd.DataFrame(rows)


def build_overall_summary(best_runs: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    """Gắn các chỉ số đo lường mean/std tương ứng với giá trị factor được chọn."""
    rows: list[dict[str, Any]] = []
    for row in best_runs.to_dict("records"):
        exp = row["experiment"]
        if exp == "baseline":
            summary = read_csv(tables_dir / "summary_exp_A.csv")
            match = summary[summary["train_size"] == 1000]
        else:
            summary = read_csv(tables_dir / f"summary_{exp}.csv")
            factor = row["factor"]
            match = summary[summary[factor].astype(str) == str(row["factor_value"])]
        summary_row = match.iloc[0].to_dict() if not match.empty else {}
        rows.append(
            {
                **row,
                "test_mse_mean": float(
                    summary_row.get("test_mse_mean", row["test_mse"])
                ),
                "test_mse_std": float(
                    summary_row.get("test_mse_std", float("nan"))
                ),
                "test_rmse_mean": float(
                    summary_row.get("test_rmse_mean", row["test_rmse"])
                ),
                "test_rmse_std": float(
                    summary_row.get("test_rmse_std", float("nan"))
                ),
                "test_mae_mean": float(
                    summary_row.get("test_mae_mean", row["test_mae"])
                ),
                "test_mae_std": float(
                    summary_row.get("test_mae_std", float("nan"))
                ),
                "val_mse_mean": float(
                    summary_row.get("val_mse_mean", row["val_mse"])
                ),
                "n_runs": int(summary_row.get("n_runs", 1)),
            }
        )
    return pd.DataFrame(rows)


def load_manifest(tables_dir: Path, run_id: str) -> dict[str, Any]:
    """Tải manifest của từng lượt chạy được tạo bởi ExperimentRunner."""
    path = tables_dir / f"manifest_{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest for best run: {path}")
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def checkpoint_path(output_root: Path, manifest: dict[str, Any]) -> Path:
    """Xác định đường dẫn checkpoint, ưu tiên đường dẫn trong thư mục output hiện tại."""
    run_id = manifest["run_id"]
    local_path = output_root / "checkpoints" / f"best_model_{run_id}.pt"
    if local_path.exists():
        return local_path
    recorded = Path(manifest["checkpoint_path"])
    if recorded.exists():
        return recorded
    return local_path


def load_best_model(
    baseline: dict[str, Any],
    output_root: Path,
    best_row: pd.Series,
    device: torch.device,
) -> torch.nn.Module:
    """Tải checkpoint tốt nhất đã chọn, tự động huấn luyện tái lập tại chỗ nếu file chưa có trên đĩa."""
    tables = output_root / "tables"
    checkpoints = output_root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(tables, str(best_row["run_id"]))
    model_config = manifest["resolved_config"]["model"]
    model = FunctionApproximatorMLP(model_config, baseline["data"]).to(device)
    path = checkpoint_path(output_root, manifest)

    if not path.exists():
        print(
            f"-> Checkpoint {path.name} chua co tren dia, tien hanh huan luyen tai lap..."
        )
        from src.data_generator import GaussianDataset
        from src.runner import ExperimentRunner
        from torch.utils.data import DataLoader, Subset

        train_data = GaussianDataset(ROOT / baseline["data"]["train_full_path"])
        val_data = GaussianDataset(ROOT / baseline["data"]["val_path"])
        test_data = GaussianDataset(ROOT / baseline["data"]["test_path"])
        generator = GaussianDataGenerator()

        train_size = int(best_row.get("train_size", 1000))
        seed = int(best_row["seed"])
        subset_indices = generator.fixed_train_subset_indices(
            len(train_data), train_size, baseline["data"]["subset_seed"]
        )
        loader_generator = torch.Generator().manual_seed(seed)
        train_cfg = manifest["resolved_config"]["training"]
        train_loader = DataLoader(
            Subset(train_data, subset_indices.tolist()),
            batch_size=train_cfg["batch_size"],
            shuffle=train_cfg["shuffle_train"],
            drop_last=train_cfg["drop_last"],
            generator=loader_generator,
        )
        val_loader = DataLoader(
            val_data, batch_size=train_cfg["batch_size"], shuffle=False
        )
        test_loader = DataLoader(
            test_data, batch_size=train_cfg["batch_size"], shuffle=False
        )

        runner = ExperimentRunner(baseline, device)
        runner.run_single_trial(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model_config=model_config,
            train_config=train_cfg,
            seed=seed,
            checkpoint_path=path,
        )
        print(f"-> Da huan luyen va luu checkpoint tai: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_dataset(
    model: torch.nn.Module,
    dataset: GaussianDataset,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    """Trả về các giá trị mục tiêu thực tế và giá trị dự đoán trên tập dữ liệu."""
    with torch.no_grad():
        inputs = dataset.features.to(device)
        predictions = model(inputs).cpu().numpy().reshape(-1).tolist()
    return dataset.targets.numpy().reshape(-1).tolist(), predictions


def generate_model_figures(
    baseline: dict[str, Any],
    output_root: Path,
    overall: pd.DataFrame,
    device: torch.device,
    grid_size: tuple[int, int],
) -> list[Path]:
    """Khởi tạo biểu đồ bề mặt, đồ thị tương quan parity và mặt sai số thặng dư cho mô hình tốt nhất."""
    best_row = overall.sort_values(["val_mse", "run_id"]).iloc[0]
    model = load_best_model(baseline, output_root, best_row, device)
    data_config = baseline["data"]
    generator = GaussianDataGenerator(
        mu=data_config["mu"],
        cov=data_config["cov"],
        x1_range=tuple(data_config["x1_range"]),
        x2_range=tuple(data_config["x2_range"]),
    )
    test = GaussianDataset(ROOT / data_config["test_path"])
    y_true, y_pred = predict_dataset(model, test, device)
    visualizer = Visualizer(
        output_root / "figures",
        dpi=baseline["visualization"]["dpi"],
        figure_format=baseline["visualization"]["format"],
    )
    return [
        visualizer.plot_surface_comparison(
            generator,
            model,
            device=device,
            grid_size=grid_size,
        ),
        visualizer.plot_parity_scatter(y_true, y_pred),
        visualizer.plot_residual_surface(
            generator,
            model,
            device=device,
            grid_size=grid_size,
        ),
    ]


def main() -> None:
    """Tạo các bảng tổng hợp và biểu đồ đối sánh tổng thể."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--skip-model-figures", action="store_true")
    parser.add_argument("--grid-size", nargs=2, type=int)
    args = parser.parse_args()

    baseline = load_yaml(ROOT / "configs/baseline.yaml")
    device = resolve_device(baseline)
    output_root = args.output_root
    tables = output_root / "tables"
    figures = output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    best_runs = choose_best_runs(tables)
    best_runs.to_csv(tables / "best_runs_by_validation.csv", index=False)
    overall = build_overall_summary(best_runs, tables)
    overall.to_csv(tables / "summary_overall.csv", index=False)

    visualizer = Visualizer(
        figures,
        dpi=baseline["visualization"]["dpi"],
        figure_format=baseline["visualization"]["format"],
    )
    comparison_path = visualizer.plot_overall_comparison(overall)
    created = [comparison_path]
    if not args.skip_model_figures:
        grid = tuple(args.grid_size or baseline["visualization"]["surface_grid_size"])
        created.extend(
            generate_model_figures(
                baseline,
                output_root,
                overall,
                device,
                (int(grid[0]), int(grid[1])),
            )
        )
    print(f"summary_rows={len(overall)}")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
