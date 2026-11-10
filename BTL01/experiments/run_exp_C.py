"""Thực thi Thực nghiệm C: Khảo sát độ sâu mạng MLP với chiều rộng cố định (fixed-width hidden-depth ablation)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_generator import GaussianDataset
from src.runner import ExperimentRunner


def load_yaml(path: Path) -> dict[str, Any]:
    """Đọc một file YAML của dự án với cơ chế phân tích cú pháp an toàn (safe parsing)."""
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve_device(config: dict[str, Any]) -> torch.device:
    """Xác định thiết bị tính toán đầu tiên khả dụng theo thứ tự ưu tiên của baseline."""
    if config["hardware"]["device"] != "auto":
        return torch.device(config["hardware"]["device"])
    for name in config["hardware"]["device_priority"]:
        if name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if name == "cpu":
            return torch.device("cpu")
    raise RuntimeError("Không tìm thấy thiết bị PyTorch khả dụng.")


def main() -> None:
    """Thực thi kịch bản Thực nghiệm C theo các tham số yêu cầu."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    baseline = load_yaml(ROOT / "configs/baseline.yaml")
    specs = load_yaml(ROOT / "configs/exp_specs.yaml")
    experiment = dict(specs["experiments"]["exp_C"])
    if args.depths is not None:
        experiment["depths"] = args.depths
    if args.seeds is not None:
        experiment["multi_seeds"] = args.seeds

    device = resolve_device(baseline)
    data_config = baseline["data"]
    train = GaussianDataset(ROOT / data_config["train_full_path"])
    val = GaussianDataset(ROOT / data_config["val_path"])
    test = GaussianDataset(ROOT / data_config["test_path"])
    runner = ExperimentRunner(baseline, device)
    runs, summary = runner.run_depth_experiment(
        train,
        val,
        test,
        experiment,
        baseline["model_baseline"],
        baseline["training_baseline"],
        output_root=args.output_root,
        epochs=args.epochs,
    )
    print(f"device={device}; runs={len(runs)}; completed="
          f"{int((runs['status'] == 'completed').sum())}; "
          f"failed={int((runs['status'] == 'failed').sum())}")
    print(f"summary_rows={len(summary)}")


if __name__ == "__main__":
    main()
