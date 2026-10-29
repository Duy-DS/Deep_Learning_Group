"""Generate Experiment A figures from persisted summary tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualizer import Visualizer


def main() -> None:
    """Read A summary CSV and write the required figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    summary = pd.read_csv(args.output_root / "tables/summary_exp_A.csv")
    visualizer = Visualizer(args.output_root / "figures")
    visualizer.plot_factor_error(
        summary,
        "train_size",
        "test_mse",
        "exp_A_error_vs_samples.png",
        "Training samples A",
    )
    visualizer.plot_train_validation_gap(
        summary,
        "train_size",
        "exp_A_train_validation_gap.png",
        "Training samples A",
    )
    print("Generated Experiment A figures")


if __name__ == "__main__":
    main()
