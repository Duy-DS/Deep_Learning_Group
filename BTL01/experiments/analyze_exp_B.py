"""Generate Experiment B figures from persisted summary tables."""

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
    """Read B summary CSV and write the required figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    tables = args.output_root / "tables"
    figures = args.output_root / "figures"
    summary = pd.read_csv(tables / "summary_exp_B.csv")
    visualizer = Visualizer(figures)
    visualizer.plot_factor_error(
        summary,
        "width",
        "test_mse",
        "exp_B_error_vs_neurons.png",
        "Hidden width B",
    )
    visualizer.plot_train_validation_gap(
        summary,
        "width",
        "exp_B_train_validation_gap.png",
        "Hidden width B",
    )
    print("Generated Experiment B figures")


if __name__ == "__main__":
    main()
