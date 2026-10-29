"""Generate Experiment C figures from persisted summary tables."""

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
    """Read C summary CSV and write the required figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    summary = pd.read_csv(args.output_root / "tables/summary_exp_C.csv")
    visualizer = Visualizer(args.output_root / "figures")
    visualizer.plot_depth_convergence_time(
        summary, "exp_C_depth_convergence_time.png"
    )
    visualizer.plot_factor_error(
        summary,
        "depth",
        "test_mse",
        "exp_C_error_vs_depth.png",
        "Hidden depth C",
    )
    print("Generated Experiment C figures")


if __name__ == "__main__":
    main()
