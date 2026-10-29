"""Contract tests for the Experiment B width suite."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from src.data_generator import GaussianDataset
import yaml

from src.runner import ExperimentRunner


ROOT = Path(__file__).resolve().parents[1]


class ExperimentBContractTests(unittest.TestCase):
    """Verify the fixed Experiment B grid before running expensive trials."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the normative baseline and Experiment B specification."""
        with (ROOT / "configs/baseline.yaml").open(encoding="utf-8") as stream:
            cls.baseline = yaml.safe_load(stream)
        with (ROOT / "configs/exp_specs.yaml").open(encoding="utf-8") as stream:
            cls.specs = yaml.safe_load(stream)
        cls.runner = ExperimentRunner(cls.baseline, torch.device("cpu"))

    def test_width_grid_has_39_trials_and_declared_seeds(self) -> None:
        """The B grid contains 13 widths crossed with the three fixed seeds."""
        trials = self.runner.build_width_trials(self.specs["experiments"]["exp_B"])

        self.assertEqual(len(trials), 39)
        self.assertEqual(
            sorted({trial["hidden_dims"][0] for trial in trials}),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64],
        )
        self.assertEqual(
            sorted({trial["seed"] for trial in trials}), [42, 100, 2026]
        )

    def test_width_trial_resolves_baseline_train_configuration(self) -> None:
        """Every B trial varies only width and keeps the B/C train size."""
        trials = self.runner.build_width_trials(self.specs["experiments"]["exp_B"])
        trial = next(item for item in trials if item["width"] == 16 and item["seed"] == 100)

        self.assertEqual(trial["hidden_dims"], [16])
        self.assertEqual(trial["train_size"], 1000)
        self.assertEqual(trial["seed"], 100)

    def test_width_suite_writes_per_run_and_summary_tables(self) -> None:
        """The suite persists rows and aggregates only successful trial results."""
        experiment = {
            "train_size": 1,
            "neuron_widths": [1],
            "extended_widths": [2],
            "multi_seeds": [42],
        }
        train = GaussianDataset(ROOT / "data/train_full.csv")
        val = GaussianDataset(ROOT / "data/val.csv")
        test = GaussianDataset(ROOT / "data/test.csv")
        fake_result = {
            "best_epoch": 2,
            "epochs_run": 2,
            "optimizer_steps": 2,
            "training_time_seconds": 0.01,
            "train_mse": 0.1,
            "train_rmse": 0.316,
            "train_mae": 0.2,
            "val_mse": 0.2,
            "val_rmse": 0.447,
            "val_mae": 0.3,
            "test_mse": 0.3,
            "test_rmse": 0.548,
            "test_mae": 0.4,
            "history": {"train_loss": [0.2, 0.1], "val_loss": [0.3, 0.2]},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                self.runner, "run_single_trial", return_value=fake_result
            ):
                runs, summary = self.runner.run_width_experiment(
                    train,
                    val,
                    test,
                    experiment,
                    self.baseline["model_baseline"],
                    self.baseline["training_baseline"],
                    output_root=Path(tmpdir),
                )
            manifest_count = len(
                list(Path(tmpdir).glob("tables/manifest_exp_B_*.json"))
            )

        self.assertIsInstance(runs, pd.DataFrame)
        self.assertEqual(len(runs), 2)
        self.assertEqual(set(runs["status"]), {"completed"})
        self.assertEqual(set(summary["width"]), {1, 2})
        self.assertEqual(manifest_count, 2)

    def test_width_suite_forwards_epoch_override(self) -> None:
        """Smoke mode changes epochs without mutating the baseline mapping."""
        experiment = {
            "train_size": 1,
            "neuron_widths": [1],
            "extended_widths": [],
            "multi_seeds": [42],
        }
        train = GaussianDataset(ROOT / "data/train_full.csv")
        val = GaussianDataset(ROOT / "data/val.csv")
        test = GaussianDataset(ROOT / "data/test.csv")
        observed: list[int] = []

        def fake_trial(*args, **kwargs):
            observed.append(args[4]["epochs"])
            return {
                "best_epoch": 1,
                "epochs_run": 1,
                "optimizer_steps": 1,
                "training_time_seconds": 0.01,
                "train_mse": 0.1,
                "train_rmse": 0.316,
                "train_mae": 0.2,
                "val_mse": 0.2,
                "val_rmse": 0.447,
                "val_mae": 0.3,
                "test_mse": 0.3,
                "test_rmse": 0.548,
                "test_mae": 0.4,
                "history": {"train_loss": [0.2], "val_loss": [0.3]},
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(self.runner, "run_single_trial", side_effect=fake_trial):
                self.runner.run_width_experiment(
                    train,
                    val,
                    test,
                    experiment,
                    self.baseline["model_baseline"],
                    self.baseline["training_baseline"],
                    output_root=Path(tmpdir),
                    epochs=3,
                )

        self.assertEqual(observed, [3])
        self.assertEqual(self.baseline["training_baseline"]["epochs"], 1000)


if __name__ == "__main__":
    unittest.main()
