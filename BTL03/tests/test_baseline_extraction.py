"""Kiểm thử parser baseline chỉ đọc outputs notebook và ghi provenance."""

import csv
from pathlib import Path

from scripts.extract_keras_baselines import NOT_RECORDED, SPECS, write_csv


def test_extractor_emits_ten_rows_and_preserves_missing_metrics(tmp_path):
    """CSV phải có đúng 10 baseline và không tự bịa output bị thiếu."""
    project_root = Path(__file__).resolve().parents[1]
    destination = tmp_path / "keras_baseline.csv"
    count = write_csv(project_root, destination)

    assert count == 10
    with destination.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert [row["exp_id"] for row in rows] == [spec.exp_id for spec in SPECS]
    assert rows[0]["total_params"] == NOT_RECORDED
    assert rows[0]["parameter_formula"] == "4*B+1; 65 only when B=16"
    assert rows[0]["notebook_printed_eval_metric"] == NOT_RECORDED
    assert rows[5]["val_loss_last_epoch"] != NOT_RECORDED
    assert rows[5]["notebook_printed_eval_loss"] == NOT_RECORDED
    assert rows[5]["input_shape_nhwc"] == "(8, 28, 28, 1)"
    assert rows[1]["notebook_printed_eval_metric_name"] == "accuracy"
    assert "same split" in rows[1]["evaluation_data_split"]
    assert rows[1]["training_time_seconds"] != NOT_RECORDED
    assert "summary_cells" in rows[1]["evidence_source"]
    assert "validation file" in rows[3]["evaluation_data_split"]
    assert "not an independent final holdout" in rows[6]["evaluation_data_split"]


def test_extractor_does_not_require_tensorflow_import():
    """Parser chỉ đọc JSON; không import hoặc thực thi notebook Keras."""
    import sys

    assert "tensorflow" not in sys.modules
    assert "keras" not in sys.modules