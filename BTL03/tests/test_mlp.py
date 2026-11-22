"""Kiểm thử Dataset Gaussian và hai kiến trúc MLP Nhóm 1."""

import numpy as np
import pytest
import torch

from src.datasets import BivariateNormalDataset
from src.models import FuncApxmMLP, MNISTMLP


def test_gaussian_grid_matches_legacy_shape_density_and_seeded_sampling():
    """Lưới gốc có 1024/100 điểm và mẫu A tái lập đúng API RandomState."""
    train_full = BivariateNormalDataset(split="train", sample_count=1000)
    test = BivariateNormalDataset(split="test")
    sampled = BivariateNormalDataset(split="train", sample_count=800, seed=42)

    x1, x2 = np.mgrid[-3.0:3.0:32j, -3.0:3.0:32j]
    points = np.column_stack([x1.flat, x2.flat])
    density = np.exp(-0.5 * np.square(points).sum(axis=1)) / (2.0 * np.pi)
    indices = np.random.RandomState(42).choice(range(1, 1001), size=800, replace=False)

    assert len(train_full.full_inputs) == 1024
    assert len(test) == 100
    assert len(sampled) == 800
    assert torch.equal(sampled.inputs, torch.tensor(points[indices], dtype=torch.float32))
    assert torch.allclose(
        sampled.targets[:, 0], torch.tensor(density[indices], dtype=torch.float32)
    )
    assert sampled[0][0].shape == (2,) and sampled[0][1].shape == (1,)
    assert sampled[0][0].dtype == sampled[0][1].dtype == torch.float32
    # Mật độ chuẩn độc lập đúng với công thức covariance đơn vị.
    reference = np.exp(-0.5 * np.square(points).sum(axis=1)) / (2.0 * np.pi)
    assert np.allclose(train_full.full_targets[:, 0].numpy(), reference.astype(np.float32))


def test_func_apxm_mlp_shape_and_keras_parameter_formula():
    """MLP Gaussian trả (N,1), sigmoid và 4B+1 tham số."""
    model = FuncApxmMLP(hidden_units=16)
    values = model(torch.zeros(5, 2))
    assert values.shape == (5, 1)
    assert torch.all((values >= 0) & (values <= 1))
    assert model.count_parameters()["total_params"] == 4 * 16 + 1


@pytest.mark.parametrize("shape", [(3, 28, 28), (3, 1, 28, 28), (3, 28, 28, 1), (3, 784)])
def test_mnist_mlp_logits_probability_parameter_parity_and_fc2_feature(shape):
    """Mọi input legacy hỗ trợ đều có logits, 669706 params và hook sau ReLU."""
    model = MNISTMLP().eval()
    x = torch.rand(shape)
    assert model.count_parameters()["total_params"] == 669_706
    assert model(x).shape == (3, 10)
    probabilities = model.predict_proba(x)
    assert probabilities.shape == (3, 10)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(3), atol=1e-6)

    model.register_feature_hooks(["relu2"])
    _, features = model.extract_features(x)
    assert features["relu2"].shape == (3, 512)
    assert bool((features["relu2"] >= 0).all())
    model.clear_hooks()


def test_mnist_mlp_rejects_invalid_shape():
    """Shape ảnh không thuộc MNIST 28×28 bị từ chối tường minh."""
    with pytest.raises(ValueError, match="MNISTMLP nhận"):
        MNISTMLP()(torch.zeros(2, 3, 28, 28))


def test_sweep_csv_replaces_same_configuration_without_duplicate(tmp_path):
    """Ghi lại cấu hình A/B/C thay hàng cũ thay vì tạo hàng trùng."""
    from scripts.run_group1_mlp import Group1MLPRunner

    path = tmp_path / "sweep.csv"
    original = {
        "Sweep": "C",
        "A": 800,
        "B": 16,
        "C": 10,
        "Train_MSE_Grid_1024": 0.03,
        "Test_MSE_Grid_100": 0.04,
        "Train_Time_Sec": 1.0,
    }
    updated = {**original, "Test_MSE_Grid_100": 0.02}
    Group1MLPRunner._append_sweep_csv(path, original)
    Group1MLPRunner._append_sweep_csv(path, updated)

    import csv

    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["Test_MSE_Grid_100"] == "0.02"


def test_sweep_reuses_only_master_run_with_complete_matching_epoch_trace(tmp_path):
    """Không coi summary partial hoặc sai A/B/C là sweep hoàn chỉnh."""
    import csv
    from pathlib import Path

    from scripts.run_group1_mlp import Group1MLPRunner

    csv_root = tmp_path / "experiments/csv_logs"
    csv_root.mkdir(parents=True)
    config_root = tmp_path / "configs"
    config_root.mkdir()
    project_configs = Path(__file__).resolve().parents[1] / "configs"
    for config_name in ("mlp_func_apxm.yaml", "mlp_mnist.yaml"):
        (config_root / config_name).write_bytes((project_configs / config_name).read_bytes())
    master = csv_root / "master_experiment_runs.csv"
    with master.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["Run_ID", "Model_Name", "Member_Alias", "Epochs", "Best_Epoch", "Test_Metric"],
        )
        writer.writeheader()
        writer.writerow({
            "Run_ID": "wrong_config",
            "Model_Name": "FuncApxmMLP_C800_B128_C30_wrong",
            "Member_Alias": "Duy-DS",
            "Epochs": 30,
            "Best_Epoch": 30,
            "Test_Metric": 0.1,
        })
        writer.writerow({
            "Run_ID": "right_config_partial",
            "Model_Name": "FuncApxmMLP_C800_B128_C10_partial",
            "Member_Alias": "Duy-DS",
            "Epochs": 10,
            "Best_Epoch": 10,
            "Test_Metric": 0.2,
        })
    partial = csv_root / "right_config_partial_epochs.csv"
    with partial.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss"])
        writer.writerows([[epoch, 0.1, 0.2] for epoch in range(1, 10)])

    runner = Group1MLPRunner(tmp_path, member_alias="Duy-DS")
    assert runner._find_complete_sweep_run("C", 800, 128, 10) is None

    complete = csv_root / "right_config_complete_epochs.csv"
    with complete.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss"])
        writer.writerows([[epoch, 0.1, 0.2] for epoch in range(1, 11)])
    with master.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["Run_ID", "Model_Name", "Member_Alias", "Epochs", "Best_Epoch", "Test_Metric"],
        )
        writer.writerow({
            "Run_ID": "right_config_complete",
            "Model_Name": "FuncApxmMLP_C800_B128_C10_complete",
            "Member_Alias": "Duy-DS",
            "Epochs": 10,
            "Best_Epoch": 10,
            "Test_Metric": 0.2,
        })
    assert runner._find_complete_sweep_run("C", 800, 128, 10) is not None


def test_gaussian_sweep_scenarios_match_legacy_a_b_c_fixed_values():
    """Sweep A/B giữ C=100; sweep C giữ A=800 và B=128 như notebook gốc."""
    from pathlib import Path

    from scripts.run_group1_mlp import Group1MLPRunner

    root = Path(__file__).resolve().parents[1]
    runner = Group1MLPRunner(root)
    config = runner.func_config
    a_scenarios = [
        ("A", value, config["default_B"], config["default_C"])
        for value in config["sweeps"]["A"]
    ]
    b_scenarios = [
        ("B", config["default_A"], value, config["default_C"])
        for value in config["sweeps"]["B"]
    ]
    c_scenarios = [
        ("C", config["default_A"], config["default_B_C"], value)
        for value in config["sweeps"]["C"]
    ]
    assert len(a_scenarios) == 10 and all(row[2:] == (16, 100) for row in a_scenarios)
    assert len(b_scenarios) == 7 and all(row[1] == 800 and row[3] == 100 for row in b_scenarios)
    assert len(c_scenarios) == 19 and all(row[1:3] == (800, 128) for row in c_scenarios)


def test_all_legacy_random_indices_map_to_the_1024_point_grid():
    """Toàn bộ pool index 1..1000 của notebook nằm trong lưới 32×32."""
    dataset = BivariateNormalDataset(split="train", sample_count=1000, seed=42)
    assert tuple(dataset.full_inputs.shape) == (1024, 2)
    assert len(dataset) == 1000