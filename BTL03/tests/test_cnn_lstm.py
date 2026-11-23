"""Kiểm thử sinh chuỗi MNIST và mô hình hồi quy CNN-LSTM."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.run_group3_cnn_lstm import Group3CNNLSTMRunner
from src.datasets import SequentialMNISTDataset
from src.models import MNISTCNNLSTM


def _make_source(samples: int = 12):
    """Tạo ảnh/nhãn nhỏ có giá trị pixel và digit dễ xác minh."""
    images = np.zeros((samples, 28, 28), dtype=np.uint8)
    for index in range(samples):
        images[index, index % 28, :,] = (index + 1) * 10
    labels = np.arange(samples, dtype=np.uint8) % 10
    return images, labels


def test_sequence_dataset_shape_zero_padding_and_target_sum():
    """Chuỗi B-T-C-H-W có zero padding và tổng label đúng theo metadata."""
    images, labels = _make_source()
    dataset = SequentialMNISTDataset(images, labels, seed=42, is_train=True)
    sequence, target = dataset[0]
    metadata = dataset.metadata(0)
    assert sequence.shape == (8, 1, 28, 28)
    assert sequence.dtype == torch.float32
    assert target.shape == (1,) and target.dtype == torch.float32
    assert metadata["frame_count"] in range(1, 9)
    assert torch.count_nonzero(sequence[metadata["frame_count"] :]) == 0
    assert target.item() == metadata["target_sum"]
    batch_x, batch_y = next(iter(DataLoader(dataset, batch_size=4)))
    assert batch_x.shape == (4, 8, 1, 28, 28)
    assert batch_y.shape == (4, 1)


def test_sequence_generation_is_repeatable_and_split_independent():
    """Cùng seed/split tái lập; train/test dùng luồng RNG tách biệt."""
    images, labels = _make_source()
    first = SequentialMNISTDataset(images, labels, seed=42, is_train=True)
    second = SequentialMNISTDataset(images, labels, seed=42, is_train=True)
    test = SequentialMNISTDataset(images, labels, seed=42, is_train=False)
    assert np.array_equal(first.source_indices, second.source_indices)
    assert np.array_equal(first.frame_counts, second.frame_counts)
    assert np.array_equal(first.source_indices, test.source_indices) is False
    assert torch.equal(first[3][0], second[3][0])


def test_cnn_lstm_parameter_parity_and_forward_shapes():
    """Kiến trúc 5D trả một tổng mỗi chuỗi và có 1,747,073 tham số."""
    model = MNISTCNNLSTM().eval()
    assert model.count_parameters()["total_params"] == 1_747_073
    with torch.inference_mode():
        prediction = model(torch.zeros(2, 8, 1, 28, 28))
    assert prediction.shape == (2, 1)
    assert prediction.dtype == torch.float32


def test_cnn_lstm_smoke_optimizer_step_has_finite_loss_and_gradients():
    """Một bước train trên chuỗi giả chạy backward/optimizer không lỗi."""
    model = MNISTCNNLSTM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    sequences = torch.zeros(1, 8, 1, 28, 28)
    targets = torch.tensor([[12.0]], dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.l1_loss(model(sequences), targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.fc3_10ways_softmax.weight.grad is not None
    optimizer.step()


@pytest.mark.parametrize(
    "bad_shape",
    [(2, 1, 28, 28), (2, 7, 1, 28, 28), (2, 8, 3, 28, 28), (2, 8, 1, 32, 32)],
)
def test_cnn_lstm_rejects_invalid_input_shape(bad_shape):
    """Forward kiểm tra rõ 5D, seq_len, grayscale và độ phân giải 28×28."""
    with pytest.raises(ValueError):
        MNISTCNNLSTM()(torch.zeros(bad_shape))


def test_runner_dataset_evidence_and_mae_helpers(tmp_path):
    """Runner sinh evidence PNG và đánh giá MAE scalar trên CPU."""
    runner = Group3CNNLSTMRunner.__new__(Group3CNNLSTMRunner)
    runner.project_root = tmp_path
    runner.config = {"seed": 42, "seq_len": 8, "figure_root": "outputs/figures"}
    figure = runner.create_dataset_evidence(smoke_test=True)
    assert figure.name == "t0206_seq_mnist_samples.png"
    assert figure.is_file()
    dataset = SequentialMNISTDataset(*_make_source(), seed=42)
    model = MNISTCNNLSTM().eval()
    value = runner._evaluate_mae(model, DataLoader(dataset, batch_size=4))
    assert isinstance(value, float) and np.isfinite(value)


def test_cnn_lstm_yaml_matches_legacy_hyperparameters():
    """Config lưu đúng sequence length, SGD, MAE, batch và epoch legacy."""
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[1] / "configs/cnn_lstm_mnist.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["seed"] == 42
    assert config["seq_len"] == 8
    assert config["optimizer"] == "SGD" and config["learning_rate"] == 0.01
    assert config["loss"] == "MAE"
    assert config["batch_size"] == 128 and config["epochs"] == 30
