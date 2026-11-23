"""Kiểm thử parity tham số và luồng tensor ba CNN Nhóm 2."""

import torch

from src.models import EmotionCNN, HumanGenderCNN, MNISTCNN


def test_cnn_parameter_counts_match_saved_keras_summaries():
    """Ba kiến trúc có đúng số tham số trong output model.summary legacy."""
    assert MNISTCNN().count_parameters()["total_params"] == 1_817_610
    assert HumanGenderCNN().count_parameters()["total_params"] == 4_063_570
    assert EmotionCNN().count_parameters()["total_params"] == 19_028_287


def test_mnist_logits_and_pool3_hook_shape():
    """MNIST trả logits 10 lớp và hook pool3 cho feature map 256×3×3."""
    model = MNISTCNN().eval()
    images = torch.zeros(3, 1, 28, 28)
    model.register_feature_hooks(["pool3"])
    with torch.inference_mode():
        logits, captured = model.extract_features(images)
    assert logits.shape == (3, 10)
    assert captured["pool3"].shape == (3, 256, 3, 3)
    assert captured["pool3"].flatten(1).shape == (3, 2304)
    model.clear_hooks()


def test_gender_and_emotion_logits_have_expected_class_dimensions():
    """WIKI và FER giữ NCHW grayscale và trả logits đúng số lớp."""
    with torch.inference_mode():
        assert HumanGenderCNN().eval()(torch.zeros(1, 1, 64, 64)).shape == (1, 2)
        assert EmotionCNN().eval()(torch.zeros(1, 1, 48, 48)).shape == (1, 7)


def test_cnn_models_reject_nhwc_inputs():
    """Validator của BaseMigratedModel phát hiện NHWC trước Conv2d."""
    import pytest

    with pytest.raises(ValueError, match="NHWC"):
        MNISTCNN()(torch.zeros(2, 28, 28, 1))


def test_all_cnn_configs_parse_and_hold_legacy_baseline_hyperparameters():
    """YAML CNN có seed/optimizer/epochs tương ứng ba notebook cơ sở."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1] / "configs"
    mnist = yaml.safe_load((root / "cnn_mnist.yaml").read_text(encoding="utf-8"))
    gender = yaml.safe_load((root / "cnn_gender.yaml").read_text(encoding="utf-8"))
    emotion = yaml.safe_load((root / "cnn_emotion.yaml").read_text(encoding="utf-8"))
    assert (mnist["seed"], mnist["optimizer"], mnist["learning_rate"], mnist["epochs"]) == (42, "SGD", 0.01, 30)
    assert (gender["optimizer"], gender["learning_rate"], gender["epochs"]) == ("Adam", 0.0001, 30)
    assert (emotion["optimizer"], emotion["learning_rate"], emotion["epochs"]) == ("Adam", 0.0001, 15)
    assert emotion["normalization"] == "preserve"


def test_runner_dummy_data_rotation_feature_and_verification_figure(tmp_path):
    """Runner hỗ trợ smoke data, pool3 extraction và ảnh xoay không cần logger."""
    from scripts.run_group2_cnn import Group2CNNRunner

    runner = Group2CNNRunner.__new__(Group2CNNRunner)
    runner.project_root = tmp_path
    mnist_config = {"data_root": "data/raw/mnist", "validation_fraction": 0.2}
    train, validation, test = runner._load_mnist(mnist_config, smoke=True)
    assert len(train) + len(validation) == 24
    model = MNISTCNN().eval()
    features, labels = runner._extract_pool3(model, test, batch_size=4)
    assert features.shape == (12, 2304)
    assert labels.shape == (12,)

    gender_config = {
        "data_root": "data/raw/wiki_gender",
        "apply_legacy_rotation": True,
        "normalization": "auto",
    }
    _, wiki_validation = runner._load_gender(gender_config, smoke=True)
    image, _ = wiki_validation[0]
    assert image.shape == (1, 64, 64)
    figure_path = runner._save_wiki_verification(wiki_validation)
    assert figure_path.name == "t0204_wiki_rot90_verify.png"
    assert figure_path.is_file()