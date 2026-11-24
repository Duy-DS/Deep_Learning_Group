"""Kiểm thử top-k, nhóm đúng/sai, cặp nhầm lẫn và artifacts T0503."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate.error_analysis import ErrorAnalyzer


class FixedLogitModel(nn.Module):
    """Mô hình kiểm thử trả logits định trước từ chỉ số mẫu."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.register_buffer("logits", torch.tensor([
            [5.0, 1.0, 0.0], [0.0, 5.0, 1.0], [0.0, 4.0, 3.0], [0.0, 2.0, 5.0],
        ]))

    def forward(self, images):
        indices = images[:, 0, 0, 0].long()
        return self.logits[indices]


def _loader():
    images = torch.zeros(4, 1, 8, 8)
    images[:, 0, 0, 0] = torch.arange(4)
    labels = torch.tensor([0, 1, 2, 2])
    return DataLoader(TensorDataset(images, labels), batch_size=2)


def test_top3_probabilities_confusions_and_case_groups(tmp_path):
    analyzer = ErrorAnalyzer(FixedLogitModel(), tmp_path, ("Angry", "Disgust", "Fear"),
                             device=torch.device("cpu"))
    results = analyzer.collect(_loader())
    assert results["top_indices"].shape == (4, 3)
    assert torch.allclose(results["top_probabilities"].sum(dim=1), torch.ones(4), atol=1e-6)
    assert torch.all(results["top_probabilities"][:, :-1] >= results["top_probabilities"][:, 1:])
    summary = analyzer.summarize(results, confidence_threshold=0.8)
    assert summary["fail_cases"] == [2]
    assert summary["confusion_pairs"] == [{"true": "Fear", "predicted": "Disgust", "count": 1}]
    assert len(summary["true_high_confidence"]) == 3


def test_analysis_exports_tables_and_case_grid(tmp_path):
    analyzer = ErrorAnalyzer(FixedLogitModel(), tmp_path, ("Angry", "Disgust", "Fear"),
                             device=torch.device("cpu"))
    result = analyzer.analyze(_loader(), "PrivateTest")
    assert result["artifacts"]["cases_csv"].is_file()
    assert result["artifacts"]["pairs_csv"].is_file()
    assert result["artifacts"]["analysis_markdown"].is_file()
    assert result["artifacts"]["true_grid"].is_file()
    assert result["artifacts"]["fail_grid"].is_file()
    content = result["artifacts"]["cases_csv"].read_text(encoding="utf-8-sig")
    assert "top_k" in content and "Disgust: " in content