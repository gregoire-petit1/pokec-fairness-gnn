import torch
from src.fairness.metrics import demographic_parity_diff, equal_opportunity_diff, group_auc_gap


def test_dp_diff_perfect_fairness():
    pred = torch.tensor([1, 1, 0, 0])
    sensitive = torch.tensor([0, 1, 0, 1])
    assert demographic_parity_diff(pred, sensitive) == 0.0


def test_dp_diff_unfair():
    pred = torch.tensor([1, 1, 0, 0])
    sensitive = torch.tensor([0, 0, 1, 1])
    assert demographic_parity_diff(pred, sensitive) == 1.0


def test_eo_diff():
    pred = torch.tensor([1, 1, 0, 0])
    y_true = torch.tensor([1, 1, 1, 1])
    sensitive = torch.tensor([0, 0, 1, 1])
    assert equal_opportunity_diff(pred, y_true, sensitive) == 1.0
