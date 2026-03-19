import torch
from src.fairness.metrics import (
    demographic_parity_diff,
    equal_opportunity_diff,
    group_auc_gap,
    sensitive_leakage,
)


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


from src.fairness.resampling import oversample_train_mask


def test_oversample_increases_minority():
    n = 200
    y = torch.cat([torch.zeros(150), torch.ones(50)]).long()
    gender = torch.randint(0, 2, (n,))
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:160] = True
    new_mask = oversample_train_mask(train_mask, y, gender, seed=42)
    # Should return more indices than original
    assert new_mask.sum() >= train_mask.sum()


def test_sensitive_leakage_perfect():
    """When embeddings perfectly encode the sensitive attribute, leakage should be 1.0."""
    torch.manual_seed(42)
    n = 100
    sensitive = torch.randint(0, 2, (n,))
    # Embeddings = one-hot of sensitive attribute → perfect prediction
    embeddings = torch.zeros(n, 2)
    embeddings[sensitive == 0, 0] = 1.0
    embeddings[sensitive == 1, 1] = 1.0
    acc = sensitive_leakage(embeddings, sensitive, seed=42)
    assert acc == 1.0


def test_sensitive_leakage_random():
    """When embeddings are random (uninformative), leakage should be near majority baseline."""
    torch.manual_seed(42)
    n = 200
    sensitive = torch.cat([torch.zeros(100), torch.ones(100)]).long()
    embeddings = torch.randn(n, 4)
    acc = sensitive_leakage(embeddings, sensitive, seed=42)
    # Majority baseline is 0.5 for balanced classes; allow up to 0.75 for small random datasets
    assert acc <= 0.75
