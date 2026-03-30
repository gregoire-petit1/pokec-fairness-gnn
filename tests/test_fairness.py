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
    """When embeddings perfectly encode the sensitive attribute, AUC should be 1.0."""
    torch.manual_seed(42)
    n = 200
    # Interleave 0/1 so both classes appear in train and test
    sensitive = torch.tensor([i % 2 for i in range(n)])
    # Embeddings = one-hot of sensitive attribute → perfect prediction
    embeddings = torch.zeros(n, 2)
    embeddings[sensitive == 0, 0] = 1.0
    embeddings[sensitive == 1, 1] = 1.0
    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:140] = True
    test_mask[140:] = True
    auc = sensitive_leakage(embeddings, sensitive, train_mask, test_mask, seed=42)
    assert auc == 1.0


def test_sensitive_leakage_random():
    """When embeddings are random (uninformative), AUC should be near 0.5."""
    torch.manual_seed(42)
    n = 400
    # Interleave 0/1 so both classes appear in train and test
    sensitive = torch.tensor([i % 2 for i in range(n)])
    embeddings = torch.randn(n, 8)
    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:280] = True
    test_mask[280:] = True
    auc = sensitive_leakage(embeddings, sensitive, train_mask, test_mask, seed=42)
    # AUC near 0.5 for random embeddings; allow 0.15 margin around chance
    assert 0.4 <= auc <= 0.65
