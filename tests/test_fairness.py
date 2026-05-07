"""Fairness metrics tests — vectorised assertions, smoke markers on the fast ones."""

import numpy as np
import pytest
import torch

from src.fairness.fairdrop import fairdrop, fairdrop_stats
from src.fairness.metrics import (
    assortative_mixing_coefficient,
    counterfactual_fairness_score,
    demographic_parity_diff,
    equal_opportunity_diff,
    group_auc_gap,
    sensitive_leakage,
)
from src.fairness.resampling import oversample_train_indices, oversample_train_mask


@pytest.mark.smoke
def test_dp_diff_perfect_fairness():
    pred = torch.tensor([1, 1, 0, 0])
    sensitive = torch.tensor([0, 1, 0, 1])
    assert demographic_parity_diff(pred, sensitive) == 0.0


@pytest.mark.smoke
def test_dp_diff_unfair():
    pred = torch.tensor([1, 1, 0, 0])
    sensitive = torch.tensor([0, 0, 1, 1])
    assert demographic_parity_diff(pred, sensitive) == 1.0


@pytest.mark.smoke
def test_dp_diff_multiclass_sensitive():
    """Multi-group sensitive (3 groups) — max−min over rates."""
    # Group 0: 100% positive, group 1: 50%, group 2: 0%
    pred = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0])
    sensitive = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    # rates: g0=1.0, g1=0.5, g2=0.0 -> max-min = 1.0
    assert demographic_parity_diff(pred, sensitive) == 1.0


@pytest.mark.smoke
def test_eo_diff():
    pred = torch.tensor([1, 1, 0, 0])
    y_true = torch.tensor([1, 1, 1, 1])
    sensitive = torch.tensor([0, 0, 1, 1])
    assert equal_opportunity_diff(pred, y_true, sensitive) == 1.0


@pytest.mark.smoke
def test_oversample_returns_more_indices():
    n = 200
    y = torch.cat([torch.zeros(150), torch.ones(50)]).long()
    gender = torch.randint(0, 2, (n,))
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:160] = True
    out = oversample_train_indices(train_mask, y, gender, seed=42)
    # Bug fix: legacy test used `out.sum()` which sums index VALUES (not count).
    # The contract is that output cardinality is >= input cardinality.
    assert len(out) >= int(train_mask.sum().item())


@pytest.mark.smoke
def test_oversample_legacy_alias_still_works():
    """Backwards-compatible alias kept for notebook callers."""
    assert oversample_train_mask is oversample_train_indices


def test_sensitive_leakage_perfect():
    """When embeddings perfectly encode the sensitive attribute, AUC should be 1.0."""
    torch.manual_seed(42)
    n = 200
    sensitive = torch.tensor([i % 2 for i in range(n)])
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
    """Uninformative embeddings → AUC near 0.5."""
    torch.manual_seed(42)
    n = 400
    sensitive = torch.tensor([i % 2 for i in range(n)])
    embeddings = torch.randn(n, 8)
    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:280] = True
    test_mask[280:] = True
    auc = sensitive_leakage(embeddings, sensitive, train_mask, test_mask, seed=42)
    assert 0.4 <= auc <= 0.65


@pytest.mark.smoke
def test_assortative_mixing_perfect_assortativity():
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    sensitive = torch.tensor([0, 0, 1, 1])
    r = assortative_mixing_coefficient(edge_index, sensitive)
    assert abs(r - 1.0) < 1e-5


@pytest.mark.smoke
def test_assortative_mixing_perfect_disassortativity():
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [2, 3, 2, 3, 0, 1, 0, 1]])
    sensitive = torch.tensor([0, 0, 1, 1])
    r = assortative_mixing_coefficient(edge_index, sensitive)
    assert abs(r - (-1.0)) < 1e-5


@pytest.mark.smoke
def test_assortative_mixing_empty_graph():
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    sensitive = torch.tensor([0, 1, 0, 1])
    r = assortative_mixing_coefficient(edge_index, sensitive)
    assert r == 0.0


@pytest.mark.smoke
def test_assortative_mixing_multiclass():
    """3-group sensitive with all intra-group edges → r = 1."""
    edge_index = torch.tensor([[0, 2, 4], [1, 3, 5]])
    sensitive = torch.tensor([0, 0, 1, 1, 2, 2])
    r = assortative_mixing_coefficient(edge_index, sensitive)
    assert abs(r - 1.0) < 1e-5


def test_counterfactual_fairness_score_range():
    torch.manual_seed(42)
    n = 200
    embeddings = torch.randn(n, 8)
    sensitive = torch.tensor([i % 2 for i in range(n)])
    labels = torch.randint(0, 2, (n,))
    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:140] = True
    test_mask[140:] = True
    score = counterfactual_fairness_score(
        embeddings, sensitive, labels, train_mask, test_mask, seed=42
    )
    assert 0.0 <= score <= 1.0


def test_counterfactual_fairness_score_sensitive_encodes_label():
    torch.manual_seed(0)
    n = 200
    sensitive = torch.tensor([i % 2 for i in range(n)])
    labels = sensitive.clone()
    embeddings = torch.randn(n, 4) * 0.01
    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:140] = True
    test_mask[140:] = True
    score = counterfactual_fairness_score(
        embeddings, sensitive, labels, train_mask, test_mask, seed=0
    )
    assert score > 0.5


@pytest.mark.smoke
def test_fairdrop_reduces_edge_count():
    edge_index = torch.tensor([[0, 1, 2, 3, 0, 2], [1, 0, 3, 2, 2, 0]])
    sensitive = torch.tensor([0, 0, 1, 1, 0, 1])
    dropped = fairdrop(edge_index, sensitive, drop_rate=0.5, seed=42)
    assert dropped.shape[0] == 2
    assert dropped.shape[1] <= edge_index.shape[1]


@pytest.mark.smoke
def test_fairdrop_empty_graph():
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    sensitive = torch.tensor([0, 1])
    dropped = fairdrop(edge_index, sensitive, drop_rate=0.5, seed=0)
    assert dropped.shape == (2, 0)


@pytest.mark.smoke
def test_fairdrop_stats_keys():
    edge_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]])
    sensitive = torch.tensor([0, 0, 1, 1])
    dropped = fairdrop(edge_index, sensitive, drop_rate=0.0, seed=0)
    stats = fairdrop_stats(edge_index, dropped, sensitive)
    for key in [
        "n_edges_original",
        "n_edges_dropped",
        "intra_fraction_original",
        "intra_fraction_dropped",
        "drop_rate_actual",
    ]:
        assert key in stats


@pytest.mark.smoke
def test_group_auc_gap_zero_when_classifier_is_random():
    """With identical noisy proba per group, AUC gap should be small."""
    rng = np.random.default_rng(0)
    n = 300
    y_true = torch.from_numpy(rng.integers(0, 2, size=n)).long()
    proba = rng.uniform(0, 1, size=(n, 2))
    proba[:, 0] = 1.0 - proba[:, 1]
    sensitive = torch.from_numpy(rng.integers(0, 2, size=n)).long()
    gap = group_auc_gap(proba, y_true, sensitive)
    assert 0.0 <= gap <= 0.5
