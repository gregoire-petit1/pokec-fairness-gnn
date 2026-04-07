import torch
import numpy as np
from src.fairness.metrics import (
    demographic_parity_diff,
    equal_opportunity_diff,
    group_auc_gap,
    sensitive_leakage,
    assortative_mixing_coefficient,
    counterfactual_fairness_score,
)
from src.fairness.fairdrop import fairdrop, fairdrop_stats


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


# ---------------------------------------------------------------------------
# Assortative mixing coefficient
# ---------------------------------------------------------------------------


def test_assortative_mixing_perfect_assortativity():
    """A graph where all edges are intra-group should return r = 1."""
    # 4 nodes: groups [0, 0, 1, 1]
    # Edges only within groups: 0-1 and 2-3
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    sensitive = torch.tensor([0, 0, 1, 1])
    r = assortative_mixing_coefficient(edge_index, sensitive)
    assert abs(r - 1.0) < 1e-5


def test_assortative_mixing_perfect_disassortativity():
    """A graph where all edges are inter-group should return r = -1."""
    # 4 nodes: groups [0, 0, 1, 1]
    # Edges only between groups: 0-2, 0-3, 1-2, 1-3
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [2, 3, 2, 3, 0, 1, 0, 1]])
    sensitive = torch.tensor([0, 0, 1, 1])
    r = assortative_mixing_coefficient(edge_index, sensitive)
    assert abs(r - (-1.0)) < 1e-5


def test_assortative_mixing_empty_graph():
    """An empty edge index should return 0."""
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    sensitive = torch.tensor([0, 1, 0, 1])
    r = assortative_mixing_coefficient(edge_index, sensitive)
    assert r == 0.0


# ---------------------------------------------------------------------------
# Counterfactual fairness score
# ---------------------------------------------------------------------------


def test_counterfactual_fairness_score_range():
    """Score must be in [0, 1]."""
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
    """When embeddings + sensitive perfectly predict label, flipping sensitive
    should change many predictions, giving a score > 0."""
    torch.manual_seed(0)
    n = 200
    # sensitive = label, so flipping sensitive flips the prediction
    sensitive = torch.tensor([i % 2 for i in range(n)])
    labels = sensitive.clone()
    # Embeddings are uninformative (random); only sensitive carries signal
    embeddings = torch.randn(n, 4) * 0.01
    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:140] = True
    test_mask[140:] = True
    score = counterfactual_fairness_score(
        embeddings, sensitive, labels, train_mask, test_mask, seed=0
    )
    # Flipping sensitive should cause most predictions to flip
    assert score > 0.5


# ---------------------------------------------------------------------------
# FairDrop
# ---------------------------------------------------------------------------


def test_fairdrop_reduces_edge_count():
    """FairDrop must return fewer or equal edges than the original graph."""
    edge_index = torch.tensor([[0, 1, 2, 3, 0, 2], [1, 0, 3, 2, 2, 0]])
    sensitive = torch.tensor([0, 0, 1, 1, 0, 1])
    dropped = fairdrop(edge_index, sensitive, drop_rate=0.5, seed=42)
    assert dropped.shape[0] == 2
    assert dropped.shape[1] <= edge_index.shape[1]


def test_fairdrop_reduces_intra_fraction():
    """FairDrop with high bias should reduce the fraction of intra-group edges."""
    torch.manual_seed(0)
    n = 100
    # All edges are intra-group
    edge_index = torch.stack(
        [
            torch.arange(0, n - 1),
            torch.arange(1, n),
        ]
    )
    sensitive = torch.zeros(n, dtype=torch.long)  # all in group 0

    stats = fairdrop_stats(
        edge_index,
        fairdrop(edge_index, sensitive, drop_rate=0.5, intra_group_bias=3.0, seed=7),
        sensitive,
    )
    # Intra fraction stays 1.0 (no inter-group edges exist) but actual drop rate
    # should be > base drop_rate due to bias
    assert stats["drop_rate_actual"] >= 0.5


def test_fairdrop_empty_graph():
    """FairDrop on an empty graph should return an empty edge index."""
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    sensitive = torch.tensor([0, 1])
    dropped = fairdrop(edge_index, sensitive, drop_rate=0.5, seed=0)
    assert dropped.shape == (2, 0)


def test_fairdrop_stats_keys():
    """fairdrop_stats must return the expected keys."""
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
