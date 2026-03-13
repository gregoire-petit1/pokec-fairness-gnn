import torch
from src.robustness.perturbations import add_feature_noise, drop_edges


def test_feature_noise_shape_preserved():
    x = torch.ones(100, 16)
    noisy = add_feature_noise(x, sigma=0.3, seed=42)
    assert noisy.shape == x.shape
    assert not torch.allclose(noisy, x)


def test_edge_drop_reduces_edges():
    edge_index = torch.randint(0, 50, (2, 200))
    dropped = drop_edges(edge_index, rate=0.3, seed=42)
    assert dropped.shape[1] < edge_index.shape[1]
    assert abs(dropped.shape[1] / edge_index.shape[1] - 0.7) < 0.05
