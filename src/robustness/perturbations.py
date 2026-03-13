"""Controlled perturbations for robustness evaluation."""

import numpy as np
import torch


def add_feature_noise(x: torch.Tensor, sigma: float, seed: int = 42) -> torch.Tensor:
    """Add Gaussian noise N(0, sigma) to all features."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    noise = torch.randn(x.shape, generator=rng) * sigma
    return x + noise


def drop_edges(edge_index: torch.Tensor, rate: float, seed: int = 42) -> torch.Tensor:
    """Randomly drop `rate` fraction of edges."""
    num_edges = edge_index.shape[1]
    num_keep = int(num_edges * (1 - rate))
    rng = np.random.default_rng(seed)
    keep_idx = rng.choice(num_edges, size=num_keep, replace=False)
    keep_idx = np.sort(keep_idx)
    return edge_index[:, keep_idx.tolist()]
