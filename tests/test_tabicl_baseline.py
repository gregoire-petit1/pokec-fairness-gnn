"""Smoke test for the TabICL baseline wrapper."""

import numpy as np
import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.smoke
def test_tabicl_predict_smoke():
    """Smoke test on synthetic tabular data: shape and probability range."""
    pytest.importorskip("tabicl")
    from src.baselines.tabicl import tabicl_predict

    rng = np.random.default_rng(0)
    n, d = 200, 16
    x = rng.standard_normal((n, d)).astype(np.float32)
    # Create a label that's recoverable from x (so TabICL has signal)
    y = (x[:, 0] + x[:, 1] > 0).astype(np.int64)
    train_idx = np.arange(140)
    test_idx = np.arange(140, n)

    device = "cuda:0" if CUDA_AVAILABLE else "cpu"
    pred, proba = tabicl_predict(x, y, train_idx, test_idx, seed=0, max_train=140, device=device)

    assert pred.shape == (60,)
    assert proba.shape == (60,)
    assert pred.dtype == np.int64
    assert (proba >= 0.0).all() and (proba <= 1.0).all()


@pytest.mark.smoke
def test_tabicl_subsamples_when_train_too_large():
    """If train_idx > max_train, subsample without crashing."""
    pytest.importorskip("tabicl")
    from src.baselines.tabicl import tabicl_predict

    rng = np.random.default_rng(0)
    n, d = 1000, 8
    x = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    train_idx = np.arange(800)
    test_idx = np.arange(800, n)

    device = "cuda:0" if CUDA_AVAILABLE else "cpu"
    pred, proba = tabicl_predict(x, y, train_idx, test_idx, seed=0, max_train=200, device=device)

    assert pred.shape == (200,)
    assert proba.shape == (200,)
