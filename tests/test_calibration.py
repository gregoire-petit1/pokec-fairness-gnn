"""Smoke tests for temperature scaling (Guo et al. 2017)."""

import pytest
import torch

from src.postprocess.calibration import (
    apply_temperature,
    calibrate_logits,
    fit_temperature,
)


@pytest.mark.smoke
def test_fit_temperature_returns_positive_scalar():
    torch.manual_seed(0)
    n, k = 200, 2
    logits = torch.randn(n, k)
    y = torch.randint(0, k, (n,))
    T = fit_temperature(logits, y)
    assert T > 0.0
    assert isinstance(T, float)


@pytest.mark.smoke
def test_overconfident_logits_get_temperature_above_one():
    """When the model is over-confident (logits scaled up), the optimal
    T should be > 1 to soften the softmax."""
    torch.manual_seed(0)
    n = 500
    # Simulate over-confident model: large-magnitude logits with some errors
    y = torch.randint(0, 2, (n,))
    base = torch.zeros(n, 2)
    base[torch.arange(n), y] = 5.0  # very high logit on the correct class
    base[torch.arange(n), 1 - y] = -5.0
    # Inject 30% errors
    err_mask = torch.rand(n) < 0.3
    base[err_mask] = base[err_mask].flip(dims=(-1,))
    T = fit_temperature(base, y)
    assert T > 1.0, f"expected T > 1 for over-confident logits, got {T:.3f}"


@pytest.mark.smoke
def test_apply_temperature_preserves_argmax():
    """Temperature scaling never changes the predicted class."""
    torch.manual_seed(0)
    logits = torch.randn(100, 5)
    proba_T1 = apply_temperature(logits, 1.0)
    proba_T2 = apply_temperature(logits, 2.0)
    proba_T05 = apply_temperature(logits, 0.5)
    assert torch.equal(proba_T1.argmax(dim=-1), proba_T2.argmax(dim=-1))
    assert torch.equal(proba_T1.argmax(dim=-1), proba_T05.argmax(dim=-1))


@pytest.mark.smoke
def test_calibrate_logits_round_trip():
    """End-to-end: fit on val, apply to val itself, returns calibrated proba."""
    torch.manual_seed(0)
    n, k = 200, 3
    logits = torch.randn(n, k)
    y = torch.randint(0, k, (n,))
    proba, T = calibrate_logits(logits, y, logits)
    assert proba.shape == (n, k)
    assert T > 0.0
    # Probabilities are valid distributions
    assert torch.allclose(proba.sum(dim=-1), torch.ones(n), atol=1e-5)
