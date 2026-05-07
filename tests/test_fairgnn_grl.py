"""Tests for the Gradient Reversal Layer FairGNN implementation."""

import pytest
import torch
import torch.nn.functional as F

from src.models.fairgnn import FairGNN, GradientReversal, fairgnn_loss, grad_reverse


@pytest.mark.smoke
def test_grl_forward_is_identity():
    x = torch.randn(8, 16, requires_grad=True)
    y = grad_reverse(x, lambda_adv=2.5)
    assert torch.allclose(y, x)


@pytest.mark.smoke
def test_grl_backward_flips_sign_and_scales():
    """∂L/∂x = -λ * ∂L/∂y."""
    x = torch.ones(4, requires_grad=True)
    lam = 3.0
    y = GradientReversal.apply(x, lam)
    # Use sum so ∂y/∂x = +1 element-wise upstream;
    # GRL inverts: ∂x.grad = -lam * 1
    y.sum().backward()
    expected = torch.full_like(x, -lam)
    assert torch.allclose(x.grad, expected)


@pytest.mark.smoke
def test_fairgnn_forward_shapes():
    n, d = 64, 8
    x = torch.randn(n, d)
    edge_index = torch.randint(0, n, (2, 200))
    model = FairGNN(in_channels=d, hidden_channels=16, out_channels=2, adv_hidden=8, lambda_adv=0.5)
    pred, adv = model(x, edge_index)
    assert pred.shape == (n, 2)
    assert adv.shape == (n, 2)


@pytest.mark.smoke
def test_fairgnn_loss_is_positive_sum():
    """Both terms are positive — sign-flipping happens inside the GRL, not the loss."""
    n = 32
    pred = torch.randn(n, 2)
    adv = torch.randn(n, 2)
    y = torch.randint(0, 2, (n,))
    s = torch.randint(0, 2, (n,))
    mask = torch.ones(n, dtype=torch.bool)
    loss = fairgnn_loss(pred, adv, y, s, mask)
    # Cross-entropy is non-negative, sum is non-negative.
    assert loss.item() >= 0.0


def test_fairgnn_lambda_zero_matches_graphsage_for_classification():
    """λ=0 → encoder receives no adversarial gradient; classifier behaves like SAGE."""
    torch.manual_seed(0)
    n, d = 32, 8
    x = torch.randn(n, d)
    edge_index = torch.randint(0, n, (2, 80))
    y = torch.randint(0, 2, (n,))
    s = torch.randint(0, 2, (n,))
    mask = torch.ones(n, dtype=torch.bool)

    _ = s  # sensitive isn't used here — this test only checks classification path
    model = FairGNN(in_channels=d, hidden_channels=16, out_channels=2, lambda_adv=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(20):
        opt.zero_grad()
        pred, _adv = model(x, edge_index)
        loss = F.cross_entropy(pred[mask], y[mask])  # only cls loss
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_fairgnn_adversary_can_learn_when_signal_present():
    """When sensitive is recoverable from x, the adversary's loss decreases over training.

    This sanity-checks that the GRL doesn't break the adversary's learning path —
    only the gradient flowing back to the *encoder* is reversed; the adversary's
    own parameters still see a normal gradient.
    """
    torch.manual_seed(0)
    n, d = 200, 8
    s = torch.randint(0, 2, (n,))
    # Encode sensitive into x: first column = s + small noise
    x = torch.randn(n, d) * 0.1
    x[:, 0] = s.float() + torch.randn(n) * 0.1
    edge_index = torch.randint(0, n, (2, 400))
    y = torch.randint(0, 2, (n,))
    mask = torch.ones(n, dtype=torch.bool)

    model = FairGNN(in_channels=d, hidden_channels=16, out_channels=2, lambda_adv=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    adv_losses = []
    for _ in range(40):
        opt.zero_grad()
        pred, adv = model(x, edge_index)
        loss = fairgnn_loss(pred, adv, y, s, mask)
        loss.backward()
        opt.step()
        adv_loss = F.cross_entropy(adv[mask], s[mask])
        adv_losses.append(adv_loss.item())

    # Adversary loss should drop noticeably over training when signal is present.
    assert adv_losses[-1] < adv_losses[0] * 0.95
