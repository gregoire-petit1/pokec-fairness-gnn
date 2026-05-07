"""Smoke tests for Kamiran & Calders 2012 reweighting."""

import numpy as np
import pytest

from src.fairness.reweighting import kamiran_calders_weights


@pytest.mark.smoke
def test_weights_sum_to_n():
    """Weights are normalised so the loss magnitude stays the same."""
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0, 1, 0, 1, 0, 1])
    w = kamiran_calders_weights(y, s)
    np.testing.assert_allclose(w.sum(), len(y), atol=1e-5)


@pytest.mark.smoke
def test_weights_equalise_joint_distribution():
    """After reweighting, the *weighted* P(s,y) should match P(s) * P(y)."""
    rng = np.random.default_rng(0)
    n = 1000
    s = rng.integers(0, 2, n)
    # Skewed P(y|s): y=1 more likely when s=1
    y = rng.binomial(1, 0.3 + 0.4 * s)
    w = kamiran_calders_weights(y, s)

    p_s = np.array([(s == k).mean() for k in (0, 1)])
    p_y = np.array([(y == k).mean() for k in (0, 1)])

    for s_val in (0, 1):
        for y_val in (0, 1):
            mask = (s == s_val) & (y == y_val)
            weighted_joint = w[mask].sum() / n
            target = p_s[s_val] * p_y[y_val]
            assert abs(weighted_joint - target) < 0.02, (
                f"cell (s={s_val}, y={y_val}): weighted={weighted_joint:.3f} target={target:.3f}"
            )


@pytest.mark.smoke
def test_weights_balanced_input_unchanged():
    """If P(s, y) is already P(s) * P(y), all weights ≈ 1."""
    n = 200
    s = np.array([0] * (n // 2) + [1] * (n // 2))
    y = np.array([0] * (n // 4) + [1] * (n // 4)) * 2  # quadrants of n/4 each
    y = np.array([0, 1, 0, 1] * (n // 4))  # interleave gives balanced (s,y) cells
    w = kamiran_calders_weights(y, s)
    # All cells equal → weights should all be ≈ 1.0
    assert np.allclose(w, 1.0, atol=0.02)


@pytest.mark.smoke
def test_weights_handle_multi_class():
    """Works for sensitive with k > 2 groups (e.g. age_group)."""
    rng = np.random.default_rng(1)
    n = 600
    s = rng.integers(0, 3, n)  # 3 sensitive groups
    y = rng.integers(0, 2, n)
    w = kamiran_calders_weights(y, s)
    assert w.shape == (n,)
    assert np.all(w > 0)
    np.testing.assert_allclose(w.sum(), n, atol=1e-5)
