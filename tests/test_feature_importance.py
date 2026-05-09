"""Smoke tests for ``src.interpretability.feature_importance``."""

from __future__ import annotations

import numpy as np
import pytest

from src.interpretability.feature_importance import (
    correlation_with_sensitive,
    rank_features_by_coef,
)


@pytest.mark.smoke
def test_rank_features_orders_by_abs_coef():
    coefs = np.array([0.1, -0.5, 0.3, 0.0, 0.7, -0.2])
    names = [f"f{i}" for i in range(6)]
    top3 = rank_features_by_coef(coefs, names, top_k=3)
    assert top3.height == 3
    assert top3["feature"].to_list() == ["f4", "f1", "f2"]
    assert top3["abs_coef"][0] >= top3["abs_coef"][1] >= top3["abs_coef"][2]
    assert top3["rank"].to_list() == [1, 2, 3]


@pytest.mark.smoke
def test_rank_features_handles_top_k_larger_than_n():
    coefs = np.array([1.0, -2.0, 0.5])
    names = ["a", "b", "c"]
    out = rank_features_by_coef(coefs, names, top_k=10)
    assert out.height == 3


@pytest.mark.smoke
def test_rank_features_validates_size_mismatch():
    coefs = np.array([1.0, 2.0])
    names = ["a", "b", "c"]
    with pytest.raises(ValueError, match="3"):
        rank_features_by_coef(coefs, names, top_k=2)


@pytest.mark.smoke
def test_correlation_with_sensitive_perfect_match():
    rng = np.random.default_rng(0)
    n = 200
    sensitive = rng.integers(0, 2, size=n).astype(np.float64)
    # f0 = sensitive (perfect correlation), f1 = noise, f2 = anti (-1 corr).
    x = np.column_stack(
        [
            sensitive,
            rng.normal(size=n),
            -sensitive + 0.001 * rng.normal(size=n),
        ]
    )
    out = correlation_with_sensitive(x, ["f0", "f1", "f2"], ["f0", "f1", "f2"], sensitive)
    corrs = dict(zip(out["feature"].to_list(), out["corr_sensitive"].to_list()))
    assert corrs["f0"] > 0.99
    assert corrs["f2"] < -0.99
    assert abs(corrs["f1"]) < 0.2  # noise should be near zero


@pytest.mark.smoke
def test_correlation_handles_zero_variance_feature():
    n = 100
    sensitive = np.array([0, 1] * (n // 2), dtype=np.float64)
    x = np.column_stack([np.ones(n), np.linspace(0, 1, n)])
    out = correlation_with_sensitive(x, ["constant", "ramp"], ["constant", "ramp"], sensitive)
    # Constant feature has zero variance → correlation defaults to 0.
    corr_const = out.filter(out["feature"] == "constant")["corr_sensitive"][0]
    assert abs(corr_const) < 1e-6
