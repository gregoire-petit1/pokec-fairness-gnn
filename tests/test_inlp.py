"""Smoke tests for INLP (Ravfogel et al. 2020)."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.postprocess.inlp import apply_projection, inlp


@pytest.mark.smoke
def test_inlp_breaks_perfect_leak():
    """When z encodes s perfectly, INLP should drive probe AUC to ~0.5."""
    rng = np.random.default_rng(0)
    n = 600
    s = rng.integers(0, 2, size=n).astype(np.int64)
    # z column 0 = s (perfect leak), other columns are noise
    z = rng.standard_normal((n, 16)).astype(np.float32)
    z[:, 0] = s.astype(np.float32) * 5.0

    # Probe BEFORE INLP — should hit ~1.0
    auc_before = roc_auc_score(
        s, LogisticRegression(max_iter=1000).fit(z, s).predict_proba(z)[:, 1]
    )
    assert auc_before > 0.95

    z_clean, _P = inlp(z, s, n_iter=10, seed=0)
    auc_after = roc_auc_score(
        s, LogisticRegression(max_iter=1000).fit(z_clean, s).predict_proba(z_clean)[:, 1]
    )
    # INLP brings probe AUC from ~1.0 to near-chance. Allow 10pp slack above 0.5
    # to absorb solver-level noise and finite-sample regression to chance.
    assert auc_after < 0.60, f"INLP failed: probe AUC still {auc_after:.3f}"


@pytest.mark.smoke
def test_inlp_preserves_orthogonal_signal():
    """Information orthogonal to s should be preserved after INLP."""
    rng = np.random.default_rng(1)
    n = 500
    # y is independent of s (uncorrelated)
    s = rng.integers(0, 2, n).astype(np.int64)
    y = rng.integers(0, 2, n).astype(np.int64)
    z = rng.standard_normal((n, 16)).astype(np.float32)
    z[:, 0] = s.astype(np.float32) * 3.0  # column 0 carries s
    z[:, 1] = y.astype(np.float32) * 3.0  # column 1 carries y, orthogonal to col 0

    z_clean, _ = inlp(z, s, n_iter=5, seed=0)

    # AFTER INLP, predicting y from z_clean should still work (col 1 is preserved)
    clf = LogisticRegression(max_iter=1000).fit(z_clean, y)
    auc_y = roc_auc_score(y, clf.predict_proba(z_clean)[:, 1])
    assert auc_y > 0.85, f"INLP destroyed orthogonal signal: y AUC = {auc_y:.3f}"


@pytest.mark.smoke
def test_inlp_handles_multiclass_sensitive():
    """Works for k > 2 sensitive groups (e.g. age_group, 3 classes)."""
    rng = np.random.default_rng(2)
    n = 600
    s = rng.integers(0, 3, n).astype(np.int64)
    # 3 dimensions encode s via one-hot, others are noise
    z = rng.standard_normal((n, 16)).astype(np.float32)
    z[:, 0] = (s == 0).astype(np.float32) * 4.0
    z[:, 1] = (s == 1).astype(np.float32) * 4.0
    z[:, 2] = (s == 2).astype(np.float32) * 4.0

    z_clean, _ = inlp(z, s, n_iter=8, seed=0)
    # Multi-class probe AUC OvR
    proba = LogisticRegression(max_iter=1000).fit(z_clean, s).predict_proba(z_clean)
    auc = roc_auc_score(s, proba, multi_class="ovr", average="macro")
    assert auc < 0.6, f"INLP failed on multi-class: AUC {auc:.3f}"


@pytest.mark.smoke
def test_apply_projection_matches_inlp_output():
    """The returned projection matrix P reproduces the projected z exactly."""
    rng = np.random.default_rng(3)
    n, d = 200, 16
    s = rng.integers(0, 2, n).astype(np.int64)
    z = rng.standard_normal((n, d)).astype(np.float32)
    z_clean, P = inlp(z, s, n_iter=3, seed=0)
    z_via_P = apply_projection(z, P)
    np.testing.assert_allclose(z_clean, z_via_P, rtol=1e-5, atol=1e-5)
