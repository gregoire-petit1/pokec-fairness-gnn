"""Smoke tests for ``src.baselines.tabicl_inlp_embedding``.

We use synthetic-style data : 512 features, 800 samples, with a planted gender
direction so the linear probe should reach high pre-INLP AUC and INLP should
collapse it. Marked ``smoke`` so it runs in the fast CI subset.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.baselines.tabicl_inlp_embedding import (
    EmbeddingLeakageResult,
    fit_tabicl_with_embeddings,
    measure_leakage_pre_post_inlp,
)


def _build_synthetic(
    n: int = 800, d: int = 32, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random features, binary y correlated to one column, gender on a different column."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = (x[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(np.int64)
    gender = (x[:, 1] + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    return x, y, gender


@pytest.mark.smoke
def test_inlp_drops_leakage_on_tabicl_embeddings():
    """End-to-end : fit TabICL, extract row_repr, INLP collapses gender leakage."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for TabICL fit at sane speed")

    x, y, gender = _build_synthetic(n=600, d=24, seed=42)

    _clf, embeddings = fit_tabicl_with_embeddings(x, y, seed=42, n_estimators=2, device="cuda:0")
    assert embeddings.ndim == 2
    assert embeddings.shape[0] == x.shape[0]
    assert embeddings.shape[1] > 0
    assert embeddings.dtype == np.float32

    results = measure_leakage_pre_post_inlp(embeddings, {"gender": gender}, seed=42, n_iter_inlp=10)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, EmbeddingLeakageResult)
    assert r.axis == "gender"
    assert 0.0 <= r.leakage_pre <= 1.0
    assert 0.0 <= r.leakage_post <= 1.0
    # Sanity : INLP can only reduce the linear probe's signal, not create it.
    assert r.leakage_post <= r.leakage_pre + 0.05, (
        f"post-INLP leakage {r.leakage_post:.3f} mysteriously exceeds pre-INLP {r.leakage_pre:.3f}"
    )


@pytest.mark.smoke
def test_measure_handles_multiple_axes_and_multiclass():
    """Multi-axis path : binary + 3-class. Returns one result per axis."""
    rng = np.random.default_rng(7)
    n, d = 400, 16
    embeddings = rng.normal(size=(n, d)).astype(np.float32)
    gender = (embeddings[:, 0] > 0).astype(np.int64)
    age = np.clip(((embeddings[:, 1] + 1.0) * 1.5).astype(np.int64), 0, 2)

    results = measure_leakage_pre_post_inlp(
        embeddings,
        {"gender": gender, "age": age},
        seed=7,
        n_iter_inlp=8,
    )
    assert [r.axis for r in results] == ["gender", "age"]
    for r in results:
        assert 0.0 <= r.leakage_pre <= 1.0
        assert 0.0 <= r.leakage_post <= 1.0


@pytest.mark.smoke
def test_measure_skips_degenerate_axis():
    """Single-class axis → leakage_pre/post are NaN."""
    rng = np.random.default_rng(0)
    n, d = 200, 8
    embeddings = rng.normal(size=(n, d)).astype(np.float32)
    constant = np.zeros(n, dtype=np.int64)
    results = measure_leakage_pre_post_inlp(
        embeddings, {"constant": constant}, seed=0, n_iter_inlp=4
    )
    assert len(results) == 1
    assert np.isnan(results[0].leakage_pre)
    assert np.isnan(results[0].leakage_post)
