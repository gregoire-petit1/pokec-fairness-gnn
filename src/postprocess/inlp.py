"""Iterative Nullspace Projection (INLP) — Ravfogel et al. 2020.

Iteratively trains a linear probe to predict the sensitive attribute from
the representation, recovers its weight directions, and projects the
representation onto the orthogonal complement of those directions. After
``n_iter`` iterations, no linear probe can recover the sensitive attribute
above chance level — by construction.

Reference:
    Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M. & Goldberg, Y. (2020).
    "Null It Out: Guarding Protected Attributes by Iterative Nullspace
    Projection". ACL 2020.

Why it matters here. EOT, DPT and Reweighting all operate at the prediction
level — they do not change the model's internal representation. The leakage
metric (linear-probe AUC predicting ``s`` from embeddings or features) is
therefore *unchanged* by those methods. INLP attacks exactly this dimension:
it removes from the representation any linearly-recoverable signal about
``s``, leaving downstream classifiers free to be both fair *and* unable to
re-encode the sensitive attribute in their features.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

_INLP_THREADS = 4


def _get_orthonormal_directions(weights: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis of the rowspace of ``weights``.

    For a binary probe ``weights`` has shape ``(1, d)`` (single direction),
    for a k-way probe shape ``(k, d)``. We orthonormalise via SVD on the
    transpose, taking the left-singular vectors with non-negligible singular
    values — the columns are an orthonormal basis of the rowspace of
    ``weights`` and span exactly the directions to project out.
    """
    U, S, _ = np.linalg.svd(weights.T, full_matrices=False)
    keep = S > 1e-8
    return U[:, keep]


def inlp(
    z: np.ndarray,
    s: np.ndarray,
    n_iter: int = 10,
    seed: int = 42,
    max_iter: int = 1000,
    early_stop_acc: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ``z`` onto a subspace where a linear probe cannot predict ``s``.

    Args:
        z: Representation matrix of shape ``(n, d)``.
        s: Sensitive attribute (any cardinality), shape ``(n,)``.
        n_iter: Maximum number of probe-and-project iterations. Each iteration
            removes up to ``k`` (= number of probe classes − 1 for binary, or
            ``n_classes`` for multiclass) directions.
        seed: Random seed for the logistic-regression solver.
        max_iter: Solver max_iter for each probe fit.
        early_stop_acc: If given, stop iterating as soon as the probe's
            *training* accuracy drops below this threshold. Defaults to
            ``1 / n_classes + 0.05`` (chance + 5pp slack), which prevents
            iteration into pure noise once the sensitive signal is gone.

    Returns:
        ``(z_clean, projection_matrix)`` — the projected representation and
        the cumulative projection matrix ``P`` such that ``z_clean = z @ P``.
        ``P`` can be reused on a new sample (e.g. test set) without refitting.
    """
    z = np.asarray(z, dtype=np.float64)
    s = np.asarray(s, dtype=np.int64).ravel()
    _, d = z.shape

    P = np.eye(d, dtype=np.float64)
    z_curr = z.copy()

    unique_s = np.unique(s)
    if unique_s.size < 2:
        return z.astype(np.float32), P.astype(np.float32)
    if early_stop_acc is None:
        early_stop_acc = 1.0 / unique_s.size + 0.05

    with threadpool_limits(limits=_INLP_THREADS):
        for _ in range(n_iter):
            clf = LogisticRegression(max_iter=max_iter, random_state=seed)
            clf.fit(z_curr, s)
            train_acc = float(clf.score(z_curr, s))
            if train_acc < early_stop_acc:
                # Probe already at chance — further iterations would project
                # noise directions and strip orthogonal information.
                break
            W = clf.coef_  # (n_classes-1, d) for binary, (n_classes, d) for multiclass
            if W.ndim == 1:
                W = W.reshape(1, -1)
            U = _get_orthonormal_directions(W)  # (d, k_eff)
            if U.size == 0:
                break
            # Cumulative orthogonal projection: z' = z @ (I - U U^T)
            step = np.eye(d, dtype=np.float64) - U @ U.T
            P = P @ step
            z_curr = z_curr @ step
    return z_curr.astype(np.float32), P.astype(np.float32)


def apply_projection(z: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
    """Apply a fitted INLP projection to a new sample. Vectorised."""
    return (np.asarray(z, dtype=np.float64) @ projection_matrix).astype(np.float32)
