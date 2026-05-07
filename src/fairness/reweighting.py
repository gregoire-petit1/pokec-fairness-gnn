"""Kamiran & Calders 2012 — pre-processing reweighting for fairness.

Each training example receives a weight ``w_i = P(s_i) * P(y_i) / P(s_i, y_i)``.
Plugging these weights into the cross-entropy loss makes the empirical training
distribution behave **as if** the sensitive attribute and the label were
statistically independent — i.e. ``P(s, y) → P(s) * P(y)`` in expectation.

Reference:
    Kamiran, F. & Calders, T. (2012). Data preprocessing techniques for
    classification without discrimination. Knowledge and Information Systems.
"""

from __future__ import annotations

import numpy as np


def kamiran_calders_weights(
    y: np.ndarray,
    sensitive: np.ndarray,
) -> np.ndarray:
    """Return per-row weights matching ``P(s, y) → P(s) * P(y)``.

    Args:
        y: Integer labels of shape ``(n,)``.
        sensitive: Integer sensitive attribute of shape ``(n,)``. Any
            cardinality is supported (binary, multi-class, intersectional
            composites encoded as ``s_a * K_b + s_b``).

    Returns:
        Float weights of shape ``(n,)``, normalised so ``sum(w) == n``
        (preserves the magnitude of the empirical loss).
    """
    y = np.asarray(y, dtype=np.int64).ravel()
    sensitive = np.asarray(sensitive, dtype=np.int64).ravel()
    n = y.size

    weights = np.ones(n, dtype=np.float64)

    # Outer loop over k_s × k_y bounded distinct (sensitive, label) pairs —
    # typically ≤ 6 cells on Pokec-z. Inner work is fully vectorised.
    s_values = np.unique(sensitive)
    y_values = np.unique(y)
    for s_val in s_values:
        for y_val in y_values:
            cell_mask = (sensitive == s_val) & (y == y_val)
            n_cell = int(cell_mask.sum())
            if n_cell == 0:
                continue
            p_s = float((sensitive == s_val).mean())
            p_y = float((y == y_val).mean())
            p_sy = n_cell / n
            weights[cell_mask] = p_s * p_y / max(p_sy, 1e-12)

    # Normalise so the magnitude of the loss is unchanged
    weights *= n / weights.sum()
    return weights.astype(np.float32)
