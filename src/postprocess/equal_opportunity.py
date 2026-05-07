"""Group-specific threshold post-processing for fair classification.

Given the probability output of any classifier and a sensitive attribute, fit
a per-group threshold so that either the positive-prediction rate (DP) or the
true-positive rate (TPR / equal opportunity) is equalised across groups.

This is a **post-process** method — applicable to frozen / pre-trained
classifiers (TabICL, locked GNN checkpoints) where in-training fairness
methods like adversarial debiasing cannot be inserted.

Reference:
    Hardt, M., Price, E. & Srebro, N. (2016). Equality of Opportunity in
    Supervised Learning. NeurIPS.
"""

from __future__ import annotations

import numpy as np

_STRATEGIES = ("demographic_parity", "equal_opportunity")


def fit_thresholds(
    proba_pos: np.ndarray,
    y_true: np.ndarray,
    sensitive: np.ndarray,
    strategy: str = "equal_opportunity",
) -> dict[int, float]:
    """Find a per-group threshold matching the global rate.

    Strategy ``"demographic_parity"`` equalises ``P(ŷ=1 | s=g)`` across groups.
    Strategy ``"equal_opportunity"`` equalises ``P(ŷ=1 | y=1, s=g)`` (the
    true-positive rate), conditional on the true label being positive — this
    is the original Hardt et al. 2016 criterion.

    Args:
        proba_pos: ``P(y=1 | x)`` for each row, shape ``(n,)``.
        y_true: Ground-truth binary labels (0/1), shape ``(n,)``.
        sensitive: Integer sensitive attribute (any cardinality), shape ``(n,)``.
        strategy: Which rate to equalise.

    Returns:
        Mapping ``group_id → threshold in [0, 1]``. A row from group ``g`` is
        predicted positive iff ``proba_pos >= thresholds[g]``.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; must be one of {_STRATEGIES}")

    proba_pos = np.asarray(proba_pos, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    sensitive = np.asarray(sensitive, dtype=np.int64).ravel()

    if strategy == "demographic_parity":
        eligible_mask = np.ones(proba_pos.size, dtype=bool)
    else:  # equal_opportunity
        eligible_mask = y_true == 1

    if eligible_mask.sum() == 0:
        # Nothing to calibrate on — fall back to default threshold for every group.
        return dict.fromkeys(np.unique(sensitive).astype(int).tolist(), 0.5)

    target_rate = float((proba_pos[eligible_mask] >= 0.5).mean())

    thresholds: dict[int, float] = {}
    # Loop over k distinct groups (k ≤ 6 on Pokec-z) — bounded, NOT a per-row loop.
    for g in np.unique(sensitive):
        mask = (sensitive == g) & eligible_mask
        n_g = int(mask.sum())
        if n_g == 0:
            thresholds[int(g)] = 0.5
            continue
        # Sort group probas descending, pick the threshold at rank k = round(target_rate * n_g)
        sorted_p = np.sort(proba_pos[mask])[::-1]
        k = int(round(target_rate * n_g))
        if k <= 0:
            thresholds[int(g)] = 1.0  # nobody passes → matches rate 0
        elif k >= n_g:
            thresholds[int(g)] = 0.0  # everybody passes → matches rate 1
        else:
            thresholds[int(g)] = float(sorted_p[k - 1])
    return thresholds


def apply_thresholds(
    proba_pos: np.ndarray,
    sensitive: np.ndarray,
    thresholds: dict[int, float],
) -> np.ndarray:
    """Apply ``thresholds[g]`` to every row from group ``g``. Vectorised."""
    proba_pos = np.asarray(proba_pos, dtype=np.float64).ravel()
    sensitive = np.asarray(sensitive, dtype=np.int64).ravel()

    # Build a per-row threshold lookup from the dict — vectorised via fancy indexing.
    if not thresholds:
        return (proba_pos >= 0.5).astype(np.int64)

    keys = np.array(list(thresholds.keys()), dtype=np.int64)
    vals = np.array(list(thresholds.values()), dtype=np.float64)
    max_g = int(max(int(sensitive.max(initial=0)), int(keys.max())))
    lookup = np.full(max_g + 1, 0.5, dtype=np.float64)
    lookup[keys] = vals
    per_row_t = lookup[sensitive]

    return (proba_pos >= per_row_t).astype(np.int64)


def calibrate_predictions(
    proba_pos_calib: np.ndarray,
    y_true_calib: np.ndarray,
    sensitive_calib: np.ndarray,
    proba_pos_eval: np.ndarray,
    sensitive_eval: np.ndarray,
    strategy: str = "equal_opportunity",
) -> tuple[np.ndarray, dict[int, float]]:
    """Fit thresholds on a calibration set, apply them to an evaluation set.

    Returns ``(predictions, fitted_thresholds)``.
    """
    thresholds = fit_thresholds(proba_pos_calib, y_true_calib, sensitive_calib, strategy)
    pred = apply_thresholds(proba_pos_eval, sensitive_eval, thresholds)
    return pred, thresholds
