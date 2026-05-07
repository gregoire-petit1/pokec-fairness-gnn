"""Post-processing fairness: per-group decision threshold calibration.

Reference:
    Hardt, M., Price, E., & Srebro, N. (2016). Equality of Opportunity in
    Supervised Learning. NeurIPS 2016.

Key idea:
    After training, the model produces class probabilities P(y=1|x).
    A standard classifier applies a global threshold t=0.5 to all nodes.
    Per-group calibration uses different thresholds (t_0, t_1) per sensitive
    group, chosen on the *validation set* to satisfy a fairness criterion
    (ΔDP=0 or ΔEO=0) while minimising accuracy loss.

    This is purely post-hoc: no model weights are changed.
    Consequence: the embedding leakage (RB) is unchanged, because the
    representations themselves are identical — only the decision boundary moves.

Calibration criteria implemented:
    - ``"dp"``  : Demographic Parity — equalize P(ŷ=1|s=0) = P(ŷ=1|s=1)
    - ``"eo"``  : Equal Opportunity  — equalize TPR_s=0 = TPR_s=1
    - ``"pareto"`` : Return the full Pareto front (ΔDP vs F1) over all
                     threshold pairs, for visualisation purposes.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score


# ── helpers ────────────────────────────────────────────────────────────────

def _pos_rate(proba_pos: np.ndarray, mask: np.ndarray, threshold: float) -> float:
    """Positive-prediction rate for a sub-population."""
    sub = proba_pos[mask]
    if len(sub) == 0:
        return 0.0
    return float((sub >= threshold).mean())


def _tpr(proba_pos: np.ndarray, y_true: np.ndarray, mask: np.ndarray, threshold: float) -> float:
    """True Positive Rate (recall) for a sub-population."""
    pos_mask = mask & (y_true == 1)
    sub = proba_pos[pos_mask]
    if len(sub) == 0:
        return 0.0
    return float((sub >= threshold).mean())


def _macro_f1(pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(f1_score(y_true, pred, average="macro", zero_division=0))


# ── apply thresholds ────────────────────────────────────────────────────────

def apply_group_thresholds(
    proba_pos: np.ndarray,
    sensitive: np.ndarray,
    thresholds: dict[int, float],
) -> np.ndarray:
    """Apply per-group thresholds to produce binary predictions.

    Args:
        proba_pos: P(y=1|x) for all nodes, shape [N].
        sensitive:  Sensitive attribute values, shape [N].
        thresholds: Mapping ``{group_value: threshold}``.

    Returns:
        Binary prediction array of shape [N].
    """
    pred = np.zeros(len(proba_pos), dtype=np.int64)
    for g, t in thresholds.items():
        mask = sensitive == g
        pred[mask] = (proba_pos[mask] >= t).astype(np.int64)
    return pred


# ── DP calibration ──────────────────────────────────────────────────────────

def calibrate_dp(
    proba_pos: np.ndarray,
    y_true: np.ndarray,
    sensitive: np.ndarray,
    mask: np.ndarray,
    grid_size: int = 101,
    min_f1_retention: float = 0.95,
) -> dict:
    """Find per-group thresholds that minimise ΔDP on the provided subset.

    Strategy:
        Grid-search over (t_0, t_1) ∈ [0,1]².
        Compute baseline F1 at (0.5, 0.5) and only consider solutions
        where F1 ≥ min_f1_retention × baseline_f1 — this prevents the
        degenerate solution t0=t1=0 (predict all positive, ΔDP=0 trivially).
        Among feasible solutions, select the pair that minimises |ΔDP|;
        break ties by highest macro F1.

    Args:
        proba_pos: P(y=1|x) for all nodes.
        y_true:    Ground-truth labels for all nodes.
        sensitive:  Binary sensitive attribute for all nodes.
        mask:       Boolean mask selecting the calibration set (val set).
        grid_size:  Number of threshold values to test per group (default 101).
        min_f1_retention: Fraction of baseline F1 that must be retained.
            Prevents degenerate all-positive/all-negative solutions.

    Returns:
        Dict with keys ``thresholds``, ``delta_dp``, ``macro_f1``, ``details``.
    """
    thresholds_grid = np.linspace(0.0, 1.0, grid_size)

    p_pos = proba_pos[mask]
    y_cal = y_true[mask]
    s_cal = sensitive[mask]

    mask_g0 = s_cal == 0
    mask_g1 = s_cal == 1

    # Baseline F1 at global threshold 0.5
    pred_base = np.zeros(len(p_pos), dtype=np.int64)
    pred_base[mask_g0] = (p_pos[mask_g0] >= 0.5).astype(np.int64)
    pred_base[mask_g1] = (p_pos[mask_g1] >= 0.5).astype(np.int64)
    baseline_f1 = _macro_f1(pred_base, y_cal)
    min_f1 = min_f1_retention * baseline_f1

    best_dp = np.inf
    best_f1 = -np.inf
    best_t0, best_t1 = 0.5, 0.5

    for t0 in thresholds_grid:
        rate0 = _pos_rate(p_pos, mask_g0, t0)
        for t1 in thresholds_grid:
            rate1 = _pos_rate(p_pos, mask_g1, t1)
            dp = abs(rate0 - rate1)

            pred = np.zeros(len(p_pos), dtype=np.int64)
            pred[mask_g0] = (p_pos[mask_g0] >= t0).astype(np.int64)
            pred[mask_g1] = (p_pos[mask_g1] >= t1).astype(np.int64)
            f1 = _macro_f1(pred, y_cal)

            if f1 < min_f1:
                continue

            # Prefer lower ΔDP; break ties by higher F1
            if dp < best_dp - 1e-5 or (abs(dp - best_dp) < 1e-5 and f1 > best_f1):
                best_dp, best_f1 = dp, f1
                best_t0, best_t1 = t0, t1

    thresholds = {0: float(best_t0), 1: float(best_t1)}
    return {
        "thresholds": thresholds,
        "delta_dp": float(best_dp),
        "macro_f1_val": float(best_f1),
        "criterion": "dp",
        "details": {
            "t0": float(best_t0),
            "t1": float(best_t1),
            "baseline_f1_val": float(baseline_f1),
            "rate_g0": _pos_rate(p_pos, mask_g0, best_t0),
            "rate_g1": _pos_rate(p_pos, mask_g1, best_t1),
        },
    }


# ── EO calibration ──────────────────────────────────────────────────────────

def calibrate_eo(
    proba_pos: np.ndarray,
    y_true: np.ndarray,
    sensitive: np.ndarray,
    mask: np.ndarray,
    grid_size: int = 101,
    min_f1_retention: float = 0.95,
) -> dict:
    """Find per-group thresholds that minimise ΔEO on the provided subset.

    Same strategy as ``calibrate_dp`` but optimises |TPR_g0 - TPR_g1|.
    Applies the same min_f1_retention constraint to exclude degenerate
    all-positive/all-negative solutions.

    Args:
        proba_pos: P(y=1|x) for all nodes.
        y_true:    Ground-truth labels for all nodes.
        sensitive:  Binary sensitive attribute for all nodes.
        mask:       Boolean mask selecting the calibration set (val set).
        grid_size:  Number of threshold values per group (default 101).
        min_f1_retention: Fraction of baseline F1 that must be retained.

    Returns:
        Dict with keys ``thresholds``, ``delta_eo``, ``macro_f1``, ``details``.
    """
    thresholds_grid = np.linspace(0.0, 1.0, grid_size)

    p_pos = proba_pos[mask]
    y_cal = y_true[mask]
    s_cal = sensitive[mask]

    mask_g0 = s_cal == 0
    mask_g1 = s_cal == 1

    # Baseline F1 at global threshold 0.5
    pred_base = np.zeros(len(p_pos), dtype=np.int64)
    pred_base[mask_g0] = (p_pos[mask_g0] >= 0.5).astype(np.int64)
    pred_base[mask_g1] = (p_pos[mask_g1] >= 0.5).astype(np.int64)
    baseline_f1 = _macro_f1(pred_base, y_cal)
    min_f1 = min_f1_retention * baseline_f1

    best_eo = np.inf
    best_f1 = -np.inf
    best_t0, best_t1 = 0.5, 0.5

    for t0 in thresholds_grid:
        tpr0 = _tpr(p_pos, y_cal, mask_g0, t0)
        for t1 in thresholds_grid:
            tpr1 = _tpr(p_pos, y_cal, mask_g1, t1)
            eo = abs(tpr0 - tpr1)

            pred = np.zeros(len(p_pos), dtype=np.int64)
            pred[mask_g0] = (p_pos[mask_g0] >= t0).astype(np.int64)
            pred[mask_g1] = (p_pos[mask_g1] >= t1).astype(np.int64)
            f1 = _macro_f1(pred, y_cal)

            if f1 < min_f1:
                continue

            if eo < best_eo - 1e-5 or (abs(eo - best_eo) < 1e-5 and f1 > best_f1):
                best_eo, best_f1 = eo, f1
                best_t0, best_t1 = t0, t1

    thresholds = {0: float(best_t0), 1: float(best_t1)}
    return {
        "thresholds": thresholds,
        "delta_eo": float(best_eo),
        "macro_f1_val": float(best_f1),
        "criterion": "eo",
        "details": {
            "t0": float(best_t0),
            "t1": float(best_t1),
            "baseline_f1_val": float(baseline_f1),
            "tpr_g0": _tpr(p_pos, y_cal, mask_g0, best_t0),
            "tpr_g1": _tpr(p_pos, y_cal, mask_g1, best_t1),
        },
    }


# ── Pareto front ────────────────────────────────────────────────────────────

def build_pareto_front(
    proba_pos: np.ndarray,
    y_true: np.ndarray,
    sensitive: np.ndarray,
    mask: np.ndarray,
    grid_size: int = 51,
) -> list[dict]:
    """Compute the fairness–accuracy Pareto front over all (t_0, t_1) pairs.

    Returns a list of non-dominated points sorted by ΔDP ascending.
    Each point: ``{"t0", "t1", "delta_dp", "delta_eo", "macro_f1", "accuracy"}``.

    Useful for visualising the full trade-off curve.

    Note: grid_size=51 → 2601 evaluations, runs in < 1 s on numpy.
    """
    thresholds_grid = np.linspace(0.0, 1.0, grid_size)

    p_pos = proba_pos[mask]
    y_cal = y_true[mask]
    s_cal = sensitive[mask]

    mask_g0 = s_cal == 0
    mask_g1 = s_cal == 1

    points = []
    for t0 in thresholds_grid:
        for t1 in thresholds_grid:
            pred = np.zeros(len(p_pos), dtype=np.int64)
            pred[mask_g0] = (p_pos[mask_g0] >= t0).astype(np.int64)
            pred[mask_g1] = (p_pos[mask_g1] >= t1).astype(np.int64)

            rate0 = _pos_rate(p_pos, mask_g0, t0)
            rate1 = _pos_rate(p_pos, mask_g1, t1)
            dp = abs(rate0 - rate1)

            tpr0 = _tpr(p_pos, y_cal, mask_g0, t0)
            tpr1 = _tpr(p_pos, y_cal, mask_g1, t1)
            eo = abs(tpr0 - tpr1)

            f1 = _macro_f1(pred, y_cal)
            acc = float((pred == y_cal).mean())

            points.append({"t0": float(t0), "t1": float(t1),
                           "delta_dp": dp, "delta_eo": eo,
                           "macro_f1": f1, "accuracy": acc})

    # Keep Pareto-non-dominated points (min ΔDP AND max F1)
    points.sort(key=lambda p: (p["delta_dp"], -p["macro_f1"]))
    pareto = []
    best_f1 = -np.inf
    for pt in points:
        if pt["macro_f1"] >= best_f1:
            pareto.append(pt)
            best_f1 = pt["macro_f1"]

    return pareto


# ── evaluation helper ───────────────────────────────────────────────────────

def evaluate_thresholds(
    proba_pos: np.ndarray,
    y_true: np.ndarray,
    sensitive: np.ndarray,
    mask: np.ndarray,
    thresholds: dict[int, float],
) -> dict:
    """Evaluate fairness and accuracy of per-group thresholds on a given split.

    Returns accuracy, macro F1, ΔDP, ΔEO.
    """
    p_pos = proba_pos[mask]
    y_eval = y_true[mask]
    s_eval = sensitive[mask]

    mask_g0 = s_eval == 0
    mask_g1 = s_eval == 1

    pred = apply_group_thresholds(p_pos, s_eval, thresholds)

    acc = float((pred == y_eval).mean())
    f1 = _macro_f1(pred, y_eval)

    rate0 = _pos_rate(p_pos, mask_g0, thresholds.get(0, 0.5))
    rate1 = _pos_rate(p_pos, mask_g1, thresholds.get(1, 0.5))
    dp = abs(rate0 - rate1)

    tpr0 = _tpr(p_pos, y_eval, mask_g0, thresholds.get(0, 0.5))
    tpr1 = _tpr(p_pos, y_eval, mask_g1, thresholds.get(1, 0.5))
    eo = abs(tpr0 - tpr1)

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "delta_dp": dp,
        "delta_eo": eo,
        "rate_g0": rate0,
        "rate_g1": rate1,
        "tpr_g0": tpr0,
        "tpr_g1": tpr1,
    }
