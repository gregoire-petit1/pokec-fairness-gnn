"""Fairness metrics: ΔDP, ΔEO, Group AUC gap, Sensitive Attribute Leakage."""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


def demographic_parity_diff(pred: torch.Tensor, sensitive: torch.Tensor) -> float:
    """Demographic Parity Difference: |P(y_hat=1|s=0) - P(y_hat=1|s=1)|."""
    groups = sensitive.unique()
    rates = []
    for g in groups:
        mask = sensitive == g
        rates.append(pred[mask].float().mean().item())
    return float(max(rates) - min(rates))


def equal_opportunity_diff(
    pred: torch.Tensor, y_true: torch.Tensor, sensitive: torch.Tensor
) -> float:
    """Equal Opportunity Difference: |TPR_g0 - TPR_g1| for binary y."""
    groups = sensitive.unique()
    tprs = []
    for g in groups:
        mask = (sensitive == g) & (y_true == 1)
        if mask.sum() == 0:
            continue
        tpr = (pred[mask] == 1).float().mean().item()
        tprs.append(tpr)
    if len(tprs) < 2:
        return 0.0
    return float(max(tprs) - min(tprs))


def group_auc_gap(proba: np.ndarray, y_true: torch.Tensor, sensitive: torch.Tensor) -> float:
    """Max AUC difference across sensitive groups."""
    groups = sensitive.unique().tolist()
    aucs = []
    for g in groups:
        mask = sensitive == g
        y_g = y_true[mask].tolist()
        p_g = proba[mask.tolist()]
        if len(set(y_g)) < 2:
            continue
        aucs.append(roc_auc_score(y_g, p_g, multi_class="ovr", average="macro"))
    if len(aucs) < 2:
        return 0.0
    return float(max(aucs) - min(aucs))


def sensitive_leakage(
    embeddings: torch.Tensor,
    sensitive: torch.Tensor,
    seed: int = 42,
) -> float:
    """Sensitive attribute leakage via logistic regression probe.

    Train a logistic regression classifier on frozen node embeddings to predict
    the sensitive attribute.  A high accuracy indicates that the sensitive
    attribute is encoded in the representation (high leakage); accuracy near the
    majority-class baseline indicates minimal leakage.

    This metric is widely used in the FairGNN / NIFTY literature to measure how
    much demographic information is captured by learned embeddings.

    Args:
        embeddings: Node embedding matrix of shape ``[N, d]``.
        sensitive: Binary sensitive attribute tensor of shape ``[N]``.
        seed: Random seed for the logistic regression solver.

    Returns:
        Classification accuracy of the probe classifier (float in [0, 1]).
    """
    X = embeddings.detach().cpu().tolist()
    y = sensitive.cpu().tolist()
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X, y)
    preds = clf.predict(X)
    accuracy = float(np.mean(np.array(preds) == np.array(y)))
    return accuracy


def compute_all_fairness_metrics(
    pred: torch.Tensor,
    y_true: torch.Tensor,
    sensitive: torch.Tensor,
    proba: np.ndarray | None = None,
) -> dict:
    """Compute all fairness metrics and return as dict."""
    result = {
        "delta_dp": demographic_parity_diff(pred, sensitive),
        "delta_eo": equal_opportunity_diff(pred, y_true, sensitive),
    }
    if proba is not None:
        result["group_auc_gap"] = group_auc_gap(proba, y_true, sensitive)
    return result
