"""Fairness metrics: ΔDP, ΔEO, Group AUC gap."""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score


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
        p_g = proba[mask.numpy()]
        if len(set(y_g)) < 2:
            continue
        aucs.append(roc_auc_score(y_g, p_g, multi_class="ovr", average="macro"))
    if len(aucs) < 2:
        return 0.0
    return float(max(aucs) - min(aucs))


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
