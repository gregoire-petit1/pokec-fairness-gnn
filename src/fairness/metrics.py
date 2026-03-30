"""Fairness metrics: ΔDP, ΔEO, Group AUC gap, Sensitive Attribute Leakage."""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


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
    # For binary classification, use P(y=1) only; for multiclass use full proba matrix.
    n_classes = proba.shape[1] if proba.ndim == 2 else 1
    is_binary = n_classes == 2
    aucs = []
    for g in groups:
        mask = sensitive == g
        y_g = y_true[mask].tolist()
        p_g = proba[mask.tolist()]
        if len(set(y_g)) < 2:
            continue
        if is_binary:
            # sklearn expects 1D scores for binary case
            auc = roc_auc_score(y_g, p_g[:, 1])
        else:
            auc = roc_auc_score(y_g, p_g, multi_class="ovr", average="macro")
        aucs.append(auc)
    if len(aucs) < 2:
        return 0.0
    return float(max(aucs) - min(aucs))


def sensitive_leakage(
    embeddings: torch.Tensor,
    sensitive: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    seed: int = 42,
) -> float:
    """Sensitive attribute leakage via logistic regression probe.

    Measures how much demographic information is encoded in frozen node
    embeddings. Implements rigorous evaluation to avoid linkage bias:

    - The probe is trained **only on train-set embeddings** to avoid
      information leakage from GNN message passing across the train/test
      boundary (linkage bias).
    - Training uses **balanced sampling** (equal positives/negatives) so
      the probe accuracy is not inflated by group imbalance.
    - Performance is reported as **AUC-ROC** on the test-set embeddings,
      which is threshold-independent and robust to class imbalance.

    A high AUC-ROC (near 1.0) means the sensitive attribute is strongly
    encoded in the embeddings. AUC-ROC near 0.5 means the embeddings are
    uninformative about the sensitive attribute (minimal leakage).

    Args:
        embeddings: Node embedding matrix of shape ``[N, d]``.
        sensitive: Binary sensitive attribute tensor of shape ``[N]``.
        train_mask: Boolean mask for training nodes (shape ``[N]``).
        test_mask: Boolean mask for test nodes (shape ``[N]``).
        seed: Random seed for the logistic regression solver and sampler.

    Returns:
        AUC-ROC of the probe classifier on the test set (float in [0, 1]).
    """
    X_all = np.array(embeddings.detach().cpu().tolist(), dtype=np.float32)
    y_all = np.array(sensitive.cpu().tolist(), dtype=np.int64)

    # Restrict to train / test sets to avoid linkage bias
    train_idx = np.where(train_mask.cpu().tolist())[0]
    test_idx = np.where(test_mask.cpu().tolist())[0]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    # Balanced sampling in the probe training set
    classes, counts = np.unique(y_train, return_counts=True)
    min_count = counts.min()
    rng = np.random.default_rng(seed)
    balanced_idx = np.concatenate(
        [rng.choice(np.where(y_train == c)[0], size=min_count, replace=False) for c in classes]
    )
    X_train_bal = X_train[balanced_idx]
    y_train_bal = y_train[balanced_idx]

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train_bal, y_train_bal)

    # AUC-ROC on the test set (threshold-independent, robust to imbalance)
    proba_test = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba_test)
    return float(auc)


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
