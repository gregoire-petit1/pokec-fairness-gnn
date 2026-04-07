"""Fairness metrics: ΔDP, ΔEO, Group AUC gap, Sensitive Attribute Leakage,
Assortative Mixing Coefficient, Counterfactual Fairness Score.

References:
    Laclau, C. et al. (2024). A Survey on Fairness for Machine Learning on Graphs.
    Dai, E. & Wang, S. (2021). Say No to Discrimination: FairGNN (WSDM 2021).
    Agarwal, C. et al. (2021). NIFTY (arXiv:2109.05228).
    Newman, M. (2003). Mixing patterns in networks. Physical Review E.
"""

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


def assortative_mixing_coefficient(
    edge_index: torch.Tensor,
    sensitive: torch.Tensor,
) -> float:
    """Assortative mixing coefficient r (Newman 2003).

    Measures the tendency of nodes to connect with others sharing the same
    sensitive attribute value. Formally:

        r = (Σ_i e_ii − Σ_i a_i b_i) / (1 − Σ_i a_i b_i)

    where e_ij is the fraction of edges running between groups i and j,
    a_i = Σ_j e_ij, and b_j = Σ_i e_ij.

    Interpretation:
        r = 1  → perfect assortativity (nodes only connect within their group)
        r = 0  → random mixing (structurally fair graph)
        r = -1 → perfect disassortativity (nodes only connect across groups)

    On Pokec-z the survey reports r ≈ 0.87 w.r.t. region (Laclau et al., 2024),
    indicating strong structural bias that GNNs can exploit.

    Args:
        edge_index: Edge index tensor of shape ``[2, E]``.
        sensitive: Binary sensitive attribute tensor of shape ``[N]``.

    Returns:
        Assortative mixing coefficient r ∈ [-1, 1].
    """
    src, dst = edge_index[0], edge_index[1]
    m = float(edge_index.shape[1])
    if m == 0:
        return 0.0

    groups = sensitive.unique()
    k = len(groups)

    # e_ij = fraction of edges from group i to group j
    e = torch.zeros(k, k, dtype=torch.float)
    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            mask = (sensitive[src] == gi) & (sensitive[dst] == gj)
            e[i, j] = mask.sum().float() / m

    a = e.sum(dim=1)  # row sums
    b = e.sum(dim=0)  # col sums

    trace_e = torch.diag(e).sum()
    sum_ab = (a * b).sum()

    denom = 1.0 - sum_ab.item()
    if abs(denom) < 1e-10:
        # Perfectly assortative or degenerate graph
        return 1.0

    r = (trace_e.item() - sum_ab.item()) / denom
    return float(r)


def counterfactual_fairness_score(
    embeddings: torch.Tensor,
    sensitive: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    seed: int = 42,
) -> float:
    """Counterfactual fairness score (inspired by NIFTY, Agarwal et al. 2021).

    Measures the fraction of test nodes whose downstream prediction changes
    when the sensitive attribute is flipped (0→1 or 1→0), while keeping the
    rest of the embedding unchanged.

    Implementation:
        1. Augment each node's embedding with its sensitive attribute value
           as an extra dimension: z_aug = [z | s].
        2. Train a logistic regression classifier on the augmented train-set
           embeddings.
        3. For each test node, compute predictions with the original sensitive
           value and with the flipped value.
        4. Return the fraction of test nodes where the two predictions differ.

    A score of 0 means the model is fully counterfactually fair w.r.t. the
    sensitive attribute (no test node changes prediction). A score of 1 means
    every test node would be affected by a change in sensitive attribute.

    Note: This metric tests *potential* sensitivity of a downstream classifier
    to the sensitive attribute via the embedding space. It is complementary to
    the RB/leakage metric: leakage measures *recoverability* of the sensitive
    attribute; counterfactual fairness measures *decision impact*.

    Args:
        embeddings: Frozen node embedding matrix of shape ``[N, d]``.
        sensitive: Binary sensitive attribute tensor of shape ``[N]``.
        labels: Ground-truth label tensor of shape ``[N]`` (for training the probe).
        train_mask: Boolean mask for training nodes (shape ``[N]``).
        test_mask: Boolean mask for test nodes (shape ``[N]``).
        seed: Random seed for the logistic regression solver.

    Returns:
        Fraction of test nodes whose predicted label changes when sensitive
        attribute is flipped (float in [0, 1]).
    """
    Z = np.array(embeddings.detach().cpu().tolist(), dtype=np.float32)
    s = np.array(sensitive.cpu().tolist(), dtype=np.float32).reshape(-1, 1)
    y = np.array(labels.cpu().tolist(), dtype=np.int64)

    # Augment embeddings with sensitive attribute
    Z_aug = np.concatenate([Z, s], axis=1)
    # Counterfactual: flip the sensitive attribute
    s_flip = 1.0 - s
    Z_flip = np.concatenate([Z, s_flip], axis=1)

    train_idx = np.where(np.array(train_mask.cpu().tolist()))[0]
    test_idx = np.where(np.array(test_mask.cpu().tolist()))[0]

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(Z_aug[train_idx], y[train_idx])

    pred_orig = clf.predict(Z_aug[test_idx])
    pred_flip = clf.predict(Z_flip[test_idx])

    violation_rate = float((pred_orig != pred_flip).mean())
    return violation_rate


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
