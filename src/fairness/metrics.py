"""Fairness metrics: ΔDP, ΔEO, Group AUC gap, Sensitive Attribute Leakage,
Assortative Mixing Coefficient, Counterfactual Fairness Score.

All metrics are vectorised — no Python loop iterates over individual nodes,
edges, or rows. The few small loops that remain iterate over distinct group
keys (typically k = 2..6) which is bounded and not a hot path.

References:
    Laclau, C. et al. (2024). A Survey on Fairness for Machine Learning on Graphs.
    Dai, E. & Wang, S. (2021). Say No to Discrimination: FairGNN (WSDM 2021).
    Agarwal, C. et al. (2021). NIFTY (arXiv:2109.05228).
    Newman, M. (2003). Mixing patterns in networks. Physical Review E.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

# Limit BLAS/OpenMP threads for sklearn probes — see explanatory commit 338f5a4.
_PROBE_N_THREADS = 4


# ---------------------------------------------------------------------------
# Output-level fairness (predictions on test set)
# ---------------------------------------------------------------------------


def demographic_parity_diff(pred: torch.Tensor, sensitive: torch.Tensor) -> float:
    """ΔDP = max_g P(ŷ=1 | s=g) − min_g P(ŷ=1 | s=g). Vectorised via bincount."""
    pred = pred.flatten().long()
    sensitive = sensitive.flatten().long()
    n_groups = int(sensitive.max().item()) + 1
    counts = torch.bincount(sensitive, minlength=n_groups).clamp(min=1)
    sums = torch.bincount(sensitive, weights=pred.float(), minlength=n_groups)
    rates = sums / counts.float()
    # Only keep groups that actually appear in the data
    valid = torch.bincount(sensitive, minlength=n_groups) > 0
    rates = rates[valid]
    return float((rates.max() - rates.min()).item())


def equal_opportunity_diff(
    pred: torch.Tensor, y_true: torch.Tensor, sensitive: torch.Tensor
) -> float:
    """ΔEO = max_g TPR_g − min_g TPR_g, restricted to y=1. Vectorised via bincount."""
    pred = pred.flatten().long()
    y_true = y_true.flatten().long()
    sensitive = sensitive.flatten().long()
    pos_mask = y_true == 1
    s_pos = sensitive[pos_mask]
    p_pos = pred[pos_mask].float()
    if s_pos.numel() == 0:
        return 0.0
    n_groups = int(sensitive.max().item()) + 1
    counts = torch.bincount(s_pos, minlength=n_groups)
    sums = torch.bincount(s_pos, weights=p_pos, minlength=n_groups)
    valid = counts > 0
    if valid.sum().item() < 2:
        return 0.0
    tprs = sums[valid] / counts[valid].float()
    return float((tprs.max() - tprs.min()).item())


def group_auc_gap(proba: np.ndarray, y_true: torch.Tensor, sensitive: torch.Tensor) -> float:
    """max_g AUC_g − min_g AUC_g.

    sklearn's ``roc_auc_score`` is the bottleneck here and is called once per
    distinct sensitive group (typically k = 2..6). The outer loop is over a
    handful of group ids, *not* over data points.
    """
    sensitive_np = sensitive.detach().cpu().numpy()
    y_np = y_true.detach().cpu().numpy()
    is_binary = proba.ndim == 2 and proba.shape[1] == 2
    aucs: list[float] = []
    for g in np.unique(sensitive_np):  # k iterations, k typically small
        m = sensitive_np == g
        y_g = y_np[m]
        if len(np.unique(y_g)) < 2:
            continue
        p_g = proba[m]
        if is_binary:
            aucs.append(roc_auc_score(y_g, p_g[:, 1]))
        else:
            aucs.append(roc_auc_score(y_g, p_g, multi_class="ovr", average="macro"))
    if len(aucs) < 2:
        return 0.0
    return float(max(aucs) - min(aucs))


# ---------------------------------------------------------------------------
# Embedding-level fairness
# ---------------------------------------------------------------------------


def sensitive_leakage(
    embeddings: torch.Tensor,
    sensitive: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    seed: int = 42,
) -> float:
    """AUC-ROC of a logistic-regression probe on **train** embeddings,
    evaluated on **test** embeddings. Balanced sampling on the train set.

    See commit 696345c for the methodological rationale (avoid linkage bias
    from cross-set message passing).
    """
    X = embeddings.detach().cpu().numpy().astype(np.float32)
    s = sensitive.detach().cpu().numpy().astype(np.int64)
    train_idx = train_mask.detach().cpu().numpy().nonzero()[0]
    test_idx = test_mask.detach().cpu().numpy().nonzero()[0]

    X_tr, s_tr = X[train_idx], s[train_idx]
    X_te, s_te = X[test_idx], s[test_idx]

    # Balanced down-sampling to min class size — vectorised via rng.choice
    classes, counts = np.unique(s_tr, return_counts=True)
    min_count = int(counts.min())
    rng = np.random.default_rng(seed)
    # Build per-class chosen indices via list comprehension across k=2..6 classes
    bal_idx = np.concatenate(
        [rng.choice(np.flatnonzero(s_tr == c), size=min_count, replace=False) for c in classes]
    )

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    with threadpool_limits(limits=_PROBE_N_THREADS):
        clf.fit(X_tr[bal_idx], s_tr[bal_idx])
        proba = clf.predict_proba(X_te)
    # Binary → 1-D scores; multi-class → full proba matrix with one-vs-rest macro AUC.
    if proba.shape[1] == 2:
        return float(roc_auc_score(s_te, proba[:, 1]))
    return float(roc_auc_score(s_te, proba, multi_class="ovr", average="macro"))


def counterfactual_fairness_score(
    embeddings: torch.Tensor,
    sensitive: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    seed: int = 42,
) -> float:
    """Fraction of test nodes whose prediction flips when sensitive is flipped.

    Augment embeddings with sensitive as an extra dim, train an LR classifier on
    the augmented train embeddings to predict y, then compare predictions with
    s vs. (1 − s) on the test set. Inspired by NIFTY (Agarwal et al. 2021).
    """
    Z = embeddings.detach().cpu().numpy().astype(np.float32)
    s = sensitive.detach().cpu().numpy().astype(np.float32).reshape(-1, 1)
    y = labels.detach().cpu().numpy().astype(np.int64)

    Z_aug = np.concatenate([Z, s], axis=1)
    Z_flip = np.concatenate([Z, 1.0 - s], axis=1)

    train_idx = train_mask.detach().cpu().numpy().nonzero()[0]
    test_idx = test_mask.detach().cpu().numpy().nonzero()[0]

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    with threadpool_limits(limits=_PROBE_N_THREADS):
        clf.fit(Z_aug[train_idx], y[train_idx])
        pred_orig = clf.predict(Z_aug[test_idx])
        pred_flip = clf.predict(Z_flip[test_idx])
    return float((pred_orig != pred_flip).mean())


# ---------------------------------------------------------------------------
# Graph-structural fairness (no model needed)
# ---------------------------------------------------------------------------


def assortative_mixing_coefficient(
    edge_index: torch.Tensor,
    sensitive: torch.Tensor,
) -> float:
    """Newman 2003 assortative mixing coefficient r ∈ [-1, 1].

    Fully vectorised: the k×k mixing matrix is built via a single
    :func:`torch.bincount` on flattened (gi*k + gj) edge group indices.
    No Python loop iterates over groups or edges.
    """
    if edge_index.shape[1] == 0:
        return 0.0

    src, dst = edge_index[0], edge_index[1]
    sensitive_long = sensitive.long()
    src_g = sensitive_long[src]
    dst_g = sensitive_long[dst]

    k = int(sensitive_long.max().item()) + 1
    m = float(edge_index.shape[1])

    # Single vectorised pass: build flat (k*k,) histogram, reshape to (k, k).
    flat_idx = src_g * k + dst_g
    e_flat = torch.bincount(flat_idx, minlength=k * k).float() / m
    e = e_flat.reshape(k, k)

    a = e.sum(dim=1)
    b = e.sum(dim=0)
    trace_e = torch.diag(e).sum()
    sum_ab = (a * b).sum()

    denom = 1.0 - sum_ab.item()
    if abs(denom) < 1e-10:
        return 1.0
    return float((trace_e.item() - sum_ab.item()) / denom)


# ---------------------------------------------------------------------------
# Convenience aggregator
# ---------------------------------------------------------------------------


def compute_all_fairness_metrics(
    pred: torch.Tensor,
    y_true: torch.Tensor,
    sensitive: torch.Tensor,
    proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all output-level fairness metrics. Embedding-level (leakage,
    CF) are computed separately because they need embeddings + train/test masks.
    """
    out: dict[str, float] = {
        "delta_dp": demographic_parity_diff(pred, sensitive),
        "delta_eo": equal_opportunity_diff(pred, y_true, sensitive),
    }
    if proba is not None:
        out["group_auc_gap"] = group_auc_gap(proba, y_true, sensitive)
    return out
