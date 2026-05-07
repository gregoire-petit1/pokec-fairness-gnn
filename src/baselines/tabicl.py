"""TabICL baseline: tabular foundation model (no graph, no in-training fairness).

Used as a control point in the fairness analysis: TabICL is pre-trained / frozen,
so no in-training fairness method (FairGNN-style adversarial debiasing) can be
applied to it. Comparing its fairness metrics to GraphSAGE / FairGNN-fixed
quantifies the *added* discrimination introduced by the graph message passing
*and* the *value* of in-training debiasing.

Reference:
    Qu, J. et al. (2025). TabICL: A Tabular Foundation Model for In-Context
    Learning on Large Data. INRIA / arXiv.
"""

from __future__ import annotations

import numpy as np
import torch
from tabicl import TabICLClassifier


def tabicl_predict(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
    max_train: int = 10_000,
    device: str | None = None,
    n_estimators: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit TabICL on a (subsampled) train set and predict on the test set.

    Args:
        x: Feature matrix of shape ``(N, d)``, float32, already normalised.
        y: Labels of shape ``(N,)``.
        train_idx: 1-D int array of training row indices into ``x``.
        test_idx: 1-D int array of test row indices into ``x``.
        seed: Random seed for the subsampler and TabICL.
        max_train: Cap on the train set size handed to TabICL — ICL context
            window is finite, very large train sets either fail or burn memory.
            10 k is a safe default on a 24 GB RTX 3090.
        device: Torch device string. Defaults to ``cuda:0`` if available, else
            ``cpu`` — TabICL >= 0.1 supports both.
        n_estimators: Number of in-context ensemble rounds (TabICL averages
            ``average_logits=True`` across them). Default 8 = TabICL package
            default. Empirical test on Pokec-z showed bumping to 32 changes
            ΔDP/Leakage by less than 1e-3 (within seed noise) at 4× the
            inference cost — the package default is well-tuned. Memory cost
            is constant in ``batch_size`` (= 8); time scales linearly with
            ``n_estimators``.

    Returns:
        Tuple ``(predictions, probabilities[:, 1])`` aligned with ``test_idx``.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    rng = np.random.default_rng(seed)
    if train_idx.size > max_train:
        train_idx = rng.choice(train_idx, size=max_train, replace=False)

    clf = TabICLClassifier(
        random_state=seed,
        device=device,
        n_estimators=n_estimators,
    )
    clf.fit(x[train_idx], y[train_idx])
    pred = clf.predict(x[test_idx])
    proba = clf.predict_proba(x[test_idx])

    # TabICL returns shape (n, n_classes) — pick column index 1 for binary tasks.
    proba_pos = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()

    return pred.astype(np.int64), proba_pos.astype(np.float32)
