"""TabICL + INLP with re-injection of cleaned embeddings into the predictor.

Extends ``tabicl_inlp_embedding`` from a leakage-only validation to a true
end-to-end pipeline: project the train/test embeddings with INLP, re-feed
them through ``icl_predictor`` to obtain final predictions, then measure
both **F1** (prediction quality) and **leakage AUC** on the cleaned
embeddings.

Pipeline
--------
1. ``TabICLClassifier.fit(x_train, y_train)`` with ``kv_cache="repr"``
   → ``cache.row_repr`` holds the training row embeddings (post-row-
   interactor, with ``y_train`` baked in via ``prepare_repr_cache``).
2. Run a custom forward on the test set, hooking
   ``row_interactor.forward`` so that we capture the un-baked test
   embeddings as they pass through.
3. Fit INLP on the (mean-pooled) train embeddings against the sensitive
   axis → projection matrix ``P``.
4. Apply ``P`` to *both* the cached train embeddings (in place) and the
   test embeddings (via the same hook) so all inputs to ``icl_predictor``
   live in the same projected sub-space.
5. Run ``icl_predictor.forward_with_repr_cache`` on the concatenated
   ``[projected_train_repr ; projected_test_repr]`` to obtain test
   logits, and softmax to probabilities.

Caveat
------
TabICL's ICL predictor was trained on un-projected representations, so
projecting all inputs introduces a distribution shift. The F1 cost of
that shift is exactly what this experiment quantifies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tabicl import TabICLClassifier

from src.postprocess.inlp import inlp


@dataclass
class ReinjectionResult:
    """End-to-end metrics : F1, accuracy retention and leakage AUC pre/post INLP."""

    axis: str
    n_train: int
    n_test: int
    embed_dim: int
    acc_baseline: float
    acc_after_inlp: float
    f1_baseline: float
    f1_after_inlp: float
    leakage_pre: float
    leakage_post: float

    @property
    def acc_drop(self) -> float:
        return self.acc_baseline - self.acc_after_inlp

    @property
    def f1_drop(self) -> float:
        return self.f1_baseline - self.f1_after_inlp

    @property
    def leakage_drop(self) -> float:
        return self.leakage_pre - self.leakage_post


def _binary_or_multiclass_auc(s_true: np.ndarray, proba: np.ndarray) -> float:
    n_classes = int(s_true.max()) + 1
    if n_classes == 2:
        return float(roc_auc_score(s_true, proba[:, 1]))
    return float(roc_auc_score(s_true, proba, multi_class="ovr", average="macro"))


def _capture_train_test_embeddings(
    clf: TabICLClassifier, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Get train embeddings from the KV cache, test embeddings via a hook.

    With ``kv_cache="repr"``, TabICL's predict path doesn't re-run
    ``row_interactor`` on the training set (those representations were
    cached during ``fit``). It only runs ``row_interactor`` on the test
    inputs and concatenates with the cached train tensor before feeding
    to ``icl_predictor``. So we read train embeddings directly from
    ``cache.row_repr`` and capture test embeddings via a forward hook.

    Returns
    -------
    (train_emb, test_emb) : np.ndarray
        Both float32, shape ``(n_train, H)`` and ``(n_test, H)``.
    """
    # Train side: cache.row_repr is (n_ensemble_members, train_size, H).
    cache = next(iter(clf.model_kv_cache_.values()))
    train_emb = cache.row_repr.float().mean(dim=0).cpu().numpy().astype(np.float32)

    # Test side: hook row_interactor during a predict_proba call. Each
    # invocation returns (B_chunk, test_size, H); we concat then mean-pool.
    captured: list[torch.Tensor] = []
    original_forward = clf.model_.row_interactor.forward

    def hook(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        captured.append(out.detach().clone())
        return out

    clf.model_.row_interactor.forward = hook
    try:
        clf.predict_proba(x_test)
    finally:
        clf.model_.row_interactor.forward = original_forward

    if not captured:
        raise RuntimeError("row_interactor was never called during predict")

    stacked = torch.cat(captured, dim=0)  # (B_total, test_size, H)
    test_emb = stacked.float().mean(dim=0).cpu().numpy().astype(np.float32)
    return train_emb, test_emb


def run_inlp_reinjection(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    sensitive_train: np.ndarray,
    sensitive_test: np.ndarray,
    *,
    axis_name: str = "gender",
    seed: int = 42,
    n_estimators: int = 4,
    n_iter_inlp: int = 15,
    device: str | None = None,
) -> ReinjectionResult:
    """Fit, capture embeddings, INLP, re-predict, measure F1 + leakage.

    Steps mirror the module docstring. Returns one
    :class:`ReinjectionResult` with both F1 and leakage AUC,
    pre- and post-INLP.

    Args
    ----
    x_train, y_train : np.ndarray
        Training features and labels.
    x_test, y_test : np.ndarray
        Test features and labels.
    sensitive_train, sensitive_test : np.ndarray
        Sensitive attribute on each side. Must align with ``y_*``.
    axis_name : str
        For book-keeping in the result.
    seed : int
        Random seed forwarded to TabICL, INLP and the LR probes.
    n_estimators : int
        TabICL ensemble size.
    n_iter_inlp : int
        Maximum INLP iterations.
    device : str | None
        Defaults to ``cuda:0`` if available.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    clf = TabICLClassifier(
        random_state=seed, device=device, n_estimators=n_estimators, kv_cache="repr"
    )
    clf.fit(x_train, y_train)

    # Baseline accuracy + macro F1 (no INLP, vanilla TabICL).
    proba_baseline = clf.predict_proba(x_test)
    pred_baseline = proba_baseline.argmax(axis=1)
    acc_baseline = float(accuracy_score(y_test, pred_baseline))
    f1_baseline = float(f1_score(y_test, pred_baseline, average="macro"))

    # Capture embeddings on a separate predict_proba pass. This re-enters the
    # forward path with row_interactor hooked so we collect both train and
    # test row embeddings at once.
    train_emb, test_emb = _capture_train_test_embeddings(clf, x_test)

    # Pre-INLP probe (use the captured train embeddings to fit, score on test
    # embeddings with the same axis).
    probe_pre = LogisticRegression(max_iter=1000, random_state=seed)
    probe_pre.fit(train_emb, sensitive_train[: train_emb.shape[0]])
    leakage_pre = _binary_or_multiclass_auc(sensitive_test, probe_pre.predict_proba(test_emb))

    # INLP on (train_emb, sensitive_train).
    s_tr = sensitive_train[: train_emb.shape[0]]
    _train_clean, projection = inlp(train_emb, s_tr, n_iter=n_iter_inlp, seed=seed)
    train_emb_clean = (train_emb @ projection).astype(np.float32)
    test_emb_clean = (test_emb @ projection).astype(np.float32)

    # Post-INLP probe.
    probe_post = LogisticRegression(max_iter=1000, random_state=seed)
    probe_post.fit(train_emb_clean, s_tr)
    leakage_post = _binary_or_multiclass_auc(
        sensitive_test, probe_post.predict_proba(test_emb_clean)
    )

    # Re-injection: replace cache.row_repr (per-ensemble-member) with their
    # projected versions, and hook row_interactor so test embeddings are
    # also projected on the next predict_proba call.
    p_torch = torch.from_numpy(projection.astype(np.float32))
    for _norm_method, cache in clf.model_kv_cache_.items():
        original = cache.row_repr  # (n_ens, train_size, H)
        cache.row_repr = torch.einsum(
            "eth,hk->etk", original.float(), p_torch.to(original.device)
        ).to(original.dtype)

    original_forward = clf.model_.row_interactor.forward

    def project_hook(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        return torch.einsum("eth,hk->etk", out.float(), p_torch.to(out.device)).to(out.dtype)

    clf.model_.row_interactor.forward = project_hook
    try:
        proba_after = clf.predict_proba(x_test)
    finally:
        clf.model_.row_interactor.forward = original_forward

    pred_after = proba_after.argmax(axis=1)
    acc_after = float(accuracy_score(y_test, pred_after))
    f1_after = float(f1_score(y_test, pred_after, average="macro"))

    return ReinjectionResult(
        axis=axis_name,
        n_train=int(train_emb.shape[0]),
        n_test=int(test_emb.shape[0]),
        embed_dim=int(train_emb.shape[1]),
        acc_baseline=acc_baseline,
        acc_after_inlp=acc_after,
        f1_baseline=f1_baseline,
        f1_after_inlp=f1_after,
        leakage_pre=leakage_pre,
        leakage_post=leakage_post,
    )
