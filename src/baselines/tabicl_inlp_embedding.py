"""TabICL + INLP applied directly on the model's row representations.

This module validates the claim that TabICL exposes its embeddings via
:class:`tabicl._model.kv_cache.TabICLCache` (the ``row_repr`` field), so INLP
can be applied as a true post-process on the model's latent representations
rather than on the raw input features ``x``. The 2-pager limit "INLP appliqué
côté TabICL est techniquement du pré-traitement" was wrong : the embeddings
are accessible.

What this module does
---------------------
* Fits :class:`TabICLClassifier` with ``kv_cache="repr"`` so the per-norm-method
  KV caches retain the row-level embeddings produced by ``row_interactor``.
* Reads ``cache.row_repr``  → tensor of shape
  ``(n_estimators_per_norm_method, train_size, embed_dim)``.
* Averages across the ensemble dimension to get one embedding per training row.
* For each sensitive axis, splits the train embeddings 50/50 (probe-train /
  probe-test) and measures the linear-probe gender-leakage AUC pre- and
  post-INLP.

The function is purely *measurement* — it does not re-inject cleaned embeddings
into the predictor (that would require a custom inference path). The point is
to demonstrate that INLP-on-embeddings is feasible and produces a competitive
or better leakage drop than INLP-on-x-brut, which justifies relaxing the
2-pager's pessimistic limit.

Conventions
-----------
* Polars / numpy / torch only. No pandas anywhere.
* No Python ``for`` loops on tensors / colonnes — the only loops are over the
  small dict of sensitive axes (≤10 keys).
* GPU CUDA by default; falls back to CPU only if ``torch.cuda.is_available()``
  is False.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tabicl import TabICLClassifier

from src.postprocess.inlp import inlp


@dataclass
class EmbeddingLeakageResult:
    """Per-axis leakage AUC, before and after INLP on TabICL embeddings."""

    axis: str
    n_train: int
    embed_dim: int
    leakage_pre: float
    leakage_post: float

    @property
    def drop(self) -> float:
        return self.leakage_pre - self.leakage_post


def _binary_or_multiclass_auc(s_true: np.ndarray, proba: np.ndarray) -> float:
    """Return ROC-AUC. Falls back to OvR for >2 classes."""
    n_classes = int(s_true.max()) + 1
    if n_classes == 2:
        # proba is (n, 2); pick positive-class column
        return float(roc_auc_score(s_true, proba[:, 1]))
    return float(roc_auc_score(s_true, proba, multi_class="ovr", average="macro"))


def fit_tabicl_with_embeddings(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int = 42,
    n_estimators: int = 4,
    device: str | None = None,
) -> tuple[TabICLClassifier, np.ndarray]:
    """Fit TabICL with KV-cache='repr' and return the (mean-pooled) train embeddings.

    Args:
        x_train: Train features ``(n_train, n_features)``, float32.
        y_train: Train labels ``(n_train,)``, int64.
        seed: Random seed forwarded to :class:`TabICLClassifier`.
        n_estimators: Ensemble size. Smaller = faster and uses less memory but
            the cache returns one embedding per ensemble member which we mean-
            pool — too small (n=1) gives a noisier embedding, too large is
            wasteful for measurement.
        device: ``"cuda:0"`` / ``"cpu"`` / etc. Defaults to first GPU if any.

    Returns:
        ``(clf, embeddings)`` where ``embeddings`` is a float32 numpy array of
        shape ``(n_train, embed_dim)`` obtained by averaging
        ``cache.row_repr`` over its ensemble dimension. ``clf`` is the fitted
        classifier (kept for caller-side usage, e.g. running ``predict_proba``
        for downstream metrics in the same fit).
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    clf = TabICLClassifier(
        random_state=seed,
        device=device,
        n_estimators=n_estimators,
        kv_cache="repr",
    )
    clf.fit(x_train, y_train)

    # Pull row_repr from the first norm method's cache.
    # Shape: (n_ensemble_members, train_size, embed_dim).
    cache = next(iter(clf.model_kv_cache_.values()))
    row_repr = cache.row_repr.detach().float().cpu().numpy()
    # Average across ensemble dim.
    embeddings = row_repr.mean(axis=0).astype(np.float32)
    return clf, embeddings


def measure_leakage_pre_post_inlp(
    embeddings: np.ndarray,
    sensitive_axes: dict[str, np.ndarray],
    *,
    seed: int = 42,
    n_iter_inlp: int = 15,
    probe_max_iter: int = 1000,
) -> list[EmbeddingLeakageResult]:
    """Probe + INLP on each sensitive axis, return per-axis leakage AUC.

    Args:
        embeddings: ``(n_train, embed_dim)`` mean-pooled TabICL row embeddings.
        sensitive_axes: dict mapping axis name → int label array of length
            ``n_train``. Multi-class supported (uses OvR macro-AUC).
        seed: Random seed for the probe-train / probe-test split shuffle, the
            INLP solver, and the LR probe.
        n_iter_inlp: Maximum INLP iterations (each strips 1 direction for
            binary, ``k`` for multi-class).
        probe_max_iter: ``max_iter`` for the LR probes (pre and post).

    Returns:
        One :class:`EmbeddingLeakageResult` per axis, in the same order as the
        input dict's iteration order.
    """
    rng = np.random.default_rng(seed)
    n_train, embed_dim = embeddings.shape
    perm = rng.permutation(n_train)
    half = n_train // 2
    pi_tr, pi_te = perm[:half], perm[half:]
    z_tr_full = embeddings[pi_tr]
    z_te_full = embeddings[pi_te]

    results: list[EmbeddingLeakageResult] = []
    for axis_name, s_arr in sensitive_axes.items():
        s = np.asarray(s_arr, dtype=np.int64).ravel()
        if s.shape[0] != n_train:
            raise ValueError(f"axis '{axis_name}' length {s.shape[0]} != n_train {n_train}")
        s_tr, s_te = s[pi_tr], s[pi_te]

        if np.unique(s_tr).size < 2 or np.unique(s_te).size < 2:
            # Degenerate split — no valid probe possible for this axis.
            results.append(
                EmbeddingLeakageResult(
                    axis=axis_name,
                    n_train=n_train,
                    embed_dim=embed_dim,
                    leakage_pre=float("nan"),
                    leakage_post=float("nan"),
                )
            )
            continue

        probe_pre = LogisticRegression(max_iter=probe_max_iter, random_state=seed)
        probe_pre.fit(z_tr_full, s_tr)
        proba_pre = probe_pre.predict_proba(z_te_full)
        auc_pre = _binary_or_multiclass_auc(s_te, proba_pre)

        z_tr_clean, projection = inlp(z_tr_full, s_tr, n_iter=n_iter_inlp, seed=seed)
        z_te_clean = (z_te_full @ projection).astype(np.float32)

        probe_post = LogisticRegression(max_iter=probe_max_iter, random_state=seed)
        probe_post.fit(z_tr_clean, s_tr)
        proba_post = probe_post.predict_proba(z_te_clean)
        auc_post = _binary_or_multiclass_auc(s_te, proba_post)

        results.append(
            EmbeddingLeakageResult(
                axis=axis_name,
                n_train=n_train,
                embed_dim=embed_dim,
                leakage_pre=auc_pre,
                leakage_post=auc_post,
            )
        )
    return results
