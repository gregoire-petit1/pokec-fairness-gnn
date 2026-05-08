"""TabICL + INLP_composite + DPT_composite (all 3 axes) with re-injection.

Equivalent of the ULTIMATE chain in ``scripts.main_experiment`` but with INLP
applied directly to TabICL's row embeddings (``TabICLCache.row_repr``) rather
than to ``x`` brut, and the cleaned embeddings re-injected into the
``icl_predictor`` for the final prediction. The composite sensitive attribute
encodes ``gender × age_group × region`` jointly (12 cells), so a single INLP
+ a single DPT calibration covers all 3 axes — including their intersections.

Pipeline
--------
1. Fit ``TabICLClassifier`` with ``kv_cache="repr"`` on the train split.
2. Read ``cache.row_repr`` (train embeddings, post-row-interactor with
   y_train baked in via ``prepare_repr_cache``).
3. Capture *test* embeddings via a hook on ``row_interactor.forward``
   during a ``predict_proba`` pass.
4. Fit INLP on the *train* embeddings against the **composite** sensitive
   attribute, mean-pooled across the ensemble dim → projection ``P``.
5. Replace ``cache.row_repr`` (per ensemble member) with ``row_repr @ P``
   *and* hook ``row_interactor.forward`` so subsequent test forwards return
   their output projected through ``P``. Both train and test inputs to
   ``icl_predictor`` therefore live in the same projected sub-space.
6. Run ``predict_proba`` on val (for DPT calibration) and on test → final
   probabilities.
7. Apply ``apply_equal_opportunity_threshold(strategy="demographic_parity",
   sensitive_tensor=composite)`` to obtain DPT_composite predictions.
8. For each of the 5 sensitive axes (the 3 marginals + 2 intersections),
   compute accuracy, macro F1, ΔDP, ΔEO, and leakage AUC on the projected
   embeddings.

Output : a single :class:`ReinjectionCompositeResult` per (dataset, seed).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tabicl import TabICLClassifier

from src.postprocess.inlp import inlp


@dataclass
class AxisMetrics:
    """Per-axis metrics after ULTIMATE-LATENT."""

    axis: str
    delta_dp: float
    delta_eo: float
    leakage_pre: float
    leakage_post: float


@dataclass
class ReinjectionCompositeResult:
    """End-to-end ULTIMATE-LATENT metrics for one (dataset, seed) run."""

    n_train: int
    n_test: int
    embed_dim: int
    composite_n_cells: int
    acc_baseline: float
    f1_baseline: float
    acc_after_ultimate: float
    f1_after_ultimate: float
    per_axis: list[AxisMetrics] = field(default_factory=list)


def _binary_or_multiclass_auc(s_true: np.ndarray, proba: np.ndarray) -> float:
    n_classes = int(s_true.max()) + 1
    if n_classes == 2:
        return float(roc_auc_score(s_true, proba[:, 1]))
    return float(roc_auc_score(s_true, proba, multi_class="ovr", average="macro"))


def _build_composite(arrays: list[np.ndarray], cardinalities: list[int]) -> np.ndarray:
    """Encode multiple categorical arrays as a single int via mixed radix."""
    out = np.zeros_like(arrays[0])
    multiplier = 1
    for arr, k in zip(reversed(arrays), reversed(cardinalities), strict=True):
        out = out + arr * multiplier
        multiplier *= k
    return out


def _delta_dp(pred: np.ndarray, sensitive: np.ndarray) -> float:
    rates = []
    for g in np.unique(sensitive):
        mask = sensitive == g
        if mask.sum() == 0:
            continue
        rates.append(float(pred[mask].mean()))
    return float(max(rates) - min(rates)) if len(rates) >= 2 else 0.0


def _delta_eo(pred: np.ndarray, y: np.ndarray, sensitive: np.ndarray) -> float:
    """TPR by group (max−min) for binary y."""
    tprs = []
    for g in np.unique(sensitive):
        mask = (sensitive == g) & (y == 1)
        if mask.sum() == 0:
            continue
        tprs.append(float(pred[mask].mean()))
    return float(max(tprs) - min(tprs)) if len(tprs) >= 2 else 0.0


def _capture_test_embeddings(clf: TabICLClassifier, x_test: np.ndarray) -> np.ndarray:
    """Hook ``row_interactor`` during predict_proba to grab test embeddings."""
    captured: list[torch.Tensor] = []
    original = clf.model_.row_interactor.forward

    def hook(*args, **kwargs):
        out = original(*args, **kwargs)
        captured.append(out.detach().clone())
        return out

    clf.model_.row_interactor.forward = hook
    try:
        clf.predict_proba(x_test)
    finally:
        clf.model_.row_interactor.forward = original

    if not captured:
        raise RuntimeError("row_interactor was never called during predict")
    stacked = torch.cat(captured, dim=0)  # (B_total, test_size, H)
    return stacked.float().mean(dim=0).cpu().numpy().astype(np.float32)


def _calibrate_dpt_thresholds(
    proba_pos_val: np.ndarray,
    composite_val: np.ndarray,
    grid_size: int = 51,
) -> dict[int, float]:
    """Per-cell threshold calibration (DPT) on val.

    For each composite cell, sweep a threshold grid and pick the one that
    most equalises the per-cell positive-prediction rate to the global rate
    on val. Simple and fast — same spirit as
    ``src.postprocess.equal_opportunity`` but without the ModelOutput
    wrapper, since we operate directly on numpy.
    """
    grid = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    global_rate = float((proba_pos_val > 0.5).mean())

    thresholds: dict[int, float] = {}
    for cell in np.unique(composite_val):
        mask = composite_val == cell
        if mask.sum() == 0:
            thresholds[int(cell)] = 0.5
            continue
        cell_proba = proba_pos_val[mask]
        # rate per threshold : mean(proba > t)
        rates = (cell_proba[None, :] > grid[:, None]).mean(axis=1)
        best = int(np.argmin(np.abs(rates - global_rate)))
        thresholds[int(cell)] = float(grid[best])
    return thresholds


def _apply_dpt_thresholds(
    proba_pos: np.ndarray,
    composite: np.ndarray,
    thresholds: dict[int, float],
) -> np.ndarray:
    """Vectorised : per-row threshold lookup via composite cell."""
    # Build a per-row threshold tensor by mapping each composite value to its t.
    cell_to_t = np.array(
        [thresholds.get(int(c), 0.5) for c in np.unique(composite)],
        dtype=np.float32,
    )
    # Map each row's composite value to its index in unique(composite),
    # then look up the threshold.
    unique_cells, inv = np.unique(composite, return_inverse=True)
    row_t = cell_to_t[inv]
    return (proba_pos > row_t).astype(np.int64)


def run_ultimate_reinjection(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    sensitive_train: dict[str, np.ndarray],
    sensitive_val: dict[str, np.ndarray],
    sensitive_test: dict[str, np.ndarray],
    *,
    composite_axes: tuple[str, ...] = ("gender", "age_group", "region"),
    seed: int = 42,
    n_estimators: int = 4,
    n_iter_inlp: int = 15,
    device: str | None = None,
) -> ReinjectionCompositeResult:
    """End-to-end ULTIMATE-LATENT for one (dataset, seed) experiment.

    ``sensitive_*`` dicts must contain the marginals listed in
    ``composite_axes`` plus any extra axes you want metrics on (typically
    the 2 intersections). All arrays are aligned with ``x_*`` / ``y_*``.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ── 1. Build the composite sensitive attribute on each split ────────────
    cards = [int(sensitive_train[ax].max()) + 1 for ax in composite_axes]
    composite_train = _build_composite(
        [sensitive_train[ax].astype(np.int64) for ax in composite_axes], cards
    )
    composite_val = _build_composite(
        [sensitive_val[ax].astype(np.int64) for ax in composite_axes], cards
    )
    composite_test = _build_composite(
        [sensitive_test[ax].astype(np.int64) for ax in composite_axes], cards
    )
    n_cells = int(composite_train.max()) + 1

    # ── 2. Fit TabICL with KV cache for embeddings ──────────────────────────
    clf = TabICLClassifier(
        random_state=seed, device=device, n_estimators=n_estimators, kv_cache="repr"
    )
    clf.fit(x_train, y_train)

    # Baseline metrics (no INLP).
    proba_test_base = clf.predict_proba(x_test)
    pred_test_base = proba_test_base.argmax(axis=1)
    acc_baseline = float(accuracy_score(y_test, pred_test_base))
    f1_baseline = float(f1_score(y_test, pred_test_base, average="macro"))

    # ── 3. Pull cached train embeddings + capture test embeddings ──────────
    cache = next(iter(clf.model_kv_cache_.values()))
    train_emb = cache.row_repr.float().mean(dim=0).cpu().numpy().astype(np.float32)
    test_emb = _capture_test_embeddings(clf, x_test)

    # Pre-INLP leakage probes (one per axis we report on, on test embeddings).
    pre_leak: dict[str, float] = {}
    train_for_probe = train_emb
    test_for_probe = test_emb
    for axis_name, s_train_arr in sensitive_train.items():
        s_arr = s_train_arr.astype(np.int64)
        if np.unique(s_arr).size < 2:
            pre_leak[axis_name] = float("nan")
            continue
        probe = LogisticRegression(max_iter=1000, random_state=seed)
        probe.fit(train_for_probe, s_arr)
        pre_leak[axis_name] = _binary_or_multiclass_auc(
            sensitive_test[axis_name].astype(np.int64),
            probe.predict_proba(test_for_probe),
        )

    # ── 4. INLP on composite attribute using train embeddings ──────────────
    _train_clean, projection = inlp(train_emb, composite_train, n_iter=n_iter_inlp, seed=seed)

    # ── 5. Project + re-inject ──────────────────────────────────────────────
    p_torch = torch.from_numpy(projection.astype(np.float32))
    for _norm_method, c in clf.model_kv_cache_.items():
        original = c.row_repr  # (n_ens, train_size, H)
        c.row_repr = torch.einsum("eth,hk->etk", original.float(), p_torch.to(original.device)).to(
            original.dtype
        )

    original_forward = clf.model_.row_interactor.forward

    def project_hook(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        return torch.einsum("eth,hk->etk", out.float(), p_torch.to(out.device)).to(out.dtype)

    clf.model_.row_interactor.forward = project_hook
    try:
        proba_val = clf.predict_proba(x_val)
        proba_test = clf.predict_proba(x_test)
    finally:
        clf.model_.row_interactor.forward = original_forward

    proba_val_pos = (
        proba_val[:, 1] if proba_val.ndim == 2 and proba_val.shape[1] >= 2 else proba_val.ravel()
    )
    proba_test_pos = (
        proba_test[:, 1]
        if proba_test.ndim == 2 and proba_test.shape[1] >= 2
        else proba_test.ravel()
    )

    # ── 6. DPT_composite calibration on val, applied on test ───────────────
    thresholds = _calibrate_dpt_thresholds(proba_val_pos, composite_val)
    pred_test_after = _apply_dpt_thresholds(proba_test_pos, composite_test, thresholds)
    acc_after = float(accuracy_score(y_test, pred_test_after))
    f1_after = float(f1_score(y_test, pred_test_after, average="macro"))

    # ── 7. Per-axis ΔDP / ΔEO / leakage on the projected embeddings ────────
    train_emb_proj = (train_emb @ projection).astype(np.float32)
    test_emb_proj = (test_emb @ projection).astype(np.float32)
    per_axis: list[AxisMetrics] = []
    for axis_name in sensitive_train:
        s_train_arr = sensitive_train[axis_name].astype(np.int64)
        s_test_arr = sensitive_test[axis_name].astype(np.int64)

        ddp = _delta_dp(pred_test_after, s_test_arr)
        deo = _delta_eo(pred_test_after, y_test, s_test_arr)

        if np.unique(s_train_arr).size >= 2:
            probe_post = LogisticRegression(max_iter=1000, random_state=seed)
            probe_post.fit(train_emb_proj, s_train_arr)
            leak_post = _binary_or_multiclass_auc(
                s_test_arr, probe_post.predict_proba(test_emb_proj)
            )
        else:
            leak_post = float("nan")

        per_axis.append(
            AxisMetrics(
                axis=axis_name,
                delta_dp=ddp,
                delta_eo=deo,
                leakage_pre=pre_leak[axis_name],
                leakage_post=leak_post,
            )
        )

    return ReinjectionCompositeResult(
        n_train=int(train_emb.shape[0]),
        n_test=int(test_emb.shape[0]),
        embed_dim=int(train_emb.shape[1]),
        composite_n_cells=n_cells,
        acc_baseline=acc_baseline,
        f1_baseline=f1_baseline,
        acc_after_ultimate=acc_after,
        f1_after_ultimate=f1_after,
        per_axis=per_axis,
    )
