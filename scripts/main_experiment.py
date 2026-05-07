"""End-to-end fairness experiment driver for Pokec-z.

Designed to be used as **the** entry point — both interactively from a thin
notebook shell **and** directly from the command line. Each ``run_*`` function
encapsulates one method (baseline, resampling, FairDrop, FairGNN, TabICL) and
returns a polars DataFrame row plus a model-output dict that downstream
code (multi-attribute fairness, leakage probes) can reuse.

Usage::

    # CLI (full pipeline, multi-seed for the baseline by default):
    python scripts/main_experiment.py --device cuda:0

    # Library-style:
    from scripts.main_experiment import run_all
    df = run_all(seed=42, device_str="cuda:0",
                 cfg_path="configs/experiment.yaml",
                 raw_dir_override="data/raw/pokec-z")

The driver is GPU-first (defaults to ``cuda:0`` when available) and
polars-only (no pandas anywhere — enforced by the ``ruff PD`` preset and
``tests/test_no_pandas_no_loops.py``).
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score

# Allow running from anywhere (repo root or scripts/) without PYTHONPATH games
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.loader import load_pokec_z  # noqa: E402
from src.data.preprocessing import preprocess  # noqa: E402
from src.data.splits import make_splits  # noqa: E402
from src.fairness.fairdrop import fairdrop  # noqa: E402
from src.fairness.metrics import (  # noqa: E402
    assortative_mixing_coefficient,
    demographic_parity_diff,
    equal_opportunity_diff,
    group_auc_gap,
    sensitive_leakage,
)
from src.fairness.resampling import oversample_train_indices  # noqa: E402
from src.fairness.reweighting import kamiran_calders_weights  # noqa: E402
from src.models.fairgnn import FairGNN, fairgnn_loss  # noqa: E402
from src.models.graphsage import GraphSAGE  # noqa: E402
from src.models.trainer import train  # noqa: E402

# ---------------------------------------------------------------------------
# Config & seeding
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Lightweight typed wrapper around configs/experiment.yaml."""

    raw_dir: str
    splits_dir: str
    split_ratios: tuple[float, float, float]
    target_col: str
    sensitive_cols: list[str]
    hidden_dim: int
    num_layers: int
    dropout: float
    lr: float
    epochs: int
    patience: int
    fairgnn_lambdas: list[float]
    fairgnn_adv_hidden: int
    noise_levels: list[float]
    edge_drop_rates: list[float]
    seeds: list[int] = field(default_factory=lambda: [3, 7, 21, 42, 99])

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            raw_dir=raw["data"]["raw_dir"],
            splits_dir=raw["data"]["splits_dir"],
            split_ratios=tuple(raw["data"]["split_ratios"]),
            target_col=raw["data"]["target_col"],
            sensitive_cols=list(raw["data"]["sensitive_cols"]),
            hidden_dim=raw["model"]["hidden_dim"],
            num_layers=raw["model"]["num_layers"],
            dropout=raw["model"]["dropout"],
            lr=raw["model"]["lr"],
            epochs=raw["model"]["epochs"],
            patience=raw["model"]["patience"],
            fairgnn_lambdas=list(raw["fairgnn"]["lambda_values"]),
            fairgnn_adv_hidden=raw["fairgnn"]["adv_hidden_dim"],
            noise_levels=list(raw["robustness"]["noise_levels"]),
            edge_drop_rates=list(raw["robustness"]["edge_drop_rates"]),
        )


def setup_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data plumbing
# ---------------------------------------------------------------------------


def load_data(raw_dir: str | Path, sensitive_cols: list[str], device: torch.device):
    """Load the FairGNN Pokec-z subset, preprocess, and move to GPU."""
    data = load_pokec_z(raw_dir)
    data = preprocess(data, sensitive_cols=sensitive_cols)
    return data.to(device)


def make_masks(
    n: int,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_mask = torch.zeros(n, dtype=torch.bool, device=device)
    val_mask = torch.zeros(n, dtype=torch.bool, device=device)
    test_mask = torch.zeros(n, dtype=torch.bool, device=device)
    train_mask[train_idx.to(device)] = True
    val_mask[val_idx.to(device)] = True
    test_mask[test_idx.to(device)] = True
    return train_mask, val_mask, test_mask


def build_sensitive_attrs(data) -> dict[str, torch.Tensor]:
    """Return the 5 sensitive tensors: gender, region, age_group, and the two
    intersectional composites — all indexed on the full node set, on whatever
    device ``data`` lives on.
    """
    n_age = int(data.age_group.max().item()) + 1
    n_reg = int(data.region.max().item()) + 1
    return {
        "gender": data.gender.long(),
        "region": data.region.long(),
        "age_group": data.age_group.clamp(min=0).long(),
        "gender_x_age": data.gender.long() * n_age + data.age_group.clamp(min=0).long(),
        "gender_x_region": data.gender.long() * n_reg + data.region.long(),
    }


# ---------------------------------------------------------------------------
# Model output container
# ---------------------------------------------------------------------------


@dataclass
class ModelOutput:
    """What every ``run_*`` function returns. Embeddings are optional because
    TabICL doesn't expose them — the multi-attribute helper falls back to a
    leakage probe on raw features when ``embeddings`` is None."""

    name: str
    acc: float
    f1: float
    pred: torch.Tensor
    proba: np.ndarray | None
    embeddings: torch.Tensor | None
    extra: dict[str, Any]


# ---------------------------------------------------------------------------
# Disk cache for ModelOutput
# ---------------------------------------------------------------------------
#
# Training the 6 underlying models (1 baseline + 1 resampling + 1 fairdrop +
# 4 fairgnn-grid + 1 tabicl) takes ~5 min on a single RTX 3090. None of the
# downstream operations (multi-attribute fairness, post-process EOT,
# threshold sweeps) require retraining: they only consume ``ModelOutput``
# fields (proba_test, proba_val, embeddings, predictions). We therefore cache
# every freshly-trained ModelOutput to disk; subsequent runs reuse them and
# only re-execute the cheap parts.
#
# Cache layout (default ``results/cache/``):
#
#     results/cache/
#       └── seed42/
#           ├── GraphSAGE.pt
#           ├── GraphSAGE_Resampling.pt
#           ├── GraphSAGE_FairDrop.pt
#           ├── FairGNN_lambda_5.0.pt
#           └── TabICL.pt
#
# When the user changes hyperparameters or the data, they should pass
# ``--no-cache`` (forces retrain) or delete the cache directory.


def _safe_filename(name: str) -> str:
    """Slugify a model name for use as a filesystem path component."""
    return re.sub(r"[^\w.\-]+", "_", name)


def _output_cache_path(cache_dir: Path, seed: int, model_name: str) -> Path:
    return cache_dir / f"seed{seed}" / f"{_safe_filename(model_name)}.pt"


def _save_model_output(out: ModelOutput, path: Path) -> None:
    """Serialise a :class:`ModelOutput` to ``path`` using ``torch.save``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "name": out.name,
        "acc": out.acc,
        "f1": out.f1,
        "pred": out.pred.detach().cpu(),
        "proba": out.proba,  # numpy
        "embeddings": (
            out.embeddings.detach().cpu() if out.embeddings is not None else None
        ),
        "extra": out.extra,  # dict; numpy arrays inside are pickled by torch.save
    }
    torch.save(state, path)


def _load_model_output(path: Path, device: torch.device) -> ModelOutput:
    """Inverse of :func:`_save_model_output`: tensors restored on ``device``."""
    state = torch.load(path, weights_only=False, map_location=device)
    pred = state["pred"].to(device)
    embeddings = (
        state["embeddings"].to(device) if state["embeddings"] is not None else None
    )
    return ModelOutput(
        name=state["name"],
        acc=state["acc"],
        f1=state["f1"],
        pred=pred,
        proba=state["proba"],
        embeddings=embeddings,
        extra=state["extra"],
    )


def _cached_run(
    fn: Callable[[], ModelOutput],
    cache_dir: Path | None,
    seed: int,
    model_name: str,
    device: torch.device,
) -> ModelOutput:
    """Run ``fn`` (a no-arg closure that trains the model) with an on-disk cache.

    If the cache file exists, the trained ``ModelOutput`` is loaded from disk
    and ``fn`` is NOT called. Otherwise ``fn`` runs, its result is saved, and
    returned. ``cache_dir=None`` disables the cache and always calls ``fn``.
    """
    if cache_dir is None:
        return fn()
    path = _output_cache_path(cache_dir, seed, model_name)
    if path.exists():
        print(f"      [cache hit] {path.name}")
        return _load_model_output(path, device)
    out = fn()
    _save_model_output(out, path)
    return out


# ---------------------------------------------------------------------------
# Per-method runners
# ---------------------------------------------------------------------------


def _train_graphsage(data, train_mask, val_mask, cfg: ExperimentConfig, seed: int) -> GraphSAGE:
    setup_seeds(seed)
    model = GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=cfg.hidden_dim,
        out_channels=int(data.y.max().item()) + 1,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(data.x.device)
    train(
        model,
        data,
        train_mask,
        val_mask,
        lr=cfg.lr,
        epochs=cfg.epochs,
        patience=cfg.patience,
    )
    return model


def _evaluate_classifier(
    model,
    data,
    test_mask: torch.Tensor,
    name: str,
    val_mask: torch.Tensor | None = None,
) -> ModelOutput:
    """Evaluate ``model`` on ``test_mask``. If ``val_mask`` is also provided,
    also computes proba on val and stashes it in ``extra["proba_val_pos"]``
    for downstream post-processing (Hardt-style threshold calibration).
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        # FairGNN forward returns a tuple; baseline GraphSAGE returns logits.
        if isinstance(logits, tuple):
            logits = logits[0]
        proba_full_t = F.softmax(logits, dim=1)
        pred = logits[test_mask].argmax(dim=1)
        proba_test = proba_full_t[test_mask].cpu().numpy().astype(np.float32)
        emb = (
            model.get_embeddings(data.x, data.edge_index)
            if hasattr(model, "get_embeddings")
            else model.encode(data.x, data.edge_index)
        )

    y_test_np = data.y[test_mask].cpu().numpy()
    pred_np = pred.cpu().numpy()
    acc = float(accuracy_score(y_test_np, pred_np))
    f1 = float(f1_score(y_test_np, pred_np, average="macro", zero_division=0))

    extra: dict[str, Any] = {}
    if val_mask is not None:
        with torch.no_grad():
            proba_val = proba_full_t[val_mask].cpu().numpy().astype(np.float32)
        extra["proba_val_pos"] = proba_val[:, 1] if proba_val.shape[1] >= 2 else proba_val.ravel()

    return ModelOutput(
        name=name, acc=acc, f1=f1, pred=pred, proba=proba_test, embeddings=emb, extra=extra
    )


def run_baseline(
    data, train_mask, val_mask, test_mask, cfg: ExperimentConfig, seed: int
) -> ModelOutput:
    """Vanilla GraphSAGE — graph + features, no fairness intervention."""
    model = _train_graphsage(data, train_mask, val_mask, cfg, seed)
    return _evaluate_classifier(model, data, test_mask, val_mask=val_mask, name="GraphSAGE")


def run_reweighted(
    data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    cfg: ExperimentConfig,
    seed: int,
    sensitive_name: str = "gender",
) -> ModelOutput:
    """GraphSAGE trained with Kamiran-Calders sample weights on (label × s).

    Pre-process method that pulls the joint distribution P(s, y) towards
    P(s) * P(y) via per-row weights in the cross-entropy loss. Works for any
    cardinality of ``s`` (binary, multi-class age_group, intersectional
    composites), unlike FairGNN which is tied to a binary adversary.
    """
    setup_seeds(seed)
    sensitive = getattr(data, sensitive_name).detach().cpu().numpy()
    y_np = data.y.detach().cpu().numpy()
    train_idx = train_mask.detach().cpu().numpy().nonzero()[0]

    # Build a full-graph weights tensor; non-train rows stay at 1.0 (unused).
    weights_full = np.ones(data.num_nodes, dtype=np.float32)
    weights_full[train_idx] = kamiran_calders_weights(
        y_np[train_idx], sensitive[train_idx]
    )
    sample_weights = torch.from_numpy(weights_full).to(data.x.device)

    model = GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=cfg.hidden_dim,
        out_channels=int(data.y.max().item()) + 1,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(data.x.device)
    train(
        model,
        data,
        train_mask,
        val_mask,
        lr=cfg.lr,
        epochs=cfg.epochs,
        patience=cfg.patience,
        sample_weights=sample_weights,
    )
    return _evaluate_classifier(
        model, data, test_mask, val_mask=val_mask,
        name=f"GraphSAGE+Reweighted@{sensitive_name}",
    )


def run_resampling(
    data, train_mask, val_mask, test_mask, cfg: ExperimentConfig, seed: int
) -> ModelOutput:
    """GraphSAGE trained on a label×gender oversampled training set."""
    setup_seeds(seed)
    over_idx = oversample_train_indices(
        train_mask.cpu(), data.y.cpu(), data.gender.cpu(), seed=seed
    )
    over_mask = torch.zeros_like(train_mask)
    over_mask[over_idx.to(over_mask.device)] = True
    model = _train_graphsage(data, over_mask, val_mask, cfg, seed)
    out = _evaluate_classifier(model, data, test_mask, val_mask=val_mask, name="GraphSAGE+Resampling")
    return out


def run_fairdrop(
    data,
    train_mask,
    val_mask,
    test_mask,
    cfg: ExperimentConfig,
    seed: int,
    drop_rate: float = 0.3,
    intra_group_bias: float = 2.0,
) -> ModelOutput:
    """Pre-process FairDrop on the gender attribute, then train GraphSAGE."""
    setup_seeds(seed)
    new_edge_index = fairdrop(
        data.edge_index.cpu(),
        data.gender.cpu(),
        drop_rate=drop_rate,
        intra_group_bias=intra_group_bias,
        seed=seed,
    ).to(data.edge_index.device)

    # Replace edges in-place for training (revert at end so subsequent calls see
    # the original graph). PyG Data is mutable; clone the tensor pointer only.
    original_ei = data.edge_index
    data.edge_index = new_edge_index
    try:
        model = _train_graphsage(data, train_mask, val_mask, cfg, seed)
        out = _evaluate_classifier(model, data, test_mask, val_mask=val_mask, name="GraphSAGE+FairDrop")
    finally:
        data.edge_index = original_ei
    return out


def run_fairgnn_grid(
    data,
    train_mask,
    val_mask,
    test_mask,
    cfg: ExperimentConfig,
    seed: int,
) -> tuple[ModelOutput, dict[float, dict[str, float]]]:
    """Sweep λ ∈ cfg.fairgnn_lambdas using the GRL-based FairGNN.

    Selects the best λ as the one with smallest ΔDP **among models with
    F1 > 0.50**, to avoid the degenerate "predict-everything-class-0"
    collapse mode that produces a trivial ΔDP=0.
    """
    grid: dict[float, dict[str, float]] = {}
    best_state = {}
    out_channels = int(data.y.max().item()) + 1

    for lam in cfg.fairgnn_lambdas:
        setup_seeds(seed)
        model = FairGNN(
            in_channels=data.x.shape[1],
            hidden_channels=cfg.hidden_dim,
            out_channels=out_channels,
            adv_hidden=cfg.fairgnn_adv_hidden,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            lambda_adv=lam,
        ).to(data.x.device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=5e-4)

        best_val_f1 = 0.0
        patience = 0
        best_st = None
        for _epoch in range(cfg.epochs):
            model.train()
            opt.zero_grad()
            pred_logits, adv_logits = model(data.x, data.edge_index)
            loss = fairgnn_loss(pred_logits, adv_logits, data.y, data.gender, train_mask)
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                pl_val, _ = model(data.x, data.edge_index)
            val_pred = pl_val[val_mask].argmax(dim=1)
            val_f1 = float(
                f1_score(
                    data.y[val_mask].cpu().numpy(),
                    val_pred.cpu().numpy(),
                    average="macro",
                    zero_division=0,
                )
            )
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_st = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= cfg.patience:
                    break

        if best_st is not None:
            model.load_state_dict(best_st)

        out = _evaluate_classifier(model, data, test_mask, val_mask=val_mask, name=f"FairGNN(λ={lam})")
        ddp = demographic_parity_diff(out.pred, data.gender[test_mask])
        grid[lam] = {"f1": out.f1, "acc": out.acc, "delta_dp": ddp}
        best_state[lam] = best_st

    # Pick best λ: smallest ΔDP among non-collapsed models.
    valid = {lam: v for lam, v in grid.items() if v["f1"] > 0.50}
    chosen = min(valid or grid, key=lambda lam: (valid or grid)[lam]["delta_dp"])

    # Re-load the best-λ model and re-evaluate to produce the final ModelOutput.
    model = FairGNN(
        in_channels=data.x.shape[1],
        hidden_channels=cfg.hidden_dim,
        out_channels=out_channels,
        adv_hidden=cfg.fairgnn_adv_hidden,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        lambda_adv=chosen,
    ).to(data.x.device)
    model.load_state_dict(best_state[chosen])
    final = _evaluate_classifier(
        model, data, test_mask, val_mask=val_mask, name=f"FairGNN(λ={chosen})"
    )
    final.extra["grid"] = grid
    final.extra["chosen_lambda"] = chosen
    return final, grid


def run_tabicl(
    data,
    train_mask,
    val_mask,
    test_mask,
    _cfg: ExperimentConfig,
    seed: int,
    max_train: int = 10_000,
) -> ModelOutput:
    """Tabular foundation model — no graph, no in-training fairness.

    The leakage probe will be computed on raw features (since TabICL has no
    exposed embedding) — this is the lower-bound proxy strength of x for the
    sensitive attribute, *before* any model. Counterfactual fairness is
    skipped (no latent space to augment).
    """
    from src.baselines.tabicl import tabicl_predict

    x_np = data.x.detach().cpu().numpy().astype(np.float32)
    y_np = data.y.detach().cpu().numpy().astype(np.int64)
    train_idx_np = train_mask.detach().cpu().numpy().nonzero()[0]
    val_idx_np = val_mask.detach().cpu().numpy().nonzero()[0]
    test_idx_np = test_mask.detach().cpu().numpy().nonzero()[0]

    # Predict on val + test in a single concatenated call so TabICL fits once.
    eval_idx_np = np.concatenate([val_idx_np, test_idx_np])
    pred_np_all, proba_pos_all = tabicl_predict(
        x_np,
        y_np,
        train_idx_np,
        eval_idx_np,
        seed=seed,
        max_train=max_train,
        device=str(data.x.device) if data.x.is_cuda else "cpu",
    )
    n_val = val_idx_np.size
    proba_val_pos = proba_pos_all[:n_val].astype(np.float32)
    proba_test_pos = proba_pos_all[n_val:].astype(np.float32)
    pred_np = pred_np_all[n_val:]
    proba_2col_test = np.stack([1.0 - proba_test_pos, proba_test_pos], axis=1)

    pred = torch.from_numpy(pred_np).long().to(data.x.device)
    acc = float(accuracy_score(y_np[test_idx_np], pred_np))
    f1 = float(f1_score(y_np[test_idx_np], pred_np, average="macro", zero_division=0))

    return ModelOutput(
        name="TabICL",
        acc=acc,
        f1=f1,
        pred=pred,
        proba=proba_2col_test,
        embeddings=None,
        extra={"max_train": max_train, "proba_val_pos": proba_val_pos},
    )


# ---------------------------------------------------------------------------
# Post-process — Hardt et al. 2016 group-specific threshold
# ---------------------------------------------------------------------------


def apply_equal_opportunity_threshold(
    out: ModelOutput,
    data,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    sensitive_name: str = "gender",
    strategy: str = "equal_opportunity",
) -> ModelOutput:
    """Wrap ``out`` with a per-group threshold calibrated on val, applied on test.

    Hardt, Price & Srebro 2016 — applicable to any classifier with proba output,
    including frozen ones (TabICL) where in-training fairness can't be inserted.
    """
    from src.postprocess.equal_opportunity import calibrate_predictions

    if "proba_val_pos" not in out.extra:
        raise ValueError(
            f"cannot post-process {out.name}: extra['proba_val_pos'] missing"
        )
    if out.proba is None:
        raise ValueError(f"cannot post-process {out.name}: proba is None")

    proba_val_pos = out.extra["proba_val_pos"]
    proba_test_pos = out.proba[:, 1] if out.proba.ndim == 2 else out.proba.ravel()

    sensitive_full = getattr(data, sensitive_name).long()
    sensitive_val_np = sensitive_full[val_mask].cpu().numpy()
    sensitive_test_np = sensitive_full[test_mask].cpu().numpy()
    y_val_np = data.y[val_mask].cpu().numpy()

    pred_test_np, thresholds = calibrate_predictions(
        proba_val_pos,
        y_val_np,
        sensitive_val_np,
        proba_test_pos,
        sensitive_test_np,
        strategy=strategy,
    )

    pred_test = torch.from_numpy(pred_test_np).long().to(out.pred.device)
    y_test_np = data.y[test_mask].cpu().numpy()
    acc = float(accuracy_score(y_test_np, pred_test_np))
    f1 = float(f1_score(y_test_np, pred_test_np, average="macro", zero_division=0))

    suffix = "EOT" if strategy == "equal_opportunity" else "DPT"
    return ModelOutput(
        name=f"{out.name}+{suffix}@{sensitive_name}",
        acc=acc,
        f1=f1,
        pred=pred_test,
        proba=out.proba,
        embeddings=out.embeddings,
        extra={**out.extra, "thresholds": thresholds, "post_strategy": strategy},
    )


# ---------------------------------------------------------------------------
# Multi-attribute fairness aggregation
# ---------------------------------------------------------------------------


def compute_multi_attr_fairness(
    out: ModelOutput,
    data,
    sensitive_attrs: dict[str, torch.Tensor],
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    seed: int,
) -> pl.DataFrame:
    """For each sensitive axis, compute ΔDP/ΔEO/AUC-gap/leakage.

    For models with embeddings (GraphSAGE / FairGNN / FairDrop / Resampling):
    the leakage probe is on the embeddings.
    For models without embeddings (TabICL): the leakage probe falls back to
    raw features ``data.x`` — this is the *lower bound* of leakage, that is
    what the columns themselves encode about the sensitive attribute before
    any modelling.
    """
    rows: list[dict[str, Any]] = []
    leakage_source = out.embeddings if out.embeddings is not None else data.x
    for sname, stensor in sensitive_attrs.items():
        st_test = stensor[test_mask]
        ddp = demographic_parity_diff(out.pred, st_test)
        deo = equal_opportunity_diff(out.pred, data.y[test_mask], st_test)
        gap = (
            group_auc_gap(out.proba, data.y[test_mask], st_test)
            if out.proba is not None
            else float("nan")
        )
        leak = sensitive_leakage(leakage_source, stensor, train_mask, test_mask, seed=seed)
        rows.append(
            {
                "model": out.name,
                "attribute": sname,
                "delta_dp": round(ddp, 4),
                "delta_eo": round(deo, 4),
                "group_auc_gap": round(gap, 4) if not np.isnan(gap) else None,
                "leakage_auc": round(leak, 4),
            }
        )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------


def run_all(
    seed: int,
    device_str: str,
    cfg_path: str | Path,
    raw_dir_override: str | None = None,
    out_csv: str | Path = "results/metrics/comparison_full.csv",
    skip_tabicl: bool = False,
    cache_dir: str | Path | None = "results/cache",
) -> pl.DataFrame:
    """Run every method, evaluate on every sensitive attribute, write CSV.

    Trained ``ModelOutput`` objects are cached on disk under ``cache_dir`` so
    re-running with only post-process changes (e.g. new EOT axes) skips the
    expensive training step. Pass ``cache_dir=None`` to force full retrain.
    """
    cfg = ExperimentConfig.from_yaml(cfg_path)
    device = torch.device(device_str)
    raw_dir = raw_dir_override or cfg.raw_dir
    cache_path = Path(cache_dir) if cache_dir is not None else None
    setup_seeds(seed)

    cache_msg = f"cache={cache_path}" if cache_path else "cache=disabled"
    print(f"[setup] device={device} | seed={seed} | raw_dir={raw_dir} | {cache_msg}")
    data = load_data(raw_dir, cfg.sensitive_cols, device)
    train_idx, val_idx, test_idx = make_splits(
        data.num_nodes,
        data.y.cpu(),
        data.gender.cpu(),
        ratios=cfg.split_ratios,
        seed=seed,
    )
    train_mask, val_mask, test_mask = make_masks(
        data.num_nodes, train_idx, val_idx, test_idx, device
    )
    sensitive_attrs = build_sensitive_attrs(data)

    print(
        f"[graph] r(gender)={assortative_mixing_coefficient(data.edge_index, data.gender):.3f}"
        f"  r(region)={assortative_mixing_coefficient(data.edge_index, data.region):.3f}"
    )

    all_dfs: list[pl.DataFrame] = []

    print("[1/5] baseline GraphSAGE …")
    base_out = _cached_run(
        lambda: run_baseline(data, train_mask, val_mask, test_mask, cfg, seed),
        cache_path,
        seed,
        "GraphSAGE",
        device,
    )
    print(f"      acc={base_out.acc:.4f}  f1={base_out.f1:.4f}")
    all_dfs.append(
        compute_multi_attr_fairness(base_out, data, sensitive_attrs, train_mask, test_mask, seed)
    )

    print("[2/5] resampling + GraphSAGE …")
    rs_out = _cached_run(
        lambda: run_resampling(data, train_mask, val_mask, test_mask, cfg, seed),
        cache_path,
        seed,
        "GraphSAGE+Resampling",
        device,
    )
    print(f"      acc={rs_out.acc:.4f}  f1={rs_out.f1:.4f}")
    all_dfs.append(
        compute_multi_attr_fairness(rs_out, data, sensitive_attrs, train_mask, test_mask, seed)
    )

    print("[3/5] FairDrop + GraphSAGE …")
    fd_out = _cached_run(
        lambda: run_fairdrop(data, train_mask, val_mask, test_mask, cfg, seed),
        cache_path,
        seed,
        "GraphSAGE+FairDrop",
        device,
    )
    print(f"      acc={fd_out.acc:.4f}  f1={fd_out.f1:.4f}")
    all_dfs.append(
        compute_multi_attr_fairness(fd_out, data, sensitive_attrs, train_mask, test_mask, seed)
    )

    print("[4/5] FairGNN λ-grid (GRL) …")

    def _run_fairgnn_unwrapped() -> ModelOutput:
        out, grid = run_fairgnn_grid(data, train_mask, val_mask, test_mask, cfg, seed)
        out.extra["grid"] = grid  # already there but explicit for cache reload
        return out

    fg_out = _cached_run(
        _run_fairgnn_unwrapped,
        cache_path,
        seed,
        "FairGNN_grid",
        device,
    )
    print(f"      grid: {fg_out.extra.get('grid')}")
    print(
        f"      best λ={fg_out.extra.get('chosen_lambda')}  acc={fg_out.acc:.4f}  f1={fg_out.f1:.4f}"
    )
    all_dfs.append(
        compute_multi_attr_fairness(fg_out, data, sensitive_attrs, train_mask, test_mask, seed)
    )

    if not skip_tabicl:
        print("[5/5] TabICL (no graph) …")
        tab_out = _cached_run(
            lambda: run_tabicl(data, train_mask, val_mask, test_mask, cfg, seed),
            cache_path,
            seed,
            "TabICL",
            device,
        )
        print(f"      acc={tab_out.acc:.4f}  f1={tab_out.f1:.4f}")
        all_dfs.append(
            compute_multi_attr_fairness(tab_out, data, sensitive_attrs, train_mask, test_mask, seed)
        )
    else:
        print("[5/5] TabICL skipped.")
        tab_out = None

    # ---- Post-process variants (Hardt et al. 2016) on multiple sensitive axes ----
    # Two strategies × three axes × two endpoint models = 12 variants.
    #   * "equal_opportunity": equalises TPR (P(ŷ=1|y=1,s)) — original Hardt criterion.
    #   * "demographic_parity": equalises positive rate P(ŷ=1|s) — directly attacks ΔDP.
    # The two strategies move different metrics (Chouldechova-Kleinberg 2017
    # incompatibility); having both in the table makes the trade-off explicit.
    print("[6] post-process threshold (multi-axis × two strategies) …")
    eot_axes = ("gender", "age_group", "region")
    eot_strategies = (
        ("equal_opportunity", "EO"),
        ("demographic_parity", "DP"),
    )
    for strategy, tag in eot_strategies:
        for axis in eot_axes:
            base_eot = apply_equal_opportunity_threshold(
                base_out, data, val_mask, test_mask,
                sensitive_name=axis, strategy=strategy,
            )
            print(
                f"      [GraphSAGE+{tag}T@{axis:<10s}] acc={base_eot.acc:.4f}  f1={base_eot.f1:.4f}"
            )
            all_dfs.append(
                compute_multi_attr_fairness(
                    base_eot, data, sensitive_attrs, train_mask, test_mask, seed
                )
            )
            if tab_out is not None:
                tab_eot = apply_equal_opportunity_threshold(
                    tab_out, data, val_mask, test_mask,
                    sensitive_name=axis, strategy=strategy,
                )
                print(
                    f"      [TabICL+{tag}T@{axis:<10s}]   acc={tab_eot.acc:.4f}  f1={tab_eot.f1:.4f}"
                )
                all_dfs.append(
                    compute_multi_attr_fairness(
                        tab_eot, data, sensitive_attrs, train_mask, test_mask, seed
                    )
                )

    # ---- Pre-process Kamiran & Calders 2012 reweighting (multi-axis) ----------
    # Each example gets weight P(s)*P(y) / P(s,y), pulling the joint towards
    # independence. Works on any cardinality of s — including the multi-class
    # age_group case where FairGNN's binary adversary doesn't apply.
    print("[7] pre-process Kamiran-Calders reweighting (multi-axis) …")
    for axis in eot_axes:
        rw_out = _cached_run(
            lambda axis=axis: run_reweighted(
                data, train_mask, val_mask, test_mask, cfg, seed, sensitive_name=axis
            ),
            cache_path,
            seed,
            f"GraphSAGE+Reweighted_{axis}",
            device,
        )
        print(
            f"      [GraphSAGE+Reweighted@{axis:<10s}] acc={rw_out.acc:.4f}  f1={rw_out.f1:.4f}"
        )
        all_dfs.append(
            compute_multi_attr_fairness(
                rw_out, data, sensitive_attrs, train_mask, test_mask, seed
            )
        )

    df = pl.concat(all_dfs, how="diagonal_relaxed")
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out_path)
    print(f"\n[done] wrote {out_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Pokec-z fairness experiment driver")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="torch device string (cuda:0, cuda:1, cpu)",
    )
    parser.add_argument("--config", default=str(_REPO_ROOT / "configs" / "experiment.yaml"))
    parser.add_argument(
        "--raw-dir",
        default=str(_REPO_ROOT / "data" / "raw" / "pokec-z"),
        help="override the raw_dir from the YAML config",
    )
    parser.add_argument(
        "--out-csv",
        default=str(_REPO_ROOT / "results" / "metrics" / "comparison_full.csv"),
    )
    parser.add_argument("--skip-tabicl", action="store_true")
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable on-disk cache for trained ModelOutputs (default: enabled). "
            "Use --no-cache to force a full retrain. Cache lives under --cache-dir."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=str(_REPO_ROOT / "results" / "cache"),
        help="Directory holding cached ModelOutput .pt files (default: results/cache).",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete cache files for the current --seed before running.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache else None
    if args.clear_cache and cache_dir is not None:
        seed_dir = cache_dir / f"seed{args.seed}"
        if seed_dir.exists():
            for f in seed_dir.glob("*.pt"):
                f.unlink()
            print(f"[clear-cache] removed cache files in {seed_dir}")

    df = run_all(
        seed=args.seed,
        device_str=args.device,
        cfg_path=args.config,
        raw_dir_override=args.raw_dir,
        out_csv=args.out_csv,
        skip_tabicl=args.skip_tabicl,
        cache_dir=cache_dir,
    )
    print(df)
    sys.exit(0)


if __name__ == "__main__":
    main()
