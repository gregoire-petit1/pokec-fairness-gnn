"""Education fairness — predict university attendance, audit ethnic axes.

The default Pokec-z target ``completed_level_of_education_indicator`` is just
a *profile-completeness* binary: anyone who declared a language tends to
declare an education level too. We confirmed empirically that ΔDP on **any**
language column (anglicky/nemecky/madarsky/...) lands in 40–50pp on this
target — the model is discriminating *fillers vs non-fillers*, not ethnic
groups. So the previous 41.9pp ΔDP@hungarian was a target artefact.

This script switches to a target with **real socio-educational signal** :
``vysokoskolske`` (=1 iff the user declared university-level education) on
the **fillers-only** subset (``completed=1``, N=31,774). On that target,
descriptive prevalence already shows real ethnic gaps :

    Roma         5.4% vs 9.9%   (-4.5pp)   ← university
    Hungarian    7.3% vs 10.0%  (-2.7pp)
    English     11.0% vs 8.5%   (+2.5pp)   ← English-speakers, an urban-young
                                              control, go the *other* way

Pipeline :
    1. Re-train GraphSAGE on the new target, splits stratified on (y, gender)
       restricted to fillers. The forward pass still uses the full graph
       (non-fillers contribute via message passing as neighbours).
    2. Measure ΔDP, ΔEO, F1, leakage AUC on each sensitive axis :
       gender, region, hungarian, roma + controls anglicky, nemecky.
    3. Apply INLP@axis + DPT@axis chain ; re-measure.
    4. Output ``results/metrics/education_fairness.csv``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.main_experiment import (  # noqa: E402
    _evaluate_classifier,
    _train_graphsage,
    apply_equal_opportunity_threshold,
    apply_inlp_to_embeddings,
)
from src.data.loader import load_pokec_z  # noqa: E402
from src.data.minorities import attach_minorities  # noqa: E402
from src.data.preprocessing import preprocess  # noqa: E402
from src.data.splits import make_splits  # noqa: E402
from src.fairness.metrics import (  # noqa: E402
    assortative_mixing_coefficient,
    demographic_parity_diff,
    equal_opportunity_diff,
    sensitive_leakage,
)

OUT_DIR = ROOT / "results" / "metrics"
RAW_DIR = ROOT / "data" / "raw" / "pokec-z"
SEED = 42
# Target: y=1 iff user declared HIGH-school OR university (i.e. "completed
# at least secondary education"). This is a real socio-educational variable
# in Slovakia, ≈62% prevalence on the mono-level filler subset, and the 4
# constituent columns are dropped from x to avoid leak.
HIGH_LEVEL_COLS = ("stredoskolske", "vysokoskolske")
LOW_LEVEL_COLS = ("zakladne", "ucnovske")
LEAK_COLS = HIGH_LEVEL_COLS + LOW_LEVEL_COLS  # all 4 must be dropped from x
FILLER_COL = "completed_level_of_education_indicator"
SENSITIVE_AXES = (
    "gender",
    "region",
    "age_group",
    "hungarian",
    "roma",
    "anglicky",
    "nemecky",
)
ETHNIC_AXES = ("hungarian", "roma")


@dataclass
class _Cfg:
    """Mirror of ExperimentConfig with the few fields _train_graphsage uses."""

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.5
    lr: float = 0.005
    epochs: int = 200
    patience: int = 30


def setup_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def attach_language_columns(data, langs: tuple[str, ...]) -> None:
    """Attach selected language columns from raw_df as long tensors on data."""
    df = data.raw_df
    for lang in langs:
        if lang not in df.columns:
            print(f"  [WARN] language {lang!r} not in raw_df, skipping")
            continue
        arr = (df.get_column(lang).cast(pl.Int64).to_numpy() > 0).astype(np.int64)
        setattr(data, lang, torch.from_numpy(arr).long().to(data.x.device))


def override_target_high_level(data) -> torch.Tensor:
    """Binary target: y=1 iff user declared lycée OR université level.

    On the mono-level filler subset, this is ≈62% prevalence and constitutes
    a real socio-educational variable (completed at least secondary education
    in Slovakia).
    """
    df = data.raw_df
    arrs = [(df.get_column(c).cast(pl.Int64).to_numpy() > 0) for c in HIGH_LEVEL_COLS]
    y_np = np.zeros(df.height, dtype=np.int64)
    for a in arrs:
        y_np |= a.astype(np.int64)
    y_t = torch.from_numpy(y_np).long().to(data.x.device)
    data.y = y_t
    return y_t


def drop_leak_features(data, leak_cols: tuple[str, ...]) -> list[str]:
    """Remove columns in ``leak_cols`` from ``data.x`` and ``data.feature_cols``.

    Returns the list of dropped columns. The 4 education-level binary columns
    (vysokoskolske/stredoskolske/zakladne/ucnovske) are mutually exclusive with
    each other AND with ``completed_level_of_education_indicator`` — keeping
    them in x means the model can perfectly read off y from x.
    """
    keep_cols = [c for c in data.feature_cols if c not in set(leak_cols)]
    keep_indices = torch.tensor(
        [data.feature_cols.index(c) for c in keep_cols], dtype=torch.long, device=data.x.device
    )
    data.x = data.x.index_select(dim=1, index=keep_indices)
    dropped = [c for c in data.feature_cols if c in set(leak_cols)]
    data.feature_cols = keep_cols
    return dropped


def filler_mask_from(data, min_age: int = 0) -> torch.Tensor:
    """Boolean tensor : True iff user has filled exactly ONE of the 4 levels.

    With ``min_age=25``, additionally restricts to users old enough to have
    completed their formal education — removes the *temporal-censoring*
    artifact where 41 % of fillers are <25-year-old students still in school.
    """
    df = data.raw_df
    n_levels = (
        df.select(sum((pl.col(c).cast(pl.Int64) > 0).cast(pl.Int64) for c in LEAK_COLS).alias("k"))
        .get_column("k")
        .to_numpy()
    )
    keep = n_levels == 1
    if min_age > 0:
        age = df.get_column("AGE").cast(pl.Int64).to_numpy()
        keep = keep & (age >= min_age)
    return torch.from_numpy(keep).to(data.x.device)


def stratified_filler_splits(
    filler_mask: torch.Tensor,
    y: torch.Tensor,
    gender: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splits restricted to fillers, stratified on (y × gender).

    Returns global node-id indices (in the full N=66,569 frame), not local
    positions within the fillers — so downstream masks index into ``data``
    directly.
    """
    filler_idx_global = torch.where(filler_mask)[0].cpu()
    n_local = filler_idx_global.shape[0]
    y_local = y[filler_idx_global].cpu()
    g_local = gender[filler_idx_global].cpu()
    train_local, val_local, test_local = make_splits(
        n=n_local, y=y_local, sensitive=g_local, seed=seed
    )
    train_idx = filler_idx_global[train_local]
    val_idx = filler_idx_global[val_local]
    test_idx = filler_idx_global[test_local]
    return train_idx, val_idx, test_idx


def build_masks(n: int, train_idx, val_idx, test_idx, device) -> tuple:
    train_mask = torch.zeros(n, dtype=torch.bool, device=device)
    val_mask = torch.zeros(n, dtype=torch.bool, device=device)
    test_mask = torch.zeros(n, dtype=torch.bool, device=device)
    train_mask[train_idx.to(device)] = True
    val_mask[val_idx.to(device)] = True
    test_mask[test_idx.to(device)] = True
    return train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# Metric computation per (method, axis)
# ---------------------------------------------------------------------------


def _row(
    *,
    method: str,
    axis: str,
    pred: torch.Tensor,
    emb: torch.Tensor | None,
    s_full: torch.Tensor,
    s_test_mask_order: torch.Tensor,
    y_test_mask_order: torch.Tensor,
    y_full: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    f1: float,
    acc: float,
    n_pos_test: int,
    assortativity: float,
) -> dict:
    pred_cpu = pred.cpu().long()
    s_cpu = s_test_mask_order.cpu().long()
    y_cpu = y_test_mask_order.cpu().long()

    delta_dp = demographic_parity_diff(pred_cpu, s_cpu)
    delta_eo = equal_opportunity_diff(pred_cpu, y_cpu, s_cpu)

    # True prevalence vs predicted rate per group (s=1) — captures whether the
    # model's prediction matches the actual base rate in each group.
    m1 = s_cpu == 1
    m0 = s_cpu == 0
    true_rate_1 = float(y_cpu[m1].float().mean()) if m1.any() else float("nan")
    true_rate_0 = float(y_cpu[m0].float().mean()) if m0.any() else float("nan")
    pred_rate_1 = float(pred_cpu[m1].float().mean()) if m1.any() else float("nan")
    pred_rate_0 = float(pred_cpu[m0].float().mean()) if m0.any() else float("nan")
    true_rate_gap = true_rate_1 - true_rate_0
    pred_rate_gap = pred_rate_1 - pred_rate_0
    # "Excess prediction gap" : if the model's gap > true gap, it amplifies
    # the disparity beyond the data; if < true gap, it under-represents it.
    excess_gap = pred_rate_gap - true_rate_gap

    # Calibration / Positive Predictive Value per group : P(y=1 | ŷ=1, s).
    # If equal across groups, the model's positives "mean the same thing"
    # for both groups (Chouldechova 2017 calibration).
    yhat1_g1 = (pred_cpu == 1) & m1
    yhat1_g0 = (pred_cpu == 1) & m0
    ppv_1 = float(y_cpu[yhat1_g1].float().mean()) if yhat1_g1.any() else float("nan")
    ppv_0 = float(y_cpu[yhat1_g0].float().mean()) if yhat1_g0.any() else float("nan")
    ppv_gap = ppv_1 - ppv_0 if not (np.isnan(ppv_1) or np.isnan(ppv_0)) else float("nan")

    if emb is not None:
        leak = sensitive_leakage(emb.cpu(), s_full.cpu(), train_mask, test_mask, seed=SEED)
    else:
        leak = float("nan")
    return {
        "method": method,
        "axis": axis,
        "n_pos_test": n_pos_test,
        "assortativity_r": float(assortativity),
        "f1": f1,
        "acc": acc,
        "true_rate_g1": true_rate_1,
        "true_rate_g0": true_rate_0,
        "pred_rate_g1": pred_rate_1,
        "pred_rate_g0": pred_rate_0,
        "true_rate_gap": true_rate_gap,
        "pred_rate_gap": pred_rate_gap,
        "excess_gap": excess_gap,
        "delta_dp": delta_dp,
        "delta_eo": delta_eo,
        "ppv_g1": ppv_1,
        "ppv_g0": ppv_0,
        "ppv_gap": ppv_gap,
        "leakage_auc": leak,
    }


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    setup_seeds(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading data + minorities + control languages...")
    data = load_pokec_z(RAW_DIR)
    data = preprocess(data, sensitive_cols=["gender", "region"])
    data = data.to(device)
    # Move age_group + age_group_known onto device (preprocess attaches but
    # doesn't move them — Data.to() doesn't see attrs added after init).
    data.age_group = data.age_group.clamp(min=0).to(device)
    attach_minorities(data, RAW_DIR)  # adds .hungarian, .roma, .sign
    for axis in ("hungarian", "roma", "sign"):
        setattr(data, axis, getattr(data, axis).to(device))
    attach_language_columns(data, ("anglicky", "nemecky"))

    n = data.y.shape[0]
    y_orig = data.y.clone()  # keep around for sanity
    y_new = override_target_high_level(data)
    dropped = drop_leak_features(data, LEAK_COLS)
    print(f"  dropped leaky features: {dropped}")

    filler_mask = filler_mask_from(data, min_age=25)
    n_fill = int(filler_mask.sum().item())
    print(
        f"  N={n:,}  mono_level_fillers_AGE>=25={n_fill:,}  "
        f"P(>=lycée | filler) = {float(y_new[filler_mask].float().mean()) * 100:.2f}%  "
        f"(prev orig target P=1) = {float(y_orig.float().mean()) * 100:.2f}%"
    )

    train_idx, val_idx, test_idx = stratified_filler_splits(
        filler_mask, y_new, data.gender, seed=SEED
    )
    train_mask, val_mask, test_mask = build_masks(n, train_idx, val_idx, test_idx, device)
    print(
        f"  train={int(train_mask.sum())}  val={int(val_mask.sum())}  test={int(test_mask.sum())}"
    )

    # ── Train GraphSAGE on new target ───────────────────────────────────────
    print("\nTraining GraphSAGE on vysokoskolske target...")
    cfg = _Cfg()
    t0 = time.time()
    model = _train_graphsage(data, train_mask, val_mask, cfg, seed=SEED)
    out = _evaluate_classifier(model, data, test_mask, name="GraphSAGE", val_mask=val_mask)
    print(
        f"  done in {time.time() - t0:.1f}s  "
        f"acc={out.acc:.4f}  f1={out.f1:.4f}  "
        f"P(ŷ=1 | test) = {float(out.pred.float().mean()) * 100:.2f}%"
    )

    # ── Per-axis fairness audit ─────────────────────────────────────────────
    rows = []
    y_test_mo = data.y[test_mask]  # mask-order alignment with out.pred
    f1_baseline = out.f1
    acc_baseline = out.acc
    print("\n=== Per-axis fairness on the new target ===")
    for axis in SENSITIVE_AXES:
        s = getattr(data, axis)
        s_test_mo = s[test_mask]
        n_pos = int(s_test_mo.sum().item())
        if n_pos < 5:
            print(f"  [SKIP] axis={axis!r}: n_pos_test={n_pos}")
            continue
        r = assortative_mixing_coefficient(data.edge_index.cpu(), s.cpu())

        # Baseline
        rows.append(
            _row(
                method="GraphSAGE",
                axis=axis,
                pred=out.pred,
                emb=out.embeddings,
                s_full=s,
                s_test_mask_order=s_test_mo,
                y_test_mask_order=y_test_mo,
                y_full=data.y,
                train_mask=train_mask,
                test_mask=test_mask,
                f1=f1_baseline,
                acc=acc_baseline,
                n_pos_test=n_pos,
                assortativity=r,
            )
        )

        # Only run INLP+DPT on the focal sensitive axes (ethnic + structural).
        # Skip control language axes — they would clutter the output.
        if axis in ("anglicky", "nemecky"):
            continue
        out_inlp = apply_inlp_to_embeddings(
            out, data, train_mask, val_mask, test_mask, sensitive_name=axis
        )
        out_chain = apply_equal_opportunity_threshold(
            out_inlp, data, val_mask, test_mask, sensitive_name=axis, strategy="demographic_parity"
        )
        rows.append(
            _row(
                method=f"GraphSAGE+INLP+DPT@{axis}",
                axis=axis,
                pred=out_chain.pred,
                emb=out_chain.embeddings,
                s_full=s,
                s_test_mask_order=s_test_mo,
                y_test_mask_order=y_test_mo,
                y_full=data.y,
                train_mask=train_mask,
                test_mask=test_mask,
                f1=out_chain.f1,
                acc=out_chain.acc,
                n_pos_test=n_pos,
                assortativity=r,
            )
        )

    table = pl.DataFrame(rows)
    with pl.Config(tbl_rows=30, fmt_str_lengths=40, float_precision=4):
        print(table)
    out_path = OUT_DIR / "education_fairness.csv"
    table.write_csv(out_path)
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
