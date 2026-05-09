"""Full fairness toolbox on a *meaningful* target — smoking × age.

Why this configuration. The default Pokec-z setup (target = profile-completeness,
sensitive = gender or region binarised) gives results that don't survive
sanity checks (target artefacts, opaque attributes, etc — see notes).
We swap to a target with real socio-sanitary stakes, ``fajcim pravidelne``
(smokes regularly, 9.3 % prevalence), and a sensitive axis with documented
real disparity, ``age_old`` (binarised : adult+senior ≥ 25 vs young < 25).

Empirical setup. The exploratory pass identified three things worth pursuing :
  • a robust ground-truth gap (+4.0pp adults+ vs young, n=4050 in test),
  • a robust amplification by GraphSAGE (pred_gap = +7.4pp, excess +3.4pp,
    z-score ~ 3.4σ),
  • a target that's not trivially leak-coupled to other features.

Pipeline.
  1. Pre-process:
       resampling (oversample on (y × age_old))
       reweighting (Kamiran-Calders weights on (y × age_old))
       fairdrop   (down-sample intra-group edges on age_old)
  2. In-training:
       fairgnn_grid (GRL adversary on age_old, λ ∈ {0.1, 0.5, 1.0, 5.0})
  3. Post-process:
       INLP@age_old             (project age_old out of embeddings, refit head)
       INLP+DPT@age_old         (chain : INLP then per-group threshold for DP)
       INLP+EOT@age_old         (chain : INLP then per-group threshold for EO)

For each method, we report : F1, acc, ΔDP, ΔEO, **excess_gap** (= pred_gap −
true_gap, the metric that matters for "is the model amplifying the real
disparity ?"), PPV gap (calibration), leakage AUC.

Run::

    .venv/bin/python scripts/run_toolbox_smoking_age.py

Output : ``results/metrics/toolbox_smoking_age.csv``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.main_experiment import (  # noqa: E402
    apply_equal_opportunity_threshold,
    apply_inlp_to_embeddings,
    run_baseline,
    run_fairdrop,
    run_fairgnn_grid,
    run_resampling,
    run_reweighted,
)
from src.data.loader import load_pokec_z  # noqa: E402
from src.data.preprocessing import preprocess  # noqa: E402
from src.data.splits import make_splits  # noqa: E402
from src.fairness.metrics import (  # noqa: E402
    demographic_parity_diff,
    equal_opportunity_diff,
    sensitive_leakage,
)

OUT_DIR = ROOT / "results" / "metrics"
RAW_DIR = ROOT / "data" / "raw" / "pokec-z"
TARGET_COL = "fajcim pravidelne"
SEED = 42


@dataclass
class _Cfg:
    """Mirror of ExperimentConfig — only the fields the runners use."""

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.5
    lr: float = 0.005
    epochs: int = 200
    patience: int = 30
    fairgnn_adv_hidden: int = 64
    fairgnn_lambdas: tuple[float, ...] = (0.1, 0.5, 1.0, 5.0)


def setup_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_smoking_target(data) -> torch.Tensor:
    """Binary ``y = 1`` iff user declared smoking regularly."""
    df = data.raw_df
    y_np = (df.get_column(TARGET_COL).cast(pl.Int64).to_numpy() > 0).astype(np.int64)
    y_t = torch.from_numpy(y_np).long().to(data.x.device)
    data.y = y_t
    return y_t


def drop_leak_features(data, leak_cols: list[str]) -> list[str]:
    """Remove smoking-related cols from x to prevent target leak."""
    keep_cols = [c for c in data.feature_cols if c not in set(leak_cols)]
    keep_indices = torch.tensor(
        [data.feature_cols.index(c) for c in keep_cols], dtype=torch.long, device=data.x.device
    )
    data.x = data.x.index_select(dim=1, index=keep_indices)
    dropped = [c for c in data.feature_cols if c in set(leak_cols)]
    data.feature_cols = keep_cols
    return dropped


def build_age_old(data) -> torch.Tensor:
    """Binary axis : adult+senior (age_group ≥ 1) vs young (age_group = 0).

    Ages with NA (age_group=-1, post-clamp = 0) are folded into ``young``.
    """
    return (data.age_group >= 1).long()


def metrics_row(
    *,
    name: str,
    out,
    data,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    sensitive_full: torch.Tensor,
) -> dict:
    """Compute the full metric vector for one ModelOutput on ``age_old``."""
    pred = out.pred.cpu().long()
    s_test = sensitive_full[test_mask].cpu().long()
    y_test = data.y[test_mask].cpu().long()

    # True / predicted rates per binary group
    m1 = s_test == 1
    m0 = s_test == 0
    true_g0 = float(y_test[m0].float().mean()) if m0.any() else float("nan")
    true_g1 = float(y_test[m1].float().mean()) if m1.any() else float("nan")
    pred_g0 = float(pred[m0].float().mean()) if m0.any() else float("nan")
    pred_g1 = float(pred[m1].float().mean()) if m1.any() else float("nan")
    true_gap = true_g1 - true_g0
    pred_gap = pred_g1 - pred_g0
    excess = pred_gap - true_gap

    # PPV per group : P(y=1 | ŷ=1, s)
    yhat1_g0 = (pred == 1) & m0
    yhat1_g1 = (pred == 1) & m1
    ppv_0 = float(y_test[yhat1_g0].float().mean()) if yhat1_g0.any() else float("nan")
    ppv_1 = float(y_test[yhat1_g1].float().mean()) if yhat1_g1.any() else float("nan")
    ppv_gap = ppv_1 - ppv_0 if not (np.isnan(ppv_0) or np.isnan(ppv_1)) else float("nan")

    # Output-level fairness
    delta_dp = demographic_parity_diff(pred, s_test)
    delta_eo = equal_opportunity_diff(pred, y_test, s_test)

    # Leakage on embeddings (when present)
    if out.embeddings is not None:
        leak = sensitive_leakage(
            out.embeddings.cpu(), sensitive_full.cpu(), train_mask, test_mask, seed=SEED
        )
    else:
        leak = float("nan")

    # Recompute F1/acc on test (out.f1 / out.acc are computed in pred-order
    # which matches test_mask boolean order, same as y_test here, so they
    # should agree — sanity-recompute for paranoia).
    f1 = float(f1_score(y_test.numpy(), pred.numpy(), average="macro", zero_division=0))
    acc = float(accuracy_score(y_test.numpy(), pred.numpy()))

    return {
        "method": name,
        "f1": f1,
        "acc": acc,
        "true_g0": true_g0,
        "true_g1": true_g1,
        "pred_g0": pred_g0,
        "pred_g1": pred_g1,
        "true_gap": true_gap,
        "pred_gap": pred_gap,
        "excess_gap": excess,
        "delta_dp": delta_dp,
        "delta_eo": delta_eo,
        "ppv_g0": ppv_0,
        "ppv_g1": ppv_1,
        "ppv_gap": ppv_gap,
        "leakage_auc": leak,
    }


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    setup_seeds(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading data...")
    data = load_pokec_z(RAW_DIR)
    data = preprocess(data, sensitive_cols=["gender", "region"])
    data.age_group = data.age_group.clamp(min=0)
    data = data.to(device)
    data.age_group = data.age_group.to(device)

    # Target = smoking regularly
    y = build_smoking_target(data)
    print(f"  target = {TARGET_COL!r}  prevalence = {float(y.float().mean()) * 100:.2f}%")

    # Drop leak cols (other smoking-related columns)
    leak = [c for c in data.feature_cols if "fajc" in c.lower() or c == "nefajcim"]
    dropped = drop_leak_features(data, leak)
    print(f"  dropped leak cols: {dropped}")

    # Sensitive = age_old (binary). Patch data.gender so the runners that
    # hardcode ``data.gender`` (run_resampling / run_fairdrop / run_fairgnn_grid)
    # use age_old instead.
    age_old = build_age_old(data)
    data.age_old = age_old
    print(
        f"  sensitive = age_old  P(s=1) = {float(age_old.float().mean()) * 100:.2f}%  "
        f"(adult+senior ≥ 25)"
    )
    gender_backup = data.gender
    data.gender = age_old  # patch in-place

    n = data.y.shape[0]
    train_idx, val_idx, test_idx = make_splits(
        n=n, y=data.y.cpu(), sensitive=age_old.cpu(), seed=SEED
    )
    train_mask = torch.zeros(n, dtype=torch.bool, device=device)
    val_mask = torch.zeros(n, dtype=torch.bool, device=device)
    test_mask = torch.zeros(n, dtype=torch.bool, device=device)
    train_mask[train_idx.to(device)] = True
    val_mask[val_idx.to(device)] = True
    test_mask[test_idx.to(device)] = True
    print(
        f"  splits  train={int(train_mask.sum())}  val={int(val_mask.sum())}  test={int(test_mask.sum())}"
    )

    cfg = _Cfg()
    cfg.fairgnn_lambdas = list(cfg.fairgnn_lambdas)  # mutable list expected
    cfg.noise_levels = []
    cfg.edge_drop_rates = []

    rows = []

    def _record(name: str, out_obj):
        row = metrics_row(
            name=name,
            out=out_obj,
            data=data,
            train_mask=train_mask,
            test_mask=test_mask,
            sensitive_full=age_old,
        )
        rows.append(row)
        print(
            f"    {name:32s}  f1={row['f1']:.3f}  ΔDP={row['delta_dp'] * 100:+5.2f}pp  "
            f"true={row['true_gap'] * 100:+5.2f}pp  pred={row['pred_gap'] * 100:+5.2f}pp  "
            f"excess={row['excess_gap'] * 100:+5.2f}pp"
        )

    # ── Baseline ────────────────────────────────────────────────────────────
    print("\n[1] Baseline GraphSAGE")
    t0 = time.time()
    baseline = run_baseline(data, train_mask, val_mask, test_mask, cfg, seed=SEED)
    print(f"  trained in {time.time() - t0:.1f}s")
    _record("GraphSAGE", baseline)

    # ── Pre-process ─────────────────────────────────────────────────────────
    print("\n[2] Resampling (oversample on y × age_old)")
    t0 = time.time()
    out = run_resampling(data, train_mask, val_mask, test_mask, cfg, seed=SEED)
    print(f"  trained in {time.time() - t0:.1f}s")
    _record("GraphSAGE+Resampling", out)

    print("\n[3] Reweighting (Kamiran-Calders on age_old)")
    t0 = time.time()
    out = run_reweighted(
        data, train_mask, val_mask, test_mask, cfg, seed=SEED, sensitive_name="age_old"
    )
    print(f"  trained in {time.time() - t0:.1f}s")
    _record("GraphSAGE+Reweighted@age_old", out)

    print("\n[4] FairDrop (down-sample intra-group edges on age_old)")
    t0 = time.time()
    out = run_fairdrop(data, train_mask, val_mask, test_mask, cfg, seed=SEED)
    print(f"  trained in {time.time() - t0:.1f}s")
    _record("GraphSAGE+FairDrop@age_old", out)

    # ── In-training ─────────────────────────────────────────────────────────
    print("\n[5] FairGNN grid (adversarial, λ ∈ {0.1, 0.5, 1.0, 5.0})")
    t0 = time.time()
    out, grid = run_fairgnn_grid(data, train_mask, val_mask, test_mask, cfg, seed=SEED)
    print(f"  trained in {time.time() - t0:.1f}s  chosen λ={out.extra.get('chosen_lambda')}")
    _record(f"FairGNN(λ={out.extra.get('chosen_lambda')})@age_old", out)

    # ── Post-process ────────────────────────────────────────────────────────
    print("\n[6] INLP@age_old (project axis from baseline embeddings + refit head)")
    t0 = time.time()
    out_inlp = apply_inlp_to_embeddings(
        baseline, data, train_mask, val_mask, test_mask, sensitive_name="age_old"
    )
    print(f"  done in {time.time() - t0:.1f}s")
    _record("GraphSAGE+INLP@age_old", out_inlp)

    print("\n[7] INLP + DPT@age_old (parity threshold on top of INLP)")
    out_dpt = apply_equal_opportunity_threshold(
        out_inlp, data, val_mask, test_mask, sensitive_name="age_old", strategy="demographic_parity"
    )
    _record("GraphSAGE+INLP+DPT@age_old", out_dpt)

    print("\n[8] INLP + EOT@age_old (equal-opportunity threshold on top of INLP)")
    out_eot = apply_equal_opportunity_threshold(
        out_inlp, data, val_mask, test_mask, sensitive_name="age_old", strategy="equal_opportunity"
    )
    _record("GraphSAGE+INLP+EOT@age_old", out_eot)

    # Restore gender
    data.gender = gender_backup

    # ── Output ──────────────────────────────────────────────────────────────
    table = pl.DataFrame(rows)
    out_path = OUT_DIR / "toolbox_smoking_age.csv"
    table.write_csv(out_path)
    print(f"\n=== Summary table → {out_path} ===")
    with pl.Config(tbl_rows=20, fmt_str_lengths=40, float_precision=4):
        print(
            table.select(
                "method",
                (pl.col("f1") * 100).round(2).alias("f1%"),
                (pl.col("true_gap") * 100).round(2).alias("true_gap_pp"),
                (pl.col("pred_gap") * 100).round(2).alias("pred_gap_pp"),
                (pl.col("excess_gap") * 100).round(2).alias("excess_pp"),
                (pl.col("delta_dp") * 100).round(2).alias("ΔDP_pp"),
                (pl.col("delta_eo") * 100).round(2).alias("ΔEO_pp"),
                (pl.col("ppv_gap") * 100).round(2).alias("PPV_gap_pp"),
                pl.col("leakage_auc").round(3).alias("leakage"),
            )
        )


if __name__ == "__main__":
    main()
