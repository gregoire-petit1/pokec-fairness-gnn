"""Minority fairness — re-extracted Hungarian / Roma axes from SNAP raw.

The FairGNN curation reduced ``spoken_languages`` to 8 binary indicator columns,
dropping ``madarsky`` (Hungarian), ``cigansky``/``romsky`` (Roma) and
``posunkovu`` (Slovak Sign Language). We re-extracted these axes from the raw
1.7 GB SNAP dump (``scripts/extract_snap_languages.py``) and now run a 4-part
fairness audit on the **minority axes that the canonical FairGNN benchmark
literally cannot see**.

Tests
-----
1. **Standard fairness**: ΔDP, ΔEO, leakage AUC, CF score on each minority
   axis — baseline GraphSAGE vs INLP@axis + DPT@axis chain (the post-hoc
   toolbox demonstrated to work on gender/region in our 2-pager).

2. **Hidden ethnic proxy**: ``region=1`` captures 4× more Hungarian speakers
   than ``region=0``. So debiasing on ``region`` (without ever touching
   ``hungarian``) might **accidentally** debias on ``hungarian``. We measure
   ΔDP@hungarian on the GraphSAGE_INLP_region cache and compare to baseline.
   If it drops, the geographic axis is acting as an ethnic proxy under the
   hood — a strong methodological point: a model can be "ethnically fair by
   geographic accident" without anyone noticing.

3. **Robustness on minorities**: drop ``rate ∈ {0, 0.1, 0.2, 0.3}`` of edges,
   re-evaluate the frozen GraphSAGE on the perturbed graph, and measure F1
   on the hungarian-only test nodes vs. non-hungarian. If the GNN relies on
   ethnic homophily for its prediction, the hungarian subset should degrade
   faster — that's the GNN-amplifies-unfairness-on-minorities hypothesis.

4. **Counterfactual**: NIFTY-style flip on the hungarian dimension —
   reuses ``counterfactual_fairness_score`` from ``src/fairness/metrics.py``.

Run::

    .venv/bin/python scripts/run_minority_fairness.py

Outputs (under ``results/metrics/``):
    minority_fairness_table.csv      ← Test 1 + 2 + 4 in long format
    minority_robustness.csv          ← Test 3 across edge-drop rates
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Re-use the orchestration helpers already validated for gender/region/age_group.
from scripts.main_experiment import (  # noqa: E402
    _load_model_output,
    apply_equal_opportunity_threshold,
    apply_inlp_to_embeddings,
)
from src.data.loader import load_pokec_z  # noqa: E402
from src.data.minorities import attach_minorities  # noqa: E402
from src.data.preprocessing import preprocess  # noqa: E402
from src.data.splits import make_splits  # noqa: E402
from src.fairness.metrics import (  # noqa: E402
    assortative_mixing_coefficient,
    counterfactual_fairness_score,
    demographic_parity_diff,
    equal_opportunity_diff,
    sensitive_leakage,
)
from src.models.graphsage import GraphSAGE  # noqa: E402
from src.robustness.perturbations import drop_edges  # noqa: E402

OUT_DIR = ROOT / "results" / "metrics"
CACHE_DIR = ROOT / "results" / "cache" / "seed42"
SEED = 42
RAW_DIR = ROOT / "data" / "raw" / "pokec-z"
MINORITY_AXES = ("hungarian", "roma")  # `sign` has 20 users → too small for stat
EDGE_DROP_RATES = (0.0, 0.1, 0.2, 0.3)


def setup_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_masks(n: int, train_idx: torch.Tensor, val_idx: torch.Tensor, test_idx: torch.Tensor):
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def _f1_macro(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float(f1_score(y.cpu().numpy(), pred.cpu().numpy(), average="macro", zero_division=0))


def _accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float(accuracy_score(y.cpu().numpy(), pred.cpu().numpy()))


# ---------------------------------------------------------------------------
# Test 1 — Standard fairness on each minority axis (baseline + INLP+DPT chain)
# ---------------------------------------------------------------------------


def test1_standard_fairness(
    data,
    baseline,
    test_idx: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    device: torch.device,
) -> pl.DataFrame:
    rows = []
    # Cache pred is in test_mask boolean order (sorted node ids), NOT test_idx order.
    # See main_experiment._save_model_output for the convention.
    y_test = data.y[test_mask]

    for axis in MINORITY_AXES:
        s = getattr(data, axis)
        s_test = s[test_mask]
        n_pos = int(s_test.sum().item())
        if n_pos < 5:
            print(f"  [SKIP] axis={axis!r}: only {n_pos} positive in test — too small")
            continue

        r = assortative_mixing_coefficient(data.edge_index.cpu(), s.cpu())

        # Baseline
        rows.append(
            _row(
                method="GraphSAGE",
                axis=axis,
                pred=baseline.pred,
                proba=baseline.proba,
                emb=baseline.embeddings,
                s_full=s,
                s_test=s_test,
                y_test=y_test,
                y_full=data.y,
                train_mask=train_mask,
                test_mask=test_mask,
                assortativity=r,
                n_minority_test=n_pos,
            )
        )

        # INLP@axis only
        out_inlp = apply_inlp_to_embeddings(
            baseline, data, train_mask, val_mask, test_mask, sensitive_name=axis
        )
        rows.append(
            _row(
                method=f"GraphSAGE+INLP@{axis}",
                axis=axis,
                pred=out_inlp.pred,
                proba=out_inlp.proba,
                emb=out_inlp.embeddings,
                s_full=s,
                s_test=s_test,
                y_test=y_test,
                y_full=data.y,
                train_mask=train_mask,
                test_mask=test_mask,
                assortativity=r,
                n_minority_test=n_pos,
            )
        )

        # INLP@axis + DPT@axis  (canonical post-hoc chain from the 2-pager)
        out_chain = apply_equal_opportunity_threshold(
            out_inlp,
            data,
            val_mask,
            test_mask,
            sensitive_name=axis,
            strategy="demographic_parity",
        )
        rows.append(
            _row(
                method=f"GraphSAGE+INLP+DPT@{axis}",
                axis=axis,
                pred=out_chain.pred,
                proba=out_chain.proba,
                emb=out_chain.embeddings,
                s_full=s,
                s_test=s_test,
                y_test=y_test,
                y_full=data.y,
                train_mask=train_mask,
                test_mask=test_mask,
                assortativity=r,
                n_minority_test=n_pos,
            )
        )

    return pl.DataFrame(rows)


def _row(
    *,
    method: str,
    axis: str,
    pred: torch.Tensor,
    proba: np.ndarray | None,
    emb: torch.Tensor | None,
    s_full: torch.Tensor,
    s_test: torch.Tensor,
    y_test: torch.Tensor,
    y_full: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    assortativity: float,
    n_minority_test: int,
) -> dict:
    """Compute the metric vector for one (method, axis) cell."""
    delta_dp = demographic_parity_diff(pred.cpu(), s_test.cpu())
    delta_eo = equal_opportunity_diff(pred.cpu(), y_test.cpu(), s_test.cpu())
    f1 = _f1_macro(pred, y_test)
    acc = _accuracy(pred, y_test)

    leak = float("nan")
    cf = float("nan")
    if emb is not None:
        leak = sensitive_leakage(emb.cpu(), s_full.cpu(), train_mask, test_mask, seed=SEED)
        cf = counterfactual_fairness_score(
            emb.cpu(), s_full.cpu(), y_full.cpu(), train_mask, test_mask, seed=SEED
        )

    return {
        "method": method,
        "axis": axis,
        "n_minority_test": n_minority_test,
        "assortativity_r": float(assortativity),
        "f1": f1,
        "acc": acc,
        "delta_dp": delta_dp,
        "delta_eo": delta_eo,
        "leakage_auc": leak,
        "cf_score": cf,
    }


# ---------------------------------------------------------------------------
# Test 2 — Hidden ethnic proxy: does INLP@region accidentally debias hungarian?
# ---------------------------------------------------------------------------


def test2_hidden_proxy(
    data,
    baseline,
    test_idx: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> pl.DataFrame:
    """Test 2 of the docstring. We use the existing GraphSAGE_INLP_region cache
    if present (produced by the canonical pipeline), otherwise compute it on
    the fly. The key comparison is ΔDP@hungarian baseline → after-INLP@region.
    """
    cache_path = CACHE_DIR / "GraphSAGE_INLP_region.pt"
    if cache_path.exists():
        print(f"  using cached {cache_path.name}")
        inlp_region = _load_model_output(cache_path, baseline.pred.device)
    else:
        print("  recomputing INLP@region on the fly")
        inlp_region = apply_inlp_to_embeddings(
            baseline,
            data,
            train_mask,
            _bool_to(data, test_mask),
            test_mask,
            sensitive_name="region",
        )

    rows = []
    # Cache pred uses test_mask boolean order, not test_idx.
    y_test = data.y[test_mask]

    for axis in ("region", "hungarian", "roma"):
        s = getattr(data, axis)
        s_test = s[test_mask]
        if s_test.sum().item() < 5:
            continue
        for variant in ("baseline", "INLP@region"):
            out = baseline if variant == "baseline" else inlp_region
            delta_dp = demographic_parity_diff(out.pred.cpu(), s_test.cpu())
            delta_eo = equal_opportunity_diff(out.pred.cpu(), y_test.cpu(), s_test.cpu())
            leak = sensitive_leakage(
                out.embeddings.cpu(), s.cpu(), train_mask, test_mask, seed=SEED
            )
            rows.append(
                {
                    "variant": variant,
                    "axis_measured": axis,
                    "delta_dp": delta_dp,
                    "delta_eo": delta_eo,
                    "leakage_auc": leak,
                    "f1": _f1_macro(out.pred, y_test),
                }
            )
    return pl.DataFrame(rows)


def _bool_to(data, test_mask):  # noqa: ARG001
    return test_mask  # placeholder; kept for signature symmetry with main_experiment


# ---------------------------------------------------------------------------
# Test 3 — Robustness on minorities (edge-drop ablation, per-subgroup F1)
# ---------------------------------------------------------------------------


def test3_robustness_minorities(
    data,
    test_idx: torch.Tensor,
    state_dict: dict,
    device: torch.device,
) -> pl.DataFrame:
    """Drop ``rate`` of edges, re-evaluate frozen GraphSAGE, measure F1 per
    subgroup on the test set. Per-axis subgroup = ``s == 1`` vs ``s == 0``.
    """
    in_dim = data.x.shape[1]
    model = GraphSAGE(in_channels=in_dim, hidden_channels=256, out_channels=2, num_layers=2).to(
        device
    )
    model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
    model.eval()

    rows = []
    y_test_full = data.y[test_idx]

    with torch.no_grad():
        for rate in EDGE_DROP_RATES:
            ei_perturbed = drop_edges(data.edge_index.cpu(), rate, seed=SEED).to(device)
            logits = model(data.x, ei_perturbed)
            pred = logits.argmax(dim=1)
            pred_test = pred[test_idx]
            f1_global = _f1_macro(pred_test, y_test_full)
            acc_global = _accuracy(pred_test, y_test_full)

            for axis in MINORITY_AXES:
                s_test = getattr(data, axis)[test_idx]
                m_pos = s_test == 1
                m_neg = s_test == 0
                if m_pos.sum().item() < 5:
                    f1_pos = float("nan")
                else:
                    f1_pos = _f1_macro(pred_test[m_pos], y_test_full[m_pos])
                f1_neg = _f1_macro(pred_test[m_neg], y_test_full[m_neg])
                rows.append(
                    {
                        "edge_drop_rate": float(rate),
                        "axis": axis,
                        "n_minority_test": int(m_pos.sum().item()),
                        "f1_minority": f1_pos,
                        "f1_majority": f1_neg,
                        "f1_gap": f1_neg - f1_pos,
                        "f1_global": f1_global,
                        "acc_global": acc_global,
                    }
                )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    setup_seeds(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading Pokec-z + minority axes...")
    data = load_pokec_z(RAW_DIR)
    data = preprocess(data, sensitive_cols=["gender", "region"])
    data = attach_minorities(data, RAW_DIR)
    data = data.to(device)

    n = data.y.shape[0]
    print(f"  n={n:,}  hungarian={int(data.hungarian.sum())}  roma={int(data.roma.sum())}")

    train_idx, val_idx, test_idx = make_splits(
        n=n, y=data.y.cpu(), sensitive=data.gender.cpu(), seed=SEED
    )
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    train_mask, val_mask, test_mask = _build_masks(n, train_idx, val_idx, test_idx)

    baseline_path = CACHE_DIR / "GraphSAGE.pt"
    if not baseline_path.exists():
        raise FileNotFoundError(f"{baseline_path} missing — run scripts/main_experiment.py first.")
    baseline = _load_model_output(baseline_path, device)

    # ── Test 1 ──────────────────────────────────────────────────────────────
    print("\n=== Test 1 — Standard fairness on minority axes ===")
    t0 = time.time()
    table1 = test1_standard_fairness(
        data, baseline, test_idx, train_mask, val_mask, test_mask, device
    )
    print(f"  done in {time.time() - t0:.1f}s")
    with pl.Config(tbl_rows=20, fmt_str_lengths=40, float_precision=4):
        print(table1)
    table1_path = OUT_DIR / "minority_fairness_table.csv"
    table1.write_csv(table1_path)
    print(f"→ {table1_path}")

    # ── Test 2 ──────────────────────────────────────────────────────────────
    print("\n=== Test 2 — Is region a hidden proxy for hungarian? ===")
    t0 = time.time()
    table2 = test2_hidden_proxy(data, baseline, test_idx, train_mask, test_mask)
    print(f"  done in {time.time() - t0:.1f}s")
    with pl.Config(tbl_rows=20, fmt_str_lengths=30, float_precision=4):
        print(table2)
    table2_path = OUT_DIR / "minority_proxy_test.csv"
    table2.write_csv(table2_path)
    print(f"→ {table2_path}")

    # ── Test 3 ──────────────────────────────────────────────────────────────
    print("\n=== Test 3 — GNN robustness on minorities under edge drop ===")
    state_dict = baseline.extra.get("model_state_dict")
    if state_dict is None:
        print("  [SKIP] baseline cache lacks model_state_dict — re-run main_experiment.py")
    else:
        t0 = time.time()
        table3 = test3_robustness_minorities(data, test_idx, state_dict, device)
        print(f"  done in {time.time() - t0:.1f}s")
        with pl.Config(tbl_rows=20, fmt_str_lengths=30, float_precision=4):
            print(table3)
        table3_path = OUT_DIR / "minority_robustness.csv"
        table3.write_csv(table3_path)
        print(f"→ {table3_path}")


if __name__ == "__main__":
    main()
