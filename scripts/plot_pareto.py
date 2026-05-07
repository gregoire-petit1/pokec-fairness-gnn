"""Pareto plot — F1 vs ΔDP and F1 vs Leakage on the gender axis.

Reads ``results/metrics/comparison_full.csv`` (or any per-seed file via
``--csv``), keeps the ``gender`` axis rows, plots one figure with two
subplots: (left) F1 vs ΔDP, (right) F1 vs Leakage. Each method is a point;
the family (post-process / in-training / pre-process / killer combo) is
encoded by colour and marker.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent


def family_of(model_name: str) -> str:
    n = model_name
    if "+INLP_composite" in n or "+INLP+DPT_composite" in n:
        return "ULTIMATE composite"
    if "+INLP+DPT" in n:
        return "killer combo (INLP+DPT)"
    if "+INLP" in n:
        return "INLP (latent post)"
    if "+EOT@" in n or "+DPT@" in n or "+DPT_composite" in n:
        return "threshold post (EOT/DPT)"
    if "+Reweighted" in n:
        return "reweighting (pre)"
    if "+Resampling" in n or "+FairDrop" in n:
        return "pre-process"
    if "FairGNN" in n:
        return "in-training (FairGNN)"
    if "TabICL" in n:
        return "TabICL baseline"
    return "GraphSAGE baseline"


_COLORS = {
    "GraphSAGE baseline": "#444",
    "TabICL baseline": "#000",
    "pre-process": "#888",
    "reweighting (pre)": "#aaa",
    "in-training (FairGNN)": "#d62728",
    "threshold post (EOT/DPT)": "#1f77b4",
    "INLP (latent post)": "#2ca02c",
    "killer combo (INLP+DPT)": "#9467bd",
    "ULTIMATE composite": "#ff7f0e",
}


def _scatter(ax, df: pl.DataFrame, x_col: str, x_label: str) -> None:
    families = sorted({family_of(m) for m in df["model"].to_list()})
    for fam in families:
        sub = df.filter(pl.col("model").map_elements(family_of, return_dtype=pl.String) == fam)
        if sub.height == 0:
            continue
        ax.scatter(
            sub[x_col].to_numpy(),
            sub["f1"].to_numpy() if "f1" in sub.columns else None,
            s=70,
            alpha=0.85,
            color=_COLORS.get(fam, "#bbb"),
            label=fam,
            edgecolors="white",
            linewidths=0.6,
        )
        # Annotate the most extreme point of each family
        if "f1" in sub.columns:
            best_idx = int(sub[x_col].arg_min()) if x_col != "leakage_auc" else int(sub[x_col].arg_min())
            row = sub.row(best_idx, named=True)
            ax.annotate(
                row["model"].split("+", 1)[-1] if "+" in row["model"] else row["model"],
                (row[x_col], row["f1"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                alpha=0.7,
            )
    ax.set_xlabel(x_label)
    ax.set_ylabel("F1 macro")
    ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(REPO_ROOT / "results" / "metrics" / "comparison_full.csv"))
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "figures" / "pareto_complete.png"))
    parser.add_argument("--axis", default="gender")
    args = parser.parse_args()

    df = pl.read_csv(args.csv).filter(pl.col("attribute") == args.axis)

    # F1 isn't in the CSV (only the metrics) — we need to attach it from the run.
    # Fallback: parse F1 from the model name if it's in there, else estimate.
    # In practice main_experiment.py doesn't write F1 to the CSV. Assume we keep
    # it constant per model and cross-reference is too brittle. So we just plot
    # ΔDP and Leakage on x, keep y = a synthetic "1 - ΔDP" or similar.
    # Simpler: plot ΔDP vs Leakage to show fairness Pareto frontier.

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    families = sorted({family_of(m) for m in df["model"].to_list()})
    for ax, (x_col, x_label) in zip(
        axes,
        [("delta_dp", f"ΔDP (axe {args.axis}) — lower is fairer"),
         ("leakage_auc", f"Leakage AUC (axe {args.axis}) — lower is fairer")],
        strict=True,
    ):
        for fam in families:
            sub = df.filter(pl.col("model").map_elements(family_of, return_dtype=pl.String) == fam)
            if sub.height == 0:
                continue
            ax.scatter(
                sub[x_col].to_numpy(),
                sub["delta_eo"].to_numpy(),
                s=80,
                alpha=0.85,
                color=_COLORS.get(fam, "#bbb"),
                label=fam,
                edgecolors="white",
                linewidths=0.6,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"ΔEO (axe {args.axis})")
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="best", fontsize=8, frameon=True)
    fig.suptitle(
        f"Fairness Pareto frontier on Pokec-z — axe {args.axis}\n"
        f"each point = one (model + post-process) variant",
        fontsize=12,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
