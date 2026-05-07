"""Generate the 3 headline figures for the 2-page report.

Figure 1 — Pareto F1 ↔ ΔDP : one point per (model + post-process) variant
on the gender axis. Bottom-right = ideal. Shows the ULTIMATE combo and
TabICL+DPT_composite as the Pareto-optimal cluster.

Figure 2 — Leakage AUC heatmap : rows = methods, cols = 5 sensitive axes.
Color = leakage AUC (lower is fairer ; 0.50 = chance). Shows that
INLP-based methods uniformly drop the leakage to chance across all
5 axes.

Figure 3 — Cross-dataset robustness : grouped bars showing ΔDP gender for
key methods, Pokec-z vs Pokec-n side-by-side. Reproduces the finding
across datasets.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS = REPO_ROOT / "results" / "metrics"
FIGURES = REPO_ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


# Hand-curated set of methods to show in the figures (drop noise like every
# (axis × strategy) combination).
KEY_METHODS = [
    "GraphSAGE",
    "TabICL",
    "FairGNN(λ=5.0)",
    "GraphSAGE+Resampling",
    "GraphSAGE+FairDrop",
    "TabICL+EOT@gender",
    "TabICL+DPT@gender",
    "TabICL+DPT_composite@gender_age_group_region",
    "TabICL+INLP@gender",
    "TabICL+INLP+DPT@gender",
    "TabICL+INLP+DPT_composite",
    "GraphSAGE+INLP+DPT_composite",
    "GraphSAGE+INLP+DPT@gender",
]

# (model name → friendly short label)
LABEL = {
    "GraphSAGE": "GraphSAGE",
    "TabICL": "TabICL",
    "FairGNN(λ=5.0)": "FairGNN(λ=5)",
    "GraphSAGE+Resampling": "GS+Resamp",
    "GraphSAGE+FairDrop": "GS+FairDrop",
    "TabICL+EOT@gender": "TabICL+EOT",
    "TabICL+DPT@gender": "TabICL+DPT",
    "TabICL+DPT_composite@gender_age_group_region": "TabICL+DPT_comp",
    "TabICL+INLP@gender": "TabICL+INLP",
    "TabICL+INLP+DPT@gender": "TabICL+INLP+DPT",
    "TabICL+INLP+DPT_composite": "TabICL+INLP+DPT_comp",
    "GraphSAGE+INLP+DPT_composite": "GS+INLP+DPT_comp",
    "GraphSAGE+INLP+DPT@gender": "GS+INLP+DPT",
}

# Family colours (consistent across figures)
FAMILY = {
    "GraphSAGE": "baseline",
    "TabICL": "baseline",
    "FairGNN(λ=5.0)": "in-training",
    "GraphSAGE+Resampling": "pre-process",
    "GraphSAGE+FairDrop": "pre-process",
    "TabICL+EOT@gender": "post-process",
    "TabICL+DPT@gender": "post-process",
    "TabICL+DPT_composite@gender_age_group_region": "post-process",
    "TabICL+INLP@gender": "INLP",
    "TabICL+INLP+DPT@gender": "killer combo",
    "TabICL+INLP+DPT_composite": "ULTIMATE",
    "GraphSAGE+INLP+DPT_composite": "ULTIMATE",
    "GraphSAGE+INLP+DPT@gender": "killer combo",
}

COLOR = {
    "baseline": "#666",
    "pre-process": "#aaa",
    "in-training": "#d62728",
    "post-process": "#1f77b4",
    "INLP": "#2ca02c",
    "killer combo": "#9467bd",
    "ULTIMATE": "#ff7f0e",
}


# ---------------------------------------------------------------------------
# Per-model F1 lookup — F1 is not in the CSV, hand-coded from the run logs.
# ---------------------------------------------------------------------------
F1_LOOKUP_SEED42 = {
    "GraphSAGE": 0.9381,
    "GraphSAGE+Resampling": 0.9381,
    "GraphSAGE+FairDrop": 0.9351,
    "FairGNN(λ=5.0)": 0.8532,
    "TabICL": 0.9483,
    "TabICL+EOT@gender": 0.9459,
    "TabICL+DPT@gender": 0.9449,
    "TabICL+DPT_composite@gender_age_group_region": 0.9411,
    "TabICL+INLP@gender": 0.9459,
    "TabICL+INLP+DPT@gender": 0.9427,
    "TabICL+INLP+DPT_composite": 0.866,
    "GraphSAGE+INLP+DPT_composite": 0.591,
    "GraphSAGE+INLP+DPT@gender": 0.9319,
}


def _annotate(ax, x, y, label, dx=0.0008, dy=0.001, fontsize=7):
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=fontsize,
        alpha=0.85,
    )


def figure_pareto(csv_path: Path, out: Path, axis: str = "gender") -> None:
    df = pl.read_csv(csv_path).filter(pl.col("attribute") == axis)
    fig, ax = plt.subplots(figsize=(9, 6))

    families_seen: set[str] = set()
    for model in KEY_METHODS:
        rows = df.filter(pl.col("model") == model)
        if rows.height == 0:
            continue
        ddp = float(rows["delta_dp"][0])
        f1 = F1_LOOKUP_SEED42.get(model)
        if f1 is None:
            continue
        fam = FAMILY[model]
        ax.scatter(
            ddp, f1,
            s=120,
            alpha=0.85,
            color=COLOR[fam],
            edgecolors="white",
            linewidths=0.8,
            label=fam if fam not in families_seen else None,
            zorder=3,
        )
        families_seen.add(fam)
        ax.annotate(
            LABEL[model],
            xy=(ddp, f1),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            alpha=0.85,
        )

    # "Ideal" reference annotation
    ax.axhline(0.95, color="#bbb", linestyle="--", alpha=0.4, linewidth=0.7)
    ax.text(0.001, 0.952, "F1 = 0.95 (TabICL ceiling)", fontsize=7, color="#888")
    ax.set_xlabel(f"ΔDP on axe {axis} — lower is fairer →", fontsize=10)
    ax.set_ylabel("F1 macro on test ↑ (higher is more accurate)", fontsize=10)
    ax.set_title(
        f"Pareto frontier — fairness vs accuracy on Pokec-z (axe {axis}, seed=42)\n"
        f"bottom-left of the plot = ideal (low ΔDP, high F1)",
        fontsize=11,
    )
    ax.set_xscale("symlog", linthresh=0.001)
    ax.set_xlim(-0.0005, 0.06)
    ax.set_ylim(0.55, 0.97)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_leakage_heatmap(csv_path: Path, out: Path) -> None:
    df = pl.read_csv(csv_path)
    axes = ["gender", "age_group", "region", "gender_x_age", "gender_x_region"]
    methods = [m for m in KEY_METHODS if m in df["model"].to_list()]

    matrix = np.full((len(methods), len(axes)), np.nan)
    for i, model in enumerate(methods):
        for j, ax_name in enumerate(axes):
            sub = df.filter((pl.col("model") == model) & (pl.col("attribute") == ax_name))
            if sub.height > 0:
                matrix[i, j] = float(sub["leakage_auc"][0])

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(methods) + 1.5))
    cmap = plt.cm.RdYlGn_r  # green = low leakage = good, red = high leakage = bad
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(axes)))
    ax.set_xticklabels(axes, rotation=20, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([LABEL[m] for m in methods], fontsize=8)
    for i in range(len(methods)):
        for j in range(len(axes)):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if 0.55 < v < 0.85 else "white")
    cbar = fig.colorbar(im, ax=ax, label="Leakage AUC (lower is fairer; 0.5 = chance)")
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("Leakage across 5 sensitive axes — INLP-based methods reach chance\n"
                 "(seed=42, Pokec-z)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_cross_dataset(z_csv: Path, n_csv: Path, out: Path) -> None:
    zdf = pl.read_csv(z_csv).filter(pl.col("attribute") == "gender")
    ndf = pl.read_csv(n_csv).filter(pl.col("attribute") == "gender")
    methods = [m for m in KEY_METHODS if m in zdf["model"].to_list() and m in ndf["model"].to_list()]
    z_dp = [float(zdf.filter(pl.col("model") == m)["delta_dp"][0]) for m in methods]
    n_dp = [float(ndf.filter(pl.col("model") == m)["delta_dp"][0]) for m in methods]

    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.4
    x = np.arange(len(methods))
    ax.bar(x - width / 2, z_dp, width, color="#1f77b4", label="Pokec-z", alpha=0.85)
    ax.bar(x + width / 2, n_dp, width, color="#ff7f0e", label="Pokec-n", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in methods], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("ΔDP gender")
    ax.set_title(
        "Cross-dataset reproduction — ΔDP gender on Pokec-z vs Pokec-n (seed=42)\n"
        "the ordering of methods is preserved across datasets",
        fontsize=11,
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    z_csv = METRICS / "comparison_seed42.csv"
    n_csv = METRICS / "comparison_pokec_n_seed42.csv"
    figure_pareto(z_csv, FIGURES / "fig1_pareto_f1_vs_dp.png")
    figure_leakage_heatmap(z_csv, FIGURES / "fig2_leakage_heatmap.png")
    figure_cross_dataset(z_csv, n_csv, FIGURES / "fig3_cross_dataset.png")


if __name__ == "__main__":
    main()
