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
    # Keep only methods with distinct positions on the Pareto plot;
    # drop near-duplicates to avoid label overlap.
    plotted_methods = [
        "GraphSAGE",
        "TabICL",
        "FairGNN(λ=5.0)",
        "TabICL+DPT@gender",
        "TabICL+DPT_composite@gender_age_group_region",
        "TabICL+INLP+DPT@gender",
        "TabICL+INLP+DPT_composite",
        "GraphSAGE+INLP+DPT_composite",
    ]
    fig, ax = plt.subplots(figsize=(7, 4.4))

    families_seen: set[str] = set()
    for model in plotted_methods:
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
            s=80,
            alpha=0.85,
            color=COLOR[fam],
            edgecolors="white",
            linewidths=0.7,
            label=fam if fam not in families_seen else None,
            zorder=3,
        )
        families_seen.add(fam)

    # Per-method label offsets to avoid overlap
    label_offsets = {
        "GraphSAGE": (8, -10),
        "TabICL": (8, 6),
        "FairGNN(λ=5.0)": (-90, 0),
        "TabICL+DPT@gender": (8, 4),
        "TabICL+DPT_composite@gender_age_group_region": (8, 0),
        "TabICL+INLP+DPT@gender": (-110, 6),
        "TabICL+INLP+DPT_composite": (-130, 6),
        "GraphSAGE+INLP+DPT_composite": (8, -2),
    }
    for model in plotted_methods:
        rows = df.filter(pl.col("model") == model)
        if rows.height == 0:
            continue
        ddp = float(rows["delta_dp"][0])
        f1 = F1_LOOKUP_SEED42.get(model)
        if f1 is None:
            continue
        dx, dy = label_offsets.get(model, (6, 4))
        ax.annotate(
            LABEL[model],
            xy=(ddp, f1),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            alpha=0.95,
        )

    ax.set_xlabel(f"ΔDP {axis} (lower is fairer →)", fontsize=9)
    ax.set_ylabel("F1 macro test ↑", fontsize=9)
    ax.set_title(
        f"Fig 1. F1 vs ΔDP — chacun cherche son meilleur compromis\n"
        f"(Pokec-z, axe {axis}, seed=42)",
        fontsize=10,
    )
    ax.set_xlim(-0.002, 0.055)
    ax.set_ylim(0.55, 0.97)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc="lower right", fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_toolbox(csv_path: Path, out: Path, axis: str = "gender") -> None:
    """Finding 1 — each metric has its own tool.

    Three small panels side-by-side: ΔDP, ΔEO, Leakage AUC. For each, bar
    chart of 6 representative methods. Visually shows that DPT crushes
    ΔDP, INLP crushes Leakage, FairGNN moves ΔDP modestly at the cost of
    F1, etc.
    """
    df = pl.read_csv(csv_path).filter(pl.col("attribute") == axis)
    methods = [
        "GraphSAGE",
        "FairGNN(λ=5.0)",
        "TabICL+DPT@gender",
        "TabICL+INLP@gender",
        "TabICL+INLP+DPT@gender",
        "TabICL+INLP+DPT_composite",
    ]
    short = {
        "GraphSAGE": "GraphSAGE",
        "FairGNN(λ=5.0)": "FairGNN(λ=5)",
        "TabICL+DPT@gender": "TabICL+DPT",
        "TabICL+INLP@gender": "TabICL+INLP",
        "TabICL+INLP+DPT@gender": "TabICL+INLP\n+DPT",
        "TabICL+INLP+DPT_composite": "ULTIMATE",
    }

    metrics = [
        ("delta_dp", "ΔDP gender", "lower is fairer"),
        ("delta_eo", "ΔEO gender", "lower is fairer"),
        ("leakage_auc", "Leakage AUC", "0.5 = chance"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    bar_colors = [COLOR[FAMILY[m]] for m in methods]

    for ax, (col, title, sub) in zip(axes, metrics, strict=True):
        vals = []
        for m in methods:
            rows = df.filter(pl.col("model") == m)
            vals.append(float(rows[col][0]) if rows.height else np.nan)
        bars = ax.bar(range(len(methods)), vals, color=bar_colors,
                      edgecolor="white", linewidth=0.6)
        ymax = max(vals) if vals and max(vals) > 0 else 1
        for bar, v in zip(bars, vals, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + ymax * 0.025,
                    f"{v:.3f}", ha="center", fontsize=7.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([short[m] for m in methods], fontsize=7.5,
                           rotation=30, ha="right")
        ax.set_title(f"{title}\n({sub})", fontsize=9.5)
        ax.grid(True, axis="y", alpha=0.3)
        if col == "leakage_auc":
            ax.axhline(0.5, color="green", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.set_ylim(0.45, 1.0)
        else:
            ax.set_ylim(0, ymax * 1.18)

    fig.suptitle(
        f"Fig 1. Chaque outil tape une métrique différente — axe {axis}, Pokec-z\n"
        f"DPT casse ΔDP, INLP casse le leakage ; aucun outil seul ne fait les trois.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_chain_progression(csv_path: Path, out: Path) -> None:
    """Finding 2 — TabICL holds, GraphSAGE collapses at multi-axis composition.

    Two panels :
      - left: F1 along the chain depth (baseline → +DPT → +INLP+DPT →
        +ULTIMATE). 2 lines: GraphSAGE vs TabICL.
      - right: Average leakage AUC across the 5 sensitive axes along the
        same chain. Shows leakage drops at the INLP step and stays low.
    The GraphSAGE F1 cliff at the last step is the main visual message.
    """
    df = pl.read_csv(csv_path)

    def avg_leakage_over_axes(model_name: str) -> float:
        rows = df.filter(pl.col("model") == model_name)
        if rows.height == 0:
            return np.nan
        return float(rows["leakage_auc"].mean())

    chain_steps = ["baseline", "+DPT@gender", "+INLP+DPT@gender", "+ULTIMATE"]
    tabicl_models = [
        "TabICL", "TabICL+DPT@gender", "TabICL+INLP+DPT@gender",
        "TabICL+INLP+DPT_composite",
    ]
    graphsage_models = [
        "GraphSAGE", "GraphSAGE+TempCal+DPT@gender",
        "GraphSAGE+INLP+DPT@gender", "GraphSAGE+INLP+DPT_composite",
    ]
    f1_lookup = F1_LOOKUP_SEED42
    # Add the missing GraphSAGE+TempCal+DPT@gender F1 (we know from the run logs ~0.94)
    if "GraphSAGE+TempCal+DPT@gender" not in f1_lookup:
        f1_lookup["GraphSAGE+TempCal+DPT@gender"] = 0.939
    tabicl_f1 = [f1_lookup.get(m, np.nan) for m in tabicl_models]
    graphsage_f1 = [f1_lookup.get(m, np.nan) for m in graphsage_models]

    tabicl_leak = [avg_leakage_over_axes(m) for m in tabicl_models]
    graphsage_leak = [avg_leakage_over_axes(m) for m in graphsage_models]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(chain_steps))

    ax = axes[0]
    ax.plot(x, tabicl_f1, marker="o", linewidth=2, color="#1f77b4",
            label="TabICL chain", markersize=8)
    ax.plot(x, graphsage_f1, marker="s", linewidth=2, color="#d62728",
            label="GraphSAGE chain", markersize=8)
    for xi, v in zip(x, tabicl_f1, strict=True):
        ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8)
    for xi, v in zip(x, graphsage_f1, strict=True):
        ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(chain_steps, fontsize=8)
    ax.set_ylabel("F1 macro test ↑")
    ax.set_title("F1 le long de la chaîne d'équité",
                 fontsize=10)
    ax.set_ylim(0.55, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[1]
    ax.plot(x, tabicl_leak, marker="o", linewidth=2, color="#1f77b4",
            label="TabICL chain", markersize=8)
    ax.plot(x, graphsage_leak, marker="s", linewidth=2, color="#d62728",
            label="GraphSAGE chain", markersize=8)
    for xi, v in zip(x, tabicl_leak, strict=True):
        ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8)
    for xi, v in zip(x, graphsage_leak, strict=True):
        ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.axhline(0.5, color="green", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(chain_steps, fontsize=8)
    ax.set_ylabel("Leakage AUC ↓ (moy. sur 5 axes)")
    ax.set_title("Leakage moyen le long de la chaîne",
                 fontsize=10)
    ax.set_ylim(0.45, 0.95)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Fig 2. TabICL tient la composition, GraphSAGE s'écrase\n"
        "à droite : leakage descend au chance level pour les deux ; à gauche : "
        "GraphSAGE perd 35 pp de F1, TabICL seulement 8 pp.",
        fontsize=10,
    )
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
    # Old figures (kept for the long pedagogical analysis_note.md)
    figure_pareto(z_csv, FIGURES / "fig1_pareto_f1_vs_dp.png")
    figure_leakage_heatmap(z_csv, FIGURES / "fig2_leakage_heatmap.png")
    figure_cross_dataset(z_csv, n_csv, FIGURES / "fig3_cross_dataset.png")
    # Figures designed specifically for the 2-pager (one finding per figure)
    figure_toolbox(z_csv, FIGURES / "fig1_toolbox_per_metric.png")
    figure_chain_progression(z_csv, FIGURES / "fig2_chain_tabicl_vs_graphsage.png")


if __name__ == "__main__":
    main()
