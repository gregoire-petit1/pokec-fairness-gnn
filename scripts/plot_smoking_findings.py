"""Plot the smoking × age fairness findings for the 2-pager.

Two figures :
    1. Horizontal bar of ``excess_gap`` per method, with the true_gap line as
       reference (green band ±1pp around 0 = "model agrees with reality").
       Methods that amplify (red), agree (green), over-correct (blue).
    2. F1 vs |excess_gap| Pareto scatter — Reweighting dominates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CSV = ROOT / "results" / "metrics" / "toolbox_smoking_age.csv"
OUT_DIR = ROOT / "report"

PLOT_LABEL = {
    "GraphSAGE": "GraphSAGE (baseline)",
    "GraphSAGE+Resampling": "+ Resampling",
    "GraphSAGE+Reweighted@age_old": "+ Reweighting (Kamiran-Calders)",
    "GraphSAGE+FairDrop@age_old": "+ FairDrop",
    "FairGNN(λ=1.0)@age_old": "FairGNN  (λ = 1.0, GRL adv)",
    "GraphSAGE+INLP@age_old": "+ INLP",
    "GraphSAGE+INLP+DPT@age_old": "+ INLP + DPT",
    "GraphSAGE+INLP+EOT@age_old": "+ INLP + EOT",
}

# Display order : amplifying → correcting → over-correcting
ORDER = [
    "GraphSAGE",
    "GraphSAGE+Resampling",
    "GraphSAGE+FairDrop@age_old",
    "GraphSAGE+Reweighted@age_old",
    "FairGNN(λ=1.0)@age_old",
    "GraphSAGE+INLP@age_old",
    "GraphSAGE+INLP+EOT@age_old",
    "GraphSAGE+INLP+DPT@age_old",
]


def color_for(excess_pp: float) -> str:
    if excess_pp > 1.5:
        return "#c0392b"  # red — amplifies the real gap
    if excess_pp < -1.5:
        return "#2980b9"  # blue — over-corrects (predicts inverse direction)
    return "#27ae60"  # green — close to faithful (within ±1.5pp)


def plot_excess_bar(df: pl.DataFrame, out_path: Path) -> None:
    """Horizontal bars of excess_gap per method, colour-coded by direction."""
    rows = []
    for m in ORDER:
        sub = df.filter(pl.col("method") == m)
        if sub.is_empty():
            continue
        rows.append((PLOT_LABEL[m], float(sub["excess_gap"][0]) * 100, float(sub["f1"][0])))

    labels, excess_pp, f1 = zip(*rows, strict=True)
    colours = [color_for(e) for e in excess_pp]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    y = list(range(len(labels)))
    ax.barh(y, excess_pp, color=colours, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvspan(-1, 1, color="#27ae60", alpha=0.08, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("excess_gap (pp)  =  pred_gap − true_gap", fontsize=10)
    ax.set_title(
        "Smoking × age (≥ 25) on Pokec — toolbox impact on algorithmic amplification\n"
        "true_gap = +4.10pp ; ideal excess = 0 (model faithful to reality)",
        fontsize=11,
    )
    # Annotate F1 to the right of each bar
    for i, (e, fv) in enumerate(zip(excess_pp, f1, strict=True)):
        # Bar value text
        offset = 0.25 if e >= 0 else -0.25
        ha = "left" if e >= 0 else "right"
        ax.text(
            e + offset, i, f"{e:+.2f}pp", va="center", ha=ha, fontsize=9, color="black"
        )
        # F1 annotation on far right
        ax.text(
            6.0, i, f"F1 = {fv:.3f}", va="center", ha="left", fontsize=9, color="#555"
        )
    ax.set_xlim(-5.5, 9.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#c0392b", label="Amplifies (excess > +1.5pp)"),
        plt.Rectangle((0, 0), 1, 1, color="#27ae60", label="Faithful (|excess| ≤ 1.5pp)"),
        plt.Rectangle((0, 0), 1, 1, color="#2980b9", label="Over-corrects (excess < −1.5pp)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_pareto(df: pl.DataFrame, out_path: Path) -> None:
    """F1 vs |excess_gap| scatter — Pareto-front view of the trade-off.

    Resampling is omitted because its point is identical to the baseline
    (the method has no effect — see fig_smoking_excess for that finding).
    """
    SKIP = {"GraphSAGE+Resampling"}
    rows = []
    for m in ORDER:
        if m in SKIP:
            continue
        sub = df.filter(pl.col("method") == m)
        if sub.is_empty():
            continue
        rows.append(
            (PLOT_LABEL[m], float(sub["f1"][0]), abs(float(sub["excess_gap"][0])) * 100)
        )

    labels, f1, abs_excess = zip(*rows, strict=True)

    # Manual label offsets to avoid the Pareto-cluster overlap (GraphSAGE
    # baseline and FairDrop sit at the same y, INLP+DPT and INLP+EOT collapse).
    LABEL_OFFSET = {
        "GraphSAGE (baseline)": (10, -10),
        "+ FairDrop": (-10, 10),
        "+ Reweighting (Kamiran-Calders)": (12, -3),
        "FairGNN  (λ = 1.0, GRL adv)": (10, 4),
        "+ INLP": (-10, 10),
        "+ INLP + DPT": (10, -8),
        "+ INLP + EOT": (-10, -10),
        "+ Resampling": (10, -10),
    }

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for label, f, e in zip(labels, f1, abs_excess, strict=True):
        if "Reweighting" in label:
            c, sz = "#27ae60", 140
        elif "FairGNN" in label:
            c, sz = "#8e44ad", 100
        elif "baseline" in label:
            c, sz = "#c0392b", 100
        elif "INLP + DPT" in label or "INLP + EOT" in label:
            c, sz = "#2980b9", 80
        else:
            c, sz = "#7f8c8d", 70
        ax.scatter(e, f, c=c, s=sz, edgecolor="black", linewidth=0.6, zorder=3)
        offset = LABEL_OFFSET.get(label, (8, -3))
        ha = "left" if offset[0] > 0 else "right"
        ax.annotate(
            label, (e, f), xytext=offset, textcoords="offset points",
            fontsize=8.5, ha=ha,
        )

    ax.set_xlabel("|excess_gap| (pp)  ←  better", fontsize=10)
    ax.set_ylabel("F1 (macro)  →  better", fontsize=10)
    ax.set_title(
        "Pareto : utility (F1) vs algorithmic-amplification correction\n"
        "Top-left = ideal.  Reweighting dominates : excess ≈ 0 at no F1 cost.",
        fontsize=11,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.3, 5.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    df = pl.read_csv(CSV)
    plot_excess_bar(df, OUT_DIR / "fig_smoking_excess.png")
    plot_pareto(df, OUT_DIR / "fig_smoking_pareto.png")


if __name__ == "__main__":
    main()
