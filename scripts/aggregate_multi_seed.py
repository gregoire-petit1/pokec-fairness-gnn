"""Aggregate per-seed comparison_seed{N}.csv into mean ± std stats.

Outputs ``results/metrics/comparison_multiseed_summary.csv`` with one row per
``(model, attribute)`` pair and columns ``delta_dp_mean``, ``delta_dp_std``,
``delta_eo_mean`` … etc. Polars-only, no pandas.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = REPO_ROOT / "results" / "metrics"
SEEDS = (3, 7, 21, 42, 99)


def main() -> None:
    frames: list[pl.DataFrame] = []
    for seed in SEEDS:
        csv_path = METRICS_DIR / f"comparison_seed{seed}.csv"
        if not csv_path.exists():
            print(f"[skip] {csv_path} (run multi-seed first)")
            continue
        df = pl.read_csv(csv_path).with_columns(pl.lit(seed).alias("seed"))
        frames.append(df)

    if not frames:
        print("no seed csv found — nothing to aggregate")
        return

    concat = pl.concat(frames, how="diagonal_relaxed")
    metric_cols = ["delta_dp", "delta_eo", "group_auc_gap", "leakage_auc"]
    agg_exprs = []
    for col in metric_cols:
        agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
        agg_exprs.append(pl.col(col).std().alias(f"{col}_std"))
    summary = concat.group_by(["model", "attribute"]).agg(agg_exprs).sort(["model", "attribute"])
    out = METRICS_DIR / "comparison_multiseed_summary.csv"
    summary.write_csv(out)
    print(f"wrote {out} — {summary.height} (model × attribute) rows over {len(frames)} seeds")


if __name__ == "__main__":
    main()
