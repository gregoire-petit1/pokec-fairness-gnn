"""ULTIMATE-LATENT runner: INLP_composite + DPT_composite via re-injection.

Multi-seed × Pokec-z/n. For each (dataset, seed) we run the full ULTIMATE
chain on TabICL row embeddings (instead of x brut) and report per-axis
metrics across the 5 sensitive axes (gender, region, age_group, gender×age,
gender×region).

Output : ``results/metrics/tabicl_inlp_reinjection_composite.csv``
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baselines.tabicl_inlp_reinjection_composite import (  # noqa: E402
    run_ultimate_reinjection,
)
from src.data.loader import load_pokec_z  # noqa: E402
from src.data.preprocessing import preprocess  # noqa: E402

OUT_DIR = ROOT / "results" / "metrics"
OUT_CSV = OUT_DIR / "tabicl_inlp_reinjection_composite.csv"


def _build_axis_arrays(data, idx: np.ndarray) -> dict[str, np.ndarray]:
    g = data.gender.cpu().numpy().astype(np.int64)
    r = data.region.cpu().numpy().astype(np.int64)
    a = data.age_group.cpu().numpy().astype(np.int64).clip(min=0)
    n_age = int(a.max()) + 1
    n_reg = int(r.max()) + 1
    full = {
        "gender": g,
        "region": r,
        "age_group": a,
        "gender_x_age": g * n_age + a,
        "gender_x_region": g * n_reg + r,
    }
    return {k: v[idx] for k, v in full.items()}


def _stratified_split(
    n: int, y: np.ndarray, gender: np.ndarray, seed: int, max_train: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """60/20/20 stratified by (y, gender), train capped at max_train."""
    strat = y * 2 + gender
    idx_train, idx_rest = train_test_split(
        np.arange(n), test_size=0.4, random_state=seed, stratify=strat
    )
    strat_rest = strat[idx_rest]
    idx_val, idx_test = train_test_split(
        idx_rest, test_size=0.5, random_state=seed, stratify=strat_rest
    )
    rng = np.random.default_rng(seed)
    if idx_train.size > max_train:
        idx_train = rng.choice(idx_train, size=max_train, replace=False)
    return idx_train, idx_val, idx_test


def run_one(raw_dir: Path, seed: int, n_estimators: int, max_train: int, device: str) -> list[dict]:
    data = load_pokec_z(raw_dir)
    data = preprocess(data, sensitive_cols=["gender", "region", "age_group"])

    x = data.x.cpu().numpy().astype(np.float32)
    y = data.y.cpu().numpy().astype(np.int64)
    gender = data.gender.cpu().numpy().astype(np.int64)

    idx_train, idx_val, idx_test = _stratified_split(x.shape[0], y, gender, seed, max_train)

    sens_train = _build_axis_arrays(data, idx_train)
    sens_val = _build_axis_arrays(data, idx_val)
    sens_test = _build_axis_arrays(data, idx_test)

    result = run_ultimate_reinjection(
        x_train=x[idx_train],
        y_train=y[idx_train],
        x_val=x[idx_val],
        y_val=y[idx_val],
        x_test=x[idx_test],
        y_test=y[idx_test],
        sensitive_train=sens_train,
        sensitive_val=sens_val,
        sensitive_test=sens_test,
        composite_axes=("gender", "age_group", "region"),
        seed=seed,
        n_estimators=n_estimators,
        device=device,
    )

    rows: list[dict] = []
    for axis_metrics in result.per_axis:
        rows.append(
            {
                "dataset": raw_dir.name,
                "seed": seed,
                "axis": axis_metrics.axis,
                "n_train": result.n_train,
                "n_test": result.n_test,
                "embed_dim": result.embed_dim,
                "composite_n_cells": result.composite_n_cells,
                "acc_baseline": result.acc_baseline,
                "f1_baseline": result.f1_baseline,
                "acc_after_ultimate": result.acc_after_ultimate,
                "f1_after_ultimate": result.f1_after_ultimate,
                "delta_dp": axis_metrics.delta_dp,
                "delta_eo": axis_metrics.delta_eo,
                "leakage_pre": axis_metrics.leakage_pre,
                "leakage_post": axis_metrics.leakage_post,
            }
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["pokec-z", "pokec-n"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 7, 21, 42, 99])
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument("--max-train", type=int, default=10_000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=str(OUT_CSV))
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} datasets={args.datasets} seeds={args.seeds} "
        f"composite=gender×age_group×region",
        flush=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for dataset in args.datasets:
        raw_dir = ROOT / "data" / "raw" / dataset
        if not raw_dir.exists():
            print(f"[SKIP] {raw_dir} missing", flush=True)
            continue
        for seed in args.seeds:
            t0 = time.time()
            print(f"[{dataset} seed={seed}] start", flush=True)
            rows = run_one(raw_dir, seed, args.n_estimators, args.max_train, device)
            for row in rows:
                print(
                    f"  {row['axis']:18s} "
                    f"acc(base→ult)={row['acc_baseline']:.3f}→{row['acc_after_ultimate']:.3f}  "
                    f"f1={row['f1_baseline']:.3f}→{row['f1_after_ultimate']:.3f}  "
                    f"ΔDP={row['delta_dp']:.4f} ΔEO={row['delta_eo']:.4f}  "
                    f"leak={row['leakage_pre']:.3f}→{row['leakage_post']:.3f}",
                    flush=True,
                )
            all_rows.extend(rows)
            pl.DataFrame(all_rows).write_csv(args.out)
            print(
                f"[{dataset} seed={seed}] done in {time.time() - t0:.1f}s "
                f"(checkpoint {len(all_rows)} rows)",
                flush=True,
            )

    if not all_rows:
        print("nothing produced", flush=True)
        return

    df = pl.DataFrame(all_rows)
    df.write_csv(args.out)
    print(f"\nwrote {args.out} rows={df.height}", flush=True)

    summary = (
        df.group_by(["dataset", "axis"])
        .agg(
            pl.col("acc_baseline").mean().alias("acc_base"),
            pl.col("acc_after_ultimate").mean().alias("acc_after"),
            pl.col("f1_baseline").mean().alias("f1_base"),
            pl.col("f1_after_ultimate").mean().alias("f1_after"),
            pl.col("delta_dp").mean().alias("delta_dp"),
            pl.col("delta_eo").mean().alias("delta_eo"),
            pl.col("leakage_pre").mean().alias("leak_pre"),
            pl.col("leakage_post").mean().alias("leak_post"),
        )
        .sort(["dataset", "axis"])
    )
    print("\n=== Mean across seeds ===")
    with pl.Config(tbl_rows=20, tbl_cols=12, float_precision=4):
        print(summary)


if __name__ == "__main__":
    main()
