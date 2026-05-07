"""Run the TabICL-row-embedding leakage validation across multi-seed × Pokec-z/n.

Validates the 2-pager claim that ``TabICLCache.row_repr`` is accessible and
that INLP fits naturally on those embeddings (a true post-process), in
contrast to our pipeline's current INLP-on-x-brut.

For each (dataset, seed, sensitive_axis) we measure :
* ``leakage_pre``   — linear-probe AUC predicting the sensitive attribute
                       from the raw row embeddings.
* ``leakage_post``  — same probe AUC after INLP projection.
* ``drop``          — ``leakage_pre - leakage_post``.

Output : ``results/metrics/tabicl_inlp_embedding.csv`` (polars DataFrame).

Usage::

    .venv/bin/python scripts/run_tabicl_inlp_embedding.py \
        --datasets pokec-z pokec-n \
        --seeds 3 7 21 42 99 \
        --n-estimators 4

Env :
    CUDA_VISIBLE_DEVICES=0 (or 1) to pick a GPU.
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

from src.baselines.tabicl_inlp_embedding import (  # noqa: E402
    fit_tabicl_with_embeddings,
    measure_leakage_pre_post_inlp,
)
from src.data.loader import load_pokec_z  # noqa: E402
from src.data.preprocessing import preprocess  # noqa: E402

OUT_DIR = ROOT / "results" / "metrics"
OUT_CSV = OUT_DIR / "tabicl_inlp_embedding.csv"


def _build_axes(data, train_idx: np.ndarray) -> dict[str, np.ndarray]:
    """Return dict of axis_name -> int array of length len(train_idx)."""
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
    return {k: v[train_idx] for k, v in full.items()}


def _stratified_train_idx(
    n: int, y: np.ndarray, gender: np.ndarray, seed: int, max_train: int
) -> np.ndarray:
    """60/20/20 stratified-by-(y, gender) split, then cap train at max_train."""
    strat = y * 2 + gender
    idx_train, _ = train_test_split(np.arange(n), test_size=0.4, random_state=seed, stratify=strat)
    rng = np.random.default_rng(seed)
    if idx_train.size > max_train:
        idx_train = rng.choice(idx_train, size=max_train, replace=False)
    return idx_train


def run_one(
    raw_dir: Path,
    seed: int,
    n_estimators: int,
    max_train: int,
    device: str,
) -> list[dict]:
    """Fit + measure on one (dataset, seed). Returns list of per-axis row dicts."""
    data = load_pokec_z(raw_dir)
    data = preprocess(data, sensitive_cols=["gender", "region", "age_group"])

    x = data.x.cpu().numpy().astype(np.float32)
    y = data.y.cpu().numpy().astype(np.int64)
    gender = data.gender.cpu().numpy().astype(np.int64)

    idx_train = _stratified_train_idx(x.shape[0], y, gender, seed, max_train)
    axes = _build_axes(data, idx_train)

    _clf, embeddings = fit_tabicl_with_embeddings(
        x[idx_train], y[idx_train], seed=seed, n_estimators=n_estimators, device=device
    )
    if embeddings.shape[0] != idx_train.size:
        # Defensive: TabICL ensemble generator may reorder; keep only matching prefix.
        idx_train = idx_train[: embeddings.shape[0]]
        axes = _build_axes(data, idx_train)

    results = measure_leakage_pre_post_inlp(embeddings, axes, seed=seed)

    rows = [
        {
            "dataset": raw_dir.name,
            "seed": seed,
            "axis": r.axis,
            "n_train": r.n_train,
            "embed_dim": r.embed_dim,
            "leakage_pre": r.leakage_pre,
            "leakage_post": r.leakage_post,
            "drop": r.drop,
        }
        for r in results
    ]
    # Free GPU memory : TabICLClassifier holds a kv-cache of cuda tensors, drop them
    # before the next (dataset, seed) iteration so memory doesn't accumulate.
    del _clf, embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["pokec-z", "pokec-n"],
        help="Subdirs under data/raw/ to process (default: both).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 7, 21, 42, 99])
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument("--max-train", type=int, default=10_000)
    parser.add_argument(
        "--device",
        default=None,
        help="Defaults to cuda:0 if available, else cpu. Use CUDA_VISIBLE_DEVICES "
        "to pick a GPU at the env level.",
    )
    parser.add_argument("--out", default=str(OUT_CSV))
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, datasets={args.datasets}, seeds={args.seeds}", flush=True)

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
                    f"pre={row['leakage_pre']:.4f}  "
                    f"post={row['leakage_post']:.4f}  "
                    f"drop={row['drop']:+.4f}",
                    flush=True,
                )
            all_rows.extend(rows)
            print(f"[{dataset} seed={seed}] done in {time.time() - t0:.1f}s", flush=True)

    if not all_rows:
        print("nothing produced — check args", flush=True)
        return

    df = pl.DataFrame(all_rows)
    df.write_csv(args.out)
    print(f"\nwrote {args.out}  rows={df.height}", flush=True)

    # Summary : mean across seeds per (dataset, axis)
    summary = df.group_by(["dataset", "axis"]).agg(
        pl.col("leakage_pre").mean().alias("pre_mean"),
        pl.col("leakage_pre").std().alias("pre_std"),
        pl.col("leakage_post").mean().alias("post_mean"),
        pl.col("leakage_post").std().alias("post_std"),
        pl.col("drop").mean().alias("drop_mean"),
    )
    print("\n=== Mean across seeds ===")
    with pl.Config(tbl_rows=20, tbl_cols=10):
        print(summary)


if __name__ == "__main__":
    main()
