"""End-to-end TabICL+INLP-on-embeddings runner: F1 + leakage on test.

Multi-seed × Pokec-z/n × multi-axis. For each (dataset, seed, axis) :
* baseline F1 (vanilla TabICL, no fairness)
* F1 after INLP-projection of embeddings (re-injected into icl_predictor)
* leakage AUC pre-INLP (probe on raw embeddings)
* leakage AUC post-INLP (probe on projected embeddings)

Output : ``results/metrics/tabicl_inlp_reinjection.csv``.
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

from src.baselines.tabicl_inlp_reinjection import run_inlp_reinjection  # noqa: E402
from src.data.loader import load_pokec_z  # noqa: E402
from src.data.preprocessing import preprocess  # noqa: E402

OUT_DIR = ROOT / "results" / "metrics"
OUT_CSV = OUT_DIR / "tabicl_inlp_reinjection.csv"


def _build_axis_arrays(data) -> dict[str, np.ndarray]:
    g = data.gender.cpu().numpy().astype(np.int64)
    r = data.region.cpu().numpy().astype(np.int64)
    a = data.age_group.cpu().numpy().astype(np.int64).clip(min=0)
    n_age = int(a.max()) + 1
    n_reg = int(r.max()) + 1
    return {
        "gender": g,
        "region": r,
        "age_group": a,
        "gender_x_age": g * n_age + a,
        "gender_x_region": g * n_reg + r,
    }


def _stratified_split(
    n: int, y: np.ndarray, gender: np.ndarray, seed: int, max_train: int
) -> tuple[np.ndarray, np.ndarray]:
    strat = y * 2 + gender
    idx_train, idx_test = train_test_split(
        np.arange(n), test_size=0.4, random_state=seed, stratify=strat
    )
    rng = np.random.default_rng(seed)
    if idx_train.size > max_train:
        idx_train = rng.choice(idx_train, size=max_train, replace=False)
    return idx_train, idx_test


def run_one(
    raw_dir: Path,
    seed: int,
    axes: list[str],
    n_estimators: int,
    max_train: int,
    device: str,
) -> list[dict]:
    data = load_pokec_z(raw_dir)
    data = preprocess(data, sensitive_cols=["gender", "region", "age_group"])

    x = data.x.cpu().numpy().astype(np.float32)
    y = data.y.cpu().numpy().astype(np.int64)
    gender = data.gender.cpu().numpy().astype(np.int64)
    axis_arrays = _build_axis_arrays(data)

    idx_train, idx_test = _stratified_split(x.shape[0], y, gender, seed, max_train)

    rows: list[dict] = []
    for axis_name in axes:
        if axis_name not in axis_arrays:
            print(f"  [skip] unknown axis '{axis_name}'", flush=True)
            continue
        s_full = axis_arrays[axis_name]
        result = run_inlp_reinjection(
            x_train=x[idx_train],
            y_train=y[idx_train],
            x_test=x[idx_test],
            y_test=y[idx_test],
            sensitive_train=s_full[idx_train],
            sensitive_test=s_full[idx_test],
            axis_name=axis_name,
            seed=seed,
            n_estimators=n_estimators,
            device=device,
        )
        rows.append(
            {
                "dataset": raw_dir.name,
                "seed": seed,
                "axis": result.axis,
                "n_train": result.n_train,
                "n_test": result.n_test,
                "embed_dim": result.embed_dim,
                "acc_baseline": result.acc_baseline,
                "acc_after_inlp": result.acc_after_inlp,
                "acc_drop": result.acc_drop,
                "f1_baseline": result.f1_baseline,
                "f1_after_inlp": result.f1_after_inlp,
                "f1_drop": result.f1_drop,
                "leakage_pre": result.leakage_pre,
                "leakage_post": result.leakage_post,
                "leakage_drop": result.leakage_drop,
            }
        )
        # Free GPU memory between axes.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["pokec-z", "pokec-n"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 7, 21, 42, 99])
    parser.add_argument(
        "--axes",
        nargs="+",
        default=["gender", "region", "age_group", "gender_x_age", "gender_x_region"],
    )
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument("--max-train", type=int, default=10_000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=str(OUT_CSV))
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} datasets={args.datasets} seeds={args.seeds} axes={args.axes}", flush=True
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
            rows = run_one(raw_dir, seed, args.axes, args.n_estimators, args.max_train, device)
            for row in rows:
                print(
                    f"  {row['axis']:18s} "
                    f"acc={row['acc_baseline']:.3f}->{row['acc_after_inlp']:.3f} "
                    f"f1={row['f1_baseline']:.3f}->{row['f1_after_inlp']:.3f}  "
                    f"leak={row['leakage_pre']:.3f}->{row['leakage_post']:.3f}",
                    flush=True,
                )
            all_rows.extend(rows)
            # Incremental save so a kill mid-run doesn't lose what's done.
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

    summary = df.group_by(["dataset", "axis"]).agg(
        pl.col("f1_baseline").mean().alias("f1_base_mean"),
        pl.col("f1_after_inlp").mean().alias("f1_after_mean"),
        pl.col("f1_drop").mean().alias("f1_drop_mean"),
        pl.col("leakage_pre").mean().alias("leak_pre_mean"),
        pl.col("leakage_post").mean().alias("leak_post_mean"),
    )
    print("\n=== Mean across seeds ===")
    with pl.Config(tbl_rows=20, tbl_cols=10):
        print(summary)


if __name__ == "__main__":
    main()
