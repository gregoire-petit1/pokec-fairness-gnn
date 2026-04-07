"""Target column sweep for Pokec-z node classification.

Evaluates GraphSAGE (multi-seed) on each candidate target column.
For each target × seed combination, reports:
  - test F1-macro
  - ΔDP  (demographic parity difference w.r.t. gender)
  - ΔEO  (equal opportunity difference w.r.t. gender)
  - leakage AUC (sensitive attribute probe on embeddings)

Results are saved to results/metrics/target_sweep.csv.

Usage:
    python scripts/target_sweep.py --raw-dir data/raw/pokec-z [--device cuda]
"""

import argparse
import csv
import os
import random
import sys
import time

import numpy as np
import torch
from sklearn.metrics import f1_score

# Allow running from both repo root and scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader import load_pokec_z
from src.data.preprocessing import preprocess
from src.data.splits import make_splits
from src.fairness.metrics import (
    demographic_parity_diff,
    equal_opportunity_diff,
    sensitive_leakage,
)
from src.models.graphsage import GraphSAGE
from src.models.trainer import train, evaluate

# ---------------------------------------------------------------------------
# Candidate targets
# ---------------------------------------------------------------------------
# Each entry: column_name -> callable(df) that returns a binary Series.
# Using callables keeps the sweep self-contained (no changes to loader.py).

TARGET_CANDIDATES: dict[str, str] = {
    # column_name: binarisation description (for the CSV)
    "I_am_working_in_field": "value > 0",  # baseline
    "completed_level_of_education_indicator": "direct (0/1)",
    "nefajcim": "direct (0/1)",
    "marital_status_indicator": "direct (0/1)",
    "stredoskolske": "direct (0/1)",
    "relation_to_children_indicator": "direct (0/1)",
    "abstinent": "direct (0/1)",
    "vysoke_skoly": "direct (0/1)",  # university
}

# Composite target: high_edu = stredoskolske OR vysoke_skoly > 0
# Handled separately below.

SENSITIVE_COLS = ["gender", "region"]
SEEDS = [3, 7, 21, 42, 99]

# Hyperparams (same as main experiment)
HIDDEN = 128
LAYERS = 2
DROPOUT = 0.5
LR = 1e-3
EPOCHS = 300
PATIENCE = 20

OUT_CSV = os.path.join("results", "metrics", "target_sweep.csv")
FIELDNAMES = [
    "target",
    "binarisation",
    "seed",
    "n_pos",
    "pct_pos",
    "test_f1_macro",
    "delta_dp",
    "delta_eo",
    "leakage_auc",
    "wall_sec",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_target_y(df, col: str) -> torch.Tensor:
    """Binarise target column; 'I_am_working_in_field' uses >0, rest direct."""
    if col == "I_am_working_in_field":
        vals = (df[col].values > 0).astype(int)
    else:
        # For direct binary columns (already 0/1), just use as-is
        vals = (df[col].fillna(0).values > 0).astype(int)
    return torch.tensor(vals, dtype=torch.long)


def build_high_edu_y(df) -> torch.Tensor:
    """Composite: stredoskolske OR vysoke_skoly."""
    s = df.get("stredoskolske", 0)
    v = df.get("vysoke_skoly", 0)
    vals = ((s + v) > 0).astype(int).values if hasattr(s, "values") else ((s + v) > 0)
    return torch.tensor(vals, dtype=torch.long)


def run_one(
    data_base,
    target_col: str,
    binarisation: str,
    seed: int,
    device: torch.device,
) -> dict:
    """Train GraphSAGE on a single (target, seed) pair and return metrics."""
    set_seed(seed)
    t0 = time.perf_counter()

    # Deep-copy data and swap the y tensor
    from copy import deepcopy

    data = deepcopy(data_base)

    if target_col == "high_edu":
        data.y = build_high_edu_y(data.raw_df).to(device)
    else:
        data.y = build_target_y(data.raw_df, target_col).to(device)

    # Remove old target col from features (if present) before preprocess
    if target_col in data.feature_cols and target_col != "I_am_working_in_field":
        keep = [c for c in data.feature_cols if c != target_col]
        data.x = data.x[:, [data.feature_cols.index(c) for c in keep]]
        data.feature_cols = keep

    n_pos = int(data.y.sum().item())
    pct_pos = n_pos / len(data.y)

    # Splits — stratify on y × gender
    gender = data.gender  # set by preprocess
    train_idx, val_idx, test_idx = make_splits(
        n=len(data.y), y=data.y.cpu(), sensitive=gender.cpu(), seed=seed
    )

    # Boolean masks
    N = data.x.shape[0]
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # Model
    model = GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=HIDDEN,
        out_channels=2,
        num_layers=LAYERS,
        dropout=DROPOUT,
    ).to(device)

    train(model, data, train_mask, val_mask, lr=LR, epochs=EPOCHS, patience=PATIENCE)

    # Evaluation
    _, test_f1 = evaluate(model, data, test_mask)

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    pred = logits[test_mask].argmax(dim=1)

    gender_test = data.gender[test_mask]
    y_test = data.y[test_mask]

    delta_dp = demographic_parity_diff(pred, gender_test)
    delta_eo = equal_opportunity_diff(pred, y_test, gender_test)

    embeddings = model.get_embeddings(data.x, data.edge_index)
    leakage = sensitive_leakage(
        embeddings.cpu(), data.gender.cpu(), train_mask.cpu(), test_mask.cpu(), seed=seed
    )

    wall_sec = time.perf_counter() - t0

    return {
        "target": target_col,
        "binarisation": binarisation,
        "seed": seed,
        "n_pos": n_pos,
        "pct_pos": round(pct_pos, 4),
        "test_f1_macro": round(test_f1, 4),
        "delta_dp": round(delta_dp, 4),
        "delta_eo": round(delta_eo, 4),
        "leakage_auc": round(leakage, 4),
        "wall_sec": round(wall_sec, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Target column sweep for Pokec-z")
    parser.add_argument(
        "--raw-dir",
        default=os.path.join("data", "raw", "pokec-z"),
        help="Path to raw Pokec-z data directory",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (cuda / cpu)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Random seeds to use",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Subset of targets to sweep (default: all)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load data once and preprocess (sensitive cols extracted, x normalised)
    print(f"Loading data from {args.raw_dir} ...")
    data_raw = load_pokec_z(args.raw_dir)
    # Preprocess in-place: extracts gender/region, normalises features
    # Note: target col (y) is set per-iteration, so base x should NOT include it.
    data_base = preprocess(data_raw, sensitive_cols=SENSITIVE_COLS)

    # Add composite high_edu target
    all_candidates = dict(TARGET_CANDIDATES)
    all_candidates["high_edu"] = "stredoskolske OR vysoke_skoly > 0"

    if args.targets:
        all_candidates = {k: v for k, v in all_candidates.items() if k in args.targets}

    print(
        f"Sweeping {len(all_candidates)} targets × {len(args.seeds)} seeds = "
        f"{len(all_candidates) * len(args.seeds)} runs\n"
    )

    # Check which columns actually exist
    df_cols = set(data_base.raw_df.columns)
    for col in list(all_candidates.keys()):
        if col not in df_cols and col not in ("high_edu", "I_am_working_in_field"):
            print(f"  WARNING: column '{col}' not found in dataset, skipping.")
            all_candidates.pop(col)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    write_header = not os.path.exists(OUT_CSV)

    with open(OUT_CSV, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        total = len(all_candidates) * len(args.seeds)
        done = 0
        for target_col, binarisation in all_candidates.items():
            for seed in args.seeds:
                done += 1
                print(f"[{done}/{total}] target={target_col}  seed={seed} ...", end=" ", flush=True)
                try:
                    row = run_one(data_base, target_col, binarisation, seed, device)
                    writer.writerow(row)
                    fh.flush()
                    print(
                        f"F1={row['test_f1_macro']:.4f}  "
                        f"ΔDP={row['delta_dp']:.4f}  "
                        f"leakage={row['leakage_auc']:.4f}  "
                        f"({row['wall_sec']:.0f}s)"
                    )
                except Exception as exc:
                    print(f"ERROR: {exc}")

    print(f"\nDone. Results saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
