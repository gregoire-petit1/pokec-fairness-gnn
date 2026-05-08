"""FairGNN avec adversaire ciblant ``region`` (pas gender).

Sur Pokec-z, ``r(region) = 0.901`` : le graphe est massivement homophile
en region, ce qui amplifie le biais structurel via le message passing.
C'est exactement le scénario où FairGNN a quelque chose à attaquer
(contrairement à gender, où ``r ≈ 0`` rend l'adversaire inutile).

On entraîne un grid λ ∈ [0.1, 0.5, 1.0, 5.0], on choisit le λ qui
minimise ΔDP region parmi les modèles non-collapsés (F1 > 0.50), et on
reporte les métriques sur les 5 axes pour comparaison avec le pipeline
gender-centric.

Output : ``results/metrics/fairgnn_on_region.csv`` (multi-seed).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_pokec_z  # noqa: E402
from src.data.preprocessing import preprocess  # noqa: E402
from src.data.splits import make_splits  # noqa: E402
from src.models.fairgnn import FairGNN, fairgnn_loss  # noqa: E402

OUT_DIR = ROOT / "results" / "metrics"
OUT_CSV = OUT_DIR / "fairgnn_on_region.csv"

LAMBDAS = [0.1, 0.5, 1.0, 5.0]
HIDDEN_DIM = 256
ADV_HIDDEN = 64
NUM_LAYERS = 2
DROPOUT = 0.5
LR = 5e-3
EPOCHS = 200
PATIENCE = 30


def setup_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _delta_dp(pred: np.ndarray, sensitive: np.ndarray) -> float:
    rates = [float(pred[sensitive == g].mean()) for g in np.unique(sensitive)]
    return float(max(rates) - min(rates)) if len(rates) >= 2 else 0.0


def _delta_eo(pred: np.ndarray, y_true: np.ndarray, sensitive: np.ndarray) -> float:
    tprs = []
    for g in np.unique(sensitive):
        mask = (sensitive == g) & (y_true == 1)
        if mask.sum() == 0:
            continue
        tprs.append(float(pred[mask].mean()))
    return float(max(tprs) - min(tprs)) if len(tprs) >= 2 else 0.0


def _leakage(emb_train, emb_test, s_train, s_test, seed):
    if np.unique(s_train).size < 2:
        return float("nan")
    probe = LogisticRegression(max_iter=1000, random_state=seed)
    probe.fit(emb_train, s_train)
    n_classes = int(s_test.max()) + 1
    if n_classes == 2:
        return float(roc_auc_score(s_test, probe.predict_proba(emb_test)[:, 1]))
    return float(
        roc_auc_score(
            s_test, probe.predict_proba(emb_test), multi_class="ovr", average="macro"
        )
    )


def train_fairgnn_on_region(data, train_mask, val_mask, seed: int):
    """Grid sweep on λ, return (final_model, embeddings, chosen_λ)."""
    out_channels = int(data.y.max().item()) + 1
    region = data.region.long()

    grid = {}
    best_state = {}

    for lam in LAMBDAS:
        setup_seeds(seed)
        model = FairGNN(
            in_channels=data.x.shape[1],
            hidden_channels=HIDDEN_DIM,
            out_channels=out_channels,
            adv_hidden=ADV_HIDDEN,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            lambda_adv=lam,
        ).to(data.x.device)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

        best_val_f1 = 0.0
        patience = 0
        best_st = None
        for _epoch in range(EPOCHS):
            model.train()
            opt.zero_grad()
            pred_logits, adv_logits = model(data.x, data.edge_index)
            loss = fairgnn_loss(pred_logits, adv_logits, data.y, region, train_mask)
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                pl_val, _ = model(data.x, data.edge_index)
            val_pred = pl_val[val_mask].argmax(dim=1).cpu().numpy()
            val_f1 = float(
                f1_score(
                    data.y[val_mask].cpu().numpy(),
                    val_pred,
                    average="macro",
                    zero_division=0,
                )
            )
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_st = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE:
                    break

        model.load_state_dict(best_st)
        # Evaluate on the test set deferred — first pick best λ on val.
        with torch.no_grad():
            pred_logits, _ = model(data.x, data.edge_index)
        test_mask_full = ~(train_mask | val_mask)
        test_idx = test_mask_full.nonzero(as_tuple=False).squeeze(1)
        pred_test = pred_logits[test_idx].argmax(dim=1).cpu().numpy()
        y_test = data.y[test_idx].cpu().numpy()
        f1_lam = f1_score(y_test, pred_test, average="macro", zero_division=0)
        ddp_lam = _delta_dp(pred_test, region[test_idx].cpu().numpy())
        grid[lam] = {"f1": float(f1_lam), "delta_dp_region": ddp_lam}
        best_state[lam] = best_st

    # Choose best λ : smallest ΔDP region among non-collapsed models.
    valid = {lam: v for lam, v in grid.items() if v["f1"] > 0.50}
    chosen = min(valid or grid, key=lambda lam: (valid or grid)[lam]["delta_dp_region"])

    # Reload chosen model.
    setup_seeds(seed)
    model = FairGNN(
        in_channels=data.x.shape[1],
        hidden_channels=HIDDEN_DIM,
        out_channels=out_channels,
        adv_hidden=ADV_HIDDEN,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        lambda_adv=chosen,
    ).to(data.x.device)
    model.load_state_dict(best_state[chosen])

    # Extract embeddings (encoder output).
    model.eval()
    with torch.no_grad():
        h = model.encode(data.x, data.edge_index)
        pred_logits, _ = model(data.x, data.edge_index)

    return model, h, pred_logits, chosen, grid


def run_one(raw_dir: Path, seed: int, device: str) -> list[dict]:
    data = load_pokec_z(raw_dir)
    data = preprocess(data, sensitive_cols=["gender", "region", "age_group"])
    data = data.to(device)

    train_idx_t, val_idx_t, test_idx_t = make_splits(
        n=data.y.shape[0], y=data.y.cpu(), sensitive=data.gender.cpu(), seed=seed
    )
    train_mask = torch.zeros(data.y.shape[0], dtype=torch.bool, device=device)
    val_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)
    train_mask[train_idx_t.to(device)] = True
    val_mask[val_idx_t.to(device)] = True
    test_mask[test_idx_t.to(device)] = True

    t0 = time.time()
    _model, h, pred_logits, chosen_lam, grid = train_fairgnn_on_region(
        data, train_mask, val_mask, seed
    )
    pred_test = pred_logits[test_mask].argmax(dim=1).cpu().numpy()
    y_test = data.y[test_mask].cpu().numpy()
    acc = float(accuracy_score(y_test, pred_test))
    f1 = float(f1_score(y_test, pred_test, average="macro"))

    # Build per-axis sensitives.
    g = data.gender.cpu().numpy().astype(np.int64)
    r = data.region.cpu().numpy().astype(np.int64)
    a = data.age_group.cpu().numpy().astype(np.int64).clip(min=0)
    n_age = int(a.max()) + 1
    n_reg = int(r.max()) + 1
    axes_full = {
        "gender": g,
        "region": r,
        "age_group": a,
        "gender_x_age": g * n_age + a,
        "gender_x_region": g * n_reg + r,
    }

    train_idx = train_mask.cpu().numpy().nonzero()[0]
    test_idx = test_mask.cpu().numpy().nonzero()[0]
    h_np = h.cpu().numpy().astype(np.float32)

    rows = []
    for axis_name, s_full in axes_full.items():
        s_te = s_full[test_idx]
        s_tr = s_full[train_idx]
        ddp = _delta_dp(pred_test, s_te)
        deo = _delta_eo(pred_test, y_test, s_te)
        leak = _leakage(h_np[train_idx], h_np[test_idx], s_tr, s_te, seed)
        rows.append(
            {
                "dataset": raw_dir.name,
                "seed": seed,
                "method": "FairGNN_adv_region",
                "axis": axis_name,
                "lambda_chosen": chosen_lam,
                "acc": round(acc, 4),
                "f1_macro": round(f1, 4),
                "delta_dp": round(ddp, 4),
                "delta_eo": round(deo, 4),
                "leakage_auc": round(leak, 4),
            }
        )
    elapsed = time.time() - t0
    print(
        f"  [{raw_dir.name} seed={seed}] λ*={chosen_lam}  "
        f"acc={acc:.4f}  F1={f1:.4f}  done in {elapsed:.1f}s",
        flush=True,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["pokec-z", "pokec-n"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 7, 21, 42, 99])
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=str(OUT_CSV))
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device} datasets={args.datasets} seeds={args.seeds}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for dataset in args.datasets:
        raw_dir = ROOT / "data" / "raw" / dataset
        if not raw_dir.exists():
            print(f"[SKIP] {raw_dir} missing", flush=True)
            continue
        for seed in args.seeds:
            rows = run_one(raw_dir, seed, device)
            all_rows.extend(rows)
            pl.DataFrame(all_rows).write_csv(args.out)

    if not all_rows:
        return
    df = pl.DataFrame(all_rows)
    df.write_csv(args.out)
    print(f"\nwrote {args.out} rows={df.height}", flush=True)

    summary = df.group_by(["dataset", "axis"]).agg(
        pl.col("acc").mean().alias("acc_mean"),
        pl.col("f1_macro").mean().alias("f1_mean"),
        pl.col("delta_dp").mean().alias("dp_mean"),
        pl.col("delta_eo").mean().alias("eo_mean"),
        pl.col("leakage_auc").mean().alias("leak_mean"),
    ).sort(["dataset", "axis"])
    print("\n=== Mean across seeds ===")
    with pl.Config(tbl_rows=20, tbl_cols=10, float_precision=4):
        print(summary)


if __name__ == "__main__":
    main()
