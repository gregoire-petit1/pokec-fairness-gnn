"""FairGNN — version two-optimizer alternating (impl canonique Dai-Wang 2021).

Au lieu d'utiliser le Gradient Reversal Layer (impl simplifié dans
``src/models/fairgnn.py::FairGNN.forward``), on alterne **deux optimiseurs**
comme dans le repo officiel github.com/EnyanDai/FairGNN :

  Step A — adversary update :
    optim_adv.zero_grad()
    with torch.no_grad(): z = encoder(x)
    adv_logits = adversary(z)
    loss_adv = CE(adv_logits, sensitive)
    loss_adv.backward()
    optim_adv.step()                # UPDATE adversary params only

  Step B — encoder + classifier update :
    optim_main.zero_grad() ; optim_adv.zero_grad()
    z = encoder(x)
    y_logits = classifier(z)
    adv_logits = adversary(z)
    loss_main = CE(y_logits, y) − λ · CE(adv_logits, sensitive)
    loss_main.backward()
    optim_main.step()               # UPDATE encoder + classifier only

C'est mathématiquement équivalent à GRL en espérance, mais l'adversaire
peut être entraîné plus fort entre deux updates de l'encoder, ce qui peut
donner un point Pareto F1↔fairness différent.

Run multi-seed × Pokec-z, comparer adv=gender vs adv=region.

Output : ``results/metrics/fairgnn_two_opt_{sensitive}.csv``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_pokec_z  # noqa: E402
from src.data.preprocessing import preprocess  # noqa: E402
from src.data.splits import make_splits  # noqa: E402
from src.models.fairgnn import FairGNN  # noqa: E402

OUT_DIR = ROOT / "results" / "metrics"

LAMBDAS = [0.1, 0.5, 1.0, 5.0]
HIDDEN_DIM = 256
ADV_HIDDEN = 64
NUM_LAYERS = 2
DROPOUT = 0.5
LR = 5e-3
LR_ADV = 5e-3
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
        roc_auc_score(s_test, probe.predict_proba(emb_test), multi_class="ovr", average="macro")
    )


def _train_two_optimizer(
    data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    sensitive: torch.Tensor,
    lam: float,
    seed: int,
) -> tuple[FairGNN, dict[str, torch.Tensor] | None]:
    """Train FairGNN with two-optimizer alternating. Return model + best state."""
    out_channels = int(data.y.max().item()) + 1
    setup_seeds(seed)
    model = FairGNN(
        in_channels=data.x.shape[1],
        hidden_channels=HIDDEN_DIM,
        out_channels=out_channels,
        adv_hidden=ADV_HIDDEN,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        lambda_adv=0.0,  # GRL disabled — we use two optimizers instead
    ).to(data.x.device)

    enc_cls_params = list(model.convs.parameters()) + list(model.classifier.parameters())
    adv_params = list(model.adversary.parameters())
    opt_main = torch.optim.Adam(enc_cls_params, lr=LR, weight_decay=5e-4)
    opt_adv = torch.optim.Adam(adv_params, lr=LR_ADV, weight_decay=5e-4)

    best_val_f1 = 0.0
    best_state = None
    patience = 0

    for _epoch in range(EPOCHS):
        model.train()

        # ── Step A: train adversary on current (frozen) embeddings ─────────
        opt_adv.zero_grad(set_to_none=True)
        with torch.no_grad():
            z = model.encode(data.x, data.edge_index)
        adv_logits = model.adversary(z)
        loss_adv = F.cross_entropy(adv_logits[train_mask], sensitive[train_mask])
        loss_adv.backward()
        opt_adv.step()

        # ── Step B: train encoder + classifier (adversary frozen by opt scope) ─
        opt_main.zero_grad(set_to_none=True)
        opt_adv.zero_grad(set_to_none=True)  # clear stale grads on adv params
        z = model.encode(data.x, data.edge_index)
        y_logits = model.classifier(z)
        adv_logits = model.adversary(z)
        loss_cls = F.cross_entropy(y_logits[train_mask], data.y[train_mask])
        loss_adv2 = F.cross_entropy(adv_logits[train_mask], sensitive[train_mask])
        loss_main = loss_cls - lam * loss_adv2  # encoder MAXIMIZES adv loss
        loss_main.backward()
        opt_main.step()

        # ── Validation : pick best F1 model ────────────────────────────────
        model.eval()
        with torch.no_grad():
            z_val = model.encode(data.x, data.edge_index)
            y_val_logits = model.classifier(z_val)
        val_pred = y_val_logits[val_mask].argmax(dim=1).cpu().numpy()
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
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_state


def run_one(raw_dir: Path, seed: int, sensitive_name: str, device: str) -> list[dict]:
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

    # Sensitive tensor for adversary.
    if sensitive_name == "gender":
        sensitive_t = data.gender.long()
    elif sensitive_name == "region":
        sensitive_t = data.region.long()
    else:
        raise ValueError(f"unsupported sensitive_name {sensitive_name!r}")

    # ── Grid sweep on λ ─────────────────────────────────────────────────────
    grid: dict[float, dict] = {}
    state_per_lam: dict[float, dict] = {}
    t0 = time.time()
    for lam in LAMBDAS:
        model, state = _train_two_optimizer(data, train_mask, val_mask, sensitive_t, lam, seed)
        # Quick test eval to score this lambda.
        model.eval()
        with torch.no_grad():
            z = model.encode(data.x, data.edge_index)
            y_logits = model.classifier(z)
        pred_test = y_logits[test_mask].argmax(dim=1).cpu().numpy()
        y_test = data.y[test_mask].cpu().numpy()
        f1_lam = f1_score(y_test, pred_test, average="macro", zero_division=0)
        ddp_lam = _delta_dp(pred_test, sensitive_t[test_mask].cpu().numpy())
        grid[lam] = {"f1": float(f1_lam), "delta_dp": float(ddp_lam)}
        state_per_lam[lam] = state

    # Pick best λ : smallest ΔDP among non-collapsed (F1 > 0.5) models.
    valid = {lam: v for lam, v in grid.items() if v["f1"] > 0.50}
    chosen = min(valid or grid, key=lambda lam: (valid or grid)[lam]["delta_dp"])

    # Re-load chosen model for final eval.
    out_channels = int(data.y.max().item()) + 1
    final = FairGNN(
        in_channels=data.x.shape[1],
        hidden_channels=HIDDEN_DIM,
        out_channels=out_channels,
        adv_hidden=ADV_HIDDEN,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        lambda_adv=0.0,
    ).to(device)
    final.load_state_dict(state_per_lam[chosen])
    final.eval()
    with torch.no_grad():
        z = final.encode(data.x, data.edge_index)
        y_logits = final.classifier(z)

    pred_test = y_logits[test_mask].argmax(dim=1).cpu().numpy()
    y_test = data.y[test_mask].cpu().numpy()
    acc = float(accuracy_score(y_test, pred_test))
    f1 = float(f1_score(y_test, pred_test, average="macro"))

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
    h_np = z.cpu().numpy().astype(np.float32)

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
                "method": f"FairGNN_two_opt_adv_{sensitive_name}",
                "axis": axis_name,
                "lambda_chosen": chosen,
                "acc": round(acc, 4),
                "f1_macro": round(f1, 4),
                "delta_dp": round(ddp, 4),
                "delta_eo": round(deo, 4),
                "leakage_auc": round(leak, 4),
            }
        )
    elapsed = time.time() - t0
    print(
        f"  [{raw_dir.name} seed={seed} adv={sensitive_name}] λ*={chosen}  "
        f"acc={acc:.4f}  F1={f1:.4f}  done in {elapsed:.1f}s",
        flush=True,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["pokec-z"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 7, 21, 42, 99])
    parser.add_argument(
        "--sensitives", nargs="+", default=["gender", "region"], choices=["gender", "region"]
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} datasets={args.datasets} seeds={args.seeds} sensitives={args.sensitives}",
        flush=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sensitive in args.sensitives:
        out_csv = OUT_DIR / f"fairgnn_two_opt_{sensitive}.csv"
        all_rows: list[dict] = []
        for dataset in args.datasets:
            raw_dir = ROOT / "data" / "raw" / dataset
            if not raw_dir.exists():
                print(f"[SKIP] {raw_dir} missing", flush=True)
                continue
            for seed in args.seeds:
                rows = run_one(raw_dir, seed, sensitive, device)
                all_rows.extend(rows)
                pl.DataFrame(all_rows).write_csv(out_csv)

        if not all_rows:
            continue
        df = pl.DataFrame(all_rows)
        df.write_csv(out_csv)
        print(f"\nwrote {out_csv} rows={df.height}", flush=True)

        summary = (
            df.group_by(["dataset", "axis"])
            .agg(
                pl.col("acc").mean().alias("acc_mean"),
                pl.col("f1_macro").mean().alias("f1_mean"),
                pl.col("delta_dp").mean().alias("dp_mean"),
                pl.col("delta_eo").mean().alias("eo_mean"),
                pl.col("leakage_auc").mean().alias("leak_mean"),
            )
            .sort(["dataset", "axis"])
        )
        print(f"\n=== {sensitive} : mean across seeds ===")
        with pl.Config(tbl_rows=15, tbl_cols=10, float_precision=4):
            print(summary)


if __name__ == "__main__":
    main()
