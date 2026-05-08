"""Smoke: train TabICL on GraphSAGE embeddings (instead of raw x).

Premier essai pour valider l'idée : les embeddings GraphSAGE encodent
l'information du graphe (homophilie region 0.901, voisins partageant la
cible). Si on les utilise comme features pour TabICL au lieu de x brut,
qu'arrive-t-il à la F1 et au leakage ?

Hypothèses :
- F1 pourrait monter (graphe ajoute info) ou baisser (graphe dilue les
  features tabulaires fortes par moyennage des voisins).
- Leakage pourrait monter beaucoup (homophilie region) → intéressant pour
  voir si TabICL+ULTIMATE peut quand même nettoyer.

Lance ::

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/_smoke_graph_emb_to_tabicl.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_pokec_z  # noqa: E402
from src.data.preprocessing import preprocess  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEED = 42
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_TRAIN = 10_000


def main() -> None:
    print(f"device={DEVICE}, seed={SEED}")

    # Load data + splits.
    data = load_pokec_z(ROOT / "data" / "raw" / "pokec-z")
    data = preprocess(data, sensitive_cols=["gender", "region", "age_group"])
    x = data.x.cpu().numpy().astype(np.float32)
    y = data.y.cpu().numpy().astype(np.int64)
    gender = data.gender.cpu().numpy().astype(np.int64)
    region = data.region.cpu().numpy().astype(np.int64)
    age_group = data.age_group.cpu().numpy().astype(np.int64).clip(min=0)

    strat = y * 2 + gender
    idx_train_full, idx_rest = train_test_split(
        np.arange(x.shape[0]), test_size=0.4, random_state=SEED, stratify=strat
    )
    _idx_val, idx_test = train_test_split(
        idx_rest, test_size=0.5, random_state=SEED, stratify=strat[idx_rest]
    )
    rng = np.random.default_rng(SEED)
    idx_train = (
        rng.choice(idx_train_full, size=MAX_TRAIN, replace=False)
        if idx_train_full.size > MAX_TRAIN
        else idx_train_full
    )
    print(f"train={idx_train.size}  test={idx_test.size}")

    # Load cached GraphSAGE embeddings.
    cache_path = ROOT / "results" / "cache" / f"seed{SEED}" / "GraphSAGE.pt"
    out = torch.load(cache_path, weights_only=False, map_location="cpu")
    graph_emb = out["embeddings"].cpu().numpy().astype(np.float32)
    print(f"GraphSAGE embeddings: shape={graph_emb.shape}")

    # Sanity : baseline GraphSAGE F1 from the cache.
    print(
        f"  cached GraphSAGE  acc={out['acc']:.4f}  F1={out['f1']:.4f}"
    )

    # Fit TabICL twice : on x brut and on graph embeddings, compare.
    from tabicl import TabICLClassifier

    print("\n=== TabICL on raw x (264 dims) ===")
    clf_x = TabICLClassifier(random_state=SEED, device=DEVICE, n_estimators=4)
    clf_x.fit(x[idx_train], y[idx_train])
    pred_x = clf_x.predict(x[idx_test])
    print(
        f"  acc={accuracy_score(y[idx_test], pred_x):.4f}  "
        f"F1={f1_score(y[idx_test], pred_x, average='macro'):.4f}"
    )

    print("\n=== TabICL on GraphSAGE embeddings (256 dims) ===")
    clf_g = TabICLClassifier(random_state=SEED, device=DEVICE, n_estimators=4)
    clf_g.fit(graph_emb[idx_train], y[idx_train])
    pred_g = clf_g.predict(graph_emb[idx_test])
    proba_g = clf_g.predict_proba(graph_emb[idx_test])
    acc_g = accuracy_score(y[idx_test], pred_g)
    f1_g = f1_score(y[idx_test], pred_g, average="macro")
    print(f"  acc={acc_g:.4f}  F1={f1_g:.4f}")

    print("\n=== TabICL on concat[x, graph_emb] (520 dims) ===")
    x_concat = np.concatenate([x, graph_emb], axis=1).astype(np.float32)
    clf_c = TabICLClassifier(random_state=SEED, device=DEVICE, n_estimators=4)
    clf_c.fit(x_concat[idx_train], y[idx_train])
    pred_c = clf_c.predict(x_concat[idx_test])
    print(
        f"  acc={accuracy_score(y[idx_test], pred_c):.4f}  "
        f"F1={f1_score(y[idx_test], pred_c, average='macro'):.4f}"
    )

    # Fairness side : per-axis ΔDP and leakage.
    print("\n=== Fairness on TabICL+graph_emb ===")
    axes = {
        "gender": gender,
        "region": region,
        "age_group": age_group,
        "gender_x_age": gender * 3 + age_group,
        "gender_x_region": gender * 2 + region,
    }

    def delta_dp(pred, sensitive):
        rates = [float(pred[sensitive == g].mean()) for g in np.unique(sensitive)]
        return float(max(rates) - min(rates)) if len(rates) >= 2 else 0.0

    for axis_name, s_full in axes.items():
        s_te = s_full[idx_test]
        s_tr = s_full[idx_train]
        ddp = delta_dp(pred_g, s_te)
        # Leakage probe on the graph embeddings.
        if np.unique(s_tr).size < 2:
            leak = float("nan")
        else:
            probe = LogisticRegression(max_iter=1000, random_state=SEED)
            probe.fit(graph_emb[idx_train], s_tr)
            n_classes = int(s_te.max()) + 1
            if n_classes == 2:
                leak = roc_auc_score(s_te, probe.predict_proba(graph_emb[idx_test])[:, 1])
            else:
                leak = roc_auc_score(
                    s_te, probe.predict_proba(graph_emb[idx_test]),
                    multi_class="ovr", average="macro",
                )
        print(f"  {axis_name:18s} ΔDP={ddp:.4f}  leak={leak:.4f}")


if __name__ == "__main__":
    main()
