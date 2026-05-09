"""Interprétabilité de notre pipeline fairness — options A et B.

A. **Feature importance via coefficients LR** (équivalent SHAP pour modèle
   linéaire) : on entraîne une LR sur ``x`` brut prédisant la cible, on
   regarde les top-k features par |coef|, on croise avec ``gender`` et
   ``region`` pour comprendre **mécaniquement** d'où vient le biais.

B. **GNNExplainer** (PyG, déjà dans `src/interpretability/explainer.py`)
   sur un échantillon de nœuds test → agrégat des feature_masks +
   edge_masks. Pour le edge_mask, on calcule la fraction d'arêtes
   **intra-region** parmi les top-10% — si elle est >> au taux baseline,
   le GNN s'appuie effectivement sur l'homophilie region pour prédire.

Run ::

    .venv/bin/python scripts/run_interpretability.py

Output : ``results/metrics/interpretability_top_features.csv`` (option A),
``results/metrics/interpretability_gnnexplainer.csv`` (option B).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_pokec_z  # noqa: E402
from src.data.preprocessing import preprocess  # noqa: E402
from src.data.splits import make_splits  # noqa: E402
from src.interpretability.explainer import explain_group_with_edges  # noqa: E402
from src.interpretability.feature_importance import (  # noqa: E402
    correlation_with_sensitive,
    rank_features_by_coef,
)
from src.models.graphsage import GraphSAGE  # noqa: E402

OUT_DIR = ROOT / "results" / "metrics"
SEED = 42
TOP_K = 15
N_NODES_GNNEXPLAINER = 50


def setup_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def option_a_lr_coefs(data, train_idx: np.ndarray) -> pl.DataFrame:
    """LR fit sur ``x`` → top-k features ranked by |coef|, croisés avec
    gender et region pour comprendre la source du biais."""
    feature_names = data.feature_cols
    x = data.x.cpu().numpy().astype(np.float32)
    y = data.y.cpu().numpy().astype(np.int64)
    gender = data.gender.cpu().numpy().astype(np.float32)
    region = data.region.cpu().numpy().astype(np.float32)

    setup_seeds(SEED)
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(x[train_idx], y[train_idx])
    coefs = clf.coef_.ravel()

    top = rank_features_by_coef(coefs, feature_names, top_k=TOP_K)
    top_features = top["feature"].to_list()

    corr_g = correlation_with_sensitive(
        x, feature_names, top_features, gender, sensitive_name="gender"
    )
    corr_r = correlation_with_sensitive(
        x, feature_names, top_features, region, sensitive_name="region"
    )

    merged = top.join(
        corr_g.select(["feature", "corr_gender", "abs_corr"]).rename(
            {"abs_corr": "abs_corr_gender"}
        ),
        on="feature",
    ).join(
        corr_r.select(["feature", "corr_region", "abs_corr"]).rename(
            {"abs_corr": "abs_corr_region"}
        ),
        on="feature",
    )
    return merged


def option_b_gnnexplainer(
    data, model: torch.nn.Module, test_idx: np.ndarray, device: str
) -> pl.DataFrame:
    """GNNExplainer sur ``N_NODES_GNNEXPLAINER`` nœuds test, agrégat."""
    setup_seeds(SEED)
    rng = np.random.default_rng(SEED)
    sampled = rng.choice(test_idx, size=min(N_NODES_GNNEXPLAINER, test_idx.size), replace=False)

    # Boolean tensor : True if edge connects two nodes with same region.
    edge_index = data.edge_index.to(device)
    region_tensor = data.region.to(device).long()
    edge_intra_region = region_tensor[edge_index[0]] == region_tensor[edge_index[1]]

    print(f"  GNNExplainer sur {len(sampled)} nœuds test...", flush=True)
    t0 = time.time()
    out = explain_group_with_edges(
        model=model,
        data=data,
        node_indices=sampled.tolist(),
        num_hops=2,
        edge_intra_attr=edge_intra_region,
        top_edge_frac=0.10,
    )
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    feat_imp = out["mean_feat_importance"].cpu().numpy()
    feature_names = data.feature_cols

    # Top-k features par importance moyenne du mask.
    rank = np.argsort(np.abs(feat_imp))[::-1][:TOP_K]
    top_feat_df = pl.DataFrame(
        {
            "rank": np.arange(1, len(rank) + 1, dtype=np.int64),
            "feature": [feature_names[int(i)] for i in rank],
            "mean_importance": feat_imp[rank].astype(np.float32),
        }
    )

    # Stats edge intra-region : baseline vs top.
    summary_row = pl.DataFrame(
        {
            "intra_region_baseline": [out["intra_attr_fraction_global"]],
            "intra_region_top10pct": [out["intra_attr_fraction_top"]],
            "amplification": [
                out["intra_attr_fraction_top"] / max(out["intra_attr_fraction_global"], 1e-9)
            ],
            "n_explained_nodes": [out["n_explained"]],
        }
    )
    return top_feat_df, summary_row


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Data + splits.
    data = load_pokec_z(ROOT / "data" / "raw" / "pokec-z")
    data = preprocess(data, sensitive_cols=["gender", "region", "age_group"])
    data = data.to(device)

    n = data.y.shape[0]
    train_idx_t, _val_idx_t, test_idx_t = make_splits(
        n=n, y=data.y.cpu(), sensitive=data.gender.cpu(), seed=SEED
    )
    train_idx = train_idx_t.cpu().numpy()
    test_idx = test_idx_t.cpu().numpy()

    # ── Option A : LR coefs + corrélation gender/region ────────────────────
    print("\n=== Option A : LR coefs + corrélation sensible ===")
    a_df = option_a_lr_coefs(data, train_idx)
    a_path = OUT_DIR / "interpretability_top_features.csv"
    a_df.write_csv(a_path)
    with pl.Config(tbl_rows=20, fmt_str_lengths=40, float_precision=4):
        print(a_df)
    print(f"→ {a_path}", flush=True)

    # ── Option B : GNNExplainer ─────────────────────────────────────────────
    print("\n=== Option B : GNNExplainer ===")
    cache_path = ROOT / "results" / "cache" / f"seed{SEED}" / "GraphSAGE.pt"
    if not cache_path.exists():
        print(f"[SKIP] GraphSAGE cache absent ({cache_path}). Lancer main_experiment d'abord.")
        return
    cached = torch.load(cache_path, weights_only=False, map_location=device)
    if "model_state_dict" not in cached:
        print("[SKIP] cached ModelOutput n'a pas de model_state_dict, GNNExplainer impossible")
        print(f"       (cache contient: {list(cached.keys())})")
        return

    model = GraphSAGE(
        in_channels=data.x.shape[1], hidden_channels=256, out_channels=2, num_layers=2
    ).to(device)
    model.load_state_dict(cached["model_state_dict"])
    model.eval()

    top_feat_b, summary_b = option_b_gnnexplainer(data, model, test_idx, device)
    b_path = OUT_DIR / "interpretability_gnnexplainer.csv"
    top_feat_b.write_csv(b_path)
    print(top_feat_b)
    print()
    print("Statistiques edge intra-region :")
    print(summary_b)
    print(f"→ {b_path}", flush=True)


if __name__ == "__main__":
    main()
