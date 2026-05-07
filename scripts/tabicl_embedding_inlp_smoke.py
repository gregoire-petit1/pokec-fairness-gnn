"""Smoke test : montrer que les embeddings TabICL sont accessibles et que INLP
fonctionne dessus comme un vrai post-process.

Steps :
  1. Fit ``TabICLClassifier`` avec ``kv_cache="repr"``.
  2. Lire ``model_kv_cache_[norm_method].row_repr`` -> tenseur d'embeddings.
  3. Mesurer leakage gender via probe LR.
  4. Fit INLP sur train embeddings -> projection matrix P.
  5. Project embeddings, re-mesurer leakage.

Si le leakage tombe vers 0.5 c'est validé.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tabicl import TabICLClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_pokec_z  # noqa: E402
from src.data.preprocessing import preprocess  # noqa: E402
from src.postprocess.inlp import inlp  # noqa: E402


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data = load_pokec_z(ROOT / "data/raw/pokec-z")
    data = preprocess(data, sensitive_cols=["gender", "region", "age_group"])

    x = data.x.cpu().numpy().astype(np.float32)
    y = data.y.cpu().numpy().astype(np.int64)
    gender = data.gender.cpu().numpy().astype(np.int64)
    n = x.shape[0]
    print(f"n={n}, d={x.shape[1]}")

    # 60/20/20 split stratified by y * gender
    strat = y * 2 + gender
    idx_train, idx_rest = train_test_split(
        np.arange(n), test_size=0.4, random_state=42, stratify=strat
    )
    # cap train at 10k for TabICL context
    rng = np.random.default_rng(42)
    if idx_train.size > 10_000:
        idx_train = rng.choice(idx_train, size=10_000, replace=False)
    print(f"train={idx_train.size}")

    print("\n=== Fit TabICL with kv_cache='repr' ===")
    clf = TabICLClassifier(random_state=42, device=device, n_estimators=4, kv_cache="repr")
    clf.fit(x[idx_train], y[idx_train])

    print("\n=== Inspect model_kv_cache_ ===")
    print(f"keys (norm methods): {list(clf.model_kv_cache_.keys())}")
    for name, cache in clf.model_kv_cache_.items():
        print(f"  [{name}] cache_type={cache.cache_type} cache_size_mb={cache.cache_size_mb}")
        print(
            f"  [{name}] row_repr shape={tuple(cache.row_repr.shape)} dtype={cache.row_repr.dtype}"
        )

    # take first norm method, average across ensemble dim
    first_cache = next(iter(clf.model_kv_cache_.values()))
    row_repr = first_cache.row_repr.float().cpu().numpy()  # (B, train_size, H)
    print(
        f"\nrow_repr ndarray: shape={row_repr.shape} "
        f"min={row_repr.min():.3f} max={row_repr.max():.3f} "
        f"mean={row_repr.mean():.3f}"
    )

    # average across ensemble dim → (train_size, H)
    emb_train = row_repr.mean(axis=0)
    print(f"averaged emb_train: shape={emb_train.shape}")

    # Vérifier que train_size match nb idx_train
    if emb_train.shape[0] != idx_train.size:
        print(f"WARNING: emb_train size {emb_train.shape[0]} != idx_train size {idx_train.size}")

    s_train = gender[idx_train[: emb_train.shape[0]]]

    # Split train embeddings into probe-train / probe-test (50/50) for honest AUC
    perm = rng.permutation(emb_train.shape[0])
    half = emb_train.shape[0] // 2
    pi_tr, pi_te = perm[:half], perm[half:]
    z_tr, z_te = emb_train[pi_tr], emb_train[pi_te]
    s_tr, s_te = s_train[pi_tr], s_train[pi_te]

    print("\n=== Pre-INLP leakage probe ===")
    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(z_tr, s_tr)
    auc_pre = roc_auc_score(s_te, probe.predict_proba(z_te)[:, 1])
    print(f"Leakage AUC gender (pre-INLP) = {auc_pre:.4f}")

    print("\n=== Apply INLP ===")
    z_tr_clean, P = inlp(z_tr, s_tr, n_iter=15, seed=42)
    z_te_clean = (z_te @ P).astype(np.float32)
    print(
        f"P shape: {P.shape}, z_clean ranges: min={z_tr_clean.min():.3f} max={z_tr_clean.max():.3f}"
    )

    print("\n=== Post-INLP leakage probe ===")
    probe2 = LogisticRegression(max_iter=1000, random_state=42)
    probe2.fit(z_tr_clean, s_tr)
    auc_post = roc_auc_score(s_te, probe2.predict_proba(z_te_clean)[:, 1])
    print(f"Leakage AUC gender (post-INLP) = {auc_post:.4f}")

    print("\n=== Verdict ===")
    print(f"  pre-INLP leakage  : {auc_pre:.4f}")
    print(f"  post-INLP leakage : {auc_post:.4f}")
    print(f"  drop              : {auc_pre - auc_post:+.4f}")
    if auc_post < 0.55:
        print("  ✓ INLP works on TabICL embeddings — claim validée.")
    else:
        print("  ✗ INLP did not bring leakage to chance — investigate.")


if __name__ == "__main__":
    main()
