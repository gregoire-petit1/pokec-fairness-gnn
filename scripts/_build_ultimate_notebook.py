"""Generate ``notebooks/ultimate_scenario.ipynb`` from this script.

Walks step-by-step through the **ULTIMATE-LATENT scenario**: TabICL fit,
capture row embeddings, INLP composite over (gender × age × region),
re-inject into icl_predictor, DPT composite calibration, final fairness +
F1 metrics on the 5 sensitive axes (3 marginals + 2 intersections).

Run ::

    .venv/bin/python scripts/_build_ultimate_notebook.py

→ notebooks/ultimate_scenario.ipynb (executable end-to-end on a GPU).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "ultimate_scenario.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = []

    nb.cells.append(
        md(
            """\
# Scénario ULTIMATE-LATENT — pas-à-pas

Cette chaîne traite **5 axes de fairness simultanément** (gender, region,
age_group + intersections gender × age et gender × region) sur un foundation
model tabulaire frozen (TabICL), en projetant le sensible **dans l'espace
latent du modèle** plutôt que sur les features brutes.

**Idée centrale.** Au lieu d'appliquer INLP sur `x` (264 dims), on récupère les
embeddings de ligne produits par `row_interactor` à l'intérieur de TabICL
(512 dims, accessibles via `TabICLCache.row_repr`), on y applique INLP, puis
on **ré-injecte** les embeddings projetés dans le predictor de TabICL.

**Étapes** :
1. Setup — chargement Pokec-z, splits stratifiés, attributs sensibles.
2. Baseline TabICL (no fairness).
3. Construction de l'attribut composite *(gender × age_group × region)* à
   12 cellules.
4. Capture des embeddings *train* (depuis le KV cache) et *test* (via hook
   sur `row_interactor.forward`).
5. INLP composite sur les embeddings *train* → matrice de projection `P`.
6. Re-injection : on remplace `cache.row_repr` par sa version projetée et on
   hook `row_interactor` pour projeter les embeddings test à l'inférence.
7. DPT composite : calibration de seuils par cellule sur val, appliqués sur
   test.
8. Métriques finales — F1, accuracy, ΔDP / ΔEO / leakage AUC sur les 5 axes.

**Caveat connu** : sur Pokec-z multi-seed, F1 = 0.884 ± 0.025 (stable) ; sur
Pokec-n, F1 = 0.657 ± 0.214 (3 seeds sur 5 collapsent). Le pipeline ULTIMATE
sur `x` brut reste préféré pour la robustesse cross-dataset. Ce notebook
illustre la mécanique, pas une recommandation."""
        )
    )

    # ── Step 1 — Setup ──────────────────────────────────────────────────────
    nb.cells.append(
        md(
            """\
## Étape 1 — Setup

Chargement Pokec-z (subset officiel FairGNN), split stratifié 60/20/20
sur `y × gender`, cap du train à 10 k pour rester dans le contexte ICL de
TabICL."""
        )
    )

    nb.cells.append(
        code(
            """\
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

from src.data.loader import load_pokec_z
from src.data.preprocessing import preprocess
from src.postprocess.inlp import inlp

SEED = 42
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_TRAIN = 10_000
print(f"device={DEVICE}, seed={SEED}, max_train={MAX_TRAIN}")"""
        )
    )

    nb.cells.append(
        code(
            """\
data = load_pokec_z(ROOT / "data" / "raw" / "pokec-z")
data = preprocess(data, sensitive_cols=["gender", "region", "age_group"])

x = data.x.cpu().numpy().astype(np.float32)
y = data.y.cpu().numpy().astype(np.int64)
gender = data.gender.cpu().numpy().astype(np.int64)
region = data.region.cpu().numpy().astype(np.int64)
age_group = data.age_group.cpu().numpy().astype(np.int64).clip(min=0)

# 60 / 20 / 20 stratified by y × gender, then cap train at MAX_TRAIN.
strat = y * 2 + gender
idx_train_full, idx_rest = train_test_split(
    np.arange(x.shape[0]), test_size=0.4, random_state=SEED, stratify=strat
)
idx_val, idx_test = train_test_split(
    idx_rest, test_size=0.5, random_state=SEED, stratify=strat[idx_rest]
)
rng = np.random.default_rng(SEED)
idx_train = (
    rng.choice(idx_train_full, size=MAX_TRAIN, replace=False)
    if idx_train_full.size > MAX_TRAIN else idx_train_full
)

print(f"train={idx_train.size}  val={idx_val.size}  test={idx_test.size}")
print(f"n_features={x.shape[1]}  positives baseline={y.mean():.3f}")"""
        )
    )

    # ── Step 2 — Baseline TabICL ────────────────────────────────────────────
    nb.cells.append(
        md(
            """\
## Étape 2 — Baseline TabICL

On entraîne TabICL avec `kv_cache="repr"` ; le cache stockera les
représentations de ligne (post-`row_interactor`) qu'on extrairera ensuite.
Mesure F1 + accuracy de référence."""
        )
    )

    nb.cells.append(
        code(
            """\
from tabicl import TabICLClassifier

clf = TabICLClassifier(
    random_state=SEED, device=DEVICE, n_estimators=4, kv_cache="repr"
)
clf.fit(x[idx_train], y[idx_train])

proba_test_base = clf.predict_proba(x[idx_test])
pred_test_base = proba_test_base.argmax(axis=1)
acc_baseline = accuracy_score(y[idx_test], pred_test_base)
f1_baseline = f1_score(y[idx_test], pred_test_base, average="macro")
print(f"Baseline TabICL :  acc={acc_baseline:.4f}  F1 macro={f1_baseline:.4f}")"""
        )
    )

    # ── Step 3 — Composite sensitive attribute ──────────────────────────────
    nb.cells.append(
        md(
            """\
## Étape 3 — Attribut sensible composite

On encode (gender × age_group × region) en un seul entier via mixed radix.
Avec gender ∈ {0,1}, age_group ∈ {0,1,2}, region ∈ {0,1}, on obtient
**12 cellules**. INLP travaillera sur ce composite (probe multi-classes à
12 classes), DPT calibrera un seuil par cellule."""
        )
    )

    nb.cells.append(
        code(
            """\
def build_composite(arrays, cardinalities):
    out = np.zeros_like(arrays[0])
    multiplier = 1
    for arr, k in zip(reversed(arrays), reversed(cardinalities), strict=True):
        out = out + arr * multiplier
        multiplier *= k
    return out

cards = [int(gender.max()) + 1, int(age_group.max()) + 1, int(region.max()) + 1]
composite_train = build_composite([gender[idx_train], age_group[idx_train], region[idx_train]], cards)
composite_val = build_composite([gender[idx_val], age_group[idx_val], region[idx_val]], cards)
composite_test = build_composite([gender[idx_test], age_group[idx_test], region[idx_test]], cards)
n_cells = int(composite_train.max()) + 1
print(f"composite cells = {n_cells}  (cards = {cards})")
print("distribution train (cell -> count) :")
unique, counts = np.unique(composite_train, return_counts=True)
for c, n in zip(unique, counts):
    print(f"  cell {int(c):2d} : {int(n):5d}")"""
        )
    )

    # ── Step 4 — Capture embeddings ─────────────────────────────────────────
    nb.cells.append(
        md(
            """\
## Étape 4 — Capture des embeddings TabICL

Avec `kv_cache="repr"`, le cache contient déjà les embeddings *train*
(post-`row_interactor` + y baked-in). Pour les embeddings *test*, on hook
`row_interactor.forward` pendant un forward pass : chaque appel renvoie un
tenseur `(B_chunk, test_size, H)` qu'on accumule puis on moyenne sur la
dimension ensemble."""
        )
    )

    nb.cells.append(
        code(
            """\
# Train embeddings : pull from cache.
cache = next(iter(clf.model_kv_cache_.values()))
train_emb = cache.row_repr.float().mean(dim=0).cpu().numpy().astype(np.float32)
print(f"train_emb : shape={train_emb.shape}  dtype={train_emb.dtype}")

# Test embeddings : hook row_interactor during a predict_proba.
captured: list[torch.Tensor] = []
original_forward = clf.model_.row_interactor.forward

def capture_hook(*args, **kwargs):
    out = original_forward(*args, **kwargs)
    captured.append(out.detach().clone())
    return out

clf.model_.row_interactor.forward = capture_hook
try:
    clf.predict_proba(x[idx_test])
finally:
    clf.model_.row_interactor.forward = original_forward

stacked = torch.cat(captured, dim=0)  # (B_total, test_size, H)
test_emb = stacked.float().mean(dim=0).cpu().numpy().astype(np.float32)
print(f"test_emb  : shape={test_emb.shape}  dtype={test_emb.dtype}")"""
        )
    )

    # ── Step 5 — INLP composite ─────────────────────────────────────────────
    nb.cells.append(
        md(
            """\
## Étape 5 — INLP composite sur les embeddings train

INLP itère un probe LR multi-classes contre `composite_train` ;
chaque itération extrait les directions du sensible et les projette à zéro.
Avec 12 classes, INLP supprime jusqu'à 11 directions de l'espace
512-dimensionnel. Le résultat : une matrice `P` qu'on peut appliquer à
*n'importe quelle* nouvelle représentation."""
        )
    )

    nb.cells.append(
        code(
            """\
# Pre-INLP leakage probe : checked that the gender direction is recoverable.
probe_pre = LogisticRegression(max_iter=1000, random_state=SEED)
probe_pre.fit(train_emb, gender[idx_train])
leakage_pre = roc_auc_score(
    gender[idx_test], probe_pre.predict_proba(test_emb)[:, 1]
)
print(f"Leakage gender (pre-INLP) on test embeddings = {leakage_pre:.4f}")

# Fit INLP on (train_emb, composite_train).
_train_clean, projection = inlp(train_emb, composite_train, n_iter=15, seed=SEED)
print(f"projection P : shape={projection.shape}")
print(f"rank reduction ≈ {projection.shape[0] - np.linalg.matrix_rank(projection)}")"""
        )
    )

    # ── Step 6 — Re-inject ──────────────────────────────────────────────────
    nb.cells.append(
        md(
            """\
## Étape 6 — Ré-injection dans `icl_predictor`

Deux opérations :
1. Remplacer `cache.row_repr` (per ensemble member) par sa version projetée
   `row_repr @ P` — pour que les embeddings train passés à l'ICL predictor
   soient les versions *cleanées*.
2. Hook `row_interactor.forward` sur le futur forward, de sorte que les
   embeddings test soient également projetés à la volée.

Ces deux modifications garantissent que train et test entrent dans
`icl_predictor` dans le **même sous-espace projeté**."""
        )
    )

    nb.cells.append(
        code(
            """\
p_torch = torch.from_numpy(projection.astype(np.float32))

# 1) Modify cache.row_repr in place, per norm method.
for _norm_method, c in clf.model_kv_cache_.items():
    original = c.row_repr  # shape (n_ensemble, train_size, H)
    c.row_repr = torch.einsum(
        "eth,hk->etk", original.float(), p_torch.to(original.device)
    ).to(original.dtype)
print("row_repr cache projeté en place.")

# 2) Hook row_interactor so test reprs get projected at inference time.
original_forward = clf.model_.row_interactor.forward

def project_hook(*args, **kwargs):
    out = original_forward(*args, **kwargs)
    return torch.einsum(
        "eth,hk->etk", out.float(), p_torch.to(out.device)
    ).to(out.dtype)

clf.model_.row_interactor.forward = project_hook

try:
    proba_val_proj = clf.predict_proba(x[idx_val])
    proba_test_proj = clf.predict_proba(x[idx_test])
finally:
    clf.model_.row_interactor.forward = original_forward

print(f"proba_val   : shape={proba_val_proj.shape}")
print(f"proba_test  : shape={proba_test_proj.shape}")"""
        )
    )

    # ── Step 7 — DPT composite ──────────────────────────────────────────────
    nb.cells.append(
        md(
            """\
## Étape 7 — Calibration DPT composite

Pour chaque cellule du composite (12 au total), on cherche le seuil qui
égalise le taux de prédiction positive de la cellule au taux global. Avec
12 cellules sur ~13 k samples val, chaque cellule a ~1 100 samples, ce qui
suffit pour calibrer un seuil stable. Une fois calibrés sur val, on applique
les seuils par cellule sur les probabilités de test."""
        )
    )

    nb.cells.append(
        code(
            """\
proba_val_pos = proba_val_proj[:, 1] if proba_val_proj.ndim == 2 else proba_val_proj.ravel()
proba_test_pos = proba_test_proj[:, 1] if proba_test_proj.ndim == 2 else proba_test_proj.ravel()

# Calibrate per-cell thresholds on val.
grid = np.linspace(0.0, 1.0, 51, dtype=np.float32)
global_rate = float((proba_val_pos > 0.5).mean())
thresholds = {}
for cell in np.unique(composite_val):
    mask = composite_val == cell
    if mask.sum() == 0:
        thresholds[int(cell)] = 0.5
        continue
    cell_proba = proba_val_pos[mask]
    rates = (cell_proba[None, :] > grid[:, None]).mean(axis=1)
    best = int(np.argmin(np.abs(rates - global_rate)))
    thresholds[int(cell)] = float(grid[best])

print("Seuils calibrés (cell -> threshold) :")
for cell, t in sorted(thresholds.items()):
    print(f"  cell {cell:2d} -> {t:.3f}")

# Apply thresholds vectorisé : per-row threshold lookup via composite cell.
unique_cells = np.unique(composite_test)
cell_to_t = np.array([thresholds.get(int(c), 0.5) for c in unique_cells], dtype=np.float32)
_, inv = np.unique(composite_test, return_inverse=True)
row_t = cell_to_t[inv]
pred_test_ultimate = (proba_test_pos > row_t).astype(np.int64)

acc_ultimate = accuracy_score(y[idx_test], pred_test_ultimate)
f1_ultimate = f1_score(y[idx_test], pred_test_ultimate, average="macro")
print(f"\\nULTIMATE-LATENT :  acc={acc_ultimate:.4f}  F1 macro={f1_ultimate:.4f}")
print(f"Baseline TabICL :  acc={acc_baseline:.4f}  F1 macro={f1_baseline:.4f}")"""
        )
    )

    # ── Step 8 — Per-axis metrics ───────────────────────────────────────────
    nb.cells.append(
        md(
            """\
## Étape 8 — Métriques fairness sur les 5 axes

ΔDP, ΔEO et leakage AUC pour gender, region, age_group, et les deux
intersections gender × age et gender × region. La chaîne ULTIMATE est
calibrée sur le **composite à 12 cellules** : on n'a pas besoin de re-fitter
pour chaque axe, le composite couvre tout.

**Lecture attendue** :
- ΔDP marginaux et intersectionnels proches de 0 (DPT calibré).
- Leakage AUC proche de 0.50 (chance) sur tous les axes (INLP composite).
- F1 légèrement dégradé vs baseline (coût de la fairness multi-axes)."""
        )
    )

    nb.cells.append(
        code(
            """\
def delta_dp(pred, sensitive):
    rates = [float(pred[sensitive == g].mean()) for g in np.unique(sensitive)]
    return float(max(rates) - min(rates)) if len(rates) >= 2 else 0.0

def delta_eo(pred, y_true, sensitive):
    tprs = []
    for g in np.unique(sensitive):
        mask = (sensitive == g) & (y_true == 1)
        if mask.sum() == 0:
            continue
        tprs.append(float(pred[mask].mean()))
    return float(max(tprs) - min(tprs)) if len(tprs) >= 2 else 0.0

# Project full embeddings for downstream leakage probes.
train_emb_proj = (train_emb @ projection).astype(np.float32)
test_emb_proj = (test_emb @ projection).astype(np.float32)

axes = {
    "gender": (gender[idx_train], gender[idx_test]),
    "region": (region[idx_train], region[idx_test]),
    "age_group": (age_group[idx_train], age_group[idx_test]),
    "gender_x_age": (
        build_composite([gender[idx_train], age_group[idx_train]], [cards[0], cards[1]]),
        build_composite([gender[idx_test], age_group[idx_test]], [cards[0], cards[1]]),
    ),
    "gender_x_region": (
        build_composite([gender[idx_train], region[idx_train]], [cards[0], cards[2]]),
        build_composite([gender[idx_test], region[idx_test]], [cards[0], cards[2]]),
    ),
}

rows = []
for axis_name, (s_tr, s_te) in axes.items():
    ddp = delta_dp(pred_test_ultimate, s_te)
    deo = delta_eo(pred_test_ultimate, y[idx_test], s_te)

    n_classes = int(s_te.max()) + 1
    probe = LogisticRegression(max_iter=1000, random_state=SEED)
    probe.fit(train_emb_proj, s_tr)
    if n_classes == 2:
        leak = roc_auc_score(s_te, probe.predict_proba(test_emb_proj)[:, 1])
    else:
        leak = roc_auc_score(
            s_te, probe.predict_proba(test_emb_proj),
            multi_class="ovr", average="macro",
        )
    rows.append({
        "axis": axis_name,
        "delta_dp": round(ddp, 4),
        "delta_eo": round(deo, 4),
        "leakage_post_inlp": round(leak, 4),
    })

df_axes = pl.DataFrame(rows)
print(df_axes)"""
        )
    )

    nb.cells.append(
        md(
            """\
## Lecture finale

| | Baseline TabICL | ULTIMATE-LATENT |
|---|---:|---:|
| Acc | *(cf. cellule étape 2)* | *(cf. cellule étape 7)* |
| F1 macro | *(idem)* | *(idem)* |
| Leakage gender | ~0.81 | ~0.55 |
| Leakage age_group | ~0.99 | ~0.63 |
| ΔDP gender | ~0.04 | ~0.006 |

**Quand utiliser cette chaîne ?** Quand on veut traiter plusieurs axes
simultanément avec un foundation model frozen et qu'on a accès à ses
embeddings internes. Sur Pokec-z, ça marche ; sur Pokec-n, c'est instable
(cf. annexe du 2-pager). Pour de la production, préférer la version
ULTIMATE sur `x` brut (`apply_inlp_composite_to_tabicl` dans
`scripts/main_experiment.py`), validée multi-seed cross-dataset à <0.01
près.

**Reproduction**. Le module `src/baselines/tabicl_inlp_reinjection_composite.py`
encapsule cette chaîne dans `run_ultimate_reinjection()` ; le runner
`scripts/run_tabicl_inlp_reinjection_composite.py` la déroule en multi-seed
× Pokec-z/n et écrit les résultats dans
`results/metrics/tabicl_inlp_reinjection_composite.csv`."""
        )
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        nbf.write(nb, f)
    print(f"wrote {OUT}  ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
