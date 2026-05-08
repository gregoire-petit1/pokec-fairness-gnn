"""Generate ``notebooks/ultimate_scenario.ipynb`` from this script.

Walks step-by-step through the **ULTIMATE scenario** (validated reference,
applied on raw features ``x`` rather than on TabICL embeddings) :

1. Setup (data, splits, sensitive attrs).
2. Baseline TabICL (no fairness).
3. Composite sensitive attribute (gender × age_group × region → 12 cells).
4. INLP composite on ``x`` brut → projection ``P``, ``x_clean``.
5. Downstream LR on ``x_clean`` → INLP-stage predictions.
6. DPT composite : per-cell threshold calibrated on val.
7. ULTIMATE predictions = (LR sur x_clean) puis DPT composite.
8. Métriques finales acc + macro F1 + ΔDP / ΔEO / leakage sur 5 axes.

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
# Scénario ULTIMATE — pas-à-pas

Cette chaîne traite **5 axes de fairness simultanément** (gender, region,
age_group + intersections gender × age et gender × region) sur Pokec-z, en
combinant TabICL (foundation tabulaire frozen) avec une chaîne de
post-process *INLP composite + DPT composite* appliquée sur les features
brutes `x`.

**Idée centrale.** Construire un attribut composite à 12 cellules
*(gender × age_group × region)*, projeter `x` orthogonalement à toutes les
directions du composite via INLP, ré-entraîner un classifieur sur `x_clean`,
puis calibrer un seuil par cellule via DPT. Une seule chaîne couvre les
5 axes, intersections incluses.

**Étapes** :
1. Setup — chargement Pokec-z, splits stratifiés, attributs sensibles.
2. Baseline TabICL (no fairness).
3. Construction de l'attribut composite à 12 cellules.
4. INLP composite sur `x` brut → matrice de projection `P`, `x_clean`.
5. Classifieur downstream sur `x_clean` → prédictions stade INLP.
6. DPT composite : calibration de seuils par cellule sur val.
7. ULTIMATE : prédictions finales après application des seuils.
8. Métriques fairness (acc, macro F1, ΔDP, ΔEO, leakage AUC) sur 5 axes.

**Validation.** Multi-seed [3, 7, 21, 42, 99] × Pokec-z/n donne F1 ≈ 0.866
± 0.005 et leakage ≈ 0.50 (chance level) sur les 5 axes simultanément.
Cross-dataset reproduction à <0.01 près. C'est le pipeline préféré pour la
robustesse cross-dataset (cf. annexe du 2-pager qui compare avec une
variante latent qui s'effondre sur Pokec-n)."""
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
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Embeddings TabICL et features Pokec ont des magnitudes hétérogènes →
# lbfgs ne converge pas toujours en 1000 itérations sur les probes LR.
# La solution reste numériquement correcte ; on masque les warnings pour
# garder les outputs lisibles.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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

TabICL est un foundation model tabulaire qui prédit en in-context learning
(no graph, no fairness). On l'utilise comme baseline contre laquelle
comparer la chaîne ULTIMATE."""
        )
    )

    nb.cells.append(
        code(
            """\
from tabicl import TabICLClassifier

clf_baseline = TabICLClassifier(random_state=SEED, device=DEVICE, n_estimators=4)
clf_baseline.fit(x[idx_train], y[idx_train])

proba_test_baseline = clf_baseline.predict_proba(x[idx_test])
pred_test_baseline = proba_test_baseline.argmax(axis=1)
acc_baseline = accuracy_score(y[idx_test], pred_test_baseline)
f1_baseline = f1_score(y[idx_test], pred_test_baseline, average="macro")
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
12 classes), DPT calibrera un seuil par cellule.

**Pourquoi un composite ?** Calibrer mono-axe (par ex. gender seul) laisse
fuiter sur les axes croisés (effet *whack-a-mole* de Crenshaw 1989). Le
composite couvre toutes les intersections en un seul fit."""
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
composite_train = build_composite(
    [gender[idx_train], age_group[idx_train], region[idx_train]], cards
)
composite_val = build_composite(
    [gender[idx_val], age_group[idx_val], region[idx_val]], cards
)
composite_test = build_composite(
    [gender[idx_test], age_group[idx_test], region[idx_test]], cards
)
n_cells = int(composite_train.max()) + 1
print(f"composite cells = {n_cells}  (cards = {cards})")
print("distribution train (cell -> count) :")
for c, n in zip(*np.unique(composite_train, return_counts=True)):
    print(f"  cell {int(c):2d} : {int(n):5d}")"""
        )
    )

    # ── Step 4 — INLP composite on x ────────────────────────────────────────
    nb.cells.append(
        md(
            """\
## Étape 4 — INLP composite sur les features `x`

INLP itère un probe LR multi-classes contre `composite_train` ; chaque
itération extrait les directions du sensible et les projette à zéro. Avec
12 classes, INLP supprime jusqu'à 11 directions de l'espace
264-dimensionnel des features Pokec. Le résultat : une matrice `P`
applicable à n'importe quelle nouvelle observation."""
        )
    )

    nb.cells.append(
        code(
            """\
# Pre-INLP leakage probe : check that the composite is recoverable from x.
probe_pre = LogisticRegression(max_iter=1000, random_state=SEED)
probe_pre.fit(x[idx_train], composite_train)
leakage_pre = roc_auc_score(
    composite_test, probe_pre.predict_proba(x[idx_test]),
    multi_class="ovr", average="macro",
)
print(f"Leakage composite (pre-INLP) on x_test = {leakage_pre:.4f}")

# Fit INLP on (x_train, composite_train).
x_train_clean, projection = inlp(x[idx_train], composite_train, n_iter=15, seed=SEED)
print(f"projection P : shape={projection.shape}")

# Project val and test through P.
x_val_clean = (x[idx_val] @ projection).astype(np.float32)
x_test_clean = (x[idx_test] @ projection).astype(np.float32)

# Post-INLP leakage probe : composite should be much harder to recover.
probe_post = LogisticRegression(max_iter=1000, random_state=SEED)
probe_post.fit(x_train_clean, composite_train)
leakage_post = roc_auc_score(
    composite_test, probe_post.predict_proba(x_test_clean),
    multi_class="ovr", average="macro",
)
print(f"Leakage composite (post-INLP) on x_test = {leakage_post:.4f}")"""
        )
    )

    # ── Step 5 — Downstream classifier on x_clean ───────────────────────────
    nb.cells.append(
        md(
            """\
## Étape 5 — Re-fit TabICL sur `x_clean`

On ré-entraîne TabICL **sur les features projetées** `x_clean`. C'est le
stade INLP de la chaîne : les prédictions ne reposent plus sur les
directions du sensible (par construction), mais TabICL conserve assez de
puissance prédictive sur les directions orthogonales pour rester
performant."""
        )
    )

    nb.cells.append(
        code(
            """\
clf_inlp = TabICLClassifier(random_state=SEED, device=DEVICE, n_estimators=4)
clf_inlp.fit(x_train_clean, y[idx_train])

proba_val_inlp = clf_inlp.predict_proba(x_val_clean)
proba_test_inlp = clf_inlp.predict_proba(x_test_clean)
pred_test_inlp = clf_inlp.predict(x_test_clean)

acc_inlp = accuracy_score(y[idx_test], pred_test_inlp)
f1_inlp = f1_score(y[idx_test], pred_test_inlp, average="macro")
print(f"Stage INLP only (TabICL on x_clean) :  acc={acc_inlp:.4f}  F1 macro={f1_inlp:.4f}")"""
        )
    )

    # ── Step 6 — DPT calibration on val ─────────────────────────────────────
    nb.cells.append(
        md(
            """\
## Étape 6 — DPT composite : calibration des seuils par cellule

Pour chaque cellule du composite (12 au total), on cherche le seuil qui
égalise le taux de prédiction positive de la cellule au taux global. Les
seuils sont calibrés sur **val** (~13 k samples → ~1 100 par cellule),
pour éviter le data leakage côté test."""
        )
    )

    nb.cells.append(
        code(
            """\
proba_val_pos = proba_val_inlp[:, 1]
proba_test_pos = proba_test_inlp[:, 1]

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

print(f"global positive rate (val) = {global_rate:.3f}")
print("Seuils calibrés (cell -> threshold) :")
for cell, t in sorted(thresholds.items()):
    print(f"  cell {cell:2d} -> {t:.3f}")"""
        )
    )

    # ── Step 7 — Apply DPT thresholds → ULTIMATE predictions ────────────────
    nb.cells.append(
        md(
            """\
## Étape 7 — Application des seuils → prédictions ULTIMATE

Chaque ligne du test est classée selon le seuil de sa cellule composite.
Vectorisé via lookup `cell -> threshold`."""
        )
    )

    nb.cells.append(
        code(
            """\
unique_cells = np.unique(composite_test)
cell_to_t = np.array(
    [thresholds.get(int(c), 0.5) for c in unique_cells], dtype=np.float32
)
_, inv = np.unique(composite_test, return_inverse=True)
row_t = cell_to_t[inv]
pred_test_ultimate = (proba_test_pos > row_t).astype(np.int64)

print(f"prédictions ULTIMATE :  positives={pred_test_ultimate.sum()} / {len(pred_test_ultimate)}")"""
        )
    )

    # ── Step 8 — Final metrics: acc + F1 + per-axis fairness ───────────────
    nb.cells.append(
        md(
            """\
## Étape 8 — Métriques finales : accuracy, macro F1, ΔDP / ΔEO / Leakage

Recalcul de l'accuracy et du macro F1 après la chaîne complète, et fairness
sur les 5 axes (3 marginaux + 2 intersections). La chaîne ULTIMATE est
calibrée sur le composite à 12 cellules : un seul fit couvre tous les axes,
intersections incluses.

**Lecture attendue** :
- ΔDP marginaux et intersectionnels proches de 0 (DPT calibré sur val).
- Leakage AUC proche de 0.50 (chance) sur tous les axes (INLP composite).
- F1 et acc dégradés vs baseline TabICL (coût de la fairness multi-axes
  intégrale)."""
        )
    )

    nb.cells.append(
        code(
            """\
# Acc + macro F1 sur les prédictions finales ULTIMATE.
acc_ultimate = accuracy_score(y[idx_test], pred_test_ultimate)
f1_ultimate = f1_score(y[idx_test], pred_test_ultimate, average="macro")

print("\\n=== Récap accuracy & macro F1 ===")
print(f"  Baseline TabICL  : acc={acc_baseline:.4f}  F1 macro={f1_baseline:.4f}")
print(f"  Stage INLP       : acc={acc_inlp:.4f}  F1 macro={f1_inlp:.4f}")
print(f"  ULTIMATE (final) : acc={acc_ultimate:.4f}  F1 macro={f1_ultimate:.4f}")


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


def leakage_auc(z_train, z_test, s_train, s_test):
    if np.unique(s_train).size < 2:
        return float("nan")
    probe = LogisticRegression(max_iter=1000, random_state=SEED)
    probe.fit(z_train, s_train)
    if int(s_test.max()) + 1 == 2:
        return float(roc_auc_score(s_test, probe.predict_proba(z_test)[:, 1]))
    return float(roc_auc_score(
        s_test, probe.predict_proba(z_test),
        multi_class="ovr", average="macro",
    ))


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
    rows.append({
        "axis": axis_name,
        "acc": round(acc_ultimate, 4),
        "f1_macro": round(f1_ultimate, 4),
        "delta_dp": round(delta_dp(pred_test_ultimate, s_te), 4),
        "delta_eo": round(delta_eo(pred_test_ultimate, y[idx_test], s_te), 4),
        "leakage_post_inlp": round(leakage_auc(x_train_clean, x_test_clean, s_tr, s_te), 4),
    })

print("\\n=== Métriques par axe (ULTIMATE) ===")
df_axes = pl.DataFrame(rows)
print(df_axes)"""
        )
    )

    nb.cells.append(
        md(
            """\
## Lecture finale

| | Baseline TabICL | ULTIMATE (validé multi-seed) |
|---|---:|---:|
| Acc | ~0.946 | ~0.866 |
| F1 macro | ~0.946 | ~0.866 |
| Leakage gender | ~0.88 | ~0.50 |
| Leakage age_group | ~0.99 | ~0.48 |
| ΔDP gender | ~0.04 | ~0.013 |
| ΔDP gender × age | ~0.09 | ~0.05 |

**Coût** : 8 pp de F1 / accuracy pour la fairness multi-axes complète.
**Bénéfice** : leakage au chance level sur les 5 axes simultanément
(incluant intersections), ΔDP < 0.05 partout.

**Reproduction** :
- `scripts/main_experiment.py` orchestre la chaîne complète multi-seed
  via `apply_inlp_composite_to_tabicl()` et `apply_composite_dpt()`.
- `results/metrics/comparison_full.csv` contient les résultats validés
  pour seed=42, multi-méthodes × multi-axes.
- `results/metrics/comparison_multiseed_summary.csv` agrège [3, 7, 21,
  42, 99] avec moyennes ± écarts-types.

**Variante exploratoire**. Une chaîne ULTIMATE-LATENT (INLP appliqué dans
l'espace latent `TabICLCache.row_repr`) gagne 1.8 pp sur Pokec-z mais
collapse sur 3/5 seeds Pokec-n. Cf. annexe du 2-pager. Le présent
notebook décrit la chaîne robuste, x-brut, recommandée pour la
production."""
        )
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        nbf.write(nb, f)
    print(f"wrote {OUT}  ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
