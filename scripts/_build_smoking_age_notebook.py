"""Build a pedagogical notebook aligned with report-v2/main.tex.

Pipeline : load → baseline GraphSAGE → Resampling → Reweighting → FairDrop
→ FairGNN (λ=1) → INLP → interprétabilité (LR coefs) → robustesse.

No DPT/EOT/chained post-process. Code lisible étudiant, peu d'abstraction,
pas de framework. Output : ``notebooks/smoking_age_pipeline.ipynb``.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "smoking_age_pipeline.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    # ── 0. Header ──────────────────────────────────────────────────────────
    cells.append(
        md(
            """# Pokec — Fairness on Graph Neural Networks
## Pipeline complet : smoking × age

Ce notebook reproduit l'expérimentation du rapport (`report-v2/main.tex`).
On entraîne un GraphSAGE pour prédire le tabagisme régulier à partir des
features et du graphe d'amitiés Pokec, puis on compare une méthode dans
chaque famille de fairness :

- **pré-traitement** : Resampling, Reweighting, FairDrop ;
- **in-training** : FairGNN (adversarial avec Gradient Reversal Layer) ;
- **post-traitement** : INLP (projection sur null-space).

L'attribut sensible est `age_old` (binaire : ≥ 25 ans vs < 25). La
métrique principale est l'**`excess_gap = pred_gap − true_gap`**, qui
mesure si le modèle amplifie la disparité réelle au-delà des données.

Tout tourne en moins de 5 minutes sur une GPU."""
        )
    )

    # ── 1. Setup ───────────────────────────────────────────────────────────
    cells.append(md("## 1. Setup\n\nImports, seeds, device."))
    cells.append(
        code(
            """import sys, time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Repo root
ROOT = Path.cwd() if Path.cwd().name == "fairness" else Path.cwd().parent
sys.path.insert(0, str(ROOT))

# Imports projet
from src.data.loader import load_pokec_z
from src.data.preprocessing import preprocess
from src.data.splits import make_splits
from src.fairness.fairdrop import fairdrop
from src.fairness.metrics import demographic_parity_diff, equal_opportunity_diff, sensitive_leakage
from src.fairness.resampling import oversample_train_indices
from src.fairness.reweighting import kamiran_calders_weights
from src.models.fairgnn import FairGNN, fairgnn_loss
from src.models.graphsage import GraphSAGE
from src.models.trainer import train as train_with_early_stopping
from src.postprocess.inlp import inlp, apply_projection
from src.robustness.perturbations import add_feature_noise, drop_edges

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device = {device}")"""
        )
    )

    # ── 2. Load + preprocess ───────────────────────────────────────────────
    cells.append(
        md(
            """## 2. Chargement des données

On charge le subset Pokec officiel de Dai & Wang (`region_job_2.csv`).
`preprocess()` extrait `gender`, `region` et `age_group` comme attributs
sensibles, retire ces colonnes du `x`, et z-score normalise les features
restantes."""
        )
    )
    cells.append(
        code(
            """data = load_pokec_z(ROOT / "data" / "raw" / "pokec-z")
data = preprocess(data, sensitive_cols=["gender", "region"])
data.age_group = data.age_group.clamp(min=0)  # NA → 0 (young)
data = data.to(device)
data.age_group = data.age_group.to(device)

print(f"N = {data.x.shape[0]:,} nœuds")
print(f"E = {data.edge_index.shape[1]:,} arêtes")
print(f"d = {data.x.shape[1]} features (avant retrait des leak cols)")"""
        )
    )

    # ── 3. Target + sensitive ──────────────────────────────────────────────
    cells.append(
        md(
            """## 3. Cible et attribut sensible

**Cible** : `fajcim pravidelne` (fume régulièrement, prévalence 9.3%).

**Attribut sensible** : `age_old` (binaire, ≥ 25 ans vs < 25).

On **retire de `x`** les 4 colonnes liées au tabac (`fajcim*`,
`nefajcim`) pour éviter qu'elles servent de proxy direct à la cible."""
        )
    )
    cells.append(
        code(
            """# Cible
TARGET = "fajcim pravidelne"
y_np = (data.raw_df[TARGET].cast(pl.Int64).to_numpy() > 0).astype(np.int64)
data.y = torch.from_numpy(y_np).long().to(device)

# Drop leak features
LEAK = [c for c in data.feature_cols if "fajc" in c.lower() or c == "nefajcim"]
keep = [c for c in data.feature_cols if c not in set(LEAK)]
keep_idx = torch.tensor([data.feature_cols.index(c) for c in keep], dtype=torch.long, device=device)
data.x = data.x.index_select(1, keep_idx)
data.feature_cols = keep
assert data.x.shape[1] == len(data.feature_cols), "x et feature_cols désynchronisés"
print(f"Leak cols retirées : {LEAK}")
print(f"Features restantes : {data.x.shape[1]}")

# Sensitive : age_old binaire (≥ 25 ans)
# Note : clamp(min=0) en cellule précédente a fait que les AGE ≤ 0 (≈24% NA)
# sont mappés sur age_group=0, donc age_old=0 mélange "vraiment jeunes" et
# "âge inconnu". C'est une limite du dataset, pas un bug — mentionnée dans
# le rapport.
data.age_old = (data.age_group >= 1).long()

print(f"Prévalence cible      : {float(data.y.float().mean())*100:.2f}%")
print(f"Prévalence age_old=1  : {float(data.age_old.float().mean())*100:.2f}%")"""
        )
    )

    # ── 4. Splits ──────────────────────────────────────────────────────────
    cells.append(
        md(
            """## 4. Splits 60/20/20

Stratifiés sur `(label × age_old)` pour préserver la prévalence dans
chaque sous-ensemble."""
        )
    )
    cells.append(
        code(
            """n = data.y.shape[0]
train_idx, val_idx, test_idx = make_splits(
    n=n, y=data.y.cpu(), sensitive=data.age_old.cpu(), seed=SEED
)
train_idx = train_idx.to(device); val_idx = val_idx.to(device); test_idx = test_idx.to(device)
train_mask = torch.zeros(n, dtype=torch.bool, device=device); train_mask[train_idx] = True
val_mask   = torch.zeros(n, dtype=torch.bool, device=device); val_mask[val_idx] = True
test_mask  = torch.zeros(n, dtype=torch.bool, device=device); test_mask[test_idx] = True

print(f"train = {int(train_mask.sum())}  val = {int(val_mask.sum())}  test = {int(test_mask.sum())}")"""
        )
    )

    # ── 5. Helper metrics ──────────────────────────────────────────────────
    cells.append(
        md(
            """## 5. Helper de métriques

Pour comparer chaque méthode, on utilise systématiquement :

- **F1 macro** : utilité du modèle sur la cible (classe minoritaire prise
  en compte).
- **`true_gap`** : disparité réelle dans le test set
  (`P(y=1 | s=1) − P(y=1 | s=0)`).
- **`pred_gap`** : disparité prédite par le modèle.
- **`excess_gap = pred_gap − true_gap`** : amplification algorithmique.
  Idéal = 0 (le modèle reflète la réalité). Positif = il amplifie ;
  négatif = il sur-corrige.
- **ΔDP, ΔEO** : métriques fairness classiques (Hardt 2016).
- **leakage AUC** : à quel point un classifieur linéaire peut prédire
  l'attribut sensible depuis les embeddings (Ravfogel 2020)."""
        )
    )
    cells.append(
        code(
            """def compute_metrics(name, pred, y_test, s_test, embeddings=None, s_full=None):
    \"\"\"Calcule toutes les métriques pour une prédiction donnée.\"\"\"
    pred = pred.cpu().long()
    y_test = y_test.cpu().long()
    s_test = s_test.cpu().long()

    m1, m0 = (s_test == 1), (s_test == 0)
    true_g0 = float(y_test[m0].float().mean())
    true_g1 = float(y_test[m1].float().mean())
    pred_g0 = float(pred[m0].float().mean())
    pred_g1 = float(pred[m1].float().mean())
    true_gap = true_g1 - true_g0
    pred_gap = pred_g1 - pred_g0

    f1 = f1_score(y_test.numpy(), pred.numpy(), average="macro", zero_division=0)
    delta_dp = demographic_parity_diff(pred, s_test)
    delta_eo = equal_opportunity_diff(pred, y_test, s_test)

    leak = float("nan")
    if embeddings is not None and s_full is not None:
        leak = sensitive_leakage(embeddings.cpu(), s_full.cpu(), train_mask, test_mask, seed=SEED)

    return {
        "method": name,
        "f1": float(f1),
        "true_gap_pp": true_gap * 100,
        "pred_gap_pp": pred_gap * 100,
        "excess_gap_pp": (pred_gap - true_gap) * 100,
        "delta_dp": delta_dp,
        "delta_eo": delta_eo,
        "leakage_auc": leak,
    }

results = []  # on accumule les lignes ici"""
        )
    )

    # ── 6. Baseline ────────────────────────────────────────────────────────
    cells.append(
        md(
            """## 6. Baseline GraphSAGE

GraphSAGE 2 couches, hidden = 256, dropout = 0.5. Adam lr = 5e-3.
Early stopping sur F1 validation (patience 30, max 200 epochs).

À la fin on récupère les embeddings (avant la dernière couche) pour les
méthodes de post-traitement et la mesure de leakage."""
        )
    )
    cells.append(
        code(
            """def make_graphsage():
    return GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=256,
        out_channels=2,
        num_layers=2,
        dropout=0.5,
    ).to(device)

# Reset seed avant init pour reproductibilité (le baseline doit être identique
# si on relance le notebook).
torch.manual_seed(SEED)
t0 = time.time()
model = make_graphsage()
train_with_early_stopping(model, data, train_mask, val_mask, lr=5e-3, epochs=200, patience=30)
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    pred_test = logits[test_mask].argmax(dim=1)
    embeddings_baseline = model.get_embeddings(data.x, data.edge_index)
print(f"trained in {time.time() - t0:.1f}s")

# Sauvegarde du state_dict baseline pour la section robustesse,
# qui réutilisera ce modèle plutôt que de le retrain.
baseline_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

# Calcul métriques
y_test_t = data.y[test_mask]
s_test_t = data.age_old[test_mask]
m = compute_metrics("GraphSAGE baseline", pred_test, y_test_t, s_test_t,
                    embeddings_baseline, data.age_old)
results.append(m)
print(m)"""
        )
    )

    # ── 7. Resampling ──────────────────────────────────────────────────────
    cells.append(
        md(
            """## 7. Pré-traitement #1 : Resampling

On oversample les cellules `(y × age_old)` pour égaliser leur taille
dans le train set. Méthode naïve : on duplique des nœuds (en termes
d'indices d'entraînement) jusqu'à ce que chaque cellule ait la taille de
la plus grande.

Limite : si la corrélation `y ↔ s` est portée par d'autres features, le
modèle ré-apprend les mêmes associations."""
        )
    )
    cells.append(
        code(
            """oversampled_idx = oversample_train_indices(
    train_mask.cpu(), data.y.cpu(), data.age_old.cpu(), seed=SEED
)
oversampled_mask = torch.zeros(n, dtype=torch.bool, device=device)
oversampled_mask[oversampled_idx.to(device)] = True

t0 = time.time()
torch.manual_seed(SEED)
model = make_graphsage()
train_with_early_stopping(model, data, oversampled_mask, val_mask, lr=5e-3, epochs=200, patience=30)
model.eval()
with torch.no_grad():
    pred_test = model(data.x, data.edge_index)[test_mask].argmax(dim=1)
print(f"trained in {time.time() - t0:.1f}s")

m = compute_metrics("+ Resampling", pred_test, y_test_t, s_test_t)
results.append(m); print(m)"""
        )
    )

    # ── 8. Reweighting ─────────────────────────────────────────────────────
    cells.append(
        md(
            """## 8. Pré-traitement #2 : Reweighting (Kamiran-Calders 2012)

On pondère chaque exemple d'entraînement par
$w_i \\propto P(s) \\, P(y) / P(s, y)$, ce qui décorrèle `y` et `s` au
niveau de la fonction de coût. C'est moins brutal que Resampling : on ne
duplique pas, on ajuste juste le poids dans la cross-entropy."""
        )
    )
    cells.append(
        code(
            """sensitive_np = data.age_old.detach().cpu().numpy()
y_np_full = data.y.detach().cpu().numpy()
train_idx_np = train_mask.detach().cpu().numpy().nonzero()[0]

weights_full = np.ones(data.num_nodes, dtype=np.float32)
weights_full[train_idx_np] = kamiran_calders_weights(
    y_np_full[train_idx_np], sensitive_np[train_idx_np]
)
sample_weights = torch.from_numpy(weights_full).to(device)

t0 = time.time()
torch.manual_seed(SEED)
model = make_graphsage()
train_with_early_stopping(model, data, train_mask, val_mask, lr=5e-3, epochs=200, patience=30,
                          sample_weights=sample_weights)
model.eval()
with torch.no_grad():
    pred_test = model(data.x, data.edge_index)[test_mask].argmax(dim=1)
print(f"trained in {time.time() - t0:.1f}s")

m = compute_metrics("+ Reweighting", pred_test, y_test_t, s_test_t)
results.append(m); print(m)"""
        )
    )

    # ── 9. FairDrop ────────────────────────────────────────────────────────
    cells.append(
        md(
            """## 9. Pré-traitement #3 : FairDrop (Spinelli et al. 2021)

On retire un sous-ensemble des arêtes intra-groupe (ami du même âge)
pour casser l'homophilie générationnelle dans le graphe. L'intuition :
si l'amplification vient du message-passing entre adultes voisins, la
réduire devrait diminuer l'excess."""
        )
    )
    cells.append(
        code(
            """new_edge_index = fairdrop(
    data.edge_index.cpu(), data.age_old.cpu(),
    drop_rate=0.3, intra_group_bias=2.0, seed=SEED,
).to(device)

original_ei = data.edge_index
data.edge_index = new_edge_index

t0 = time.time()
torch.manual_seed(SEED)
model = make_graphsage()
train_with_early_stopping(model, data, train_mask, val_mask, lr=5e-3, epochs=200, patience=30)
model.eval()
with torch.no_grad():
    pred_test = model(data.x, data.edge_index)[test_mask].argmax(dim=1)
print(f"trained in {time.time() - t0:.1f}s")

# Restaurer edge_index original
data.edge_index = original_ei

m = compute_metrics("+ FairDrop", pred_test, y_test_t, s_test_t)
results.append(m); print(m)"""
        )
    )

    # ── 10. FairGNN ────────────────────────────────────────────────────────
    cells.append(
        md(
            """## 10. In-training : FairGNN (Dai & Wang 2021)

GraphSAGE encoder + classifieur cible + classifieur adversaire qui essaie
de prédire `age_old` depuis les embeddings. Un **Gradient Reversal Layer**
inverse le gradient de l'adversaire à la passe arrière, donc l'encoder
apprend à *tromper* l'adversaire (à supprimer le signal `age_old` des
embeddings).

On utilise `λ = 1.0` (compromis fairness/utilité)."""
        )
    )
    cells.append(
        code(
            """LAMBDA_ADV = 1.0
EPOCHS_FGNN = 200

torch.manual_seed(SEED)
model = FairGNN(
    in_channels=data.x.shape[1], hidden_channels=256,
    out_channels=2, adv_hidden=64, num_layers=2, dropout=0.5,
    lambda_adv=LAMBDA_ADV,
).to(device)
opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

best_val_f1, best_state, patience = 0.0, None, 0
t0 = time.time()
for epoch in range(EPOCHS_FGNN):
    model.train(); opt.zero_grad()
    pred_logits, adv_logits = model(data.x, data.edge_index)
    loss = fairgnn_loss(pred_logits, adv_logits, data.y, data.age_old, train_mask)
    loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pl_val, _ = model(data.x, data.edge_index)
    val_pred = pl_val[val_mask].argmax(dim=1)
    val_f1 = f1_score(data.y[val_mask].cpu().numpy(), val_pred.cpu().numpy(),
                      average="macro", zero_division=0)
    if val_f1 > best_val_f1:
        best_val_f1, best_state, patience = val_f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        patience += 1
        if patience >= 30:
            break

if best_state is not None:
    model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    pred_logits, _ = model(data.x, data.edge_index)
    pred_test = pred_logits[test_mask].argmax(dim=1)
print(f"trained in {time.time() - t0:.1f}s  best_val_f1={best_val_f1:.4f}")

m = compute_metrics(f"FairGNN (λ={LAMBDA_ADV})", pred_test, y_test_t, s_test_t)
results.append(m); print(m)"""
        )
    )

    # ── 11. INLP ───────────────────────────────────────────────────────────
    cells.append(
        md(
            """## 11. Post-traitement : INLP (Ravfogel 2020)

À partir des embeddings du baseline GraphSAGE, on identifie itérativement
les directions linéaires qui encodent `age_old` et on les **projette
hors** des embeddings. Après convergence, plus aucun classifieur linéaire
ne peut récupérer l'attribut sensible.

On ré-entraîne ensuite une **régression logistique** sur les embeddings
projetés pour prédire `y`, et on évalue.

> ⚠️ **Note méthodologique.** Contrairement aux autres méthodes qui
> entraînent un GraphSAGE complet (encoder + tête de classification
> jointes), INLP travaille sur les embeddings d'un baseline figé puis
> apprend une **tête linéaire fraîche**. Le coût F1 d'INLP mélange donc
> deux effets : (i) l'effet de la projection sur le null-space de
> l'attribut sensible, et (ii) le passage d'une tête non-linéaire
> profonde à une simple LR. C'est conforme à la pratique INLP standard
> (Ravfogel 2020) et nous gardons cette comparaison telle quelle, mais
> il faut garder cette nuance en tête."""
        )
    )
    cells.append(
        code(
            """z = embeddings_baseline.detach().cpu().numpy().astype(np.float32)
s_np = data.age_old.detach().cpu().numpy()
y_np_full = data.y.detach().cpu().numpy()
train_idx_np = train_mask.detach().cpu().numpy().nonzero()[0]
test_idx_np = test_mask.detach().cpu().numpy().nonzero()[0]

# Fit projection sur train, applique partout
_, P = inlp(z[train_idx_np], s_np[train_idx_np], n_iter=10, seed=SEED)
z_clean = apply_projection(z, P)

# Refit head sur embeddings projetés
clf = LogisticRegression(max_iter=1000, random_state=SEED)
clf.fit(z_clean[train_idx_np], y_np_full[train_idx_np])
pred_test_np = clf.predict(z_clean[test_idx_np])
pred_test = torch.from_numpy(pred_test_np).long().to(device)

# Leakage post-INLP
emb_clean_t = torch.from_numpy(z_clean).float()
m = compute_metrics("+ INLP", pred_test, y_test_t, s_test_t,
                    emb_clean_t, data.age_old)
results.append(m); print(m)"""
        )
    )

    # ── 12. Récap ──────────────────────────────────────────────────────────
    cells.append(
        md(
            """## 12. Tableau récapitulatif

Lecture humaine :

- **baseline** amplifie (excess > 0) ;
- **Resampling** : aucun effet ;
- **FairDrop** : effet marginal ;
- **Reweighting** : excess proche de 0, F1 stable → gagnant ;
- **FairGNN** : excess proche de 0 mais coût F1 ;
- **INLP** : leakage chute mais le modèle sur-corrige (excess négatif)."""
        )
    )
    cells.append(
        code(
            """recap = pl.DataFrame(results).select(
    "method",
    pl.col("f1").round(3),
    pl.col("true_gap_pp").round(2),
    pl.col("pred_gap_pp").round(2),
    pl.col("excess_gap_pp").round(2),
    pl.col("delta_dp").round(3),
    pl.col("delta_eo").round(3),
    pl.col("leakage_auc").round(3),
)
print(recap)"""
        )
    )

    # ── 13. Interprétabilité ───────────────────────────────────────────────
    cells.append(
        md(
            """## 13. Interprétabilité — pourquoi le modèle amplifie

On entraîne une régression logistique balancée pour prédire le tabagisme
à partir des 264 features. Pour chaque top feature (par |coef|), on
calcule la corrélation Pearson avec `age_old` : une feature à fort coef
**et** fortement corrélée à l'âge est un médiateur pro-amplification.

Pour un modèle linéaire, les coefficients sont équivalents aux SHAP
values (à un facteur d'échelle), donc inutile d'invoquer SHAP."""
        )
    )
    cells.append(
        code(
            """X_full = data.x.cpu().numpy().astype(np.float32)
y_full = data.y.cpu().numpy()
age_old_np = data.age_old.cpu().numpy().astype(np.float32)

clf_int = LogisticRegression(max_iter=2000, random_state=SEED, C=0.1, class_weight="balanced")
clf_int.fit(X_full, y_full)
coefs = clf_int.coef_.ravel()
top_idx = np.argsort(np.abs(coefs))[::-1][:10]

interp = []
for i in top_idx:
    name = data.feature_cols[i]
    feat = X_full[:, i]
    corr = float(np.corrcoef(feat, age_old_np)[0, 1])
    interp.append({"feature": name, "coef": float(coefs[i]), "corr_age_old": corr})

interp_df = pl.DataFrame(interp).select(
    "feature",
    pl.col("coef").round(3),
    pl.col("corr_age_old").round(3),
)
print(interp_df)"""
        )
    )

    # ── 14. Robustesse ─────────────────────────────────────────────────────
    cells.append(
        md(
            """## 14. Robustesse — perturbations contrôlées

On reprend le baseline GraphSAGE (déjà entraîné), et on perturbe en
inférence :

- **bruit gaussien** sur les features (σ ∈ {0, 0.1, 0.3, 0.5}) ;
- **edge dropping** aléatoire (rate ∈ {0, 0.1, 0.3, 0.5}).

Si le modèle est robuste, F1 doit rester stable pour des perturbations
modérées."""
        )
    )
    cells.append(
        code(
            """# Réutiliser le baseline déjà entraîné (sauvegardé en cellule baseline)
# plutôt que de retraîner — garantit qu'on teste la robustesse du *même*
# modèle que celui dont on a mesuré l'amplification.
baseline_model = make_graphsage()
baseline_model.load_state_dict(baseline_state_dict)
baseline_model.eval()

robust_rows = []
with torch.no_grad():
    for sigma in [0.0, 0.1, 0.3, 0.5]:
        x_noisy = add_feature_noise(data.x.cpu(), sigma=sigma, seed=SEED).to(device)
        pred = baseline_model(x_noisy, data.edge_index)[test_mask].argmax(dim=1)
        f1 = f1_score(y_test_t.cpu().numpy(), pred.cpu().numpy(), average="macro", zero_division=0)
        robust_rows.append({"perturbation": f"feature_noise σ={sigma}", "f1": float(f1)})

    for rate in [0.0, 0.1, 0.3, 0.5]:
        ei_drop = drop_edges(data.edge_index.cpu(), rate=rate, seed=SEED).to(device)
        pred = baseline_model(data.x, ei_drop)[test_mask].argmax(dim=1)
        f1 = f1_score(y_test_t.cpu().numpy(), pred.cpu().numpy(), average="macro", zero_division=0)
        robust_rows.append({"perturbation": f"edge_drop rate={rate}", "f1": float(f1)})

robust_df = pl.DataFrame(robust_rows).select("perturbation", pl.col("f1").round(3))
print(robust_df)"""
        )
    )

    # ── 15. Conclusion ─────────────────────────────────────────────────────
    cells.append(
        md(
            r"""## 15. Conclusion

Sur Pokec, cible *fume régulièrement* × axe *âge ≥ 25*, le baseline
GraphSAGE amplifie effectivement la disparité réelle de **+3.6 pp**.
Sur ce cas :

| Méthode | excess gap | F1 | verdict |
|---|---:|---:|---|
| Baseline | +3.6 pp | 0.63 | amplifie |
| Reweighting | +0.6 pp | 0.63 | **gagnant** (fidèle, gratuit) |
| FairGNN (λ=1) | +0.1 pp | 0.56 | quasi-parfait mais −7 pp F1 |
| INLP | −2.1 pp | 0.59 | sur-corrige |

La méthode la plus simple gagne. C'est le finding principal du rapport.
Le test de robustesse ci-dessus est mené uniquement sur le baseline ;
en théorie Reweighting préserve cette robustesse parce qu'il ne modifie
ni l'encoder ni le graphe — il agit seulement sur les poids des
exemples dans la loss — mais on ne le démontre pas explicitement ici
(extension naturelle pour un travail futur : refaire le sweep
σ × edge\_drop sur le modèle Reweighted)."""
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {OUT}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
