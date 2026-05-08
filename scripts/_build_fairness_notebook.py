"""Generate ``notebooks/fairness_findings.ipynb`` — narrative du 2-pager.

3 findings au cœur du notebook :

1. Toolbox post-process (DPT, EOT, INLP, INLP+DPT) bat FairGNN canonical
   sur la fairness, à F1 équivalent.
2. Le bon axe à débiaiser dépend du graphe : ``r(s)`` Newman révèle que
   region (r=0.9) est l'axe que le graphe amplifie, pas gender (r≈0).
3. ULTIMATE composite (INLP_composite + DPT_composite à 12 cellules)
   atteint le chance level sur 5 axes mais coûte 35 pp F1 sur GraphSAGE.

Run ::

    .venv/bin/python scripts/_build_fairness_notebook.py

→ notebooks/fairness_findings.ipynb (executable end-to-end sur GPU).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "fairness_findings.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = []

    # ─── Title + intro ─────────────────────────────────────────────────────
    nb.cells.append(
        md(
            """\
# Fairness sur Pokec-z — quel outil pour quelle métrique, sur quel axe ?

Ce notebook reprend les **3 findings** du 2-pager et les déroule pas-à-pas
sur Pokec-z (subset officiel FairGNN).

**Findings** :
1. **Toolbox post-process** (DPT, EOT, INLP, INLP+DPT) bat FairGNN
   canonical sur la fairness, à F1 équivalent.
2. **Le bon axe à débiaiser dépend du graphe** : on mesure le coefficient
   d'assortativité de Newman `r(s)` ; sur Pokec-z, `r(region) = 0.901`
   (le graphe est massivement homophile en region) alors que
   `r(gender) ≈ 0`. Le graphe amplifie region, pas gender — c'est region
   qu'il faut débiaiser, pas gender.
3. **ULTIMATE composite** (INLP_composite + DPT_composite sur l'attribut
   joint à 12 cellules) atteint le **chance level** (leakage ≈ 0.50) sur
   les 5 axes simultanément, mais coûte 35 pp de F1 sur GraphSAGE.

**Conclusion meta** : faire de la fairness, c'est faire des choix
normatifs (quelle métrique, quels axes, à quel coût). Et la question
ouverte reste : *quis custodiet ipsos custodes ?*"""
        )
    )

    # ─── §1 Setup ──────────────────────────────────────────────────────────
    nb.cells.append(
        md(
            """\
## §1 — Setup

Chargement Pokec-z, splits stratifiés `y × gender` 60/20/20, attributs
sensibles (gender, region, age_group + intersections)."""
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
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

from src.data.loader import load_pokec_z
from src.data.preprocessing import preprocess
from src.data.splits import make_splits
from src.fairness.metrics import assortative_mixing_coefficient
from src.models.fairgnn import FairGNN
from src.postprocess.inlp import inlp

SEED = 42
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"device={DEVICE}, seed={SEED}")"""
        )
    )

    nb.cells.append(
        code(
            """\
data = load_pokec_z(ROOT / "data" / "raw" / "pokec-z")
data = preprocess(data, sensitive_cols=["gender", "region", "age_group"])
data = data.to(DEVICE)

n = data.y.shape[0]
print(f"nœuds = {n}, arêtes = {data.edge_index.shape[1] // 2}, features = {data.x.shape[1]}")

# Splits stratifiés y × gender 60/20/20.
train_idx, val_idx, test_idx = make_splits(n=n, y=data.y.cpu(), sensitive=data.gender.cpu(), seed=SEED)
train_mask = torch.zeros(n, dtype=torch.bool, device=DEVICE)
val_mask = torch.zeros_like(train_mask)
test_mask = torch.zeros_like(train_mask)
train_mask[train_idx.to(DEVICE)] = True
val_mask[val_idx.to(DEVICE)] = True
test_mask[test_idx.to(DEVICE)] = True

# Attributs sensibles + composite à 12 cellules.
gender = data.gender.long()
region = data.region.long()
age_group = data.age_group.clamp(min=0).long()
composite = gender * 6 + age_group * 2 + region  # 12 cellules

n_age = int(age_group.max().item()) + 1
n_reg = int(region.max().item()) + 1
gender_x_age = gender * n_age + age_group
gender_x_region = gender * n_reg + region

print(f"train={train_mask.sum().item()}  val={val_mask.sum().item()}  test={test_mask.sum().item()}")
print(f"composite : {int(composite.max().item()) + 1} cellules")"""
        )
    )

    # ─── §2 r(s) — finding 2 setup ─────────────────────────────────────────
    nb.cells.append(
        md(
            """\
## §2 — Coefficient d'assortativité `r(s)` de Newman (2003)

Pour chaque axe sensible, on mesure à quel point les arêtes du graphe
connectent des nœuds de la même classe au-delà de ce qu'on attendrait du
hasard.

- **r ≈ 0** : le graphe ignore l'attribut.
- **r → 1** : graphe parfaitement homophile.
- **r < 0** : hétérophilie.

**Pourquoi c'est crucial pour la fairness** : si `r(s)` est élevé, le
message passing du GNN va amplifier le signal sensible dans les
embeddings — c'est l'axe qu'il faut débiaiser. Si `r(s)` est bas, le
graphe ne fait rien, et le débat fairness se joue uniquement au niveau
des sorties (post-process suffit)."""
        )
    )

    nb.cells.append(
        code(
            """\
print("Coefficients d'assortativité Newman sur Pokec-z :")
for axis_name, s in [("gender", gender), ("region", region), ("age_group", age_group)]:
    r = float(assortative_mixing_coefficient(data.edge_index, s))
    print(f"  r({axis_name:10s}) = {r:+.4f}")"""
        )
    )

    nb.cells.append(
        md(
            """\
**Lecture** :
- `r(gender) ≈ 0` → le graphe Pokec-z est quasi-aléatoire vs gender.
  Hommes et femmes sont autant amis entre eux qu'avec des gens du sexe
  opposé. Le message passing **n'amplifie rien sur gender**.
- `r(region) = 0.9` → ~90 % des arêtes intra-région. Pokec-z est en
  pratique un graphe **de régions** faiblement inter-connecté. Le GNN va
  saturer ses embeddings en region.
- `r(age_group) = 0.35` → légère homophilie générationnelle (les jeunes
  ont des amis jeunes).

→ **Le bon axe à débiaiser sur Pokec-z, c'est region**, pas gender."""
        )
    )

    # ─── §3 Baseline GraphSAGE ────────────────────────────────────────────
    nb.cells.append(
        md(
            """\
## §3 — Baseline GraphSAGE

On entraîne un GraphSAGE simple (2 couches, hidden=256, dropout=0.5) sur
la cible `completed_level_of_education_indicator` et on extrait
embeddings + prédictions de référence."""
        )
    )

    nb.cells.append(
        code(
            """\
from torch_geometric.nn import SAGEConv
import torch.nn as nn

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def encode(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x, edge_index):
        return self.classifier(self.encode(x, edge_index))


def setup_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


setup_seeds(SEED)
model = GraphSAGE(data.x.shape[1], 256, 2).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

best_val_f1, best_state, patience = 0.0, None, 0
for epoch in range(200):
    model.train()
    opt.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(data.x, data.edge_index)[val_mask].argmax(dim=1)
    val_f1 = f1_score(data.y[val_mask].cpu().numpy(), val_pred.cpu().numpy(), average="macro")
    if val_f1 > best_val_f1:
        best_val_f1, best_state, patience = val_f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        patience += 1
        if patience >= 30:
            break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    h_baseline = model.encode(data.x, data.edge_index)
    proba_baseline = F.softmax(model(data.x, data.edge_index), dim=1)
    pred_baseline = proba_baseline.argmax(dim=1)

acc_b = accuracy_score(data.y[test_mask].cpu().numpy(), pred_baseline[test_mask].cpu().numpy())
f1_b = f1_score(data.y[test_mask].cpu().numpy(), pred_baseline[test_mask].cpu().numpy(), average="macro")
print(f"GraphSAGE baseline :  acc={acc_b:.4f}  F1 macro={f1_b:.4f}")"""
        )
    )

    # ─── §4 Helpers + métriques baseline ──────────────────────────────────
    nb.cells.append(
        md(
            """\
**Helpers** : ΔDP, ΔEO, leakage AUC vectorisés."""
        )
    )

    nb.cells.append(
        code(
            """\
def delta_dp(pred, sensitive):
    pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
    s = sensitive.cpu().numpy() if torch.is_tensor(sensitive) else sensitive
    rates = [float(pred[s == g].mean()) for g in np.unique(s)]
    return float(max(rates) - min(rates)) if len(rates) >= 2 else 0.0

def delta_eo(pred, y_true, sensitive):
    pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
    y = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    s = sensitive.cpu().numpy() if torch.is_tensor(sensitive) else sensitive
    tprs = []
    for g in np.unique(s):
        mask = (s == g) & (y == 1)
        if mask.sum() == 0: continue
        tprs.append(float(pred[mask].mean()))
    return float(max(tprs) - min(tprs)) if len(tprs) >= 2 else 0.0

def leakage_auc(z_train, z_test, s_train, s_test):
    s_train = s_train.cpu().numpy() if torch.is_tensor(s_train) else s_train
    s_test = s_test.cpu().numpy() if torch.is_tensor(s_test) else s_test
    if np.unique(s_train).size < 2:
        return float("nan")
    probe = LogisticRegression(max_iter=1000, random_state=SEED)
    probe.fit(z_train, s_train)
    if int(s_test.max()) + 1 == 2:
        return float(roc_auc_score(s_test, probe.predict_proba(z_test)[:, 1]))
    return float(roc_auc_score(
        s_test, probe.predict_proba(z_test), multi_class="ovr", average="macro"
    ))

# Métriques baseline sur 5 axes.
h_train = h_baseline[train_mask].detach().cpu().numpy()
h_test = h_baseline[test_mask].detach().cpu().numpy()
y_test = data.y[test_mask].cpu().numpy()
pred_test = pred_baseline[test_mask].cpu().numpy()

axes = {
    "gender": gender, "region": region, "age_group": age_group,
    "gender_x_age": gender_x_age, "gender_x_region": gender_x_region,
}

print("=== GraphSAGE baseline — fairness sur 5 axes ===")
print(f"{'axis':18s} {'ΔDP':>8s} {'ΔEO':>8s} {'leakage':>8s}")
baseline_metrics = {}
for axis_name, s in axes.items():
    s_te = s[test_mask].cpu().numpy()
    s_tr = s[train_mask].cpu().numpy()
    ddp = delta_dp(pred_test, s_te)
    deo = delta_eo(pred_test, y_test, s_te)
    leak = leakage_auc(h_train, h_test, s_tr, s_te)
    baseline_metrics[axis_name] = {"ddp": ddp, "deo": deo, "leak": leak}
    print(f"{axis_name:18s} {ddp:>8.4f} {deo:>8.4f} {leak:>8.4f}")"""
        )
    )

    # ─── §4 Finding 1 — Toolbox @gender ────────────────────────────────────
    nb.cells.append(
        md(
            """\
## §4 — Finding 1 — Toolbox post-process @ axe gender

Trois familles d'outils, chacune sa métrique cible :
- **DPT** (Demographic Parity Threshold) : seuil par groupe → réduit ΔDP.
- **EOT** (Equal Opportunity Threshold, Hardt 2016) : seuil par groupe sur
  le sous-ensemble `y=1` → réduit ΔEO.
- **INLP** (Ravfogel 2020) : projette les embeddings orthogonalement aux
  directions du sensible → réduit le leakage.

Et leur composition **INLP+DPT** : règle ΔDP **et** leakage simultanément
(les deux opérations sont orthogonales — l'une touche les embeddings,
l'autre les seuils)."""
        )
    )

    nb.cells.append(
        code(
            """\
def calibrate_dpt(proba_pos_val, sensitive_val, grid_size=51):
    grid = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    global_rate = float((proba_pos_val > 0.5).mean())
    th = {}
    for g in np.unique(sensitive_val):
        mask = sensitive_val == g
        if mask.sum() == 0: th[int(g)] = 0.5; continue
        rates = (proba_pos_val[mask][None, :] > grid[:, None]).mean(axis=1)
        th[int(g)] = float(grid[int(np.argmin(np.abs(rates - global_rate)))])
    return th

def calibrate_eot(proba_pos_val, y_val, sensitive_val, grid_size=51):
    grid = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    pos = y_val == 1
    global_tpr = float((proba_pos_val[pos] > 0.5).mean())
    th = {}
    for g in np.unique(sensitive_val):
        mask = (sensitive_val == g) & pos
        if mask.sum() == 0: th[int(g)] = 0.5; continue
        tprs = (proba_pos_val[mask][None, :] > grid[:, None]).mean(axis=1)
        th[int(g)] = float(grid[int(np.argmin(np.abs(tprs - global_tpr)))])
    return th

def apply_per_group_thresholds(proba_pos, sensitive, thresholds):
    unique = np.unique(sensitive)
    cell_to_t = np.array([thresholds.get(int(g), 0.5) for g in unique], dtype=np.float32)
    _, inv = np.unique(sensitive, return_inverse=True)
    return (proba_pos > cell_to_t[inv]).astype(np.int64)

# DPT et EOT calibrés sur val
proba_val_pos = proba_baseline[val_mask, 1].detach().cpu().numpy()
proba_test_pos = proba_baseline[test_mask, 1].detach().cpu().numpy()
gender_val = gender[val_mask].cpu().numpy()
gender_test = gender[test_mask].cpu().numpy()
y_val = data.y[val_mask].cpu().numpy()

th_dpt = calibrate_dpt(proba_val_pos, gender_val)
pred_dpt = apply_per_group_thresholds(proba_test_pos, gender_test, th_dpt)
print(f"GraphSAGE+DPT@gender :  ΔDP={delta_dp(pred_dpt, gender_test):.4f}")

th_eot = calibrate_eot(proba_val_pos, y_val, gender_val)
pred_eot = apply_per_group_thresholds(proba_test_pos, gender_test, th_eot)
print(f"GraphSAGE+EOT@gender :  ΔEO={delta_eo(pred_eot, y_test, gender_test):.4f}")"""
        )
    )

    nb.cells.append(
        code(
            """\
# INLP @gender : projette les embeddings de tout le graphe sur l'orthogonal du sensible.
gender_train_np = gender[train_mask].cpu().numpy()
_, P_gender = inlp(h_train, gender_train_np, n_iter=15, seed=SEED)

h_clean_train = (h_train @ P_gender).astype(np.float32)
h_clean_test = (h_test @ P_gender).astype(np.float32)

# Re-fit un classifier linéaire sur les embeddings cleanés (pred y depuis h_clean).
clf_inlp = LogisticRegression(max_iter=2000, random_state=SEED)
clf_inlp.fit(h_clean_train, data.y[train_mask].cpu().numpy())
proba_inlp_val = clf_inlp.predict_proba((h_baseline[val_mask].detach().cpu().numpy() @ P_gender))[:, 1]
proba_inlp_test = clf_inlp.predict_proba(h_clean_test)[:, 1]
pred_inlp = (proba_inlp_test > 0.5).astype(np.int64)

# INLP+DPT : DPT calibré sur les probas INLP.
th_inlp_dpt = calibrate_dpt(proba_inlp_val, gender_val)
pred_inlp_dpt = apply_per_group_thresholds(proba_inlp_test, gender_test, th_inlp_dpt)

leak_inlp = leakage_auc(h_clean_train, h_clean_test, gender_train_np, gender_test)
print(f"GraphSAGE+INLP@gender :       ΔDP={delta_dp(pred_inlp, gender_test):.4f}  leak={leak_inlp:.4f}")
print(f"GraphSAGE+INLP+DPT@gender :   ΔDP={delta_dp(pred_inlp_dpt, gender_test):.4f}  leak={leak_inlp:.4f}")"""
        )
    )

    # ─── §5 FairGNN canonical ─────────────────────────────────────────────
    nb.cells.append(
        md(
            """\
## §5 — FairGNN canonical (two-optimizer alternating)

L'in-training adversarial — encoder + classifier + adversaire qui cherche
à prédire le sensible depuis les embeddings ; le encoder est entraîné à
contrer l'adversaire (min-max). Implémentation canonique Dai-Wang 2021 :
deux optimiseurs en alternance.

Pour ce notebook, on entraîne avec une seule λ (=1.0, valeur centrale du
grid sweep). Pour la version multi-λ multi-seed, voir
`scripts/run_fairgnn_two_opt.py`."""
        )
    )

    nb.cells.append(
        code(
            """\
def train_fairgnn_two_opt(data, train_mask, val_mask, sensitive, lam=1.0, epochs=200, patience=30):
    out_channels = int(data.y.max().item()) + 1
    setup_seeds(SEED)
    model = FairGNN(
        in_channels=data.x.shape[1], hidden_channels=256, out_channels=out_channels,
        adv_hidden=64, num_layers=2, dropout=0.5, lambda_adv=0.0,
    ).to(data.x.device)

    enc_cls = list(model.convs.parameters()) + list(model.classifier.parameters())
    adv = list(model.adversary.parameters())
    opt_main = torch.optim.Adam(enc_cls, lr=5e-3, weight_decay=5e-4)
    opt_adv = torch.optim.Adam(adv, lr=5e-3, weight_decay=5e-4)

    best_val_f1, best_state, pat = 0.0, None, 0
    for _ in range(epochs):
        model.train()
        # Step A: train adversary (encoder frozen).
        opt_adv.zero_grad(set_to_none=True)
        with torch.no_grad():
            z = model.encode(data.x, data.edge_index)
        loss_adv = F.cross_entropy(model.adversary(z)[train_mask], sensitive[train_mask])
        loss_adv.backward()
        opt_adv.step()

        # Step B: train encoder + classifier (adversary frozen via opt scope).
        opt_main.zero_grad(set_to_none=True); opt_adv.zero_grad(set_to_none=True)
        z = model.encode(data.x, data.edge_index)
        loss_cls = F.cross_entropy(model.classifier(z)[train_mask], data.y[train_mask])
        loss_adv2 = F.cross_entropy(model.adversary(z)[train_mask], sensitive[train_mask])
        loss = loss_cls - lam * loss_adv2
        loss.backward()
        opt_main.step()

        model.eval()
        with torch.no_grad():
            pred = model.classifier(model.encode(data.x, data.edge_index))[val_mask].argmax(dim=1)
        val_f1 = f1_score(data.y[val_mask].cpu().numpy(), pred.cpu().numpy(), average="macro")
        if val_f1 > best_val_f1:
            best_val_f1, best_state, pat = val_f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= patience: break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        h = model.encode(data.x, data.edge_index)
        pred = model.classifier(h).argmax(dim=1)
    return model, h, pred


_, h_fairgnn, pred_fairgnn = train_fairgnn_two_opt(data, train_mask, val_mask, gender, lam=1.0)
pred_fg_test = pred_fairgnn[test_mask].cpu().numpy()
acc_fg = accuracy_score(y_test, pred_fg_test)
f1_fg = f1_score(y_test, pred_fg_test, average="macro")
leak_fg = leakage_auc(
    h_fairgnn[train_mask].detach().cpu().numpy(),
    h_fairgnn[test_mask].detach().cpu().numpy(),
    gender_train_np, gender_test,
)
print(f"FairGNN(λ=1.0, two-opt, adv=gender) :")
print(f"  acc={acc_fg:.4f}  F1={f1_fg:.4f}  ΔDP={delta_dp(pred_fg_test, gender_test):.4f}  leak={leak_fg:.4f}")"""
        )
    )

    nb.cells.append(
        md(
            """\
**Récap Finding 1 — toolbox post-process bat FairGNN sur la fairness, à
F1 équivalent** :"""
        )
    )

    nb.cells.append(
        code(
            """\
import polars as pl

rows = [
    ("GraphSAGE baseline",      f1_b, baseline_metrics["gender"]["ddp"], baseline_metrics["gender"]["leak"]),
    ("FairGNN(λ=1, two-opt)",   f1_fg, delta_dp(pred_fg_test, gender_test), leak_fg),
    ("GraphSAGE+DPT@gender",    f1_b, delta_dp(pred_dpt, gender_test), baseline_metrics["gender"]["leak"]),
    ("GraphSAGE+EOT@gender",    f1_b, delta_dp(pred_eot, gender_test), baseline_metrics["gender"]["leak"]),
    ("GraphSAGE+INLP@gender",   f1_score(y_test, pred_inlp, average="macro"), delta_dp(pred_inlp, gender_test), leak_inlp),
    ("GraphSAGE+INLP+DPT@gender", f1_score(y_test, pred_inlp_dpt, average="macro"), delta_dp(pred_inlp_dpt, gender_test), leak_inlp),
]
print(pl.DataFrame(rows, schema=["méthode", "F1", "ΔDP gender", "leakage gender"], orient="row"))"""
        )
    )

    # ─── §6 Finding 2 — bon axe à débiaiser ───────────────────────────────
    nb.cells.append(
        md(
            """\
## §6 — Finding 2 — Le bon axe à débiaiser = celui que le graphe amplifie

Sur Pokec-z, on vient de voir `r(region) = 0.9` : c'est region que le
graphe encode et amplifie, pas gender. On répète la toolbox sur axe
region pour voir.

**Test** : DPT/INLP/INLP+DPT @region vs FairGNN canonical avec adversaire
sur region (au lieu de gender)."""
        )
    )

    nb.cells.append(
        code(
            """\
region_val = region[val_mask].cpu().numpy()
region_test = region[test_mask].cpu().numpy()
region_train_np = region[train_mask].cpu().numpy()

# Post-process @ region
th_dpt_r = calibrate_dpt(proba_val_pos, region_val)
pred_dpt_r = apply_per_group_thresholds(proba_test_pos, region_test, th_dpt_r)

_, P_region = inlp(h_train, region_train_np, n_iter=15, seed=SEED)
h_clean_train_r = (h_train @ P_region).astype(np.float32)
h_clean_test_r = (h_test @ P_region).astype(np.float32)
clf_inlp_r = LogisticRegression(max_iter=2000, random_state=SEED)
clf_inlp_r.fit(h_clean_train_r, data.y[train_mask].cpu().numpy())
proba_inlp_val_r = clf_inlp_r.predict_proba((h_baseline[val_mask].detach().cpu().numpy() @ P_region))[:, 1]
proba_inlp_test_r = clf_inlp_r.predict_proba(h_clean_test_r)[:, 1]
pred_inlp_r = (proba_inlp_test_r > 0.5).astype(np.int64)

th_inlp_dpt_r = calibrate_dpt(proba_inlp_val_r, region_val)
pred_inlp_dpt_r = apply_per_group_thresholds(proba_inlp_test_r, region_test, th_inlp_dpt_r)
leak_inlp_r = leakage_auc(h_clean_train_r, h_clean_test_r, region_train_np, region_test)

# FairGNN avec adversaire sur region
_, h_fairgnn_r, pred_fairgnn_r = train_fairgnn_two_opt(data, train_mask, val_mask, region, lam=1.0)
pred_fgr_test = pred_fairgnn_r[test_mask].cpu().numpy()
leak_fgr = leakage_auc(
    h_fairgnn_r[train_mask].detach().cpu().numpy(),
    h_fairgnn_r[test_mask].detach().cpu().numpy(),
    region_train_np, region_test,
)

rows_r = [
    ("GraphSAGE baseline",        f1_b, baseline_metrics["region"]["ddp"], baseline_metrics["region"]["leak"]),
    ("FairGNN(adv=gender)",       f1_fg, delta_dp(pred_fg_test, region_test), leakage_auc(h_fairgnn[train_mask].detach().cpu().numpy(), h_fairgnn[test_mask].detach().cpu().numpy(), region_train_np, region_test)),
    ("FairGNN(adv=region)",       f1_score(y_test, pred_fgr_test, average="macro"), delta_dp(pred_fgr_test, region_test), leak_fgr),
    ("GraphSAGE+DPT@region",      f1_b, delta_dp(pred_dpt_r, region_test), baseline_metrics["region"]["leak"]),
    ("GraphSAGE+INLP@region",     f1_score(y_test, pred_inlp_r, average="macro"), delta_dp(pred_inlp_r, region_test), leak_inlp_r),
    ("GraphSAGE+INLP+DPT@region", f1_score(y_test, pred_inlp_dpt_r, average="macro"), delta_dp(pred_inlp_dpt_r, region_test), leak_inlp_r),
]
print(pl.DataFrame(rows_r, schema=["méthode", "F1", "ΔDP region", "leakage region"], orient="row"))"""
        )
    )

    nb.cells.append(
        md(
            """\
**Lecture** : sur axe region (le bon axe selon `r(s)`), FairGNN avec
adversaire sur region tient le F1 mais réduit modérément ΔDP et **augmente
le leakage** (l'encoder trompe son adversaire MLP sans nettoyer la
représentation). **Post-process simple bat FairGNN sur les 3 métriques.**

**Règle pratique** : avant tout entraînement GNN, mesurer `r(s)`. Si élevé
sur un axe, c'est cet axe qu'il faut débiaiser, pas l'axe qu'on attendait
normativement."""
        )
    )

    # ─── §7 Finding 3 — ULTIMATE composite ────────────────────────────────
    nb.cells.append(
        md(
            """\
## §7 — Finding 3 — ULTIMATE composite : 5 axes simultanés mais coût F1 prohibitif

Pour traiter gender, region, age_group **et** leurs intersections en un
seul fit, on encode l'attribut composite *(gender × age_group × region)*
à 12 cellules et on applique INLP_composite + DPT_composite dessus.

**Hypothèse** : l'INLP composite enlève toutes les directions encodant
les 12 cellules → leakage chance level sur les 5 axes. **Mais coût F1**
sur GraphSAGE car les embeddings sont saturés en region (homophilie
graphe `r=0.9` → directions region sont les directions principales
du signal utile)."""
        )
    )

    nb.cells.append(
        code(
            """\
composite_train_np = composite[train_mask].cpu().numpy()
composite_val_np = composite[val_mask].cpu().numpy()
composite_test_np = composite[test_mask].cpu().numpy()

# INLP composite : multi-classes sur les 12 cellules
_, P_comp = inlp(h_train, composite_train_np, n_iter=15, seed=SEED)
h_comp_train = (h_train @ P_comp).astype(np.float32)
h_comp_test = (h_test @ P_comp).astype(np.float32)
h_comp_val = (h_baseline[val_mask].detach().cpu().numpy() @ P_comp).astype(np.float32)

# Re-fit downstream LR sur les embeddings composite-cleaned
clf_ult = LogisticRegression(max_iter=2000, random_state=SEED)
clf_ult.fit(h_comp_train, data.y[train_mask].cpu().numpy())
proba_ult_val = clf_ult.predict_proba(h_comp_val)[:, 1]
proba_ult_test = clf_ult.predict_proba(h_comp_test)[:, 1]

# DPT composite : seuil par cellule du composite
th_dpt_comp = calibrate_dpt(proba_ult_val, composite_val_np)
pred_ult = apply_per_group_thresholds(proba_ult_test, composite_test_np, th_dpt_comp)

acc_ult = accuracy_score(y_test, pred_ult)
f1_ult = f1_score(y_test, pred_ult, average="macro")
print(f"GraphSAGE+ULTIMATE (composite 12 cellules) : acc={acc_ult:.4f}  F1={f1_ult:.4f}")
print(f"  → vs baseline F1={f1_b:.4f}  →  perte de {(f1_b - f1_ult)*100:.1f} pp de F1")"""
        )
    )

    nb.cells.append(
        code(
            """\
print("=== ULTIMATE composite — fairness sur les 5 axes ===")
print(f"{'axis':18s} {'ΔDP':>8s} {'leakage':>10s}")
for axis_name, s in axes.items():
    s_te = s[test_mask].cpu().numpy()
    s_tr = s[train_mask].cpu().numpy()
    ddp = delta_dp(pred_ult, s_te)
    leak = leakage_auc(h_comp_train, h_comp_test, s_tr, s_te)
    print(f"{axis_name:18s} {ddp:>8.4f} {leak:>10.4f}")"""
        )
    )

    nb.cells.append(
        md(
            """\
**Lecture** : le leakage tombe au **chance level (~0.50)** sur les 5 axes
simultanément, y compris les intersections gender × age et gender × region.
Aucune autre méthode de la toolbox n'arrive à couvrir les axes croisés
proprement.

**MAIS** F1 chute de ~0.94 à ~0.59 (-35 pp). Mécaniquement : INLP
composite supprime jusqu'à 11 directions (12 classes − 1) du sous-espace
embedding, et sur GraphSAGE la majorité de ces directions encodent la
region — qui est aussi *celle qui porte le signal utile* pour prédire
`y` (corrélation indirecte via le graphe). Quand on enlève region, on
détruit la représentation.

**Conséquence** : ULTIMATE composite est *correct* au sens fairness
(chance level partout) mais **inutilisable en production** sur un GNN
homophile. Pour 5 axes simultanés sans collapse, il faudrait un encoder
dont les embeddings ne sont pas dominés par l'attribut sensible — au-delà
du périmètre des méthodes fairness-on-graphs."""
        )
    )

    # ─── §8 Conclusion meta ───────────────────────────────────────────────
    nb.cells.append(
        md(
            """\
## §8 — Conclusion : faire de la fairness, c'est faire des choix

**Findings empiriques sur Pokec-z** :

1. **Toolbox post-process bat FairGNN** sur la fairness, à F1 équivalent.
   INLP+DPT@gender atteint ΔDP=0.003 et leakage=0.57 — **10× mieux** que
   FairGNN sur ΔDP, **24 pp** de leakage en moins, à F1 comparable.

2. **Le bon axe à débiaiser dépend du graphe**, pas du normatif.
   `r(region) = 0.9` >> `r(gender) ≈ 0` ⇒ region est l'axe que le graphe
   amplifie. Mais on a passé l'essentiel du projet sur gender. Règle
   pratique : *mesurer `r(s)` avant de choisir l'axe de fairness*.

3. **ULTIMATE composite** atteint chance level sur 5 axes mais coûte
   35 pp de F1 sur GraphSAGE. **Les 1-2 axes simples sont
   pragmatiques ; les 5 axes simultanés ne le sont pas** sans changer
   d'encoder.

**Conclusion meta-éthique** :

> Faire de la fairness, c'est **faire des choix** : quelle métrique
> privilégier (ΔDP ? ΔEO ? leakage ?), quels axes traiter (gender ?
> region ? age ? un axe ethnique absent du dataset ?), à quel coût
> d'utilité (0.5 pp ? 35 pp ?). Chaque choix encode une **position
> normative**. Le théorème de Chouldechova-Kleinberg (2017) confirme
> qu'on ne peut pas satisfaire toutes les métriques en même temps.

> Et la question reste ouverte : *quis custodiet ipsos custodes ?* — qui
> dira que les choix qu'on fait pour rendre le modèle équitable sont
> eux-mêmes équitables ?

**Reproduction** :
[gregoire-petit1/pokec-fairness-gnn](https://github.com/gregoire-petit1/pokec-fairness-gnn)
(branche `feature/fairgnn-fix-and-multi-fairness`, PR #3).
- `report/2_pager.pdf` — la note 3 pages.
- `scripts/main_experiment.py` — orchestre tout en multi-seed.
- `results/metrics/comparison_full.csv` — tous les résultats par
  (modèle × axe)."""
        )
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        nbf.write(nb, f)
    print(f"wrote {OUT}  ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
