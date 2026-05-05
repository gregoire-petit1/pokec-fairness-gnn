---
tags:
  - protocole
  - dataset
  - pokec-z
  - iadata708
status: complete
cours: IADATA708
date: 2026-04-07
---

# 01 — Protocole Expérimental : Pokec-z Fairness GNN

> [!abstract] Vue d'ensemble
> Ce document décrit l'intégralité du protocole expérimental mis en place pour analyser les biais de fairness dans les Graph Neural Networks (GNN) appliqués au réseau social Pokec-z. Il couvre la sélection de la variable cible, le preprocessing, les métriques, les architectures modèles, et le protocole d'évaluation multi-seed.

---

## 1. Dataset : Pokec-z

**Référence** : Takac & Zabovsky (2012). *Data analysis in public social networks*. International Scientific Conference & International Workshop Present Day Trends of Innovations.

| Propriété | Valeur |
|---|---|
| Nœuds | 66 569 |
| Arêtes | 729 129 |
| Features (après preprocessing) | 264 |
| Réseau | Social slovaque — région "z" |
| Tâche | Classification de nœuds (binaire) |
| Variable cible | `completed_level_of_education_indicator` |
| Taux positif | 47.7 % |

### Attributs sensibles

| Attribut | Type | Distribution |
|---|---|---|
| `gender` | Binaire | 48.7 % genre = 1 |
| `region` | Binaire | — |

### Statistiques structurelles

| Métrique | Valeur | Interprétation |
|---|---|---|
| Label homophily | 0.523 | Quasi-aléatoire (proche de 0.5) |
| r(gender) | −0.046 | Mélange quasi-aléatoire par genre |
| r(region) | 0.901 | **Forte assortatività structurelle** |

> [!warning] Biais structurel région
> Le coefficient d'assortatività région r = 0.901 indique que les nœuds se connectent quasi-exclusivement au sein de leur région. Ce signal structural est exploitable par les GNN via la propagation de messages, même si `region` est retirée des features — c'est le cœur de l'analyse du leakage structurel.

---

## 2. Sélection de la Variable Cible

### Méthode : Sweep sur 8 candidats × 5 seeds

Un sweep exhaustif a été conduit sur GPU (RTX 3090) : **40 runs** au total.

Script : `scripts/target_sweep.py`
Sortie : `results/metrics/target_sweep.csv`

**Seeds** : `[3, 7, 21, 42, 99]`
**Modèle** : GraphSAGE (hidden=128, 2 couches, dropout=0.5, lr=1e-3, epochs=300, patience=20)

### Tableau de sélection (moyennes sur 5 seeds)

| Candidat | % positif | F1-macro | ΔDP | Commentaire |
|---|---|---|---|---|
| `I_am_working_in_field` | 13.2 % | 0.514 | 0.007 | **Inutilisable** — 86.8% = −1, bruit de label |
| `completed_level_of_education_indicator` | **47.7 %** | **0.939** | **0.037** | ✅ **Sélectionné** |
| `marital_status_indicator` | ~50 % | ~0.82 | ~0.02 | F1 correct mais biais faible |
| `nefajcim` | ~30 % | ~0.71 | ~0.01 | Déséquilibre modéré |
| `stredoskolske` | ~40 % | ~0.75 | ~0.03 | Redondant avec éducation |
| `vysoke_skoly` | ~15 % | ~0.63 | ~0.04 | Trop déséquilibré |
| `relation_to_children_indicator` | ~45 % | ~0.78 | ~0.01 | Biais trop faible |
| `abstinent` | ~20 % | ~0.68 | ~0.01 | Minoritaire + biais faible |

> [!success] Critères de sélection
> La variable `completed_level_of_education_indicator` est retenue car elle satisfait simultanément :
> 1. **Fort F1** (0.939) → signal prédictif réel, pas du sur-apprentissage trivial
> 2. **ΔDP visible** (0.037) → biais de fairness mesurable et non trivial
> 3. **Distribution équilibrée** (47.7 % positif) → pas de biais trivial de classe
> 4. **Cohérence littérature** → variable éducation standard dans les benchmarks fairness GNN

> [!danger] Rejet de `I_am_working_in_field`
> Bien que cette variable soit utilisée comme baseline dans la littérature (Dai & Wang, 2021), le dataset Pokec-z contient 86.8% de valeurs −1 pour ce champ. Le F1 de 0.514 résulte d'une prédiction majoritaire triviale, pas d'un apprentissage réel. Son utilisation est **méthodologiquement invalide** pour ce projet.

---

## 3. Pipeline de Preprocessing

### Étapes (dans `src/data/preprocessing.py`)

```
Données brutes (raw_df)
        │
        ▼
1. Catégorisation de l'âge (AGE → age_group)
        │
        ▼
2. Extraction des attributs sensibles (gender, region → data.gender, data.region)
        │
        ▼
3. Suppression des attributs sensibles des features (data.x)
        │
        ▼
4. Z-score normalization (μ=0, σ=1 par feature)
        │
        ▼
Features finales : 264 dimensions
```

### Détail des étapes

#### 3.1 Catégorisation de l'âge

| Catégorie | Tranche | Code |
|---|---|---|
| `young` | [0, 25) | 0 |
| `adult` | [25, 40) | 1 |
| `senior` | [40, +∞) | 2 |

> [!bug] Biais potentiel AGE=0
> Les nœuds avec `AGE=0` (valeur manquante non documentée) sont affectés à la catégorie `young` par la convention `pd.cut(..., right=False)`. Cela peut sur-représenter artificiellement les jeunes dans le dataset. Ce biais n'est pas corrigé dans la version actuelle — **limite connue** (voir §8).

#### 3.2 Suppression des attributs sensibles

`gender` et `region` sont retirés de `data.x` mais conservés comme attributs séparés (`data.gender`, `data.region`) pour les métriques de fairness. Cela garantit que le modèle ne peut pas accéder directement à ces informations via les features.

#### 3.3 Z-score normalization

$$x_{\text{norm}} = \frac{x - \mu}{\sigma + \epsilon}, \quad \epsilon = 10^{-8}$$

La normalisation est appliquée **par feature** sur l'ensemble du graphe (pas uniquement sur le train set), ce qui constitue une légère fuite d'information — **limite connue** (voir §8).

---

## 4. Métriques de Fairness

### 4.1 Demographic Parity Difference (ΔDP)

Mesure l'écart de taux de prédictions positives entre groupes :

$$\Delta\text{DP} = \left| P(\hat{y}=1 \mid s=0) - P(\hat{y}=1 \mid s=1) \right|$$

- Valeur 0 → fairness parfaite (parity démographique)
- Valeur 1 → biais maximal

### 4.2 Equal Opportunity Difference (ΔEO)

Mesure l'écart de True Positive Rate (TPR) entre groupes parmi les vrais positifs :

$$\Delta\text{EO} = \left| \text{TPR}_{s=0} - \text{TPR}_{s=1} \right| = \left| P(\hat{y}=1 \mid y=1, s=0) - P(\hat{y}=1 \mid y=1, s=1) \right|$$

### 4.3 AUC Gap

Écart maximum d'AUC-ROC entre groupes sensibles :

$$\text{AUC gap} = \max_{i,j} \left| \text{AUC}_{s=i} - \text{AUC}_{s=j} \right|$$

### 4.4 Sensitive Attribute Leakage (RB)

Implémente la probe LR de **Laclau et al. (2024), équation 2** :

> Entraîner une régression logistique sur les embeddings du **train set**, évaluer son AUC-ROC sur le **test set** pour prédire l'attribut sensible.

$$\text{Leakage} = \text{AUC-ROC}_{\text{test}}\left(\mathcal{LR}_{\text{train}(\mathbf{Z})} \rightarrow s\right)$$

Garanties d'implémentation (`src/fairness/metrics.py::sensitive_leakage`) :
- **Pas de linkage bias** : probe entraînée uniquement sur embeddings train
- **Sampling équilibré** : même nombre de positifs/négatifs dans la probe
- **Metric threshold-free** : AUC-ROC (pas d'accuracy)

| Valeur | Interprétation |
|---|---|
| ≈ 0.5 | Embeddings n'encodent pas l'attribut sensible |
| ≈ 1.0 | Attribut sensible fortement encodé dans les embeddings |

### 4.5 Assortative Mixing Coefficient r (Newman 2003)

$$r = \frac{\sum_i e_{ii} - \sum_i a_i b_i}{1 - \sum_i a_i b_i}$$

où $e_{ij}$ est la fraction d'arêtes entre groupe $i$ et groupe $j$, $a_i = \sum_j e_{ij}$, $b_j = \sum_i e_{ij}$.

| Valeur | Interprétation |
|---|---|
| r = 1 | Assortatività parfaite (homophilie totale) |
| r = 0 | Mélange aléatoire (fairness structurelle) |
| r = −1 | Disassortatività parfaite (hétérophilie totale) |

### 4.6 Counterfactual Fairness Score (CF)

Inspiré de **NIFTY (Agarwal et al., 2021)** :

$$\text{CF} = \frac{1}{|\mathcal{V}_{\text{test}}|} \sum_{v \in \mathcal{V}_{\text{test}}} \mathbf{1}\left[\hat{y}(z_v, s_v) \neq \hat{y}(z_v, 1-s_v)\right]$$

**Implémentation** (`src/fairness/metrics.py::counterfactual_fairness_score`) :
1. Augmenter chaque embedding avec $s$ : $z^{\text{aug}} = [z \mid s]$
2. Entraîner une LR probe sur les embeddings augmentés (train set)
3. Prédire avec $s$ original vs $s$ flippé pour chaque nœud test
4. Retourner la fraction de nœuds dont la prédiction change

> [!note] CF comme proxy
> Ce score est un **proxy** de la fairness contrefactuelle, pas une mesure directe sur le graphe (qui nécessiterait de recalculer la propagation de messages avec $s$ flippé). Il mesure la sensibilité d'un classifieur *downstream* à l'attribut sensible via l'espace d'embedding — voir §8 pour les limites.

---

## 5. Architecture des Modèles

### 5.1 GraphSAGE (Baseline)

**Référence** : Hamilton et al. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS.

```
Input features (264-dim)
        │
   SAGEConv → ReLU → Dropout(0.5)
        │
   SAGEConv → ReLU → Dropout(0.5)
        │
   SAGEConv → logits (2-dim)
```

**Hyperparamètres** :

| Paramètre | Valeur |
|---|---|
| `hidden_dim` | 256 |
| `num_layers` | 2 |
| `dropout` | 0.5 |
| `lr` | 0.001 |
| `epochs` | 200 |
| `patience` (early stopping) | 20 |

**Rôle** : Modèle de référence *sans* contrainte de fairness explicite. Sert à mesurer le biais de base introduit par la propagation de messages sur un graphe structurellement biaisé (r(region) = 0.901).

### 5.2 FairGNN (Débiaisage Adversarial)

**Référence** : Dai, E. & Wang, S. (2021). *Say No to Discrimination: Learning Fair Graph Neural Networks with Limited Sensitive Attribute Information*. WSDM 2021.

Architecture encodeur partagé + double tête :

```
Input features (264-dim)
        │
   SAGEConv × num_layers → embeddings h (256-dim)
        │              │
   Classifieur      Adversaire
  Linear(256→2)   Linear(256→64)→ReLU→Linear(64→2)
        │              │
  pred_logits      adv_logits (prédit s)
```

**Fonction de perte** :

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} - \lambda \cdot \mathcal{L}_{\text{adv}}$$

où $\mathcal{L}_{\text{cls}}$ = cross-entropy sur la tâche principale, $\mathcal{L}_{\text{adv}}$ = cross-entropy pour prédire l'attribut sensible.

**Valeurs de λ testées** : `{0.1, 0.5, 1.0, 5.0}`

> [!tip] Rôle de λ
> Plus λ est grand, plus le gradient adversarial est fort : le modèle est davantage pénalisé pour encoder l'attribut sensible dans ses embeddings. **Trade-off performance ↔ fairness** clé à analyser.

### 5.3 FairDrop (Preprocessing Structurel)

**Référence** : Spinelli, I. et al. (2021). *FairDrop: Biased Edge Dropout for Enhancing Fairness in Graph Representation Learning*. IEEE TNNLS.

**Principe** : Dropout biaisé sur les arêtes *intra-groupe* (même attribut sensible) pour réduire l'assortatività structurelle.

$$p_{\text{drop}}(u,v) = \begin{cases} p_{\text{base}} \times \text{bias} & \text{si } s_u = s_v \text{ (intra-groupe)} \\ p_{\text{base}} & \text{sinon (inter-groupe)} \end{cases}$$

**Hyperparamètres** :
- `drop_rate` (base) : testé dans `{0.1, 0.3, 0.5}` (contexte robustesse)
- `intra_group_bias` : 2.0 (par défaut — arêtes intra-groupe 2× plus susceptibles d'être supprimées)

> [!info] FairDrop vs robustesse
> Les perturbations d'arêtes aléatoires (`src/robustness/perturbations.py::drop_edges`) et FairDrop (`src/fairness/fairdrop.py`) utilisent toutes deux un dropout d'arêtes, mais avec des objectifs opposés : **FairDrop est ciblé** (réduit le biais structurel), **drop_edges est aléatoire** (teste la robustesse).

### 5.4 Resampling (Preprocessing Tabulaire)

**Principe** : Oversampling des combinaisons minoritaires (label × groupe sensible) dans le train set pour équilibrer les gradients d'apprentissage.

**Implémentation** (`src/fairness/resampling.py::oversample_train_mask`) :
1. Construire la stratification $\text{strat} = y \times 10 + s$
2. Identifier la taille du groupe majoritaire
3. Oversampler avec remplacement (`sklearn.utils.resample`) les groupes minoritaires jusqu'à équilibrage

> [!note] Limitation du resampling sur graphe
> Le resampling produit des **index dupliqués** dans le mask d'entraînement. Sur graphe, les nœuds dupliqués partagent exactement le même voisinage : l'oversampling ne génère pas de nouvelles informations structurelles (contrairement aux méthodes SMOTE-graph). C'est une approximation pragmatique.

---

## 6. Protocole d'Évaluation

### 6.1 Splits

**Stratégie** : Stratification sur **label × genre** (2×2 = 4 strates)

```
N = 66 569 nœuds
├── Train  60% = ~39 941 nœuds
├── Val    20% = ~13 314 nœuds
└── Test   20% = ~13 314 nœuds
```

Implémentation : `src/data/splits.py::make_splits`
Splits persistés : `data/splits/{train,val,test}.pt`

### 6.2 Multi-seed

| Usage | Seeds |
|---|---|
| Baseline GraphSAGE (robustesse statistique) | `[3, 7, 21, 42, 99]` |
| Expériences downstream (FairGNN, FairDrop, Resampling) | `42` (fixe) |

> [!info] Justification du protocole multi-seed
> Les GNN sur grands graphes présentent une variance non négligeable selon l'initialisation. Les 5 seeds permettent de calculer des intervalles de confiance sur les métriques de performance et de fairness pour le modèle baseline. Les comparaisons avec les méthodes de fairness sont faites à seed fixe (42) pour isoler l'effet de chaque méthode.

### 6.3 Infrastructure

| Composant | Version/Spec |
|---|---|
| GPU | RTX 3090 (24 GB VRAM) |
| PyTorch | 2.10+cu128 |
| PyTorch Geometric | dernière compatible |
| Python | 3.12 |
| Gestionnaire de paquets | `uv` |
| Config | `configs/experiment.yaml` |

### 6.4 Reproductibilité

```python
# Dans scripts/target_sweep.py::set_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

Splits sauvegardés en `.pt` pour garantir la même partition à chaque run.

---

## 7. Robustesse

Deux types de perturbations contrôlées sont testés pour évaluer la stabilité des métriques de fairness sous bruit :

### 7.1 Bruit sur les features

$$x_{\text{perturb}} = x + \mathcal{N}(0, \sigma^2 \cdot I)$$

**Niveaux** : σ ∈ `{0.1, 0.3, 0.5}`

### 7.2 Suppression aléatoire d'arêtes

Suppression uniforme de `rate` × E arêtes.

**Taux** : `{0.1, 0.3, 0.5}`

> [!warning] Robustesse ≠ fairness
> Ces perturbations testent la **stabilité** des métriques (les scores de fairness varient-ils sous bruit ?), pas l'amélioration de la fairness. FairDrop (§5.3) utilise un drop ciblé avec un objectif différent.

---

## 8. Limites Méthodologiques Connues

> [!bug] AGE = 0 → catégorie "young"
> Les valeurs `AGE=0` non documentées sont silencieusement assignées à `young` par `pd.cut([0, 25), right=False)`. Sans documentation sur la signification de cette valeur dans Pokec (donnée non renseignée vs vraiment 0 an), ce choix introduit un biais potentiel de sur-représentation de la catégorie jeune.

> [!bug] Normalisation sur tout le graphe
> La z-score normalization est calculée sur l'ensemble des nœuds (train + val + test) plutôt que sur le train set uniquement. Cela constitue une légère fuite d'information depuis val/test vers le preprocessing. L'impact est probablement faible sur un graphe de 66k nœuds mais reste une impureté méthodologique.

> [!bug] GNNExplainer sur 5 nœuds seulement
> L'analyse d'explicabilité (`src/interpretability/explainer.py::explain_group`) est conduite sur un **petit échantillon de nœuds** (5 nœuds par groupe dans l'implémentation actuelle). Les masques d'importance features agrégés ont donc une variance élevée et ne sont pas généralisables à l'ensemble du graphe.

> [!bug] CF comme proxy indirect
> Le score de Counterfactual Fairness (§4.6) ne recalcule **pas** la propagation de messages avec l'attribut sensible flippé. Il teste seulement la sensibilité d'un classifieur LR entraîné sur les embeddings — il mesure l'impact potentiel via l'espace latent, pas l'impact réel sur les prédictions GNN. C'est un proxy, pas une mesure contrefactuelle stricte.

> [!bug] Splits fixés à seed=42 pour les downstream
> Les comparaisons entre méthodes (FairGNN λ ∈ {0.1, 0.5, 1.0, 5.0}, FairDrop, Resampling) sont toutes évaluées avec un seul seed. La variance inter-seed des méthodes de fairness n'est pas caractérisée.

> [!bug] Homophily de label quasi-aléatoire (0.523)
> Une homophily de 0.523 signifie que la structure du graphe ne prédit pas le label mieux que le hasard. La tâche repose donc principalement sur les features (264-dim), pas sur la structure. L'utilité de l'architecture GNN vs MLP est discutable — non évaluée dans le protocole actuel.

---

## 9. Références

| Source | Utilisation |
|---|---|
| Takac & Zabovsky (2012) | Dataset Pokec |
| Hamilton et al. (2017) | GraphSAGE |
| Dai & Wang (2021), WSDM | FairGNN (débiaisage adversarial) |
| Spinelli et al. (2021), IEEE TNNLS | FairDrop (dropout biaisé) |
| Agarwal et al. (2021), NIFTY (arXiv:2109.05228) | Counterfactual Fairness |
| Newman (2003), Physical Review E | Coefficient d'assortatività r |
| Laclau et al. (2024), Survey Fairness ML on Graphs | Leakage probe (eq.2), r(region)≈0.87 |

---

## Notes liées

- [[02 - Résultats Fairness]] — Résultats des expériences par modèle (ΔDP, ΔEO, leakage, CF)
- [[03 - Biais Structurel & Leakage]] — Analyse approfondie du leakage via r(region) et la propagation de messages
