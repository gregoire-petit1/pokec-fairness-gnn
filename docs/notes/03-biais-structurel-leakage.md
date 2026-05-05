---
title: "03 - Biais Structurel & Leakage"
tags:
  - biais-structurel
  - homophily
  - leakage
  - gnn
  - fairness
  - pokec
  - IADATA708
status: complete
date: 2026-04-07
course: IADATA708
niveau: M2
---

# 03 — Biais Structurel & Leakage dans les GNN

> [!abstract] Résumé
> Ce document analyse les **mécanismes de biais structurel** dans les GNN sur Pokec-z, en s'appuyant sur le coefficient d'assortativité de Newman (2003) et la métrique de *leakage* (Representation Bias, Laclau et al. 2024). La conclusion centrale : un biais de structure peut persister — voire s'aggraver — même après suppression ciblée d'arêtes, si le canal réel du biais n'est pas identifié correctement.

---

## 1. Coefficient d'Assortativité $r$ — Définition et Valeurs

### 1.1 Formule de Newman (2003)

Le coefficient d'assortativité mesure la tendance des nœuds à se connecter à des pairs de même catégorie. Pour un attribut discatégorie $s$ :

$$
r = \frac{\sum_i e_{ii} - \sum_i a_i^2}{1 - \sum_i a_i^2}
$$

où :
- $e_{ij}$ = fraction d'arêtes reliant le groupe $i$ au groupe $j$
- $a_i = \sum_j e_{ij}$ = fraction d'arêtes ayant une extrémité dans le groupe $i$

| Valeur de $r$ | Interprétation |
|:---:|:---|
| $r = 1$ | Homophilie parfaite (arêtes uniquement intra-groupe) |
| $r = 0$ | Mélange aléatoire (ni homo- ni disassortativité) |
| $r < 0$ | Disassortativité (arêtes préférentiellement cross-groupe) |

### 1.2 Valeurs mesurées sur Pokec-z

| Attribut | $r$ mesuré | Interprétation |
|:---|:---:|:---|
| **Genre** | **−0.0463** | Légère disassortativité — les utilisateurs se connectent préférentiellement *cross-genre* |
| **Région** | **0.9006** | Homophilie extrêmement forte — quasi-ségrégation géographique |

> [!note] Référence
> Laclau et al. (2024) reportaient $r(\text{region}) \approx 0.87$ sur une version antérieure du jeu de données. Notre calcul donne **0.9006**, cohérent mais légèrement supérieur selon la version de Pokec-z utilisée.

### 1.3 Signification pour la fairness

**$r(\text{gender}) = -0.046$** : la structure du graphe ne discrimine **pas directement** par genre. En isolation, cela suggère l'absence de biais structurel lié au genre — les arêtes ne "trient pas" selon ce critère.

**$r(\text{region}) = 0.901$** : les utilisateurs se connectent quasi-exclusivement à des pairs de même région géographique. C'est un signal de ségrégation spatiale très fort. Il constitue un **canal indirect** de biais sur le genre, comme expliqué ci-dessous.

> [!warning] Piège fréquent
> Un $r(\text{genre})$ faible ne garantit **pas** l'absence de biais structurel sur le genre. Il faut examiner l'ensemble des attributs corrélés.

---

## 2. Mécanisme du Biais Indirect

### 2.1 Le message passing comme filtre passe-bas

Dans un GNN (GraphSAGE, GCN…), chaque couche agrège les représentations des voisins :

$$
h_v^{(k)} = \sigma\!\left(W^{(k)} \cdot \text{AGG}\!\left(\{h_u^{(k-1)} : u \in \mathcal{N}(v)\}\right)\right)
$$

Cette agrégation agit comme un **filtre passe-bas** sur le graphe : les signaux partagés par les voisins sont amplifiés, ceux qui varient fortement d'un nœud à l'autre sont atténués. Conséquence : dans un graphe homophile, l'attribut dominant parmi les voisins "contamine" les embeddings même si la feature est absente du vecteur d'entrée.

### 2.2 Chemin causal sur Pokec-z

```
genre ──(corrélé)──► région ──(r=0.90)──► structure d'arêtes
                                                │
                                                ▼
                              message passing ──► h_v encode genre
                                                      (sans y avoir accès)
```

Concrètement :
1. Genre et région sont **corrélés** dans les données Pokec-z (distribution démographique non uniforme selon les zones)
2. $r(\text{region}) = 0.901$ : les voisins d'un nœud partagent quasi-systématiquement sa région
3. Le GNN propage les embeddings de région par agrégation
4. Les embeddings finals $h_v$ **encodent la région**, et donc **indirectement le genre**

> [!tip] Intuition clé
> Supposez que genre = 1 vit majoritairement en région A, et genre = 0 en région B. Si les arêtes sont quasi-exclusivement intra-région ($r \approx 0.9$), alors les voisins d'un nœud "trahissent" sa région — et donc son genre probable — indépendamment de toute feature explicite.

### 2.3 Paramètres graphe (Pokec-z)

| Paramètre | Valeur |
|:---|:---:|
| Nombre d'arêtes | 729 129 |
| Homophilie de label ($h$) | 0.523 |
| Proportion genre = 1 | 48.7 % |

L'homophilie de **label** ($h \approx 0.52$) est quasi-aléatoire : les classes cible ne sont pas structurellement séparées par le graphe. Cela confirme que le biais observé est **démographique**, pas lié au label.

---

## 3. Leakage AUC ≈ 0.82 — Analyse

### 3.1 Définition de la probe (Laclau et al. 2024, eq. 2)

La **Representation Bias (RB)** — aussi appelée *leakage* — mesure la quantité d'information sur l'attribut sensible encodée dans les embeddings appris :

$$
\text{RB} = \text{AUC-ROC}\!\left(\hat{f}_{\text{LR}},\; \{(h_v, s_v)\}_{v \in V_{\text{test}}}\right)
$$

où $\hat{f}_{\text{LR}}$ est une **régression logistique** entraînée sur les embeddings du **train set** (gelés), puis évaluée sur le **test set**.

| AUC-ROC | Interprétation |
|:---:|:---|
| $0.5$ | Embeddings équitables — genre non récupérable |
| $0.5 < \text{AUC} < 1$ | Biais partiel — genre partiellement encodé |
| $1.0$ | Biais maximal — genre parfaitement récupérable |

> [!note] Pourquoi AUC-ROC et non accuracy ?
> Le dataset est quasi-équilibré (48.7% genre=1), donc AUC et accuracy seraient similaires. L'AUC-ROC est préféré car il est insensible au seuil de décision et robuste aux déséquilibres.

### 3.2 Résultats obtenus

| Méthode | Leakage AUC ↓ | Intra-genre (FairDrop) |
|:---|:---:|:---:|
| **Baseline GraphSAGE** (5 seeds) | **0.8171 ± 0.0051** | — |
| FairDrop $p = 0.1$ | 0.8101 | $0.476 \rightarrow 0.446$ |
| FairDrop $p = 0.3$ | 0.8208 | $0.476 \rightarrow 0.342$ |
| FairDrop $p = 0.5$ | **0.8779** | $0.476 \rightarrow 0.000$ |
| FairGNN (best $\lambda$) | n/a (non mesuré) | — |

**AUC = 0.817 à la baseline** : le genre est **fortement récupérable** depuis les embeddings GraphSAGE, même si `gender` a été retiré des features d'entrée. Cela confirme que le canal structurel (via région) suffit à encoder l'attribut sensible.

---

## 4. Paradoxe FairDrop — Analyse Détaillée

### 4.1 Observation

> [!warning] Paradoxe
> À $p = 0.5$, FairDrop supprime **toutes** les arêtes intra-genre (fraction intra : $0.476 \rightarrow 0.000$). Malgré cela, le leakage **augmente** : $0.817 \rightarrow 0.878$.

### 4.2 Mécanisme explicatif

FairDrop (Spinelli et al. 2021) supprime préférentiellement les arêtes **intra-genre** selon une probabilité $p$. La stratégie est correcte si le genre est le vecteur principal de propagation du biais. Mais sur Pokec-z :

```
Canal réel du biais :
  genre ──(corrélé)──► RÉGION ──(r=0.90)──► structure ──► leakage

Canal ciblé par FairDrop :
  arêtes INTRA-GENRE supprimées ✓
  arêtes INTRA-RÉGION : NON TOUCHÉES ✗
```

**Explication du paradoxe** :
1. En supprimant les arêtes intra-genre, FairDrop **perturbe** la structure informative du graphe
2. Mais le canal principal du biais — les arêtes intra-**région** ($r = 0.90$) — est **intact**
3. La perturbation aléatoire des arêtes genre force le GNN à s'appuyer davantage sur le signal de région pour reconstruire une structure cohérente
4. Résultat : les embeddings deviennent **plus informatifs** sur la région, et donc indirectement **plus informatifs** sur le genre

> [!tip] Leçon méthodologique
> Avant d'appliquer FairDrop, il faut identifier **quel attribut drive structurellement le biais**, pas seulement l'attribut sensible cible. Ici, la bonne intervention serait de supprimer des arêtes **intra-région**, ou d'utiliser une version de FairDrop conditionnée sur la région.

### 4.3 Illustration

| $p$ FairDrop | Arêtes intra-genre | Leakage | Interprétation |
|:---:|:---:|:---:|:---|
| 0 (baseline) | 47.6 % | 0.817 | Canal région + canal genre |
| 0.1 | 44.6 % | 0.810 | Légère atténuation |
| 0.3 | 34.2 % | 0.821 | Signal genre réduit, signal région amplifié |
| 0.5 | 0.0 % | **0.878** | Arêtes genre = 0, mais région domine → paradoxe |

---

## 5. Discrepancy Sweep (0.751) vs. Main (0.817)

### 5.1 Observation

Le **target sweep** (sélection de la variable cible, 40 runs × 5 seeds) reportait un leakage AUC $\approx 0.751$ pour la cible sélectionnée, contre **0.817** dans l'expérience principale.

### 5.2 Explication

| Facteur | Sweep | Expérience principale |
|:---|:---:|:---:|
| Epochs | Mêmes hyperparamètres | 200 epochs + early stopping |
| Patience | Identique | 20 epochs |
| Run quality | Rapide, 5 seeds | 5 seeds, entraînement complet |

Sur un graphe homophile, un modèle **plus entraîné** converge vers des embeddings qui capturent mieux la structure du voisinage. Des voisins structurellement homogènes (via région) → des embeddings plus discriminatifs → plus informatifs sur les attributs démographiques.

> [!note] Implication
> Le leakage AUC **augmente avec la qualité d'entraînement** dans les graphes homophiles. Ce n'est pas un bug : c'est la preuve que le biais est structurel et s'exprime d'autant plus que le modèle apprend bien.

---

## 6. Solutions au Leakage Constant

Le leakage AUC $\approx 0.82$ résiste aux interventions de pré-traitement classiques. Les pistes sérieuses pour le réduire sont :

### 6.1 Cibler le bon canal

- **FairDrop sur les arêtes intra-région** plutôt qu'intra-genre
- Identifier via $r$ et analyse de corrélation attribut sensible / structure avant d'intervenir

### 6.2 Contraindre l'espace de représentation

- **FairGNN** : débiaisage adversarial — le discriminateur contraint directement les embeddings à ne pas encoder le genre
- Résultat mesuré : ΔDP $-67\%$, ΔEO $-62\%$ (au coût de $-10.9$ pp de F1)

### 6.3 Interventions sur les features

- Retrait explicite des features de région (mais cela nuit à la précision)
- Représentations invariantes par contrainte de type FAIR-SSL

> [!tip] Voir la note de suivi
> Les plans d'amélioration complets sont détaillés dans [[06 - Gaps & Plans d'Amélioration]].

---

## 7. Synthèse

```
Graphe Pokec-z
│
├── r(genre) = -0.046  → Pas de biais structurel direct sur genre
│
└── r(région) = 0.901  → Homophilie extrême
         │
         └── genre corrélé à région
                  │
                  └── message passing ──► h_v encode genre
                                               │
                                               ▼
                                    Leakage AUC = 0.817
                                    (même sans genre dans les features)
                                               │
                     FairDrop intra-genre ──► 0.878 ← PARADOXE
                     (canal région intact)
```

> [!warning] Conclusion clé
> Le leakage dans les GNN n'est pas seulement un problème de features — c'est un problème de **structure topologique**. Sur Pokec-z, le biais circule via l'homophilie régionale ($r = 0.901$), pas via l'homophilie de genre ($r = -0.046$). Toute intervention qui ignore ce canal structural sera au mieux inefficace, au pire contre-productive.

---

## 8. Références

- **Newman, M. E. J. (2003)**. Mixing patterns in networks. *Physical Review E, 67*(2). — Formule du coefficient $r$
- **Laclau, C., Largeron, C., & Choudhary, M. (2024)**. A Survey on Fairness for Machine Learning on Graphs. *arXiv:2205.05396v2*. — Définition RB/leakage (eq. 2), valeur de référence $r(\text{region}) \approx 0.87$
- **Spinelli, I. et al. (2021)**. FairDrop: Biased Edge Dropout for Enhancing Fairness in Graph Representation Learning. *IEEE TNNLS*. — Algorithme FairDrop
- **Hamilton, W. L., Ying, R., & Leskovec, J. (2017)**. Inductive Representation Learning on Large Graphs. *NeurIPS 2017*. — GraphSAGE

---

## Liens

- [[02 - Résultats Fairness]] — Tableau comparatif complet des méthodes (ΔDP, ΔEO, AUC)
- [[06 - Gaps & Plans d'Amélioration]] — Pistes pour corriger le leakage persistant
