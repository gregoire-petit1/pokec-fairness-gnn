# IADATA708 — Pokec Fairness GNN : Index

> **Cours** : IADATA708 — Algorithmic Fairness, Interpretability & Robustness  
> **Dataset** : Pokec-z (66 569 nœuds, 729 129 arêtes, 264 features)  
> **Tâche** : Node classification — `completed_level_of_education_indicator`  
> **Branche Git** : `feature/gregoire-experiments`  
> **GitHub** : https://github.com/gregoire-petit1/pokec-fairness-gnn  
> **Statut** : Expériences complètes ✅ — Gaps mesure secondaires 🟡

---

## Navigation

| Note | Contenu |
|------|---------|
| [[01 - Protocole Expérimental]] | Setup, seeds, splits, hyperparamètres, pipeline complet |
| [[02 - Résultats Fairness]] | Tableaux de métriques (Acc, F1, ΔDP, ΔEO, leakage) par méthode |
| [[03 - Biais Structurel & Leakage]] | Homophilie région (r=0.90), leakage, paradoxes FairDrop/Resampling |
| [[04 - Interprétabilité GNNExplainer]] | Feature importance, analyse par groupe, top features |
| [[05 - Robustesse]] | Bruit gaussien + edge drop — quasi-immunité structurelle |
| [[06 - Gaps & Plans d'Amélioration]] | 7 gaps identifiés, priorités, plan d'action |

---

## Résumé Exécutif

### Baseline GraphSAGE (5 seeds : 3, 7, 21, 42, 99)

| Métrique | Valeur |
|----------|--------|
| Accuracy | 0.9381 ± 0.0012 |
| F1-score | 0.9380 ± 0.0011 |
| ΔDP (unfairness) | 0.0414 ± 0.0011 |
| ΔEO | 0.0221 |
| Sensitive Leakage | 0.8171 ± 0.0051 |

### Comparaison des Méthodes

| Méthode | Acc | ΔDP | Leakage | Verdict |
|---------|-----|-----|---------|---------|
| Baseline | 0.938 | 0.0414 | 0.817 | Référence |
| FairDrop p=0.5 | ~0.937 | 0.0380 (-8%) | **0.878 (+7%)** | Paradoxe leakage |
| FairGNN λ=1.0 | 0.827 | **0.0137 (-67%)** | n/a | Meilleur fairness |
| Resampling | 0.938 | **0.0553 (+34%)** | n/a | Contre-productif |

---

## 4 Paradoxes Clés

### 1. Resampling empire ΔDP (+34%)
Le rééquilibrage du label ignore la **corrélation genre/label encodée dans la structure du graphe**. Les arêtes intra-région transmettent le biais indépendamment de la distribution des labels.

### 2. FairDrop augmente le leakage à p=0.5
FairDrop cible les arêtes intra-groupe par genre. Or le biais vient des arêtes **intra-région** (r_region = 0.906). En supprimant les arêtes de genre, on force le modèle à s'appuyer encore plus sur les features corrélées au genre → leakage augmente.

### 3. CF très bas (0.003) vs ΔDP élevé (0.041)
- **ΔDP** = écart agrégé entre groupes (mesure populationnelle)
- **CF** = sensibilité à un flip individuel du genre (mesure locale)

Ces deux métriques capturent des aspects orthogonaux de l'équité. Un modèle peut être équitable individuellement mais inégal en groupe si la distribution des features diffère structurellement entre groupes.

### 4. FairGNN λ non-monotone (λ=5 empire vs λ=1)
La sur-régularisation adversariale à λ=5 destabilise l'entraînement : le discriminateur ne converge plus, le signal fairness devient bruité. **Optimum à λ=1.0**, pas de gain au-delà.

---

## Structure du Graphe — Diagnostics Clés

```
r(gender)  = -0.0463   → légère disassortativité (quasi-aléatoire)
r(region)  =  0.9006   → homophilie extrême ← SOURCE PRINCIPALE DU BIAIS
label homophily = 0.523 → quasi-random
label balance   = 0.477 → quasi-équilibré
gender balance  = 0.487 → quasi-équilibré
```

**Conséquence** : Les algorithmes qui agissent sur les arêtes de genre (FairDrop) attaquent le mauvais signal. Le vrai levier est la ségrégation régionale.

---

## Résultats Robustesse (Baseline)

| Perturbation | Δ F1 |
|---|---|
| Bruit σ=0.1 | -0.21% |
| Bruit σ=0.3 | -1.03% |
| Bruit σ=0.5 | -2.15% |
| Edge drop 10% | +0.10% |
| Edge drop 30% | ±0.00% |
| Edge drop 50% | -0.06% |

GraphSAGE est quasi-immunisé à la suppression d'arêtes grâce à l'agrégation de voisinage.

---

## Gaps Prioritaires

| # | Gap | Priorité | Effort |
|---|-----|----------|--------|
| 1 | Leakage FairGNN manquant | 🔴 Haute | Faible (1 ligne) |
| 2 | Leakage Resampling manquant | 🔴 Haute | Faible (1 ligne) |
| 3 | CF scores pour toutes les méthodes | 🟡 Moyenne | Moyen |
| 7 | ΔDP + leakage dans boucle robustesse | 🟡 Moyenne | Moyen |

Voir [[06 - Gaps & Plans d'Amélioration]] pour la liste complète.

---

## Fichiers Clés

```
notebooks/main_experiment_executed.ipynb   # Notebook exécuté (GPU RTX 3090)
results/metrics/results_summary.csv        # 4 méthodes, métriques complètes
results/figures/                           # 8 figures PNG
report/analysis_note.md                    # Rapport final
```
