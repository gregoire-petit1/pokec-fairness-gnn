---
title: "04 - Interprétabilité GNNExplainer"
tags:
  - interpretabilite
  - gnnexplainer
  - counterfactual
  - fairness
  - iadata708
status: complete
date: 2026-04-07
cours: IADATA708
projet: Pokec-Fairness-GNN
---

# 04 — Interprétabilité GNNExplainer & Fairness Contrefactuelle

> [!abstract] Résumé
> Ce module explore **deux dimensions complémentaires** de l'interprétabilité du GNN entraîné sur Pokec :
> 1. **GNNExplainer** : quelles features et quelles arêtes le modèle utilise pour chaque groupe de genre ?
> 2. **Counterfactual Fairness (CF)** : la prédiction d'un individu changerait-elle si son genre était différent ?
> 
> → Voir aussi : [[02 - Résultats Fairness]] · [[06 - Gaps & Plans d'Amélioration]]

---

## 1. GNNExplainer — Principe théorique

GNNExplainer (Ying et al., 2019) est une méthode d'explication **post-hoc locale** pour les Graph Neural Networks. Elle identifie, pour chaque nœud cible, le sous-graphe et le sous-ensemble de features les plus pertinents pour la prédiction.

### Formulation mathématique

L'objectif est de maximiser l'**information mutuelle** entre la prédiction $\hat{y}$ et le sous-graphe masqué $(G_S, X_S)$ :

$$
\max_{G_S,\, X_S} \; I\!\left(\hat{y},\; (G_S, X_S)\right)
$$

Ce qui se décompose par la règle de Bayes en :

$$
I\!\left(\hat{y},\; G_S\right) = H(\hat{y}) - H(\hat{y} \mid G = G_S, X = X_S)
$$

En pratique, GNNExplainer apprend :
- un **masque d'arêtes** $M_E \in [0,1]^{|\mathcal{E}|}$ — importance de chaque arête dans le sous-graphe local
- un **masque de features** $M_F \in [0,1]^{d}$ — importance de chaque feature nœud

Ces masques sont continus et optimisés **par descente de gradient**, ce qui rend la méthode différente des approches par perturbation discrète.

> [!info] Post-hoc vs ante-hoc
> GNNExplainer décrit **le comportement appris** du modèle — ce n'est pas une explication du processus génératif des données. On obtient une approximation locale du modèle, pas une vérité causale.

---

## 2. Setup expérimental

### Paramètres utilisés

| Paramètre | Valeur | Commentaire |
|-----------|--------|-------------|
| Nœuds expliqués | 5 par groupe (`gender=0`, `gender=1`) | Sélection **aléatoire** dans le test set |
| Epochs GNNExplainer | 100 par nœud | Standard pour convergence du masque |
| Output | Masque features + masque arêtes | Par nœud et par groupe |
| Figure produite | `results/figures/feature_importance.png` | Top-15 features, moyennées par groupe |

### Limitations importantes

> [!warning] Estimation très bruitée — 5 nœuds/groupe
> Avec **5 nœuds seulement par groupe**, les importances moyennées sont extrêmement sensibles au hasard de l'échantillonnage. La pratique standard en interprétabilité GNN est d'utiliser **50 à 100 nœuds** pour obtenir des estimations stables.
> 
> Les résultats ci-dessous sont donc **indicatifs, non conclusifs**. Ils permettent d'identifier des pistes, pas de conclure sur un traitement différencié avéré.

> [!caution] Biais de sélection aléatoire
> La sélection aléatoire de 5 nœuds n'est pas représentative de la distribution du graphe. Une sélection **par centralité** (betweenness, degré) ou **par représentativité d'embedding** serait plus robuste pour caractériser le comportement typique du modèle.

---

## 3. Feature importance par groupe

### Résultats observés

![[feature_importance.png]]

La figure présente le **top-15 des features** les plus importantes selon le masque GNNExplainer, moyennées sur 5 nœuds pour chaque groupe de genre.

### Analyse de la divergence inter-groupes

> [!example] Lecture de la figure
> - **gender=0** : `humor` domine nettement (~0.65), suivi de `stredoskolske`, `mam_vazny_vztah`, `chovatelstvo`. Les importances sont **plus dispersées** entre features.
> - **gender=1** : le profil est **plus homogène** et légèrement plus élevé en moyenne. `oldies`, `I_like_books_indicator`, `basketbal` et `stredoskolske` ressortent fortement (~0.62–0.64).
> - **Features communes top-5** : `chovatelstvo`, `mam_vazny_vztah`, `stredoskolske` apparaissent dans les deux groupes → socle commun de décision.

> [!tip] Interprétation fairness
> Si le modèle **utilisait des critères significativement différents** selon le genre pour prédire la même cible (région), cela constituerait un **traitement différencié** — forme de biais algorithmique. Ici, les profils se ressemblent, mais avec 5 nœuds/groupe, on ne peut pas trancher. Une analyse sur 50+ nœuds permettrait de tester statistiquement la divergence.

---

## 4. Counterfactual Fairness — Analyse du paradoxe

### Méthode (inspirée NIFTY, Agarwal et al. 2021)

La **fairness contrefactuelle** mesure : *si l'on flip le genre d'un individu (0→1 ou 1→0) tout en gardant ses autres features identiques, est-ce que la prédiction change ?*

**Pipeline** :
1. Augmenter l'embedding GNN avec l'attribut sensible $s$ → $z_s = [z \| s]$
2. Entraîner un **probe** (régression logistique) sur $z_s$ pour prédire $\hat{y}$
3. Générer la version contrefactuelle $z_{s'} = [z \| s']$ avec $s' = 1 - s$
4. Mesurer le **taux de flip** : fraction des nœuds test dont la prédiction change entre $s$ et $s'$

$$
CF = \frac{1}{|V_{test}|} \sum_{v \in V_{test}} \mathbf{1}\!\left[\hat{y}(z_s^v) \neq \hat{y}(z_{s'}^v)\right]
$$

### Résultats

![[counterfactual_fairness.png]]

| Modèle | CF (taux de flip) | Interprétation |
|--------|-------------------|----------------|
| **Baseline** | **0.0034** (0.3%) | LOW — quasi aucun nœud ne change de prédiction |
| **FairDrop (structure)** | **0.0069** (0.7%) | LOW — légèrement plus sensible |

> [!success] CF très bas = bonne fairness individuelle
> Au sens contrefactuel strict, les deux modèles sont **très équitables** : changer le genre d'un individu ne modifie la prédiction que pour < 1% des nœuds. C'est a priori une bonne nouvelle.

### Le paradoxe : CF faible mais ΔDP élevé

> [!danger] Paradoxe apparent CF vs ΔDP
> La **Demographic Parity** du baseline est $\Delta DP = 0.0414$ (4.1% d'écart dans le taux de prédictions positives entre groupes gender=0 et gender=1). Or le CF est seulement 0.3%.
>
> **Comment les deux coexistent-ils ?**
>
> Ce n'est pas une contradiction — ces deux métriques mesurent des choses différentes :
>
> | Métrique | Niveau de mesure | Ce qu'elle capture |
> |----------|------------------|--------------------|
> | $\Delta DP$ | **Groupe** (agrégé) | Écart de taux de prédiction positive entre groupes |
> | $CF$ | **Individu** (local) | Sensibilité de *ma* prédiction à *mon* genre |
>
> **Explication mécaniste** : sur une cible approximativement équilibrée (47.7% de positifs), le modèle peut produire un ΔDP non nul sans qu'aucun individu ne change de décision lorsque son genre est flipé — parce que l'**effet genre est capturé de façon non-linéaire dans les embeddings de voisinage**, pas directement via la feature brute `gender`. Le modèle a appris une représentation latente qui corrèle avec le genre via la structure du graphe (homophilie), mais le probe LR ne détecte pas de sensibilité directe au flip individuel.
>
> En d'autres termes : **le biais est distribué** — il vient de la composition des groupes de voisins (qui est dans quel groupe), pas du traitement individuel direct.

### FairDrop double le CF : effet contre-intuitif

> [!warning] FairDrop augmente la sensibilité individuelle
> FairDrop supprime aléatoirement des arêtes **intra-genre** pour réduire l'homophilie. Ce faisant, il réduit le biais structurel de groupe (ΔDP amélioré), mais en contrepartie il rend les embeddings **plus sensibles au flip individuel du genre** (CF passe de 0.003 à 0.007).
>
> C'est un **trade-off fairness groupe vs fairness individuelle** : améliorer la parité démographique peut légèrement dégrader l'équité contrefactuelle, et inversement.

---

## 5. Ce qui manque — Lacunes identifiées

> [!failure] Coverage incomplète des méthodes
> Les mesures CF n'ont été réalisées que pour **Baseline** et **FairDrop (structure)**. Les méthodes suivantes n'ont **pas encore de CF mesuré** :
> - Resampling (class-balance)
> - FairGNN
> - FairDrop (feature)
>
> Sans ces valeurs, on ne peut pas conclure sur la fairness contrefactuelle globale de l'expérience.

> [!failure] GNNExplainer insuffisant statistiquement
> - Taille d'échantillon : 5 nœuds/groupe → **insuffisant pour toute conclusion**
> - Standard recommandé : **50–100 nœuds** avec test de divergence (KL-divergence ou test de permutation sur les distributions d'importance)
> - Sélection : aléatoire → devrait être **stratifiée** (degré, centralité, région)

> [!failure] Divergence feature importance non quantifiée
> La figure `feature_importance.png` permet une comparaison visuelle, mais aucune **métrique de divergence** (ex. KL-divergence, distance cosine entre vecteurs d'importance) n'a été calculée pour quantifier objectivement si les deux groupes sont traités différemment.

---

## 6. Synthèse et prochaines étapes

```
GNNExplainer (5 nœuds/groupe)
  ├── Features communes dominantes : chovatelstvo, mam_vazny_vztah, stredoskolske
  ├── Profils similaires → pas de traitement différencié flagrant
  └── ⚠ Pas conclusif : N=5 trop petit

Counterfactual Fairness
  ├── Baseline CF = 0.003 (LOW)
  ├── FairDrop CF = 0.007 (LOW)
  ├── Paradoxe CF↓ / ΔDP↑ → effet structurel d'homophilie, pas individuel
  └── ⚠ CF manquant pour Resampling, FairGNN
```

| Action prioritaire | Impact attendu |
|--------------------|----------------|
| GNNExplainer sur 50 nœuds/groupe | Estimations stables, conclusions valides |
| Quantifier divergence importance (KL) | Détecter traitement différencié objectivement |
| Calculer CF pour toutes les méthodes | Vue complète fairness individuelle |
| Sélection par centralité | Réduire le biais d'échantillonnage |

→ Voir [[06 - Gaps & Plans d'Amélioration]] pour le plan d'action complet.

---

## Références

- Ying, R., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019). **GNNExplainer: Generating Explanations for Graph Neural Networks**. *NeurIPS 2019*. [arXiv:1903.03894](https://arxiv.org/abs/1903.03894)
- Agarwal, C., Lakkaraju, H., & Zitnik, M. (2021). **Towards a Unified Framework for Fair and Stable Graph Representation Learning (NIFTY)**. *UAI 2021*. [arXiv:2102.13186](https://arxiv.org/abs/2102.13186)
- Dai, E., & Wang, S. (2021). **Say No to the Discrimination: Learning Fair Graph Neural Networks with Limited Sensitive Attribute Information (FairGNN)**. *WSDM 2021*.
- Spinelli, I., Scardapane, S., & Uncini, A. (2021). **Fairdrop: Biased Edge Dropout for Enhancing Fairness in Graph Representation Learning**. *IEEE TAI*.
