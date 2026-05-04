---
tags:
  - robustesse
  - perturbation
  - graphsage
  - iadata708
  - pokec
status: complete
cours: IADATA708
date: 2026-04-07
---

# 05 — Robustesse du Modèle aux Perturbations

> **Contexte** : Ce document analyse la robustesse du modèle GraphSAGE entraîné sur Pokec face à deux types de perturbations : le bruit gaussien sur les features et le dropout aléatoire d'arêtes. Baseline : Acc=0.9365, F1=0.9364 (seed=42).

---

## 1. Définitions des Perturbations

### 1.1 Feature Noise (Bruit Gaussien Additif)

On perturbe chaque vecteur de features en ajoutant un bruit gaussien centré :

$$x' = x + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)$$

- $\sigma$ contrôle l'intensité du bruit
- Appliqué indépendamment sur chaque dimension de feature
- Simule des erreurs de mesure, données manquantes ou corrompues

### 1.2 Edge Dropout (Suppression Aléatoire d'Arêtes)

On supprime aléatoirement et uniformément une fraction $p$ des arêtes du graphe avant inférence :

- Tirage uniforme sans remplacement parmi toutes les arêtes
- **Distinction importante** : ce n'est **pas** FairDrop — ici la suppression est *aveugle* (pas ciblée par groupe démographique)
- Simule des données relationnelles incomplètes, bruits de crawling, ou graphes partiels

---

## 2. Résultats — Feature Noise

### Tableau des résultats

| σ | Accuracy | Macro F1 | ΔF1 vs baseline |
|---|----------|----------|-----------------|
| 0.0 (baseline) | 0.9365 | 0.9364 | — |
| 0.1 | 0.9345 | 0.9344 | −0.0020 (−0.21%) |
| 0.3 | 0.9268 | 0.9268 | −0.0096 (−1.03%) |
| 0.5 | 0.9164 | 0.9163 | −0.0201 (−2.15%) |

### Analyse

> [!success] Dégradation progressive et régulière
> La relation entre σ et la perte de F1 est monotone et quasi-linéaire. Le modèle se dégrade **proprement** : pas de collapse soudain, comportement prédictible.

La dégradation reste **modérée** même pour un bruit intense (σ=0.5 → −2.15% seulement). Deux mécanismes expliquent cette robustesse :

1. **Dilution par agrégation** : GraphSAGE agrège les représentations des voisins à chaque couche. Un bruit ajouté sur une seule feature d'un nœud est dilué par la moyenne/concaténation sur tous ses voisins. Avec un degré moyen élevé (graphe social), l'agrégation agit comme un filtre passe-bas.

2. **Dimensionnalité élevée** : Le vecteur de features contient **264 dimensions**. Le bruit gaussien i.i.d. sur chaque dimension a une énergie totale $\|\varepsilon\|^2 \approx 264\sigma^2$, mais cette énergie est répartie uniformément. Les directions discriminatives dans l'espace de features ne sont pas toutes impactées de façon critique.

> [!tip] Interprétation géométrique
> En haute dimension, un bruit gaussien isotrope perturbe peu la direction dominante du signal utile (loi des grands nombres). La projection du bruit sur les directions les plus informatives reste faible en valeur attendue.

---

## 3. Résultats — Edge Dropout

### Tableau des résultats

| Taux dropout | Accuracy | Macro F1 | ΔF1 vs baseline |
|---|----------|----------|-----------------|
| 0.0 (baseline) | 0.9365 | 0.9364 | — |
| 0.1 | 0.9374 | 0.9373 | **+0.0009 (+0.10%)** |
| 0.3 | 0.9365 | 0.9364 | ±0.0000 |
| 0.5 | 0.9359 | 0.9358 | −0.0006 (−0.06%) |

### Analyse

> [!warning] Effet inattendu à 10% de dropout
> À `edge_drop=0.1`, le F1 est **légèrement supérieur** au baseline (+0.10%). Contre-intuitif, mais explicable : le dropout aléatoire d'arêtes agit ici comme une **régularisation implicite**, réduisant le sur-ajustement aux connexions spécifiques vues à l'entraînement.

Le résultat le plus frappant est la **quasi-immunité** du modèle : même en supprimant 50% des arêtes aléatoirement, le F1 ne baisse que de −0.06%.

**Explication principale : homophilie forte et redondance structurelle**

Le graphe Pokec présente une homophilie de région $r(\text{region}) = 0.9$ : 90% des arêtes relient des nœuds de même région géographique. Cette propriété implique :

- **Redondance élevée** : chaque nœud possède de nombreuses arêtes vers des voisins "similaires". En supprimer 50% aléatoirement laisse statistiquement encore ~45% des arêtes homophiles.
- **Signal préservé** : les arêtes restantes encodent encore fortement la structure de région, qui est la feature la plus prédictive du label cible.
- **Robustesse par design** : GraphSAGE agrège sur les voisins disponibles — un sous-ensemble aléatoire de voisins homophiles est presque aussi informatif que l'ensemble complet.

> [!info] Intuition probabiliste
> Si $p_h = 0.9$ est la probabilité qu'une arête soit homophile, et qu'on garde chaque arête avec probabilité $(1-\text{dropout})$, la fraction d'arêtes homophiles restantes est toujours $p_h = 0.9$ (le dropout est aveugle). Le ratio signal/bruit structurel est conservé.

---

## 4. Comparaison Feature Noise vs Edge Dropout

| Dimension | Feature Noise (σ=0.5) | Edge Dropout (50%) |
|---|---|---|
| Dégradation F1 | −2.15% | −0.06% |
| Sensibilité | Modérée | Quasi-nulle |
| Mécanisme de résistance | Dilution par agrégation + haute dim. | Homophilie + redondance structurelle |
| Effet régularisant | Non observé | Oui, à faible taux |

> [!note] Conclusion comparative
> Le modèle est **beaucoup plus sensible aux perturbations de features qu'aux perturbations de structure**. Les features ont un gradient direct vers la décision de classification, tandis que les arêtes ne font qu'influencer l'agrégation des représentations — une couche d'indirection qui amortit les perturbations.

**Pourquoi les features sont plus critiques** :
- Les features entrent directement dans les transformations linéaires des couches GNN
- Une perturbation σ=0.5 sur une feature "region" (binaire encodée) peut suffire à la faire changer de signe → erreur de classification directe
- Les arêtes n'agissent que via l'opérateur d'agrégation, qui est intrinsèquement lissant

---

## 5. Limites de cette Analyse de Robustesse

> [!bug] Ce qui manque : robustesse des métriques de fairness
> On a mesuré la robustesse de la **performance prédictive** (Accuracy, F1), mais **pas la robustesse des métriques d'équité** (ΔDP, taux de leakage d'attribut sensible). Questions ouvertes :
> - Est-ce que le biais mesuré sous perturbation reste stable ?
> - Le bruit gaussien réduit-il ou amplifie-t-il le ΔDP ?
> - Les groupes défavorisés sont-ils plus fragilisés par les perturbations ?

> [!danger] Attaques adversariales vs perturbations aléatoires
> Cette analyse porte uniquement sur des **perturbations aléatoires non ciblées**. Les attaques adversariales (FGSM, PGD, attaques sur graphes comme Nettack/Metattack) sont fondamentalement différentes : elles cherchent le pire cas, pas le cas moyen. Un modèle robuste aux perturbations aléatoires peut être très vulnérable aux attaques adversariales.

> [!question] Perturbations ciblées intra vs inter-groupe
> Le dropout aléatoire préserve le ratio homophile global. Mais que se passe-t-il si on cible spécifiquement :
> - Les arêtes **inter-groupes** (entre régions différentes) → perte de signal de frontière
> - Les arêtes **intra-groupe minoritaire** → fragmentation des petits groupes
>
> Ces perturbations ciblées pourraient révéler des fragilités cachées par l'analyse globale. Lien avec FairDrop : voir [[03 - Biais Structurel & Leakage]].

---

## 6. Récapitulatif

```
Robustesse globale du modèle GraphSAGE sur Pokec

Feature Noise :  σ=0.1 → -0.21%  |  σ=0.3 → -1.03%  |  σ=0.5 → -2.15%
Edge Dropout :   10%   → +0.10%  |    30% → ±0.00%  |    50% → -0.06%

Verdict : Robuste aux deux types de perturbations aléatoires.
Raison principale : homophilie r=0.9 + agrégation GraphSAGE + 264 features.
```

> [!abstract] À retenir pour l'examen
> 1. La robustesse aux perturbations **n'implique pas** la robustesse aux attaques adversariales.
> 2. L'homophilie forte est à double tranchant : elle rend le modèle robuste structurellement, mais elle **encode aussi le biais démographique** (voir leakage).
> 3. La légère amélioration à edge_drop=0.1 est un effet de **régularisation**, pas une anomalie.

---

## Liens

- [[02 - Résultats Fairness]] — métriques ΔDP et équité par groupe
- [[03 - Biais Structurel & Leakage]] — homophilie, leakage d'attribut sensible, FairDrop
- [[06 - Gaps & Plans d'Amélioration]] — pistes pour améliorer robustesse et équité
