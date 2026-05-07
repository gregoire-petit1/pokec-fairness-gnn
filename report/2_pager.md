# Pokec-z — fairness multi-axes par composition post-hoc

**Mini-projet IADATA708.** Branche `feature/fairgnn-fix-and-multi-fairness`.

## 1. Protocole expérimental

**Données.** Pokec-z (subset officiel FairGNN, *Žilinský kraj*) : 66 569
nœuds, ~729 k arêtes dirigées, 264 features tabulaires.
Reproduction cross-dataset sur **Pokec-n** (~67 k nœuds, 883 k arêtes).
Cible : `completed_level_of_education_indicator` (binaire, 47.7 %
positif, sélectionnée par sweep multi-seed parmi 8 candidates).
Attributs sensibles : `gender`, `region` (binaires), `age_group`
(young/adult/senior), plus les intersections `gender × age_group` et
`gender × region` — pour sortir du mono-axe (Crenshaw 1989).

**Splits & seeds.** Stratifié `y × gender` 60/20/20 ; multi-seed
`[3, 7, 21, 42, 99]` agrégé en mean ± std.

**Méthodes** (5 familles) :

- *Baselines* : GraphSAGE (2 SAGE, hidden=256, dropout=0.5), TabICL
  (foundation tabulaire INRIA 2025, no graph, frozen).
- *Pre-process* : Resampling, FairDrop (intra-group), Reweighting
  Kamiran-Calders 2012.
- *In-training* : **FairGNN avec Gradient Reversal Layer** (Ganin 2015) —
  réimplémentation propre de Dai & Wang 2021. La version originale
  combinait `l_cls − λ·l_adv` dans un seul optimiseur, ce qui n'est pas
  du min-max et collapsait F1 = 0.4834 à certains λ.
- *Post-process* : EOT (Hardt 2016), DPT (variante Demographic Parity),
  composite DPT cellulaire, INLP (Ravfogel 2020) sur embeddings et
  features, et **les compositions INLP+DPT** — clé du finding.

**Métriques.** ΔDP, ΔEO, AUC gap (sortie), *Sensitive Leakage AUC*
(probe LR train→test, balanced sampling). Garde-fous : ruff propre,
50+ tests pytest, no-pandas/no-loops enforced.

## 2. Résultats clés

Sur Pokec-z : `r(gender) = −0.046` (homophilie quasi-nulle) mais
`r(region) = 0.901`. Le graphe **n'est pas** homophile sur l'attribut
sensible direct.

| Méthode (seed=42) | F1 | ΔDP gender | Leakage gender |
|---|---:|---:|---:|
| GraphSAGE | 0.938 | 0.043 | 0.81 |
| TabICL | 0.948 | 0.041 | 0.88 |
| FairGNN(λ=5.0, GRL) | 0.853 | 0.009 | 0.86 |
| TabICL+DPT@gender | 0.945 | 0.004 | 0.88 |
| **TabICL+DPT_composite** | 0.941 | **0.001** | 0.88 |
| TabICL+INLP+DPT@gender | 0.943 | 0.002 | 0.71 |
| **TabICL+INLP+DPT_composite ★** | 0.866 | **0.011** | **0.50** |

**Trois findings non-triviaux** :

1. **TabICL bat GraphSAGE en F1 brut** (0.948 vs 0.938) — l'inverse de
   la narrative dominante. Sur ce dataset, le graphe n'apporte rien à la
   prédiction. Cohérent avec `r(target) ≈ 0`.
2. **Une chaîne post-hoc pareto-domine le gold-standard in-training**.
   FairGNN(λ=5.0) sacrifie 9 pp de F1 pour ΔDP=0.009 ; TabICL+DPT@gender
   atteint ΔDP=0.004 à F1=0.95 — strictement meilleur sur les deux axes,
   sans aucun retraining.
3. **L'ULTIMATE combo** TabICL+INLP_composite+DPT_composite atteint le
   leakage **chance level (0.50 ± 0.01)** sur les **5 axes
   simultanément**, ΔDP < 0.05 partout, à 8 pp de F1. La même chaîne sur
   GraphSAGE collapse à F1=0.59 — le GNN avait tellement encodé le
   sensible (via `r(region)=0.9`) qu'INLP détruit la représentation.

![Figure 1 — Pareto F1 vs ΔDP](results/figures/fig1_pareto_f1_vs_dp.png)

*Fig 1. Pareto F1 vs ΔDP sur l'axe gender. TabICL+DPT_comp domine
en haut-gauche ; FairGNN(λ=5) Pareto-dominé.*

![Figure 2 — Leakage AUC sur 5 axes](results/figures/fig2_leakage_heatmap.png)

*Fig 2. Leakage AUC sur 5 axes sensibles. Les méthodes INLP_composite
(2 dernières lignes) atteignent uniformément le chance level (0.50,
vert) sur les 5 axes ; les autres laissent leakage > 0.6 partout.*

## 3. Compromis performance ↔ équité ↔ robustesse

Compromis principal documenté : **performance vs équité** (Fig 1).
Trois points clés.

- **Pas de free-lunch sur fairness multi-axes**. Le single-axis (DPT@gender,
  INLP@gender) atteint ΔDP gender quasi-nul mais laisse ΔDP gender × age
  à 0.10 — *whack-a-mole intersectionnel*, Crenshaw 1989 rendu empirique.
- **Composer les axes coûte du F1**. ULTIMATE combo perd 8 pp pour
  satisfaire les 5 axes ensemble — inévitable sous Chouldechova-Kleinberg
  2017 dès que les taux de base diffèrent.
- **Robustesse aux perturbations**. La baseline GraphSAGE est robuste
  à un bruit features σ ≤ 0.3 et edge drop ≤ 30 % (Section 7 du repo
  amont, héritée). La chaîne post-hoc **préserve** cette robustesse :
  elle ne touche pas l'encoder.

## 4. Validation et limites

**Validation forte.** Reproduction sur **Pokec-n** : tous les chiffres
clés se reproduisent à écart < 0.01 ; ranking des méthodes strictement
identique. Multi-seed `[3, 7, 21, 42, 99]` : std < 0.02 sur le leakage.
Calibration Guo 2017 testée : T = 1.020 sur GraphSAGE → no-op,
l'objection « calibration différentielle » ne tient pas.

**Limites principales** — (1) *mono-dataset family* : Pokec-z ≈ Pokec-n,
deux subsets du même réseau slovaque (Bail/Credit re-graphifiés
nécessaires pour un workshop paper) ; (2) *probe linéaire* : INLP
garantit l'invariance contre un classifieur linéaire ; un MLP pourrait
recouvrir du signal résiduel ; (3) *pré-traitement implicite* : INLP sur
`x` brut côté TabICL = pré-traitement de l'input (côté GraphSAGE c'est
purement post-hoc sur embeddings) ; (4) *sémantique opaque* :
`region=0/1` binarisé sans documentation par les auteurs FairGNN ; (5)
*catégories réifiées* : la fairness ML opère par agrégation de
catégories préalables, *historiquement contingentes* (Hoffmann 2019).
Une fairness individuelle (Dwork 2012) sortirait de ce périmètre.

**Outils d'IA** (mention exigée) : assistance algorithmique pour la
réimplémentation FairGNN-GRL, la migration pandas → polars, TabICL,
INLP / calibration / reweighting, et la rédaction. Code revu, testé,
exécuté par les auteurs.

**Références.** Hardt-Price-Srebro 2016 ; Ravfogel et al. 2020 ; Ganin &
Lempitsky 2015 ; Dai & Wang 2021 ; Kamiran & Calders 2012 ; Newman
2003 ; Chouldechova 2017 ; Crenshaw 1989 ; Hoffmann 2019 ; Qu et al.
2025 ; Laclau et al. 2024.
