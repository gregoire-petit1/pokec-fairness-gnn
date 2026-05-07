# Pokec-z — Fairness multi-axes par composition d'outils post-hoc

**Mini-projet IADATA708.** Branche `feature/fairgnn-fix-and-multi-fairness`.

## 1. Setup

**Données.** Pokec-z (subset officiel FairGNN, *Žilinský kraj*) : 66 569
nœuds, ~729 k arêtes, 264 features tabulaires. Reproduction sur **Pokec-n**.
Cible : `completed_level_of_education_indicator` (binaire, 47.7 %
positif). Attributs sensibles : `gender`, `region` (binaires),
`age_group` (3 classes), plus les intersections `gender × age_group` et
`gender × region` — soit **5 axes** simultanés.

**Splits.** Stratifié `y × gender` 60/20/20. Multi-seed `[3, 7, 21, 42, 99]`.

**Méthodes** (toolbox de 5 familles) :

- *Baselines* : GraphSAGE (2 SAGE, hidden=256, dropout=0.5), **TabICL**
  (foundation tabulaire INRIA 2025, no graph, frozen).
- *Pre-process* : Resampling, FairDrop, Reweighting Kamiran-Calders 2012.
- *In-training* : **FairGNN avec Gradient Reversal Layer** — réimpl propre
  de Dai & Wang 2021. La version originale combinait `l_cls − λ·l_adv`
  dans un seul optimiseur (pas du min-max), F1 collapsait à 0.4834 à
  certains λ.
- *Post-process* : EOT (Hardt 2016), DPT (Demographic Parity Threshold),
  **INLP** (Ravfogel 2020) sur embeddings et features, et **les
  compositions INLP+DPT et leur version multi-axes simultanée**.

**Métriques.** ΔDP, ΔEO, AUC gap (sortie), **Sensitive Leakage AUC**
(probe LR train→test, Laclau et al. 2024). 50+ tests pytest, ruff propre,
no-pandas/no-loops enforced.

## 2. Finding 1 — Une toolbox, chaque outil sa métrique

Résultat empirique principal : **les méthodes de fairness ne sont pas
substituables**, chacune attaque une dimension différente du problème.

![Fig 1](results/figures/fig1_toolbox_per_metric.png)

*Fig 1. Mêmes méthodes, 3 métriques. **DPT** (bleu) écrase ΔDP à 0.004
mais ne touche pas le leakage. **INLP** (vert) tombe le leakage à 0.71
mais ne change pas ΔDP. **FairGNN** (rouge) bouge un peu les deux au
prix de F1. Seul l'**ULTIMATE** (orange = TabICL+INLP_composite+
DPT_composite) **règle les trois en même temps**.*

Concrètement, pour un ingénieur qui voudrait une checklist :

| Si tu veux réduire... | Utilise | Pourquoi |
|---|---|---|
| **ΔDP** (taux de prédictions ≈ entre groupes) | **DPT** post-process | calibre un seuil par groupe, c'est l'objectif direct |
| **ΔEO** (TPR ≈ entre groupes parmi y=1) | **EOT** post-process | même méca mais sur le sous-set y=1 |
| **Leakage** (sensible récupérable depuis embeddings) | **INLP** | projette l'espace orthogonal aux directions du sensible |
| **Tout en même temps** | **INLP_composite + DPT_composite** | les deux sont orthogonaux et se composent sans conflit |

Le théorème de Chouldechova-Kleinberg 2017 dit que ΔDP=0 et ΔEO=0 sont
incompatibles dès que les taux de base diffèrent par groupe. On le
confirme : DPT@age_group baisse ΔDP de 0.075 à 0.044 mais double ΔEO
(0.034 → 0.074). Choix normatif, pas algorithmique.

## 3. Finding 2 — TabICL tient la composition multi-axes, GraphSAGE s'écrase

Quand on demande la fairness sur les **3 axes simultanément** (`gender +
age_group + region`, encodés en attribut composite à 12 cellules), il
faut composer INLP_composite (latent) + DPT_composite (sortie). C'est
notre chaîne ULTIMATE.

![Fig 2](results/figures/fig2_chain_tabicl_vs_graphsage.png)

*Fig 2. F1 et leakage le long de la chaîne. À gauche : TabICL garde
F1=0.87 ; GraphSAGE chute à 0.59 (perte de 35 pp de F1). À droite : les
deux chaînes atteignent leakage = chance level (0.5).*

**Pourquoi GraphSAGE s'écrase et pas TabICL ?** `r(region) = 0.901` —
le graphe est très homophile en region. Le message passing recombine
les features d'un nœud avec celles de ses voisins (qui partagent
region) ; les embeddings GraphSAGE *encodent intensément la region*.
Quand INLP supprime ces directions, la représentation est dévastée. À
l'inverse, TabICL ne consomme jamais le graphe → ses embeddings ne sont
pas dominés par la region → INLP ne casse rien d'essentiel.

**Multi-seed [3, 7, 21, 42, 99]** confirme : ΔDP gender ULTIMATE = 0.006
± 0.005 ; leakage gender = 0.500 ± 0.010 ; F1 = 0.87 ± marginal sur
TabICL. **Reproduction sur Pokec-n** : tous les chiffres reproduisent à
écart < 0.01.

## 4. Compromis perf ↔ équité ↔ robustesse

- **Perf vs équité** : 8 pp de F1 pour la fairness multi-axes complète
  sur TabICL. **Acceptable**. Sur GraphSAGE, c'est −35 pp = inutilisable.
- **Pas de free-lunch intersectionnel** : DPT@gender seul atteint ΔDP
  gender = 0.004 mais laisse ΔDP gender×age à 0.10 (whack-a-mole).
- **Robustesse** : la chaîne post-hoc préserve la robustesse héritée
  (GraphSAGE robuste à bruit features σ ≤ 0.3 et edge drop ≤ 30 %
  d'après le repo amont) parce qu'elle ne modifie pas l'encoder.

## 5. Limites

- *Mono-dataset family* — Pokec-z et Pokec-n sont 2 subsets du même
  réseau slovaque. Bail/Credit re-graphifiés nécessaires pour un workshop
  paper.
- *Probe linéaire* — INLP garantit l'invariance contre un classifieur
  linéaire. Un MLP probe pourrait recouvrir du signal résiduel.
- *Pré-traitement implicite* — INLP appliqué sur `x` brut côté TabICL
  est techniquement du pré-traitement (côté GraphSAGE on opère sur les
  embeddings, c'est purement post-hoc).
- *Sémantique opaque* — `region=0/1` binarisé sans documentation par les
  auteurs FairGNN.
- *Catégories réifiées* — la fairness ML opère par agrégation de
  catégories préalables ; une fairness individuelle (Dwork 2012)
  sortirait du périmètre.

**Outils d'IA** (mention exigée) : assistance algorithmique pour la
réimplémentation FairGNN-GRL, la migration pandas → polars, l'intégration
TabICL, l'ajout des modules INLP / calibration / reweighting, et la
rédaction. Code revu, testé, exécuté par les auteurs.

**Références.** Hardt-Price-Srebro 2016 ; Ravfogel et al. 2020 ; Ganin &
Lempitsky 2015 ; Dai & Wang 2021 ; Kamiran & Calders 2012 ; Chouldechova
2017 ; Crenshaw 1989 ; Qu et al. 2025 (TabICL) ; Laclau et al. 2024.
