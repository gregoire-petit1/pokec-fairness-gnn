# Pokec-z — Quel outil pour quelle métrique, sur quel axe ?

**Mini-projet IADATA708.** Branche `feature/fairgnn-fix-and-multi-fairness`.

## 1. Setup

**Données.** Pokec-z (subset officiel FairGNN, *Žilinský kraj*) : 66 569
nœuds, ~729 k arêtes, 264 features tabulaires. **Pokec-n** = subset sœur
sur une autre région slovaque (reproduction intra-dataset, pas
cross-dataset au sens strict).

**Cible.** `completed_level_of_education_indicator` (binaire, 47.7 %
positif). **Attributs sensibles** : `gender`, `region` (binaires),
`age_group` (3 classes), plus les intersections gender × age_group et
gender × region — soit **5 axes** simultanés.

**Splits.** Stratifié `y × gender` 60/20/20. Multi-seed
`[3, 7, 21, 42, 99]`.

**Méthodes** :

- *Baseline* : GraphSAGE (2 SAGE, hidden=256, dropout=0.5).
- *Pre-process* : Resampling, FairDrop, Reweighting Kamiran-Calders 2012.
- *In-training* : **FairGNN** (Dai & Wang 2021), méthode adversariale,
  entraînement two-optimizer alternating canonique.
- *Post-process* : **EOT** (Hardt 2016, ΔEO), **DPT** (ΔDP), **INLP**
  (Ravfogel 2020, leakage sur embeddings), composition mono-axe
  **INLP+DPT**, et la chaîne composite multi-axes **INLP_composite +
  DPT_composite** sur l'attribut joint à 12 cellules.

**Métriques.** ΔDP, ΔEO, AUC gap (sortie), **Sensitive Leakage AUC**
(probe LR train→test, Laclau et al. 2024). 50+ tests pytest, ruff propre,
no-pandas/no-loops enforced.

## 2. Finding 1 — La toolbox post-process bat FairGNN sur la fairness

Premier résultat : les méthodes de fairness ne sont pas substituables.
Chaque famille de post-process attaque **une partie spécifique** du
problème, et FairGNN ne fait ni l'une ni l'autre proprement.

| Méthode | ΔDP gender | Leakage gender |
|---|---:|---:|
| GraphSAGE baseline | 0.043 | 0.812 |
| FairGNN (canonical, two-optimizer) | 0.030 | 0.828 |
| **GraphSAGE+EOT@gender** | 0.019 | 0.812 |
| **GraphSAGE+DPT@gender** | 0.019 | 0.812 |
| **GraphSAGE+INLP@gender** | 0.043 | **0.573** |
| **GraphSAGE+INLP+DPT@gender** | **0.003** | **0.573** |

À F1 essentiellement équivalent (~0.94 dans tous les cas, dispersion
multi-seed dans le bruit), les chaînes post-process atteignent :

- **DPT/EOT** (seuil par groupe) → réduit ΔDP/ΔEO 2× à 4× vs FairGNN,
  sans toucher aux embeddings (leakage inchangé).
- **INLP** (projection sur embeddings) → réduit le leakage 0.81 → 0.57,
  sans toucher aux seuils (ΔDP inchangé).
- **INLP+DPT** : l'un opère sur les embeddings, l'autre sur les seuils,
  les deux opérations sont orthogonales et se composent sans conflit.
  Résultat : ΔDP=0.003 ET leakage=0.57 — **10× mieux** que FairGNN sur
  ΔDP, **24 pp** de leakage en moins.

**Pourquoi FairGNN reste partiel.** L'encoder apprend à tromper un
adversaire MLP spécifique sans supprimer le signal sous-jacent. Un autre
probe (la LR de notre métrique de leakage) peut encore l'extraire.
L'in-training adversariale donne une garantie *empirique* contre un
adversaire spécifique ; INLP donne une garantie *formelle* contre toute
attaque linéaire.

Le théorème de Chouldechova-Kleinberg (2017) confirme par ailleurs
l'incompatibilité de ΔDP=0 et ΔEO=0 dès que les taux de base diffèrent :
DPT@age_group baisse ΔDP de 0.075 à 0.044 mais double ΔEO. **Choisir une
métrique, c'est choisir une éthique.**

## 3. Finding 2 — Le bon axe à fairner = celui que le graphe amplifie

Le second résultat est méthodologique et **inverse l'intuition initiale**.
On a passé l'essentiel du projet à fairner `gender` (axe protégé reconnu).
Mesurer le coefficient d'assortativité de Newman (2003) sur les 3 axes
sensibles révèle qu'on s'est trompé d'axe :

| Axe | r(s) | Interprétation |
|---|---:|---|
| gender | **−0.046** | graphe quasi-aléatoire vs gender → message passing n'amplifie rien |
| age_group | **+0.352** | légère homophilie générationnelle |
| region | **+0.901** | graphe massivement homophile en region |

`r(region) = 0.9` signifie que ~90 % des arêtes connectent deux personnes
de la même région : Pokec-z est en pratique un graphe de régions,
faiblement inter-connecté. Un GNN va amplifier cette structure dans ses
embeddings — *c'est exactement le scénario où les méthodes
fairness-on-graphs sont conçues pour intervenir*. Or notre setup typique
applique FairGNN avec l'adversaire sur gender, là où le graphe ne fait
quasiment rien.

**Test empirique** : FairGNN avec adversaire sur region (multi-seed
Pokec-z) au lieu de gender :

| Adversaire | F1 | ΔDP region | Leakage region |
|---|---:|---:|---:|
| FairGNN(adv=gender, λ=5) | 0.853 | 0.073 | 0.666 |
| FairGNN(adv=region) | 0.937 | 0.033 | **0.761** ← *augmente* |
| **GraphSAGE+DPT@region** | 0.939 | **0.025** | 0.641 |
| **GraphSAGE+INLP+DPT@region** | 0.932 | **0.020** | **0.524** |

Cibler le bon axe (region) avec FairGNN récupère le F1 et réduit
modérément ΔDP, mais **augmente le leakage** : l'encoder trompe le MLP
adversaire sans nettoyer la représentation. **Post-process simple bat
FairGNN sur les 3 métriques même quand FairGNN cible le bon axe.**

**Règle pratique** : avant tout entraînement GNN, mesurer `r(s)` pour
chaque attribut sensible. Si `r(s)` est élevé, c'est cet axe que le graphe
encode et donc celui qu'il faut fairner — **indépendamment de l'axe
attendu normativement**. Si `r(s)` est faible sur tous les axes, le
graphe n'est ni source ni amplificateur de biais : la fairness se règle
sur les sorties, pas sur les embeddings.

## 4. Finding 3 — ULTIMATE composite : 5 axes simultanés mais coût F1 prohibitif

Pour traiter gender, region, age_group et leurs intersections
**simultanément**, on encode l'attribut composite *(gender × age_group ×
region)* à 12 cellules et on applique INLP_composite + DPT_composite
dessus.

**Le pipeline ULTIMATE atteint le chance level (leakage ≈ 0.50) sur les
5 axes simultanément**, y compris gender × age_group et gender × region.
Aucune autre méthode de notre toolbox n'y arrive — INLP+DPT mono-axe
laisse fuiter sur les axes croisés (gender×age 0.85, gender×region 0.73).

**Mais le coût F1 sur GraphSAGE est prohibitif.** GraphSAGE+ULTIMATE
chute à F1=**0.59** (−35 pp vs baseline 0.94). Mécaniquement : les
embeddings GraphSAGE sont saturés par l'homophilie region (cohérent avec
`r(region) = 0.9`) ; INLP composite, en supprimant les directions
encodant les 12 cellules, supprime aussi la majorité du signal utile
pour prédire `y`. La représentation s'effondre.

**Conséquence pratique** : ULTIMATE composite est *correct* au sens
fairness (chance level partout) mais *inutilisable* en production sur un
GNN homophile. Pour un cas réel multi-axes, il faut soit accepter un
compromis sur le nombre d'axes traités simultanément (rester en mono-axe
INLP+DPT), soit utiliser un encoder dont les embeddings ne sont pas
saturés par l'attribut sensible dominant — au-delà du périmètre des
méthodes fairness-on-graphs étudiées ici.

Multi-seed `[3, 7, 21, 42, 99]` × Pokec-z/n confirme la stabilité du
finding : ΔDP gender ULTIMATE = 0.006 ± 0.005, leakage ≈ 0.50 ± 0.01,
F1 collapse stable. Reproduction Pokec-n à <0.01 près (intra-dataset).

## 5. Compromis perf ↔ équité ↔ robustesse

**Perf vs équité.** Le coût en F1 dépend du périmètre de fairness :
- Mono-axe simple (DPT ou EOT seul) : <0.5 pp de F1, une métrique réglée.
- Mono-axe combiné (INLP+DPT) : ~0.5 pp de F1, ΔDP **et** leakage réglés.
- Multi-axes composite (ULTIMATE) : 35 pp de F1 sur GraphSAGE —
  prohibitif. Le bénéfice (chance level sur 5 axes) ne compense pas la
  perte d'utilité.

**Pas de free-lunch intersectionnel.** DPT@gender seul atteint ΔDP gender
= 0.019 mais laisse ΔDP gender×age_group à 0.085. C'est l'effet
*whack-a-mole* prédit par l'argument intersectionnel de Crenshaw (1989).
Pour traiter les axes croisés, il faut explicitement construire
l'attribut composite et calibrer dessus, à charge de payer le coût
associé.

**Robustesse.** La chaîne post-hoc préserve la robustesse héritée du
modèle de base parce qu'elle ne modifie pas l'encoder. GraphSAGE reste
robuste à un bruit features σ ≤ 0.3 et à un edge drop ≤ 30 % (mesures
issues du repo amont) avant comme après INLP+DPT. Avantage non-trivial
sur l'in-training fairness, qui modifie l'encoder et peut dégrader sa
robustesse de façon imprévue.

## 6. Limites

**Limite philosophique.** Les métriques de fairness encodent toutes une
position normative implicite : ΔDP=0 incarne l'anti-classification
stricte, ΔEO=0 incarne une méritocratie conditionnée à la vérité, la
calibration par groupe assume que les écarts marginaux sont OK. Le
théorème d'incompatibilité montre qu'on ne peut pas les satisfaire
toutes à la fois. Notre toolbox fournit les outils, elle ne tranche pas
le débat normatif.

**Importation culturelle des catégories.** La fairness ML hérite des
catégories US des années 60 (gender, race binaires) et notre dataset les
reproduit en réduisant `gender` et `region` à 0/1 sans documentation
sémantique (Hoffmann 2019 ; Hanna et al. 2020). Le subset FairGNN code
`spoken_languages` via 8 indicateurs binaires tous internationaux ou
langue majoritaire ; le hongrois, le tchèque et le romani sont absents
alors que le SNAP original les encode en texte libre. Conséquence : la
littérature fairness-on-graphs travaille sur un subset structurellement
aveugle aux axes ethniques slovaques.

**Limites techniques.** La garantie d'invariance d'INLP n'est valide que
contre un classifieur linéaire ; un MLP probe non-linéaire pourrait
recouvrir du signal résiduel. Le discriminateur FairGNN est binaire dans
la formulation standard de Dai & Wang ; pour fairner `age_group`
(3 classes) en in-training il faudrait redimensionner la tête de
discriminateur, non implémenté.

**Limite de généralisation.** Pokec-z et Pokec-n sont deux subsets du
même réseau Pokec sur des régions slovaques différentes — c'est de la
reproduction intra-dataset, pas du cross-dataset au sens strict (pour ça
il faudrait Bail / Credit re-graphifiés). La cible
`completed_level_of_education_indicator` est faiblement homophile en
gender ; nos conclusions sur "post-process bat in-training" sont
conditionnelles à cette propriété — sur un graphe fortement homophile à
l'attribut sensible attendu, le classement pourrait s'inverser.

**Pour conclure.** Le choix d'axe sensible est probablement la limite la
plus structurante. Dans le contexte slovaque, l'axe ethnique — minorité
hongroise et surtout minorité gitane — est beaucoup plus prévalent comme
source de discrimination effective (logement, éducation, embauche) que
l'axe du sexe sur lequel on a passé l'essentiel de l'étude. Le verrou
principal n'est pas algorithmique mais en amont, au niveau de la curation
des données. Tant qu'un dataset slovaque sans label ethnique reste le
standard, l'évaluation des méthodes restera détachée des axes qui
comptent vraiment dans le pays d'origine de la donnée.

<!-- PAGEBREAK -->

## Annexes

### A.1 Outils d'IA

Assistance algorithmique pour la migration pandas → polars, l'intégration
des modules INLP / calibration / reweighting, le portage de FairGNN
canonical (two-optimizer alternating depuis le repo Dai-Wang 2021), et la
rédaction. Code revu, testé, exécuté par les auteurs.

### A.2 Références

Dai, E. & Wang, S. (2021). *Say No to the Discrimination — Learning Fair
Graph Neural Networks with Limited Sensitive Attribute Information*.
WSDM. — Hardt, M., Price, E. & Srebro, N. (2016). *Equality of
Opportunity in Supervised Learning*. NeurIPS. — Ravfogel, S. et al.
(2020). *Null It Out — Guarding Protected Attributes by Iterative
Nullspace Projection*. ACL. — Ganin, Y. & Lempitsky, V. (2015).
*Unsupervised Domain Adaptation by Backpropagation* (origine du
Gradient Reversal Layer). — Kamiran, F. & Calders, T. (2012). *Data
Preprocessing Techniques for Classification without Discrimination*.
KAIS. — Chouldechova, A. (2017) ; Kleinberg, J. et al. (2017). Théorèmes
d'incompatibilité ΔDP / ΔEO. — Crenshaw, K. (1989). *Demarginalizing the
Intersection of Race and Sex*. — Hoffmann, A. L. (2019). *Where
Fairness Fails*. — Hanna, A. et al. (2020). *Towards a Critical Race
Methodology in Algorithmic Fairness*. FAccT. — Newman, M. E. J. (2003).
*Mixing Patterns in Networks*. — Laclau, C., Largeron, C. & Choudhary,
M. (2024). *A Survey on Fairness for Machine Learning on Graphs*.

### A.3 Comparaison méthodes × axe gender (Pokec-z, seed=42)

Source : `results/metrics/comparison_full.csv`.

| Modèle | Acc | F1 | ΔDP | ΔEO | Leakage |
|---|---:|---:|---:|---:|---:|
| GraphSAGE | 0.9383 | 0.9381 | 0.0429 | 0.0231 | 0.8120 |
| GraphSAGE+Resampling | 0.9383 | 0.9381 | 0.0423 | 0.0229 | 0.8109 |
| GraphSAGE+FairDrop | 0.9386 | 0.9384 | 0.0442 | 0.0253 | 0.8250 |
| FairGNN canonical (two-opt, multi-seed μ) | 0.937 | 0.937 | 0.030 | 0.014 | 0.828 |
| **GraphSAGE+EOT@gender** | 0.9393 | 0.9392 | **0.0191** | **0.0001** | 0.8120 |
| **GraphSAGE+DPT@gender** | 0.9393 | 0.9392 | **0.0191** | 0.0220 | 0.8120 |
| **GraphSAGE+INLP@gender** | 0.9317 | 0.9316 | 0.0428 | 0.0243 | **0.5726** |
| **GraphSAGE+INLP+DPT@gender** | 0.9320 | 0.9319 | **0.0032** | 0.0078 | **0.5726** |
| GraphSAGE+ULTIMATE composite | 0.6155 | 0.5915 | 0.0090 | 0.0248 | **0.4996** |

### A.4 Comparaison méthodes × axe region (Pokec-z, seed=42)

Source : `results/metrics/comparison_full.csv` et
`results/metrics/fairgnn_on_region.csv`.

| Modèle | Acc | F1 | ΔDP region | ΔEO region | Leakage region |
|---|---:|---:|---:|---:|---:|
| GraphSAGE | 0.9383 | 0.9381 | 0.0532 | 0.0042 | 0.6411 |
| FairGNN canonical (adv=gender, two-opt μ) | 0.937 | 0.937 | 0.043 | 0.018 | 0.821 |
| FairGNN canonical (adv=region, two-opt μ) | 0.938 | 0.938 | 0.033 | 0.006 | **0.764** ← *augmente* |
| **GraphSAGE+DPT@region** | 0.9392 | 0.9390 | **0.0246** | 0.0222 | 0.6411 |
| **GraphSAGE+INLP@region** | 0.9336 | 0.9335 | 0.0487 | 0.0035 | **0.5237** |
| **GraphSAGE+INLP+DPT@region** | 0.9323 | 0.9322 | **0.0200** | 0.0219 | **0.5237** |
| GraphSAGE+ULTIMATE composite | 0.6155 | 0.5915 | 0.0122 | 0.0343 | **0.4959** |

### A.5 ULTIMATE composite — un seul fit règle 5 axes (Pokec-z, seed=42)

Une seule chaîne `INLP_composite + DPT_composite` calibrée sur l'attribut
joint à 12 cellules ramène le leakage **simultanément** au chance level
(~0.50) sur les 5 axes — y compris les intersections.

| Modèle | Attribut | ΔDP | ΔEO | AUC-gap | Leakage |
|---|---|---:|---:|---:|---:|
| GraphSAGE+ULTIMATE (Acc=0.616, F1=0.592) | gender | 0.0090 | 0.0248 | 0.0085 | 0.4996 |
|  | region | 0.0122 | 0.0343 | 0.0103 | 0.4959 |
|  | age_group | 0.0304 | 0.0161 | 0.0324 | 0.5049 |
|  | gender × age | 0.0534 | 0.0371 | 0.0532 | 0.4983 |
|  | gender × region | 0.0405 | 0.0687 | 0.0388 | 0.4983 |

Coût : F1 = 0.59 sur GraphSAGE (−35 pp vs baseline). Le gain de fairness
intersectionnelle ne compense pas la perte d'utilité — la chaîne reste un
outil d'analyse, pas de production.
