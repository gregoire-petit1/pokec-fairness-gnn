# Pokec-z — Fairness multi-axes par composition d'outils post-hoc

**Mini-projet IADATA708.** Branche `feature/fairgnn-fix-and-multi-fairness`.

## 1. Setup

**Données.** Pokec-z (subset officiel FairGNN, *Žilinský kraj*) : 66 569
nœuds, ~729 k arêtes, 264 features tabulaires. Reproduction sur **Pokec-n**.
Cible : `completed_level_of_education_indicator` (binaire, 47.7 % positif).
Attributs sensibles : `gender`, `region` (binaires), `age_group` (3 classes),
plus les intersections `gender × age_group` et `gender × region` — soit
**5 axes** simultanés.

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

Notre résultat empirique principal sur cette première dimension est que
**les méthodes de fairness ne sont pas substituables**. Chaque famille
attaque une partie spécifique du problème, et il faut souvent en composer
plusieurs pour traiter les trois métriques en même temps.

![Fig 1](results/figures/fig1_toolbox_per_metric.png)

*Fig 1. Mêmes méthodes, 3 métriques. **DPT** (bleu) écrase ΔDP à 0.004
mais ne touche pas le leakage. **INLP** (vert) tombe le leakage à 0.71
mais ne change pas ΔDP. **FairGNN** (rouge) bouge un peu les deux au
prix de F1. Seul l'**ULTIMATE** (orange = TabICL+INLP_composite+
DPT_composite) règle les trois en même temps.*

Pour un ingénieur qui voudrait une checklist, le mapping métrique → outil
est le suivant :

| Si tu veux réduire... | Utilise | Pourquoi |
|---|---|---|
| **ΔDP** (taux de prédictions ≈ entre groupes) | **DPT** post-process | calibre un seuil par groupe, c'est l'objectif direct |
| **ΔEO** (TPR ≈ entre groupes parmi y=1) | **EOT** post-process | même méca mais sur le sous-set y=1 |
| **Leakage** (sensible récupérable depuis embeddings) | **INLP** | projette l'espace orthogonal aux directions du sensible |
| **Tout en même temps** | **INLP_composite + DPT_composite** | les deux sont orthogonaux et se composent sans conflit |

Cette toolbox a une conséquence directe : **un foundation model + 30 lignes
de post-process Pareto-domine FairGNN sur Pokec-z**. TabICL+EOT atteint
F1=0.946 et ΔDP=0.007 contre FairGNN qui plafonne à F1=0.853 et ΔDP=0.009.
On gagne 9 points de F1 *et* on est légèrement plus équitable, à un coût
d'entraînement quasi-nul (TabICL est frozen, le calibrage des seuils prend
moins d'une seconde). C'est un résultat qui invite à la prudence avant
d'engager une méthode in-training adversariale sophistiquée.

Le théorème de Chouldechova-Kleinberg (2017) dit par ailleurs que ΔDP=0 et
ΔEO=0 sont mathématiquement incompatibles dès que les taux de base diffèrent
entre groupes — ce qu'on vérifie expérimentalement : DPT@age_group baisse
ΔDP de 0.075 à 0.044 mais double ΔEO (0.034 → 0.074). Choisir une métrique
n'est pas un choix algorithmique, c'est un **choix normatif**.

## 3. Finding 2 — TabICL tient la composition multi-axes, GraphSAGE s'écrase

Le second résultat porte sur la **fairness sur plusieurs variables
simultanément** (gender + age_group + region, encodés en attribut composite
à 12 cellules). Pour traiter cinq axes à la fois, il faut composer
INLP_composite (latent) + DPT_composite (sortie). C'est notre chaîne
ULTIMATE.

![Fig 2](results/figures/fig2_chain_tabicl_vs_graphsage.png)

*Fig 2. F1 et leakage le long de la chaîne. À gauche : TabICL garde
F1=0.87 ; GraphSAGE chute à 0.59 (perte de 35 pp de F1). À droite : les
deux chaînes atteignent leakage = chance level (0.5).*

**Pourquoi GraphSAGE s'écrase et pas TabICL ?** Les coefficients
d'assortativité de Newman (2003) racontent toute l'histoire : `r(gender) =
−0.046` (le graphe n'est pas du tout homophile en gender), mais `r(region) =
0.901` (le graphe est massivement homophile en region). Le message passing
de GraphSAGE recombine les features d'un nœud avec celles de ses voisins ;
quand ces voisins partagent presque toujours la même region, l'embedding
final encode la region beaucoup plus intensément que la feature brute ne le
laissait penser. Quand INLP_composite supprime ensuite les directions
sensibles, il enlève une part majeure de l'information utile et la
représentation s'effondre. TabICL, qui ne consomme jamais le graphe, n'a
pas ce problème : ses embeddings ne sont pas dominés par la region, donc
INLP ne casse rien d'essentiel.

**La règle pratique qui en sort** est utilisable avant tout entraînement :
mesurer `r(s)` pour chaque attribut sensible. Si `r(s)` est faible, le
graphe n'aide pas à prédire la cible *et* n'amplifie pas le biais marginal,
auquel cas un foundation model tabulaire + post-process est presque
toujours le bon choix. Si `r(s)` est élevé (le cas typique des datasets
fairness-on-graphs publiés), les méthodes in-training graphiques redeviennent
nécessaires parce que le post-process, qui n'opère que sur les sorties, ne
peut pas atteindre l'espace latent biaisé. Sur Pokec-z, `r(gender) = -0.046`
aurait dû nous orienter vers TabICL avant même de lancer GraphSAGE.

La validation multi-seed `[3, 7, 21, 42, 99]` confirme la stabilité : ΔDP
gender ULTIMATE = 0.006 ± 0.005 ; leakage gender = 0.500 ± 0.010 ; F1 =
0.87 à dispersion marginale sur TabICL. La reproduction sur Pokec-n donne
les mêmes chiffres à un écart inférieur à 0.01.

## 4. Compromis perf ↔ équité ↔ robustesse

**Sur l'axe perf vs équité**, payer 8 points de F1 pour obtenir la
fairness multi-axes complète sur TabICL est un compromis défendable. Sur
GraphSAGE, perdre 35 points de F1 pour le même résultat de fairness rend
la chaîne inutilisable en production — c'est exactement le genre de
situation où le coût de la fairness fait basculer le choix d'architecture.

**Il n'y a pas de free-lunch intersectionnel.** Calibrer DPT uniquement sur
gender atteint ΔDP gender = 0.004 mais laisse ΔDP gender×age_group à 0.10,
voire le pousse plus haut sur FairGNN où on observe 0.154 sur l'axe croisé
alors que le marginal gender est à 0.009 — c'est l'effet whack-a-mole que
prédit l'argument intersectionnel de Crenshaw (1989). Si on veut traiter
les axes croisés, il faut explicitement construire l'attribut composite
et calibrer dessus, pas espérer que le débiaising mono-axe suffira.

**Sur la robustesse**, la chaîne post-hoc préserve la robustesse héritée du
modèle de base parce qu'elle ne modifie pas l'encoder : GraphSAGE reste
robuste à un bruit features σ ≤ 0.3 et à un edge drop ≤ 30 % (mesures
issues du repo amont) avant comme après l'application d'INLP+DPT. C'est un
avantage non-trivial des méthodes post-hoc sur l'in-training fairness, qui
modifie l'encoder et peut dégrader sa robustesse de façon imprévue.

## 5. Limites

**Limite philosophique.** Les métriques de fairness encodent toutes une
position normative implicite : ΔDP=0 incarne l'anti-classification stricte
(aucune corrélation avec le sensible n'est légitime), ΔEO=0 incarne une
position méritocratique (les corrélations conditionnées à la vérité sont
acceptables), la calibration par groupe assume que les écarts marginaux
sont OK tant que la prédiction est correcte par groupe. Choisir une
métrique, c'est choisir une éthique — et le théorème d'incompatibilité
montre qu'on ne peut pas toutes les satisfaire à la fois. Notre toolbox
fournit les outils, mais elle ne tranche pas le débat normatif.

**Importation culturelle des catégories.** La fairness ML hérite des
catégories du civil rights act US des années 60 (gender, race binaires)
et notre dataset les reproduit en réduisant `gender` et `region` à 0/1
sans documentation sémantique (Hoffmann 2019 ; Hanna et al. 2020).
Concrètement, le subset FairGNN de Pokec-z code `spoken_languages` via
8 indicateurs binaires — anglais, allemand, russe, français, espagnol,
italien, slovaque, japonais — qui sont tous des langues internationales
ou la langue majoritaire. Le hongrois (`madarsky`), le tchèque (`cesky`)
et le romani sont absents, alors que le dataset SNAP original les encode
en texte libre. La curation a probablement filtré ces langues parce
qu'elles ont un taux d'occurrence faible dans Žilinský kraj (0.13 %
d'hongrois, peu de gitans), mais le résultat est que le subset sur
lequel toute la littérature fairness-on-graphs travaille est, par
construction, aveugle aux axes ethniques.

**Limites techniques.** La garantie d'invariance d'INLP n'est valide que
contre un classifieur linéaire ; un MLP probe non-linéaire pourrait
recouvrir du signal résiduel. Côté FairGNN, le discriminateur adversarial
est binaire dans la formulation standard de Dai & Wang : pour faire de la
fairness sur `age_group` (3 classes) en in-training, il faudrait
redimensionner la tête de discriminateur, ce qu'on n'a pas implémenté.
Côté TabICL, ULTIMATE applique INLP sur `x` brut par simplicité ; or les
embeddings sont accessibles via `TabICLCache.row_repr`. Validation
multi-seed × Pokec-z/n : INLP sur embeddings tombe le leakage gender à
**0.61-0.63** vs 0.71 sur `x` brut, avec reproduction cross-dataset à
<0.02 près (`results/metrics/tabicl_inlp_embedding.csv`). Ré-injection
dans le prédicteur pour valider F1 retention : prochaine étape.

**Limite de généralisation.** Pokec-z et Pokec-n sont deux subsets du
même réseau slovaque, et la cible `completed_level_of_education_indicator`
est faiblement homophile en gender (`r ≈ -0.046`). Nos conclusions sur
"TabICL bat GraphSAGE" et "post-process suffit" sont conditionnelles à
cette propriété : sur un graphe fortement homophile à l'attribut
sensible, le classement s'inverserait probablement.

**Pour conclure.** Si on prend du recul sur l'ensemble du projet, le
choix d'axe sensible est probablement la limite la plus structurante.
Dans un contexte d'Europe centrale, et particulièrement en Slovaquie,
l'axe ethnique — minorité hongroise et surtout minorité gitane — est
beaucoup plus prévalent comme source de discrimination effective que
l'axe du sexe sur lequel on a passé la majeure partie de l'étude.
Logement, accès à l'éducation, embauche : c'est sur ces axes-là que les
disparités mesurables sont les plus fortes en Slovaquie. Pouvoir mesurer
la fairness sur cet axe-là aurait rendu l'analyse beaucoup plus utile
socialement, mais le dataset ne le permet pas. Notre travail montre donc
ce qu'on peut techniquement faire avec les outils disponibles, mais le
verrou principal n'est plus algorithmique — il est en amont, au niveau
de la collecte et de la curation des données. Tant qu'un dataset
fairness-on-graphs slovaque sans label ethnique reste le standard de
la littérature, l'évaluation des méthodes restera détachée des axes de
discrimination qui comptent vraiment dans le pays d'origine de la donnée.

**Outils d'IA** (mention exigée) : assistance algorithmique pour la
réimplémentation FairGNN-GRL, la migration pandas → polars, l'intégration
TabICL, l'ajout des modules INLP / calibration / reweighting, et la
rédaction. Code revu, testé, exécuté par les auteurs.

**Références.** Hardt-Price-Srebro 2016 ; Ravfogel et al. 2020 ; Ganin &
Lempitsky 2015 ; Dai & Wang 2021 ; Kamiran & Calders 2012 ; Chouldechova
2017 ; Crenshaw 1989 ; Hoffmann 2019 ; Hanna et al. 2020 (FAccT) ;
Newman 2003 ; Qu et al. 2025 (TabICL) ; Laclau et al. 2024.
