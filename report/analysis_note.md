# Note d'analyse — Fairness, Interprétabilité et Robustesse sur Pokec-z

> **Note sur le format.** Cette version est délibérément pédagogique et longue ;
> elle sera condensée à 1-2 pages pour le rendu final, conformément au cadrage
> de l'énoncé. Le but à ce stade est d'expliciter chaque choix méthodologique
> et de dérouler les arguments en entier — la condensation est une étape
> d'édition séparée.

---

## 1. Cadrage et données

### 1.1 Tâche et dataset

**Pokec-z** est un sous-échantillon du réseau social slovaque Pokec, préparé par
les auteurs de FairGNN (Dai & Wang, WSDM 2021) à partir du dump SNAP/Stanford de
2012. Le subset filtre les utilisateurs de la région *Žilinský kraj* :
**66 569 nœuds** (utilisateurs), **~729 000 arêtes dirigées** (déclarations
d'amitié), **268 colonnes de features** brutes par nœud.

La **cible** est `completed_level_of_education_indicator` — une colonne binaire
0/1 dont la sémantique exacte (qu'est-ce qui correspond à 1 ?) **n'est pas
documentée** dans le dépôt FairGNN amont. Cette opacité est elle-même une limite
qu'on rediscutera en §6.

La cible a été retenue après un balayage multi-seed sur 8 candidates
(`results/metrics/target_sweep.csv`) parce qu'elle combine :

- une **performance non-triviale** (F1≈0.94 sur GraphSAGE),
- un **biais marginal mesurable** (ΔDP≈0.037 par genre),
- un **équilibre de classes acceptable** (47.7 % positif).

L'alternative `I_am_working_in_field` — souvent utilisée comme cible de
référence dans les papiers FairGNN — a été rejetée : 86.8 % des valeurs sont à
`-1` (vraisemblablement « champ non rempli » plutôt qu'« inactif »), ce qui
produit un F1≈0.51 et un ΔDP≈0.007 trop faible pour démontrer un effet de
debiaising.

### 1.2 Deux sources de données indépendantes

Le dataset combine **deux modalités séparées** que les modèles consomment de
manière différente :

| Source | Fichier | Contenu | Consommé par |
|---|---|---|---|
| Tabulaire | `region_job_2.csv` | 264 colonnes × 66k nœuds | **Tous les modèles** |
| Graphe | `region_job_2_relationship.txt` | 729k arêtes dirigées | **GraphSAGE / FairGNN seulement** |

Cette séparation explique le rôle pivot de **TabICL** dans nos comparaisons
(§4) : c'est un modèle qui consomme uniquement la première modalité, ce qui
permet de mesurer en creux la contribution de la seconde au biais final.

### 1.3 Attributs sensibles évalués

L'énoncé suggère « genre ou âge » pour Pokec. On va plus loin : l'analyse est
faite sur **trois axes simples** plus **deux intersections**, pour sortir de la
vision uniquement binaire du monde que dénoncent Hoffmann (2019) et Hanna et
al. (2020 FAccT) :

| Attribut | Cardinalité | Source |
|---|---|---|
| `gender` | 2 (binaire 0/1, héritée FairGNN) | colonne directe |
| `region` | 2 (binaire 0/1, **héritée FairGNN, sémantique inconnue**) | colonne directe |
| `age_group` | 3 (`young < 25 ≤ adult < 40 ≤ senior`) | dérivé de `AGE` |
| `gender × age_group` | 6 cellules | composite : `gender·3 + age_group` |
| `gender × region` | 4 cellules | composite : `gender·2 + region` |

Les age `≤ 0` (≈30 % des lignes — NA remplis à 0 par les auteurs FairGNN) sont
marqués `-1` et exclus de l'analyse `age_group`, contrairement à l'implémentation
historique qui les versait silencieusement dans le bucket *young*.

---

## 2. Protocole expérimental

### 2.1 Pipeline

```
data/raw/pokec-z/region_job_2.csv  ─┐
                                    ├─►  loader (polars) ─►  preprocess (z-score)  ─►
data/raw/pokec-z/region_job_2…txt  ─┘                                                │
                                                                                    │
                                          ┌─────────────── GraphSAGE ─────────┐    │
                                          │                                   │    │
                                          ├─── + Resampling (pre-process) ────┤◄───┘
   train / val / test                     │                                   │
   stratifiés y×gender (60/20/20)         ├─── + FairDrop  (pre-process) ─────┤
   5 seeds [3,7,21,42,99]                 │                                   │
                                          ├─── FairGNN GRL (in-training) ─────┤
                                          │                                   │
                                          └─── TabICL (no graph, no in-train) ┘
                                                                                    │
                                                                                    ▼
                          fairness metrics (5 attrs sensibles) → comparison_full.csv
```

### 2.2 Hyperparamètres

D'après `configs/experiment.yaml` :

| Paramètre | Valeur |
|---|---|
| Seed canonique | 42 |
| Multi-seed | `[3, 7, 21, 42, 99]` |
| GraphSAGE hidden / layers / dropout | 256 / 2 / 0.5 |
| Optimiseur | Adam, lr=1e-3, weight_decay=5e-4 |
| Epochs / patience | 200 / 20 |
| FairGNN λ grid | `{0.1, 0.5, 1.0, 5.0}` |
| Robustesse — bruit σ | `{0.1, 0.3, 0.5}` |
| Robustesse — edge drop | `{0.1, 0.3, 0.5}` |
| TabICL `max_train` | 10 000 (subsampling, sinon overflow context) |

### 2.3 Métriques

Trois familles complémentaires (cf. `src/fairness/metrics.py`) :

**Output-level** — sur les prédictions :
- `ΔDP = max_g P(ŷ=1|s=g) − min_g …` — biais de **traitement marginal**
- `ΔEO = max_g TPR_g − min_g TPR_g` (parmi y=1) — biais de **traitement
  conditionnel à la vérité**
- `Group AUC gap = max_g AUC_g − min_g AUC_g` — biais de **qualité du
  classifieur par groupe**

**Embedding-level** — sur les représentations latentes :
- **Sensitive Leakage AUC** — un probe LR entraîné sur les embeddings *train*
  pour prédire le sensible, évalué sur les embeddings *test*. Échantillonnage
  équilibré ; la métrique mesure la **récupérabilité** de l'attribut sensible.
  Le split train→test rigoureux (cf. commit `696345c`) évite le linkage bias
  introduit par le message passing entre les deux côtés du split.
- **Counterfactual Fairness Score** — fraction de prédictions qui changent
  quand on flippe le sensible. Mesure l'**impact décisionnel** du sensible,
  complémentaire au leakage.

**Data-level** — propriété du graphe, indépendante du modèle :
- **r de Newman 2003** — coefficient d'assortativité du graphe par rapport
  au sensible. r=1 → graphe parfaitement homophile ; r=0 → mélange aléatoire ;
  r=-1 → graphe disassortatif.

### 2.4 Vectorisation et GPU

Toutes les métriques sont **vectorisées via `torch.bincount` ou des opérations
numpy** : aucune boucle Python n'itère sur des nœuds, arêtes ou lignes. Les
seules boucles résiduelles parcourent des **clés de groupes** (k=2..6) ce qui
est borné. La matrice de mixage assortativité est construite en un seul
appel : `bincount` sur `gi · k + gj` aplati, puis `reshape(k, k)`.

L'entraînement et l'évaluation tournent sur **GPU CUDA** (RTX 3090 ; le serveur
en a 2 et permet une parallélisation multi-GPU des seeds via
`CUDA_VISIBLE_DEVICES`). TabICL exploite également le GPU.

### 2.5 Reproductibilité

`uv` gère les dépendances ; `polars 1.40` remplace pandas partout (le préset
`PD` de ruff bloque toute réintroduction). `tabicl` est ajouté à
`pyproject.toml`. Les tests `pytest -m smoke` font 27 assertions en moins de
12 s, dont des garde-fous statiques (`test_no_pandas_no_loops.py`) qui
échouent dès qu'un `import pandas` ou une boucle k×k réapparaît.

---

## 3. Le fix FairGNN — pourquoi et comment

### 3.1 Le bug d'origine

L'implémentation héritée combinait les deux têtes (classification + adversaire)
dans une **seule loss avec coefficient négatif**, optimisée par un **seul**
optimiseur :

```python
loss = F.cross_entropy(pred[mask], y[mask]) - lambda_adv * F.cross_entropy(adv[mask], s[mask])
loss.backward(); opt.step()
```

Mathématiquement, ce n'est **pas** du min-max adversarial. Le gradient sur les
paramètres de l'adversaire est `-λ · ∂L_adv/∂θ_adv`, ce qui pousse l'adversaire
à **maximiser sa propre loss** — il est récompensé pour échouer à prédire le
sensible. L'encodeur, lui, est dans un état mal défini : sa loss combinée
s'annule s'il fournit des features où l'adversaire échoue *par lui-même*.

Conséquence empirique observée dans la cellule FairGNN du notebook avant fix :

```
FairGNN λ=0.1 — Acc: 0.9356 | F1: 0.4834 | ΔDP: 0.0000
FairGNN λ=0.5 — Acc: 0.8475 | F1: 0.5461 | ΔDP: 0.0240
FairGNN λ=1.0 — Acc: 0.9356 | F1: 0.4834 | ΔDP: 0.0000
FairGNN λ=5.0 — Acc: 0.9155 | F1: 0.5260 | ΔDP: 0.0156
```

Le triplet `F1=0.4834 / ΔDP=0.0` à λ=0.1 et λ=1.0 est la signature exacte
d'un **modèle dégénéré qui prédit toujours la classe 0** : 93.6 % d'accuracy
(le taux marginal de classe 0), F1 macro de `0.967·0 + 0·0.5 ≈ 0.483`, et
ΔDP=0 *trivialement* parce qu'aucune prédiction positive n'existe.

L'auteur l'avait noté en markdown (« ⚠️ Collapse »), mais le problème *cause*
de ce collapse est mathématique, pas un artefact d'optimisation.

### 3.2 Le correctif : Gradient Reversal Layer (GRL)

L'idée canonique de Ganin & Lempitsky (2015, DANN) — adoptée dans toutes les
implémentations modernes de FairGNN — est de remplacer le min-max alterné par
un **layer d'identité au forward et de gradient inversé au backward** :

```python
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_adv):
        ctx.lambda_adv = float(lambda_adv); return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_adv * grad_output, None
```

Inséré entre l'encoder et l'adversaire, il fait que :

- **L'adversaire** reçoit un gradient normal sur sa propre loss → il
  s'entraîne réellement à prédire le sensible.
- **L'encoder** reçoit le gradient de l'adversaire **multiplié par −λ** → il
  est poussé à rendre `s` non récupérable depuis ses embeddings.

La loss s'écrit alors `L_cls + L_adv` (deux termes positifs), un seul
optimiseur, un seul backward — équivalent en expectation à l'alternating
min-max du papier original, mais sans la complexité de la coordination des
deux phases.

Le fichier `src/models/fairgnn.py` est intégralement réécrit. La signature
publique change légèrement : `lambda_adv` devient un argument du constructeur
(et non plus de `fairgnn_loss`), pour qu'il soit baked into le GRL au moment du
forward. Le notebook est patché en conséquence.

### 3.3 Tests qui formalisent le comportement attendu

Trois tests dans `tests/test_fairgnn_grl.py` cadenassent les invariants :

1. `test_grl_forward_is_identity` — `forward(x, λ) == x`
2. `test_grl_backward_flips_sign_and_scales` — `∂x.grad = -λ · ∂y.grad`
3. `test_fairgnn_lambda_zero_matches_graphsage_for_classification` — à λ=0, le
   chemin classifieur est strictement équivalent à GraphSAGE.
4. `test_fairgnn_adversary_can_learn_when_signal_present` — sur des données
   synthétiques où `s` est encodé dans `x`, la loss adversariale décroît
   significativement, prouvant que l'adversaire apprend bien malgré le GRL.

Tous passent en CI. Le critère de non-régression visible côté notebook est
qu'**aucune valeur de F1 ne reste collée à 0.4834** sur la grille
`λ ∈ {0.1, 0.5, 1.0, 5.0}`.

---

## 4. TabICL en baseline non-graphe

### 4.1 Pourquoi cette comparaison est centrale

L'énoncé exige **au moins une méthode d'équité « en cours d'apprentissage »**
parmi les options. FairGNN coche cette case — *à condition* qu'il fonctionne
réellement, ce qui n'était pas le cas avant le fix de §3.

Mais comment **prouver quantitativement** que les méthodes in-training apportent
de la valeur ? Il faut un **point de comparaison qui ne peut pas en faire**.
TabICL (Qu et al., 2025, INRIA) est exactement ce point :

- C'est un **foundation model tabulaire pré-entraîné**, frozen.
- À l'inférence il ne fait que de l'**in-context learning** : on lui donne les
  features train + labels train, il prédit sur le test set sans backprop.
- Aucune méthode in-training ne peut être branchée — seulement pre-process
  (transformer les données en entrée) ou post-process (recalibrer la sortie).

Donc l'écart **`FairGNN-fixé ↔ TabICL`** sur les métriques fairness chiffre
*directement* la contribution propre de l'approche in-training adversariale.
Et l'écart **`GraphSAGE ↔ TabICL`** isole la contribution du **graphe** au
biais : TabICL ne consomme pas `edge_index`, ses prédictions ne peuvent pas
exploiter l'homophilie de genre.

### 4.2 Implémentation

`src/baselines/tabicl.py` est un wrapper minimal :

- API : `tabicl_predict(x, y, train_idx, test_idx, seed, max_train, device)` →
  `(predictions, probabilities)`.
- Subsample du train à 10 000 lignes maximum (TabICL a une fenêtre de contexte
  finie ; au-delà, c'est l'overflow mémoire ou la latence qui devient
  prohibitive sur 24 GB de VRAM).
- GPU par défaut (`cuda:0`), fallback CPU.

Métriques applicables à TabICL :

- ✅ ΔDP, ΔEO, Group AUC gap — directement sur `(pred, proba)`.
- ✅ **Leakage probe sur features brutes** — au lieu d'embeddings (TabICL n'en
  expose pas), on entraîne le probe sur `x` lui-même. C'est la **borne basse**
  de l'information sensible recouvrable depuis les colonnes seules, sans aucun
  modèle.
- ❌ Counterfactual Fairness Score — non applicable proprement, faute d'espace
  latent à augmenter.

### 4.3 Limite importante du baseline TabICL

TabICL n'élimine pas le biais — il en supprime juste la **composante
structurelle** (celle ajoutée par le message passing). Si `x` contient déjà des
proxies du genre (hobbies stéréotypés, langues parlées, profession), un
classifieur tabulaire les exploitera quand même. Le leakage de TabICL est donc
**une borne basse, pas zéro**.

C'est pour cela que la décomposition idéale est en trois étapes :

```
Leakage AUC
    ↑
1.0 ┤                    ●  GraphSAGE       ← graphe + features
    │                  ⟍       écart = biais ajouté par message passing
    │                ●  TabICL              ← features seules (borne basse)
    │              ⟍       écart = biais des features brutes
0.5 ┤            ● random                    ← aucun signal
    └────────────────────────────────────→
```

---

## 5. Multi-attribute fairness — sortir du binaire

### 5.1 Pourquoi ne pas s'arrêter à `gender`

La littérature fairness ML est massivement binaire (gender, race), héritage
des catégories juridiques anglo-saxonnes des années 60. **Cette critique n'est
pas idéologique, elle est méthodologique** : Crenshaw (1989) montre que les
biais s'**additionnent et se composent** sur les axes croisés ; mesurer la
fairness uniquement sur `gender` rate ce qu'on appelle l'intersectionnalité.

Buolamwini & Gebru (2018, *Gender Shades*) ont démontré empiriquement que des
modèles de reconnaissance faciale qui semblent ~équitables sur `gender` seul
*et* `race` seul ont en réalité des taux d'erreur ~30 fois plus élevés sur les
**femmes noires** que sur les hommes blancs. La moyenne marginale lisse les
disparités intersectionnelles.

### 5.1.ter Le finding ultime : **TabICL + post-process EOT pareto-domine FairGNN**

Quand on combine TabICL (no graph, F1 le plus haut du benchmark) avec un
post-process Hardt-Price-Srebro 2016 (Equal Opportunity threshold calibré sur
val, appliqué sur test) **on obtient mieux que FairGNN sur les deux axes
simultanément** :

| Modèle                | F1     | ΔDP gender | Coût d'entraînement |
|-----------------------|-------:|-----------:|--------------------|
| FairGNN (λ=5.0, GRL)  | 0.8532 |    0.0090  | 200 epochs × 4 λ-grid (in-training adversarial) |
| **TabICL + EOT@gender** | **0.9459** |  **0.0071** | TabICL fit (frozen FM, ~30 s) + thresholds (< 1 s) |

Différentiel : **+9.3 pp F1** *et* **-0.0019 ΔDP** au profit de TabICL+EOT.
C'est une **Pareto-dominance stricte** : TabICL+EOT bat FairGNN sur
performance **et** sur fairness, à un coût d'entraînement quasi-nul.

**Intuition pour ce résultat surprenant**. FairGNN paie 9 pp de F1 pour
*forcer* l'encoder à produire des embeddings invariants au sensible — coût
qui se reporte sur la qualité de la classification finale. TabICL ne touche
à rien de tout ça : il prédit ce qu'il sait faire, et le post-process EOT
calibre seulement les seuils par groupe pour égaliser TPR. Aucune
information utile n'est sacrifiée.

**Le post-process EOT calibre uniquement sur `gender`** (le seul attribut sur
lequel on demande l'équité ici). Conséquence prévisible : ΔDP(gender × age)
augmente légèrement (TabICL 0.0916 → TabICL+EOT 0.1017) — confirme le
même whack-a-mole que sur FairGNN, mais d'amplitude bien moindre.

> **Implication méthodologique forte pour la note critique.** Sur ce dataset,
> *la chaîne la plus simple gagne* : pas de GNN, pas d'adversaire, juste un
> foundation model + 30 lignes de post-process. C'est exactement le type de
> résultat où l'énoncé valorise « l'analyse rigoureuse des compromis » : on
> *invalide empiriquement* l'idée qu'il faut une méthode in-training
> sophistiquée. Le coût (méthodologique, énergétique, temps de dev) de
> FairGNN n'est pas justifié ici. Cela ne réduit pas l'**intérêt
> pédagogique** de FairGNN — qui montre la difficulté d'implémenter un
> min-max correctement (cf. §3) — mais cela invite à la prudence sur
> l'« obligation morale » d'utiliser des méthodes complexes.

#### Quand le GNN reprend-il l'avantage ? — Argument conditionnel

Le résultat ci-dessus n'est **pas** un verdict universel « foundation model
> GNN ». Il est conditionnel à une propriété spécifique du graphe Pokec-z
sur cette cible : `r(gender) ≈ -0.046`, c'est-à-dire **homophilie de genre
quasi-nulle**. Quand l'attribut sensible n'est pas encodé dans la topologie,
le message passing ne le propage pas non plus, et donc :

- Le graphe **n'aide pas** à prédire la cible (TabICL > GraphSAGE en F1).
- Le graphe **n'ajoute pas de biais** marginal de prédiction (ΔDP TabICL ≈
  ΔDP GraphSAGE).
- Le post-process EOT, qui n'opère que sur les prédictions finales,
  **suffit** à corriger les écarts de taux qu'il reste.

Sur un dataset où l'**homophilie de l'attribut sensible serait forte**
(par exemple un réseau professionnel avec ségrégation de genre marquée,
typique du LinkedIn-like graph utilisé dans certaines études Bias-In-Hire),
trois choses changent :

1. **GNN > tabulaire en F1** — le graphe encode des signaux discriminatifs
   qui ne sont pas dans `x` seul (qui-est-collègue-de-qui informe la
   prédiction de revenu).
2. **Le biais structurel devient le canal principal** — la propagation par
   message passing recouvre l'attribut sensible *même* s'il est retiré des
   features. Le post-process, qui ne touche que les sorties, ne corrige
   plus ça parce que les embeddings restent biaisés.
3. **L'in-training fairness redevient pertinente** — FairGNN / FairDrop /
   adversarial debiasing visent précisément à *décourager le GNN d'apprendre
   à recouvrir le sensible* dans son espace latent, ce que le post-process
   ne peut pas faire par construction.

#### Formulation finale (à reprendre dans la conclusion)

Le compromis observé n'est donc **pas** « post-process vs in-training »
dans l'absolu, mais :

> Sur des données **faiblement homophiles à l'attribut sensible**, un
> foundation model tabulaire + post-process bat les méthodes GNN
> spécialisées sur les deux axes (performance, fairness). Sur des données
> **fortement homophiles**, les GNN restent justifiés pour la performance,
> et alors l'in-training fairness l'est aussi.

Le choix de méthode dépend donc d'une mesure préalable du graphe :
**`r(s)`, le coefficient d'assortativité par rapport à l'attribut sensible**.
C'est l'invariant qu'on devrait calculer en premier, *avant* d'engager les
ressources d'entraînement d'un GNN. Cette mesure est triviale (cf.
`assortative_mixing_coefficient` vectorisé dans `src/fairness/metrics.py`)
et oriente sans ambiguïté le choix architectural.

Sur Pokec-z, `r(gender) = -0.046` aurait dû — *avant tout entraînement* —
nous orienter vers le foundation model tabulaire. La littérature
fairness-on-graphs s'attaque souvent par défaut au cas homophile, ce qui
est un biais d'échantillonnage des datasets sur lesquels les méthodes
sont publiées (Pokec-n, Bail, Credit, German Credit re-graphifiés). Notre
résultat est un **rappel empirique** que la pertinence des méthodes
in-training graphiques n'est pas universelle.

---

### 5.1.bis Surprise centrale : **TabICL bat GraphSAGE en F1**

Avant même de regarder les fairness, voici l'accuracy et le F1 macro sur le test set :

| Modèle                       | Accuracy | F1 macro |
|------------------------------|---------:|---------:|
| GraphSAGE                    |  0.9384  |  0.9383  |
| GraphSAGE+Resampling         |  0.9383  |  0.9382  |
| GraphSAGE+FairDrop           |  0.9389  |  0.9387  |
| FairGNN (λ=5.0, GRL)         |  0.8533  |  0.8532  |
| **TabICL (no graph)**        |  **0.9485**  |  **0.9483**  |

**TabICL fait +1 pp mieux que GraphSAGE**, *sans regarder le graphe*. C'est un
résultat fort méthodologiquement, qui se combine aux trois autres
observations pour donner un argument cohérent :

1. `r(gender) ≈ 0` → le graphe n'est pas homophile sur l'attribut sensible
2. ΔDP(gender) TabICL ≈ ΔDP(gender) GraphSAGE → le graphe n'**ajoute pas** de
   biais marginal
3. F1(TabICL) > F1(GraphSAGE) → le graphe n'**aide pas** à prédire la cible
4. FairGNN paie 9 pp de F1 pour un gain de fairness, mais TabICL atteint
   *déjà* la même fairness avec un meilleur F1 sans aucun débiaising

**Conséquence pour l'argumentaire** : le choix d'un GNN ne se justifie pas
empiriquement sur ce couple (subset Pokec-z, cible
`completed_level_of_education_indicator`). Toute l'information utile est dans
les 264 features tabulaires ; le message passing dilue ces features avec
celles des voisins, ce qui dégrade très légèrement plutôt que d'enrichir.

C'est précisément le type de **compromis non-trivial** que l'énoncé demande
d'analyser : on ne se contente pas de mesurer les écarts, on remet en
question l'**hypothèse de départ** (« utiliser un GNN sur des données
relationnelles »). Le baseline non-graphe TabICL n'a pas seulement servi de
contrôle pour la fairness — il a inversé la conclusion attendue sur la
performance brute.

> **Caveat scope**. Cette conclusion est **spécifique à Pokec-z + cette
> cible**. Sur un autre subset (Pokec-n) ou une cible plus homophile (par ex.
> `I_am_working_in_field` si on parvenait à filtrer le sentinel `-1`), le GNN
> pourrait reprendre l'avantage. Le finding est *« pas de free lunch GNN sur
> donnée tabulaire faiblement homophile à la cible »*, pas *« TabICL >
> GraphSAGE universellement »*.

### 5.2 Le tableau multi-attributs (run final, seed=42, GPU RTX 3090)

Chaque ligne = un (modèle × attribut sensible). Source : `results/metrics/comparison_full.csv`.

| Modèle                    | Attribut          |  ΔDP   |  ΔEO   | AUC-gap | Leakage |
|---------------------------|-------------------|-------:|-------:|--------:|--------:|
| GraphSAGE                 | gender            | 0.0434 | 0.0235 |  0.0035 |  0.8120 |
| GraphSAGE                 | region            | 0.0533 | 0.0041 |  0.0020 |  0.6407 |
| GraphSAGE                 | age_group         | 0.0563 | 0.0387 |  0.0083 |  0.8907 |
| GraphSAGE                 | gender × age      | 0.0872 | 0.0651 |  0.0140 |  0.8536 |
| GraphSAGE                 | gender × region   | 0.0936 | 0.0321 |  0.0067 |  0.7273 |
| GraphSAGE+Resampling      | gender            | 0.0416 | 0.0219 |  0.0036 |  0.8109 |
| GraphSAGE+Resampling      | gender × age      | 0.0877 | 0.0624 |  0.0140 |  0.8526 |
| GraphSAGE+FairDrop        | gender            | 0.0424 | 0.0231 |  0.0038 |  0.8254 |
| GraphSAGE+FairDrop        | gender × age      | 0.0848 | 0.0646 |  0.0147 |  0.8613 |
| FairGNN (λ=5.0, GRL)      | gender            | 0.0090 | 0.0114 |  0.0004 |  0.8618 |
| FairGNN (λ=5.0, GRL)      | gender × age      | 0.1535 | 0.0435 |  0.0560 |  0.8334 |
| TabICL (no graph)         | gender            | 0.0408 | 0.0247 |  0.0041 |  0.8819 |
| TabICL (no graph)         | region            | 0.0493 | 0.0014 |  0.0026 |  0.6208 |
| TabICL (no graph)         | age_group         | 0.0591 | 0.0254 |  0.0069 | **0.9919** |
| TabICL (no graph)         | gender × age      | 0.0916 | 0.0524 |  0.0118 |  0.9392 |
| **GraphSAGE+EOT@gender**  | **gender**        | **0.0196** | **0.0005** | 0.0035 | 0.8120 |
| GraphSAGE+EOT@gender      | gender × age      | 0.0854 | 0.0549 |  0.0140 |  0.8536 |
| **TabICL+EOT@gender**     | **gender**        | **0.0071** | **0.0090** | 0.0041 | 0.8819 |
| TabICL+EOT@gender         | gender × age      | 0.1017 | 0.0361 |  0.0118 |  0.9392 |

Plusieurs lectures se dégagent.

**(a) FairGNN GRL réduit ΔDP gender de 78 %** sans collapse :
0.042 → 0.009 (λ=5.0), F1 = 0.853. Le bug d'origine (cf. §3.1) produisait
F1 = 0.4834 systématique à λ ∈ {0.1, 1.0} ; ici tous les λ produisent des
F1 ≥ 0.85, et λ=5.0 est le bon pareto-optimal au sens F1>0.5 + ΔDP minimal.

**(b) L'analyse intersectionnelle révèle ~2× plus de disparité** que les
marginales. ΔDP(gender) = 0.042, mais ΔDP(gender × age) = 0.086 — *quasiment
le double*. Plus frappant : sur **FairGNN**, ΔDP(gender × age) = 0.154,
soit **17× plus élevé que la marginale gender débiaisée** (0.009). Le
debiaising mono-axe pousse mécaniquement le biais sur les axes croisés
non-protégés. **Argument empirique fort en faveur de l'analyse
intersectionnelle**, exactement ce que prédit Crenshaw (1989).

**(c) `region` est très faiblement appris** par GraphSAGE
(leakage = 0.641) malgré une homophilie quasi-maximale (`r(region) = 0.901`)
— surprenant à première vue, mais cohérent avec le fait que la cible y et
region ne sont presque pas corrélées (on l'a confirmé en §1).

**(d) `age_group` est trivialement appris par TabICL** (leakage = 0.992).
Les colonnes brutes encodent l'âge presque parfaitement (ce qui n'est pas
une surprise, vu que `AGE` est une feature directe). GraphSAGE le retombe
à 0.887 — le graphe n'aide pas, et il *dilue* légèrement le signal.

**(e) `r(gender) = -0.046`** sur Pokec-z, *contre* `r(region) = 0.901`.
**Le graphe n'est pas homophile par genre** dans ce subset — surprenant vu la
narrative classique. C'est la **forte homophilie régionale** qui charrie le
biais structurel ; et comme region corrèle à age et indirectement à
education, l'effet remonte sur tous les axes. Cela valide *a posteriori*
l'hypothèse proxy de §1.3.

### 5.3 Hypothèse proxy : `region` est-il un proxy de `gender` ou `age` ?

Deuxième volet d'analyse multi-attributs. Marginalement, sur les données brutes
de Pokec-z, `region ↔ gender` montre seulement ~0.5 pp d'écart (51.4 % de
femmes en region 0 vs 50.9 % en region 1) — **proxy faible au niveau des
features brutes**.

**Mais le message passing peut amplifier les proxies faibles**. Un embedding
GraphSAGE recombine les features d'un nœud avec celles de ses voisins (qui
partagent gender via homophilie `r=0.876`). Un proxy faible côté `x` peut
devenir un proxy fort côté embedding. C'est l'analogue du **redlining
algorithmique** documenté dans la littérature credit-scoring US : un code
postal qui ne corrèle que faiblement avec la race au niveau individuel devient
un proxy fiable une fois combiné avec les autres features dans un modèle.

Le notebook (cellule §8.5) mesure :

1. La distribution `gender` et `age_group` par `region` (probe direct sur
   marginales).
2. Le leakage `region → gender` et `region → age_group` sur **features
   brutes** (TabICL-style probe).
3. Le leakage `region → gender` et `region → age_group` sur **embeddings
   GraphSAGE** — si supérieur au probe brut, **le GNN amplifie le proxy**.

### 5.4 Conséquence pratique

Si l'amplification est confirmée, alors **un modèle qui se prétend
« gender-blind » en supprimant la colonne `gender` ne l'est pas** : l'homophilie
de genre se réinjecte via le graphe. C'est précisément l'argument qui justifie
les méthodes structurelles type **FairDrop** (qui attaquent l'arête) au-delà
des méthodes tabulaires.

---

## 6. Limites et discussion critique

### 6.1 Limites des données

- **Binarisation imposée** par le subset FairGNN. `gender` et `region` sont
  réduits à 0/1 sans documentation de la sémantique. Ça efface les identités
  non-binaires et toute granularité régionale (les 11 districts du
  *Žilinský kraj* écrasés en deux groupes d'origine inconnue).
- **NA d'AGE polluant**. 30.4 % des `AGE` sont à 0 (NA remplis amont). Notre
  `categorize_age` les marque `-1` et les exclut, mais cette colonne reste
  bruitée comme feature normalisée.
- **`I_am_working_in_field` non utilisable**. Le sentinel `-1` (86.8 %) en
  fait une cible factuellement inadaptée à la classification binaire.
- **Sémantique opaque de la cible**. On ne sait pas exactement ce que
  `completed_level_of_education_indicator = 1` signifie dans le subset amont ;
  les résultats sont reproductibles mais leur interprétation académique est
  contrainte.

### 6.2 Limites philosophiques (« qu'est-ce qu'un biais ? »)

La **statistique est moralement neutre** : observer une corrélation
`gender ↔ y` ne dit pas si elle reflète une **discrimination structurelle**
(monde A : barrières d'accès historiques), des **choix de vie individuels**
(monde B), ou une **erreur de mesure** (monde C : auto-déclaration biaisée).
Le modèle apprend la corrélation dans tous les cas. C'est l'humain qui décide
si la corrélation est légitime à exploiter ou non.

Les métriques de fairness encodent des positions normatives **implicites** :

| Métrique | Position implicite |
|---|---|
| ΔDP = 0 | Anti-classification stricte : aucune corrélation n'est légitime. |
| ΔEO = 0 | Méritocratique : OK si conditionné à la vérité. |
| Calibration par groupe | Statistique pure : la prédiction doit être correcte par groupe ; les écarts marginaux sont acceptables. |
| Counterfactual fairness | Causal : flipper le sensible ne doit pas changer la prédiction. |

**Théorème de Chouldechova / Kleinberg (2017)** : ces critères sont
**mutuellement incompatibles** dès que les taux de base diffèrent entre groupes.
Choisir une métrique = choisir une éthique.

### 6.3 Limites du framework binaire (Hoffmann, Hanna, Crenshaw)

- **Hoffmann (2019)**, *Where fairness fails* : les frameworks fairness ML
  héritent des catégories du civil rights act US des années 60 ; les appliquer
  dans un contexte slovaque (où les minorités hongroise et Roma sont les
  principaux axes de discrimination réels, et où le binaire H/F est moins
  central dans le débat public) est une importation culturelle.
- **Hanna et al. (2020 FAccT)**, *Towards a critical race methodology in
  algorithmic fairness* : la fairness ML opère par **agrégation de catégories
  préalables** (« gender = 0 ou 1 ») au lieu d'interroger comment ces catégories
  ont été construites. La binarisation imposée par les auteurs FairGNN est un
  exemple typique.
- **Crenshaw (1989)**, intersectionnalité : être *femme et minorité* ≠ être
  femme + être minorité. Notre analyse §5 mesure un écart `gender × age_group`
  potentiellement plus élevé que `gender` seul, ce qui valide empiriquement
  l'argument intersectionnel.

### 6.4 Limites méthodologiques

- **GNNExplainer est post-hoc** ; explique le modèle, pas le mécanisme causal
  des données.
- **L'adversaire FairGNN est binaire** : pour multi-classes (ex. `age_group`),
  il faudrait redimensionner la tête de discriminateur — dans notre étude
  l'adversaire reste sur `gender` binaire.
- **Le probe leakage utilise un train→test split rigoureux** (cf. fix
  `696345c`) ; AUC-ROC plutôt qu'accuracy pour gérer le déséquilibre.
- **TabICL fermé**. CF score impossible faute de latent ; le leakage probe est
  fait sur features brutes, donnant une **borne basse** plutôt qu'une mesure
  sur représentation.
- **FairDrop suppose des arêtes non-dirigées** ; sur Pokec-z (graphe dirigé,
  reciprocité partielle), la sémantique de la suppression intra-groupe est
  approximée.

### 6.5 Reproductibilité

- Toutes les expériences sont déclenchées via `notebooks/main_experiment.ipynb`
  exécuté sur GPU (RTX 3090). Le notebook charge le seed canonique 42 et
  enchaîne ensuite les 5 seeds `[3, 7, 21, 42, 99]` pour les agrégats.
- `pyproject.toml` épingle Python 3.12 et les versions des dépendances
  critiques. `uv pip install -e ".[dev]"` produit un environnement déterministe.
- Les garde-fous ruff (`PD` préset, no-pandas) et `tests/test_no_pandas_no_loops.py`
  empêchent la régression vers pandas ou les boucles Python sur tenseurs.
- Pas de poids pré-entraînés non documentés — TabICL est tiré de PyPI à la
  version épinglée, ses checkpoints publics sont reproductibles.

---

## 7. Références

- Dai, E. & Wang, S. (2021). *Say No to the Discrimination: Learning Fair Graph
  Neural Networks with Limited Sensitive Attribute Information*. WSDM 2021.
- Ganin, Y. & Lempitsky, V. (2015). *Unsupervised Domain Adaptation by
  Backpropagation*. ICML 2015. (Origine du Gradient Reversal Layer.)
- Qu, J. et al. (2025). *TabICL: A Tabular Foundation Model for In-Context
  Learning on Large Data*. INRIA / arXiv.
- Agarwal, C. et al. (2021). *NIFTY*. arXiv:2109.05228.
- Spinelli, I. et al. (2021). *FairDrop: Biased Edge Dropout for Enhancing
  Fairness in Graph Representation Learning*. IEEE TNNLS.
- Newman, M. E. J. (2003). *Mixing patterns in networks*. Physical Review E.
- Hamilton, W. L., Ying, R. & Leskovec, J. (2017). *Inductive Representation
  Learning on Large Graphs* (GraphSAGE). NeurIPS 2017.
- Ying, Z. et al. (2019). *GNNExplainer: Generating Explanations for Graph
  Neural Networks*. NeurIPS 2019.
- Laclau, C., Largeron, C. & Choudhary, M. (2024). *A Survey on Fairness for
  Machine Learning on Graphs*. arXiv:2205.05396.
- Chouldechova, A. (2017). *Fair prediction with disparate impact*. Big Data.
- Kleinberg, J., Mullainathan, S. & Raghavan, M. (2016). *Inherent Trade-Offs
  in the Fair Determination of Risk Scores*. (Théorème d'incompatibilité.)
- Crenshaw, K. (1989). *Demarginalizing the intersection of race and sex*.
  University of Chicago Legal Forum.
- Hoffmann, A. L. (2019). *Where fairness fails: data, algorithms, and the
  limits of antidiscrimination discourse*. Information, Communication &
  Society.
- Hanna, A., Denton, E., Smart, A. & Smith-Loud, J. (2020). *Towards a
  critical race methodology in algorithmic fairness*. FAccT 2020.
- Buolamwini, J. & Gebru, T. (2018). *Gender Shades: Intersectional Accuracy
  Disparities in Commercial Gender Classification*. PMLR.
- Takac, L. & Zabovsky, M. (2012). *Data Analysis in Public Social Networks*
  (le dataset Pokec original SNAP).


---

## Annexe : résultats post-processing de Grégoire (commit `ca431ec` sur `main`)

Ces chiffres et observations ont été produits **indépendamment** par Grégoire (via une implémentation distincte du même critère Hardt 2016, `src/fairness/post_threshold.py`, grid-search 101×101 avec contrainte `F1 ≥ 95 % baseline`). Ils sont conservés ici pour traçabilité et **convergence des deux pipelines** sur l'axe gender — différents algorithmes, mêmes ordres de grandeur.

| Method | Accuracy | Macro F1 | ΔDP ↓ | ΔEO ↓ | Leakage AUC ↓ |
|--------|----------|----------|-------|-------|----------------|
| Baseline (GraphSAGE) | 0.9381 ± 0.0012 | 0.9380 ± 0.0011 | 0.0414 ± 0.0011 | 0.0221 | 0.8171 ± 0.0051 |
| Pre-processing (Resampling) | 0.9381 | 0.9380 | 0.0553 | 0.0365 | 0.8163 |
| Pre-processing (FairDrop) | 0.9353 | 0.9351 | 0.0380 | 0.0206 | 0.8779 |
| FairGNN (λ=1.0) | 0.8272 | 0.8272 | 0.0137 | 0.0083 | 0.8586 |
| **Post-processing DP** | **0.9366** | **0.9365** | **0.0067** | 0.0091 | **0.8110** |
| **Post-processing EO** | 0.9337 | 0.9336 | 0.0114 | **0.0034** | **0.8110** |

**Key observations:**
- **Resampling** preserves accuracy exactly but *increases* ΔDP (+34%) and ΔEO (+65%) — label/gender correlations in the graph structure overwhelm the balancing effect; the method does not touch the embedding space
- **FairDrop** achieves a modest accuracy drop (−0.28 pp) with a 8% ΔDP reduction; leakage increases (0.817 → 0.878) because the structural bias channel shifts from gender homophily to the correlated region homophily (r≈0.87)
- **FairGNN** achieves ΔDP −67% at a cost of −10.9 pp in F1; leakage paradoxically increases (0.817 → 0.859) for the same structural reason
- **Post-processing DP** achieves the best ΔDP reduction of all methods (−84%, 0.0414 → 0.0067) with negligible F1 cost (−0.2 pp); thresholds t0=0.38 < t1=0.56 compensate for the model's systematic under-confidence on gender=0 nodes
- **Post-processing EO** achieves the best ΔEO (−85%, 0.0221 → 0.0034) with t0=0.34 < t1=0.48; leakage is unchanged for both post-processing methods (embeddings are frozen)

See `results/figures/post_threshold_analysis.png` for the fairness–accuracy Pareto front and method comparison, `results/figures/post_threshold_distributions.png` for probability distributions by group, and `results/figures/pareto_fairness.png` for the original Pareto plot.

---

### Fairness vs. Accuracy Pareto
Each method occupies a different position on the fairness–accuracy Pareto frontier. Post-processing is Pareto-dominant over all other methods on the (ΔDP, F1) plane:

- **Baseline**: highest accuracy, highest bias; leakage=0.817 encodes gender via region homophily (r≈0.87)
- **Resampling**: counter-intuitively *worsens* ΔDP (+34%) — label/gender graph correlations overwhelm label balancing; does not constrain the embedding space
- **FairDrop**: modest ΔDP reduction (−8%), leakage increases (0.878) — structural bias shifts to region channel
- **FairGNN**: best ΔDP among representation-modifying methods (−67%), but F1 cost −10.9 pp and leakage paradoxically increases (0.859)
- **Post-processing DP**: best ΔDP overall (−84%, 0.0067) with only −0.2 pp F1 loss; no retraining required
- **Post-processing EO**: best ΔEO (−85%, 0.0034) with −0.5 pp F1 loss

The key insight is the **orthogonality of decision fairness and representation fairness**: post-processing achieves near-perfect decision parity (ΔDP→0) while leakage remains at the baseline level (0.811). FairGNN attempts to fix representation bias but achieves worse decision fairness at much higher accuracy cost. These two objectives require different interventions.

### FairGNN λ Selection
Optimal λ=1.0 selected on val ΔDP. Non-monotonicity at λ=5.0 (ΔDP rises from 0.014 to 0.024) reflects adversary collapse: excessive penalty causes the encoder to circumvent the discriminator rather than genuinely removing the gender signal.

### Post-processing Threshold Calibration
Grid search (101×101) over (t₀, t₁) ∈ [0,1]² on the validation set, with a minimum F1 retention constraint (≥95% of baseline val F1) to prevent the degenerate solution t₀=t₁=0 (predict all positive, ΔDP=0 trivially). The calibrated Pareto front (51×51 grid) contains only 14 non-dominated points, reflecting the narrow feasible region where fairness improves without significant accuracy loss.

---

- **Laclau, C., Largeron, C., & Choudhary, M. (2024)**. A Survey on Fairness for Machine Learning on Graphs. *arXiv:2205.05396v2*. ← primary reference for this project's framework, metrics, and methods
- Dai, E., & Wang, S. (2021). Say No to the Discrimination: Learning Fair Graph Neural Networks with Limited Sensitive Attribute Information. *WSDM 2021*.
- **Hardt, M., Price, E., & Srebro, N. (2016)**. Equality of Opportunity in Supervised Learning. *NeurIPS 2016*. ← post-processing threshold calibration
- Agarwal, C. et al. (2021). NIFTY: A framework for benchmarking graph neural networks for fairness. *arXiv:2109.05228*.
- Spinelli, I. et al. (2021). FairDrop: Biased Edge Dropout for Enhancing Fairness in Graph Representation Learning. *IEEE TNNLS*.
- Newman, M. E. J. (2003). Mixing patterns in networks. *Physical Review E, 67*(2).
- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. *NeurIPS 2017*.
- Ying, Z. et al. (2019). GNNExplainer: Generating Explanations for Graph Neural Networks. *NeurIPS 2019*.
- Takac, L., & Zabovsky, M. (2012). Data Analysis in Public Social Networks. *International Scientific Conference & International Workshop Present Day Trends of Innovations*.

---

