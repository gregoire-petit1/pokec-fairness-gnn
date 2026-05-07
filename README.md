# pokec-fairness-gnn — branche `feature/fairgnn-fix-and-multi-fairness`

Node classification on the **Pokec-z** social network with fairness analysis,
GNNExplainer interpretability, and robustness evaluation.

Mini-projet pour le cours **IADATA708 — Algorithmic Fairness, Interpretability
and Robustness**.

> **Cette branche** réimplémente FairGNN avec un Gradient Reversal Layer (le
> code historique combinait `l_cls − λ·l_adv` dans un seul optimiseur, ce qui
> n'est pas du min-max adversarial et provoquait un collapse F1=0.4834 à
> certains λ), migre l'ensemble du pipeline pandas → polars, vectorise les
> métriques fairness, ajoute un baseline non-graphe **TabICL** (foundation
> model tabulaire) et étend l'analyse à 5 attributs sensibles incluant des
> intersections (`gender × age_group`, `gender × region`).

---

## Tâche

Classification binaire de nœuds : prédire
`completed_level_of_education_indicator` (déjà 0/1 dans le subset FairGNN).
Cible sélectionnée parmi 8 candidates via un sweep multi-seed
(`results/metrics/target_sweep.csv`) — F1≈0.94, ΔDP≈0.037 sur la baseline.

**Attributs sensibles évalués** : `gender`, `region`, `age_group` plus les
intersections `gender × age_group` et `gender × region` — pour sortir de
l'analyse mono-axe critiquée par Hoffmann (2019), Hanna et al. (2020 FAccT)
et Crenshaw (1989).

---

## Méthodes

| Composant | Implémentation |
|---|---|
| Baseline | GraphSAGE (2 couches, hidden=256) |
| Baseline non-graphe (contrôle) | **TabICL** (foundation model tabulaire, INRIA 2025) |
| Fairness — pre-process | Oversampling par groupe `label × gender` |
| Fairness — pre-process | **FairDrop** (suppression biaisée d'arêtes intra-groupe) |
| Fairness — in-training | **FairGNN avec Gradient Reversal Layer** (Ganin 2015) |
| Interprétabilité | GNNExplainer (PyG) |
| Robustesse | Bruit features (Gaussien) + edge drop |

---

## Métriques

| Métrique | Description |
|---|---|
| ΔDP | Demographic Parity Difference (max-min sur groupes) |
| ΔEO | Equal Opportunity Difference (TPR par groupe, conditionné y=1) |
| Group AUC gap | Max-min de l'AUC par groupe |
| Leakage AUC | Probe LR train-set → test-set, AUC-ROC ; balanced sampling. Multi-class via OvR macro |
| CF score | Counterfactual fairness (NIFTY-style) — sensible flippé puis re-prédit |
| r de Newman | Coefficient d'assortativité du graphe par rapport au sensible |

---

## Architecture

```
pokec-fairness-gnn/
├── configs/experiment.yaml             # Hyperparamètres
├── src/                                # Code testé et linté
│   ├── data/{loader,preprocessing,splits}.py
│   ├── models/{graphsage,trainer,fairgnn}.py     # FairGNN avec GRL
│   ├── fairness/{metrics,resampling,fairdrop}.py # vectorisé, polars-backed
│   ├── baselines/tabicl.py             # foundation model wrapper (no graph)
│   ├── interpretability/explainer.py
│   └── robustness/perturbations.py
├── scripts/
│   ├── main_experiment.py              # ★ Driver end-to-end (CLI ou import)
│   ├── _build_thin_notebook.py         # Régénère le notebook (coquille)
│   ├── target_sweep.py                 # Choix de la target via grid 8×5
│   └── run_main.sh
├── notebooks/main_experiment.ipynb     # Coquille mince qui appelle main_experiment
├── tests/                              # 40 tests pytest, dont des garde-fous lint
├── report/analysis_note.md             # Note d'analyse (pédagogique → ≤2 p)
└── results/
    ├── figures/                        # EDA, Pareto, robustness, GNNExplainer
    └── metrics/{comparison_full,target_sweep,results_summary}.csv
```

---

## Quickstart

**Requis** : Python 3.12, [`uv`](https://github.com/astral-sh/uv), GPU CUDA
recommandé (testé sur 2× RTX 3090).

```bash
# 1. Clone et environnement
git clone https://github.com/gregoire-petit1/pokec-fairness-gnn.git
cd pokec-fairness-gnn
git checkout feature/fairgnn-fix-and-multi-fairness
uv venv .venv --python 3.12
uv pip install -e ".[dev]"

# 2. Données — subset FairGNN de Pokec-z
mkdir -p data/raw/pokec-z
# Télécharger region_job_2.csv et region_job_2_relationship.txt depuis :
#   https://github.com/EnyanDai/FairGNN/tree/main/dataset/pokec
# et les placer dans data/raw/pokec-z/

# 3. Tests (smoke = <12s, full = ~25s)
.venv/bin/pytest tests/ -m smoke
.venv/bin/pytest tests/

# 4. Lint (no-pandas + no-loops + ruff PD preset)
.venv/bin/ruff check src/ tests/ scripts/

# 5. Pipeline complet en CLI (5 modèles × 5 attributs sensibles)
.venv/bin/python scripts/main_experiment.py --device cuda:0
# → écrit results/metrics/comparison_full.csv

# 6. Notebook (coquille mince, n'exécute que main_experiment.run_all)
.venv/bin/jupyter notebook notebooks/main_experiment.ipynb
```

---

## Contraintes (enforced via tests)

- **No pandas anywhere** (`tests/test_no_pandas_no_loops.py::test_no_pandas_imports_anywhere`
  + ruff `PD` preset).
- **No Python loops on tensors / data rows** (`test_assortative_mixing_has_no_double_for_loop`
  + revues ad-hoc des nouveaux fichiers).
- **GPU CUDA par défaut** ; multi-GPU possible via `CUDA_VISIBLE_DEVICES` et
  multiprocessing pour paralléliser les seeds.

---

## Références

- Dai, E. & Wang, S. (2021). *Say No to the Discrimination*. WSDM.
- Ganin, Y. & Lempitsky, V. (2015). *Unsupervised Domain Adaptation by
  Backpropagation* (origine du Gradient Reversal Layer).
- Qu, J. et al. (2025). *TabICL: A Tabular Foundation Model for In-Context
  Learning on Large Data*. INRIA.
- Hamilton, W. L., Ying, R. & Leskovec, J. (2017). *Inductive Representation
  Learning on Large Graphs* (GraphSAGE). NeurIPS.
- Spinelli, I. et al. (2021). *FairDrop*. IEEE TNNLS.
- Newman, M. E. J. (2003). *Mixing patterns in networks*. Physical Review E.
- Hoffmann (2019), Hanna et al. (2020 FAccT), Crenshaw (1989) pour les
  limites des frameworks fairness binaires (cf. `report/analysis_note.md`).

