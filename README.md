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
> model tabulaire) et étend l'analyse à **5 attributs sensibles** (gender,
> region, age_group + intersections gender×age et gender×region) avec
> reproduction sur **Pokec-n**. Le livrable principal est
> `report/2_pager.pdf`.

---

## Findings principaux

1. **La toolbox de fairness n'est pas substituable**. Chaque famille de
   méthode attaque une métrique : **DPT** baisse ΔDP, **EOT** baisse ΔEO,
   **INLP** baisse le leakage. Aucun outil seul ne traite les trois.
   La composition `INLP_composite + DPT_composite` règle simultanément
   ΔDP, ΔEO et leakage sur 5 axes sensibles.
2. **TabICL+EOT Pareto-domine FairGNN** sur Pokec-z : F1 = 0.946 vs 0.853,
   ΔDP gender = 0.007 vs 0.009 — un foundation model tabulaire frozen plus
   30 lignes de post-process bat la méthode in-training adversariale sur
   les deux axes, à coût d'entraînement quasi-nul.
3. **GraphSAGE s'écrase sur la composition multi-axes** (F1 0.93 → 0.59),
   TabICL tient (F1 0.93 → 0.87). Mécanisme : `r(region) = 0.901` rend les
   embeddings GraphSAGE saturés en signal regional ; INLP les démolit.
   Règle pratique : **mesurer `r(s)` (assortativité Newman) avant tout
   entraînement GNN** pour choisir entre tabulaire et graphe.

Cf. `report/2_pager.pdf` pour les chiffres complets, multi-seed, cross-dataset.

---

## Tâche

Classification binaire de nœuds : prédire
`completed_level_of_education_indicator` (déjà 0/1 dans le subset FairGNN).
Cible sélectionnée parmi 8 candidates via un sweep multi-seed
(`results/metrics/target_sweep.csv`) — F1 ≈ 0.94, ΔDP ≈ 0.04 sur la baseline.

**Attributs sensibles évalués** : `gender`, `region`, `age_group`, plus les
intersections `gender × age_group` et `gender × region` — pour sortir de
l'analyse mono-axe critiquée par Hoffmann (2019), Hanna et al. (2020 FAccT)
et Crenshaw (1989).

---

## Méthodes

| Famille | Implémentation |
|---|---|
| Baseline graphe | GraphSAGE (2 couches, hidden=256, dropout=0.5) |
| Baseline non-graphe (contrôle) | **TabICL** (foundation model tabulaire INRIA 2025, frozen) |
| Pre-process | Resampling par groupe `label × gender` |
| Pre-process | **FairDrop** (Spinelli et al. 2021) |
| Pre-process | **Reweighting Kamiran-Calders** (2012) |
| In-training | **FairGNN avec Gradient Reversal Layer** (Dai-Wang 2021 + Ganin 2015) |
| Post-process | **DPT** Demographic Parity Threshold (calibrage par groupe) |
| Post-process | **EOT** Equal Opportunity Threshold (Hardt-Price-Srebro 2016) |
| Post-process | **INLP** Iterative Nullspace Projection (Ravfogel 2020), embeddings + features |
| Post-process | **Temperature scaling** (Guo 2017) |
| Composition | `INLP_composite + DPT_composite` sur attribut joint à 12 cellules — règle 5 axes simultanément |
| Interprétabilité | GNNExplainer (PyG) |
| Robustesse | Bruit features (Gaussien) + edge drop |

---

## Métriques

| Métrique | Description |
|---|---|
| ΔDP | Demographic Parity Difference (max-min sur groupes) |
| ΔEO | Equal Opportunity Difference (TPR par groupe, conditionné y=1) |
| Group AUC gap | Max-min de l'AUC par groupe |
| **Sensitive Leakage AUC** | Probe LR train→test sur embeddings (ou features), AUC-ROC. Multi-class via OvR macro. Référence : Laclau et al. 2024 |
| CF score | Counterfactual fairness (NIFTY-style) — sensible flippé puis re-prédit |
| `r` Newman | Coefficient d'assortativité du graphe vs sensible — pivot pour choisir GNN ou tabulaire |

---

## Architecture

```
pokec-fairness-gnn/
├── configs/experiment.yaml
├── src/
│   ├── data/{loader,preprocessing,splits}.py        # polars + numpy + torch
│   ├── models/{graphsage,trainer,fairgnn}.py        # FairGNN avec GRL
│   ├── fairness/
│   │   ├── metrics.py                               # ΔDP, ΔEO, AUC gap, leakage, r-Newman
│   │   ├── resampling.py
│   │   ├── fairdrop.py
│   │   ├── reweighting.py                           # Kamiran-Calders
│   │   └── post_threshold.py                        # grid-search seuils par groupe
│   ├── postprocess/
│   │   ├── equal_opportunity.py                     # DPT et EOT (Hardt 2016)
│   │   ├── inlp.py                                  # Ravfogel 2020
│   │   └── calibration.py                           # Guo 2017
│   ├── baselines/
│   │   ├── tabicl.py                                # wrapper foundation model
│   │   └── tabicl_inlp_embedding.py                 # INLP sur row_repr cache
│   ├── interpretability/explainer.py
│   └── robustness/perturbations.py
├── scripts/
│   ├── main_experiment.py                           # ★ Driver end-to-end multi-axes
│   ├── run_main.sh / run_multi_seed.sh
│   ├── aggregate_multi_seed.py                      # Agrège les seeds en stats
│   ├── target_sweep.py                              # Choix de la cible
│   ├── plot_figures.py / plot_pareto.py             # Toutes les figures du PDF
│   ├── build_pdf.py                                 # Construit report/2_pager.pdf
│   └── run_tabicl_inlp_embedding.py                 # Validation INLP sur embeddings TabICL
├── notebooks/main_experiment.ipynb                  # Coquille → main_experiment.run_all
├── tests/                                           # 50+ pytest, ruff PD préset, no-pandas/no-loops
├── report/
│   ├── 2_pager.pdf / 2_pager.md                     # ★ Livrable principal
│   └── analysis_note.md                             # Note longue (référence interne)
└── results/
    ├── figures/{fig1_toolbox,fig2_chain,fig3_cross_dataset,...}.png
    ├── metrics/{comparison_full,multiseed_summary,pokec_n_seed42,...}.csv
    └── cache/                                       # Cache torch.save des ModelOutput
```

---

## Quickstart

**Requis** : Python 3.12, [`uv`](https://github.com/astral-sh/uv), GPU CUDA
recommandé (testé sur 2× RTX 3090).

```bash
# 1. Clone + environnement
git clone https://github.com/gregoire-petit1/pokec-fairness-gnn.git
cd pokec-fairness-gnn
git checkout feature/fairgnn-fix-and-multi-fairness
uv venv .venv --python 3.12
uv pip install -e ".[dev]"

# 2. Données — subset FairGNN
mkdir -p data/raw/pokec-z data/raw/pokec-n
# Pokec-z : region_job_2.csv  + region_job_2_relationship.txt
# Pokec-n : region_job.csv    + region_job_relationship.txt
# Source : https://github.com/EnyanDai/FairGNN/tree/main/dataset/pokec

# 3. Tests (smoke <12 s ; full ~30 s)
.venv/bin/pytest tests/ -m smoke
.venv/bin/pytest tests/

# 4. Lint (no-pandas + no-loops + ruff PD préset)
.venv/bin/ruff check src/ tests/ scripts/
.venv/bin/ruff format --check src/ tests/ scripts/

# 5. Pipeline end-to-end (5 modèles × 5 attributs sensibles, ~10 min sur 3090)
.venv/bin/python scripts/main_experiment.py --device cuda:0 --cache
# → results/metrics/comparison_full.csv

# 6. Multi-seed (5 seeds [3,7,21,42,99], parallèle 2 GPU)
bash scripts/run_multi_seed.sh
.venv/bin/python scripts/aggregate_multi_seed.py
# → results/metrics/comparison_multiseed_summary.csv

# 7. Reproduction Pokec-n (cross-dataset)
.venv/bin/python scripts/main_experiment.py --raw-dir data/raw/pokec-n \
  --device cuda:1 --out comparison_pokec_n_seed42.csv

# 8. Régénérer le 2-pager (figures + PDF)
.venv/bin/python scripts/plot_figures.py
.venv/bin/python scripts/build_pdf.py
# → report/2_pager.pdf

# 9. Validation embeddings TabICL (INLP sur row_repr du cache, multi-seed × Pokec-z/n)
.venv/bin/python scripts/run_tabicl_inlp_embedding.py
# → results/metrics/tabicl_inlp_embedding.csv
```

---

## Contraintes (enforced via tests)

- **No pandas anywhere** (`tests/test_no_pandas_no_loops.py::test_no_pandas_imports_anywhere`
  + ruff `PD` préset).
- **No Python loops on tensors / data rows** (`test_assortative_mixing_has_no_double_for_loop`
  + revues ad-hoc des nouveaux fichiers).
- **GPU CUDA par défaut** ; multi-GPU possible via `CUDA_VISIBLE_DEVICES` et
  multiprocessing pour paralléliser les seeds.

---

## Références

- Hardt, M., Price, E. & Srebro, N. (2016). *Equality of Opportunity in
  Supervised Learning*. NeurIPS.
- Ravfogel, S. et al. (2020). *Null It Out: Guarding Protected Attributes by
  Iterative Nullspace Projection*. ACL.
- Ganin, Y. & Lempitsky, V. (2015). *Unsupervised Domain Adaptation by
  Backpropagation* — origine du Gradient Reversal Layer.
- Dai, E. & Wang, S. (2021). *Say No to the Discrimination*. WSDM.
- Kamiran, F. & Calders, T. (2012). *Data preprocessing techniques for
  classification without discrimination*. KAIS.
- Chouldechova, A. (2017) ; Kleinberg, J. et al. (2017). Théorèmes
  d'incompatibilité ΔDP / ΔEO / calibration.
- Qu, J. et al. (2025). *TabICL: A Tabular Foundation Model for In-Context
  Learning on Large Data*. INRIA.
- Hamilton, W. L., Ying, R. & Leskovec, J. (2017). *Inductive Representation
  Learning on Large Graphs* (GraphSAGE). NeurIPS.
- Spinelli, I. et al. (2021). *FairDrop*. IEEE TNNLS.
- Newman, M. E. J. (2003). *Mixing patterns in networks*. Physical Review E.
- Laclau, C. et al. (2024). *A Survey on Fairness for Machine Learning on
  Graphs*. arXiv:2205.05396.
- Guo, C. et al. (2017). *On Calibration of Modern Neural Networks*. ICML.
- Hoffmann (2019), Hanna et al. (2020 FAccT), Crenshaw (1989) pour les
  limites des frameworks fairness binaires (cf. `report/2_pager.md` §5).
