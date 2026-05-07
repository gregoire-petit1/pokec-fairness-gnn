"""Build a fresh, thin notebook that delegates all heavy lifting to
``scripts/main_experiment.py``. Cells are short and explanatory — the prof
runs ``Run All`` and gets the same numbers we get from the CLI.

Run from the repo root::

    python scripts/_build_thin_notebook.py

The output is written in place to ``notebooks/main_experiment.ipynb``.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("notebooks/main_experiment.ipynb")


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


CELLS = [
    md(
        """\
# Pokec-z — Fairness, Interprétabilité, Robustesse (mini-projet IADATA708)

Ce notebook **n'est qu'une coquille de présentation** : tout le code
exécutable est dans `scripts/main_experiment.py` et `src/`. La logique
expérimentale est testée (`pytest tests/`), lintée (`ruff check`) et
exécutable directement en CLI :

```bash
python scripts/main_experiment.py --device cuda:0
```

Les sections ci-dessous appellent les fonctions du module et affichent les
résultats. Pour l'analyse complète : `report/analysis_note.md`.
"""
    ),
    md("## 1. Setup"),
    code(
        """\
%load_ext autoreload
%autoreload 2

import os
import sys

# Allow `import scripts.main_experiment` from the notebooks/ working dir
sys.path.insert(0, os.path.abspath(".."))

import polars as pl
import torch

from scripts.main_experiment import (
    ExperimentConfig,
    build_sensitive_attrs,
    compute_multi_attr_fairness,
    load_data,
    make_masks,
    run_all,
    run_baseline,
    run_fairdrop,
    run_fairgnn_grid,
    run_resampling,
    run_tabicl,
    setup_seeds,
)
from src.data.splits import make_splits
from src.fairness.metrics import assortative_mixing_coefficient

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 42
print(f"Device: {DEVICE} | CUDA: {torch.cuda.is_available()} | "
      f"GPUs: {torch.cuda.device_count()}")
"""
    ),
    md("## 2. Configuration et données"),
    code(
        """\
cfg = ExperimentConfig.from_yaml("../configs/experiment.yaml")
print(cfg)

DATA_AVAILABLE = os.path.exists(os.path.join("..", cfg.raw_dir, "region_job_2.csv"))

if DATA_AVAILABLE:
    data = load_data(os.path.join("..", cfg.raw_dir), cfg.sensitive_cols, DEVICE)
    print(f"Nodes: {data.num_nodes:,} | Edges: {data.num_edges:,} | Features: {data.x.shape[1]}")
    print(f"r(gender)={assortative_mixing_coefficient(data.edge_index, data.gender):.3f}  "
          f"r(region)={assortative_mixing_coefficient(data.edge_index, data.region):.3f}")
else:
    print("Raw data not available — see README for download instructions.")
"""
    ),
    md(
        """\
## 3. Splits et attributs sensibles

5 attributs sensibles évalués pour chaque modèle : `gender`, `region`,
`age_group`, plus les intersections `gender × age_group` et
`gender × region`.
"""
    ),
    code(
        """\
if DATA_AVAILABLE:
    setup_seeds(SEED)
    train_idx, val_idx, test_idx = make_splits(
        data.num_nodes, data.y.cpu(), data.gender.cpu(),
        ratios=cfg.split_ratios, seed=SEED,
    )
    train_mask, val_mask, test_mask = make_masks(
        data.num_nodes, train_idx, val_idx, test_idx, DEVICE
    )
    sensitive_attrs = build_sensitive_attrs(data)
    print({k: tuple(v.shape) for k, v in sensitive_attrs.items()})
"""
    ),
    md("## 4. Pipeline complet (5 modèles × 5 attributs sensibles)"),
    code(
        """\
df = run_all(
    seed=SEED,
    device_str=str(DEVICE),
    cfg_path="../configs/experiment.yaml",
    raw_dir_override=os.path.join("..", cfg.raw_dir),
    out_csv="../results/metrics/comparison_full.csv",
)
df
"""
    ),
    md(
        """\
## 5. Lecture du tableau

Lignes : 5 modèles (`GraphSAGE`, `GraphSAGE+Resampling`, `GraphSAGE+FairDrop`,
`FairGNN(λ_best)`, `TabICL`).
Colonnes : ΔDP, ΔEO, AUC-gap, leakage AUC.

**Lectures clés** (voir `report/analysis_note.md` pour l'analyse complète) :

- L'écart `GraphSAGE − TabICL` chiffre la **discrimination structurelle ajoutée
  par le message passing** (l'homophilie du graphe).
- L'écart `FairGNN − TabICL` chiffre la **valeur de l'in-training fairness**
  *au-delà* de ce qu'un foundation model atteint déjà sur les features brutes.
- Les axes intersectionnels (`gender × …`) révèlent typiquement plus de
  disparité que les marginales seules — argument empirique contre l'analyse
  mono-axe.

Pour les figures (Pareto, robustness curves, GNNExplainer) : voir
``results/figures/`` — elles sont générées par les scripts dédiés
(`scripts/main_experiment.py`, `scripts/target_sweep.py`).
"""
    ),
    md(
        """\
## 6. Reproductibilité & traçabilité

- Code source : `src/` + `scripts/` (testés via `pytest`, lintés via `ruff`)
- Note d'analyse : `report/analysis_note.md`
- CSV de résultats : `results/metrics/comparison_full.csv`
- Garde-fous : `tests/test_no_pandas_no_loops.py` empêche toute régression vers
  pandas ou les boucles Python sur tenseurs.
"""
    ),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, indent=1))
    print(f"Wrote thin notebook → {NB_PATH} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
