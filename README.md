# pokec-fairness-gnn

Node classification on the **Pokec-z** social network with fairness analysis, GNNExplainer interpretability, and robustness evaluation.

Mini-project for the course **IADATA708 — Algorithmic Fairness, Interpretability and Robustness**.

---

## Task

Binary node classification: predict whether a user is employed in a field (`I_am_working_in_field > 0`).  
Sensitive attribute: `gender` (binary, 0/1).

---

## Methods

| Component | Method |
|-----------|--------|
| Baseline | GraphSAGE (2 layers, hidden=256) |
| Fairness — pre-processing | Oversampling by label × gender group |
| Fairness — in-training | FairGNN adversarial debiasing (λ grid search) |
| Interpretability | GNNExplainer — per-group feature importance |
| Robustness | Feature noise (Gaussian) + edge drop |

---

## Fairness Metrics

| Metric | Description |
|--------|-------------|
| ΔDP | Demographic Parity Difference |
| ΔEO | Equal Opportunity Difference |
| AUC gap | Max AUC difference across gender groups |
| Leakage AUC | AUC-ROC of a logistic regression probe trained on train-set embeddings and evaluated on test-set embeddings to predict gender — avoids linkage bias |

---

## Results

| Method | Accuracy | Macro F1 | ΔDP | ΔEO | Leakage AUC |
|--------|----------|----------|-----|-----|-------------|
| GraphSAGE baseline | 92.78% ± 0.22% | 51.72% ± 0.37% | 0.0013 | 0.0187 | ~0.73 |
| Resampling | 92.08% | 52.11% | 0.0106 | 0.0356 | — |
| FairGNN (λ=0.5) | 84.75% | 54.61% | 0.0240 | 0.0379 | — |

> Multi-seed evaluation over 5 seeds `[3, 7, 21, 42, 99]`. See notebook for full results and Pareto plots.

---

## Project Structure

```
pokec-fairness-gnn/
├── configs/
│   └── experiment.yaml          # All hyperparameters
├── src/
│   ├── data/
│   │   ├── loader.py            # Pokec-z CSV → PyG Data
│   │   ├── preprocessing.py     # Feature normalization, sensitive attr extraction
│   │   └── splits.py            # Stratified train/val/test splits
│   ├── models/
│   │   ├── graphsage.py         # GraphSAGE backbone
│   │   ├── trainer.py           # Training loop with early stopping
│   │   └── fairgnn.py           # FairGNN adversarial debiasing
│   ├── fairness/
│   │   ├── metrics.py           # ΔDP, ΔEO, AUC gap, leakage probe
│   │   └── resampling.py        # Oversample by label × sensitive group
│   ├── interpretability/
│   │   └── explainer.py         # GNNExplainer wrapper
│   └── robustness/
│       └── perturbations.py     # Feature noise + edge drop
├── notebooks/
│   └── main_experiment.ipynb   # Full reproducible experiment
├── tests/                       # 18 unit tests (pytest)
├── report/
│   └── analysis_note.md        # Experimental analysis (1-2 pages)
└── results/
    ├── figures/                 # EDA, Pareto, robustness, leakage plots
    └── metrics/                 # results_summary.csv
```

---

## Quickstart

**Requirements:** Python 3.12, [uv](https://github.com/astral-sh/uv)

```bash
# 1. Clone and set up environment
git clone https://github.com/gregoire-petit1/pokec-fairness-gnn.git
cd pokec-fairness-gnn
uv venv .venv --python 3.12
uv pip install torch torchvision torch-geometric scikit-learn pandas numpy \
               matplotlib seaborn pyyaml pytest ipykernel

# 2. Download data (FairGNN subset of Pokec-z)
# Place region_job_2.csv and region_job_2_relationship.txt in:
mkdir -p data/raw/pokec-z
# Download from: https://github.com/EnyanDai/FairGNN/tree/main/dataset/pokec

# 3. Run tests
.venv/bin/pytest tests/ -v

# 4. Register Jupyter kernel and open notebook
.venv/bin/python -m ipykernel install --user --name pokec-fairness
jupyter notebook notebooks/main_experiment.ipynb
```

---

## Data

The raw data files are **not included** in this repository (too large for git).  
Download the Pokec-z FairGNN subset from the [EnyanDai/FairGNN](https://github.com/EnyanDai/FairGNN/tree/main/dataset/pokec) repository:

- `region_job_2.csv` — node features (66,569 nodes, 266 columns)
- `region_job_2_relationship.txt` — edge list (729,129 edges)

Place both files in `data/raw/pokec-z/`.

---

## References

- Dai, E., & Wang, S. (2021). [Say No to the Discrimination: Learning Fair Graph Neural Networks with Limited Sensitive Attribute Information](https://arxiv.org/abs/2009.01454). *WSDM 2021*.
- Agarwal, C. et al. (2021). [NIFTY: A framework for benchmarking graph neural networks for fairness](https://arxiv.org/abs/2109.05228). *arXiv*.
- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216). *NeurIPS 2017*.
- Ying, Z. et al. (2019). [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894). *NeurIPS 2019*.
