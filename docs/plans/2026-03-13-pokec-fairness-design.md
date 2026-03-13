# Design — IADATA708 Mini-Project
**Algorithmic Fairness, Interpretability & Robustness on Pokec-z**
Date: 2026-03-13

---

## 1. Objective

Node classification task on the Pokec-z social network dataset.
The goal is to predict a user's **region** (geographic location) while analyzing
fairness constraints with respect to two sensitive attributes: **gender** and **age**.

This project covers all four requirements from the course:
- Baseline model
- Fairness method (x2: pre-processing + in-training)
- Interpretability tool (GNNExplainer)
- Robustness evaluation (feature noise + edge removal)

---

## 2. Dataset

| Property | Value |
|----------|-------|
| Dataset | Pokec-z (official subsample) |
| Nodes | ~67k users |
| Task | Node classification (predict region) |
| Sensitive attribute 1 | `gender` (binary: 0/1) |
| Sensitive attribute 2 | `age` categorized: young (<25), adult (25–40), senior (>40) |

**Splits:** 60/20/20 train/val/test, stratified on label AND gender. Saved as artifacts.

**Preprocessing:**
- Remove sensitive attributes from input features
- One-hot encode categorical features
- Normalize continuous features (z-score)

---

## 3. Project Structure

```
project/
├── data/
│   ├── raw/pokec-z/          # READ ONLY
│   └── splits/               # Saved train/val/test indices
├── notebooks/
│   └── main_experiment.ipynb # Main deliverable
├── src/
│   ├── data/                 # Loading, preprocessing, splits
│   ├── models/               # GraphSAGE, FairGNN
│   ├── fairness/             # Metrics + fairness methods
│   ├── interpretability/     # GNNExplainer wrapper
│   └── robustness/           # Controlled perturbations
├── configs/
│   └── experiment.yaml       # Centralized hyperparameters
├── results/
│   └── figures/, metrics/
└── report/
    └── analysis_note.md      # 1-2 page report
```

---

## 4. Models & Methods

### 4.1 Baseline — GraphSAGE
- 2 layers, mean aggregation
- hidden_dim=256, dropout=0.5, lr=1e-3, epochs=200
- Early stopping on val F1-macro (patience=20)
- Sensitive attributes excluded from input features

### 4.2 Fairness Method 1 — Pre-processing: Resampling
- Oversample minority nodes per group (gender × age category) in the training set
- No architecture change
- Serves as a lightweight fairness baseline

### 4.3 Fairness Method 2 — In-training: FairGNN (adversarial)
- Architecture: shared GraphSAGE encoder + main classifier + adversarial discriminator on sensitive attribute
- Loss: `L_total = L_classification - λ * L_adversarial`
- Grid search on λ ∈ {0.1, 0.5, 1.0, 5.0} over val set
- Reference: Dai & Wang, 2021 — *Say No to the Discrimination: Learning Fair GNNs*

### 4.4 Interpretability — GNNExplainer
- Edge mask + feature mask per prediction
- Comparative analysis: do explanations differ across sensitive groups?
- Visualization of explanation subgraphs for representative nodes per group

### 4.5 Robustness
| Perturbation | Levels |
|---|---|
| Gaussian noise on features | σ ∈ {0.1, 0.3, 0.5} |
| Random edge removal | 10%, 30%, 50% |

Evaluation: track degradation of both accuracy and ΔDP.
Key question: *Is fairness more fragile than performance under perturbation?*

---

## 5. Metrics

### Performance
- Accuracy
- F1-macro

### Fairness (computed per sensitive attribute)
- **ΔDP** — Demographic Parity Difference
- **ΔEO** — Equal Opportunity Difference
- **Group AUC gap**

### Summary visualization
- Pareto curves: accuracy vs ΔDP for all 3 configurations
- Robustness curves: metric degradation vs perturbation level

---

## 6. Experimental Protocol

| Parameter | Value |
|-----------|-------|
| Random seed | 42 (all: torch, numpy, random) |
| Package manager | uv + pyproject.toml (Python 3.12) |
| Framework | PyTorch Geometric (PyG) |
| Split strategy | Stratified on label + gender |
| Evaluation | Val set for model selection, test set for final report only |

---

## 7. Notebook Structure

1. Setup & imports
2. Data loading + EDA (sensitive attribute distributions, graph homophily)
3. Baseline GraphSAGE
4. Pre-processing (resampling) + evaluation
5. FairGNN + λ grid search + evaluation
6. GNNExplainer analysis
7. Robustness experiments
8. Synthesis: Pareto curves + comparative table

---

## 8. Analysis Note Outline (1-2 pages)

1. **Experimental protocol** — splits, metrics, hyperparameters
2. **Key results** — accuracy/ΔDP/ΔEO table per method
3. **Trade-off analysis** — FairGNN at optimal λ vs baseline, Pareto discussion
4. **Limitations** — homophily not addressed, Pokec collection bias, GNNExplainer post-hoc nature

---

## 9. Grading Coverage

| Criterion | Weight | Coverage |
|-----------|--------|----------|
| Implementation + reproducibility | 40% | Fixed seed, saved splits, self-contained notebook |
| Experimental protocol quality | 20% | Stratified splits, val-based selection, config file |
| Results analysis + trade-off discussion | 30% | Pareto curves, λ sensitivity, robustness vs fairness |
| Clarity of analysis note | 10% | Structured 4-section report |
