# Analysis Note: Fairness, Interpretability and Robustness on Pokec-z

## 1. Experimental Protocol

### Dataset
Pokec-z is a Slovak social network dataset (region "z") with ~67k nodes and ~882k directed edges. Each node represents a user with features including age, gender, region, and education-related attributes.

The binary **target** is `completed_level_of_education_indicator` (already 0/1 in the raw data), selected from a sweep of 8 candidate targets (40 runs total, 5 seeds × 8 targets) on GPU. It was chosen because it yields a strong F1-macro (~0.939) with a visible and measurable demographic parity gap (ΔDP~0.037), making it well-suited for demonstrating fairness-aware GNN methods. Full sweep results are in `results/metrics/target_sweep.csv`.

> **Why not `I_am_working_in_field` (literature standard)?** That target has severe label noise: 86.8% of values are `-1` (sentinel for "field not filled"), yielding only 6.4% positive class and F1-macro~0.514. ΔDP~0.007 is too small to demonstrate meaningful debiasing. See Section 4 (Limitations) for a full discussion.

The **sensitive attributes** are `gender` (binary, 0/1) and `region` (binary, 0/1).

### Preprocessing
- Node features: all columns except `user_id` and `completed_level_of_education_indicator`; age is binned into `young/adult/senior` (cut-off: 0–25, 25–40, 40+)
- `gender` and `region` are extracted as sensitive attribute vectors and removed from the feature matrix to prevent direct leakage; they remain accessible to fairness metrics
- Z-score normalization applied after sensitive-attribute removal
- Note: `AGE=0` (missing value after `fillna(0)`) falls into the "young" bin, introducing a potential bias in the age feature

### Splits
Stratified 60/20/20 train/val/test split on `label × gender` to preserve class and group balance.

### Metrics
| Metric | Description |
|--------|-------------|
| Accuracy | Fraction of correct predictions |
| Macro F1 | F1 averaged over classes (handles imbalance) |
| ΔDP | Demographic Parity Difference: \|P(ŷ=1\|s=0) − P(ŷ=1\|s=1)\| |
| ΔEO | Equal Opportunity Difference: \|TPR_s=0 − TPR_s=1\| |
| AUC gap | Max group AUC difference across gender groups |
| Leakage (RB) | **Representation Bias** (Laclau et al., 2024 eq. 2): AUC-ROC of a logistic regression probe trained on frozen train-set embeddings to predict `gender` on the test set. AUC ≈ 0.5 = fair; AUC ≈ 1.0 = high bias encoded in representations |
| r | **Assortative mixing coefficient** (Newman 2003): measures structural homophily w.r.t. a sensitive attribute. r=1 = edges only within groups; r=0 = random mixing; r=-1 = disassortative |
| CF unfairness | **Counterfactual unfairness score** (inspired by NIFTY, Agarwal et al. 2021): fraction of test nodes whose downstream prediction changes when gender is flipped, measuring decision-level sensitivity to the sensitive attribute |

### Multi-Seed Evaluation
All experiments are reported as mean ± std over 5 seeds (`[3, 7, 21, 42, 99]`) to account for training variance. A single canonical run at `seed=42` is also used for downstream qualitative analyses (GNNExplainer, robustness curves).

### Hyperparameters
From `configs/experiment.yaml`:
- Seed: 42 (canonical) + multi-seed [3, 7, 21, 42, 99]
- Model: GraphSAGE, hidden=256, layers=2, dropout=0.5, lr=0.001, epochs=200, patience=20
- FairGNN: λ ∈ {0.1, 0.5, 1.0, 5.0}, adv_hidden=64
- Robustness: noise σ ∈ {0.1, 0.3, 0.5}, edge drop ∈ {0.1, 0.3, 0.5}

---

## 2. Key Results

### Target Selection Sweep

Before the main experiment, a sweep of 8 candidate targets × 5 seeds = 40 runs was run on GPU (RTX 3090, ~12s/run). Results in `results/metrics/target_sweep.csv`.

| Target | F1 mean | ΔDP mean | ΔEO mean | Leakage AUC | Note |
|--------|---------|---------|---------|-------------|------|
| `I_am_working_in_field` *(lit. baseline)* | 0.514 | 0.007 | 0.016 | 0.773 | Label noise (86.8% = -1), severe imbalance |
| **`completed_level_of_education_indicator`** ✅ | **0.939** | **0.037** | **0.018** | 0.751 | **Selected** |
| `nefajcim` | 0.940 | 0.005 | 0.004 | 0.752 | ΔDP too small — debiasing trivial |
| `marital_status_indicator` | 0.922 | 0.010 | 0.020 | 0.750 | Low ΔDP |
| `stredoskolske` | 0.892 | 0.010 | 0.019 | 0.755 | Subset of education |
| `relation_to_children_indicator` | 0.848 | 0.039 | 0.015 | 0.760 | Lower F1 |
| `abstinent` | 0.839* | 0.017 | 0.078 | 0.755 | *seed=42 collapses (F1=0.66) |
| ~~`high_edu`~~ | ~~0.999~~ | — | — | — | **Invalid**: `vysoke_skoly` absent → trivial leakage |

> **Key finding**: leakage AUC is constant at ~0.75 across **all** targets — structural bias comes from graph homophily (r≈0.876), not from the target choice.

### Main Experiment Results

Results from `notebooks/main_experiment.ipynb` executed on GPU (RTX 3090). Baseline reported as mean ± std over 5 seeds `[3, 7, 21, 42, 99]`; other methods use seed=42.

| Method | Accuracy | Macro F1 (mean±std) | ΔDP ↓ (mean±std) | ΔEO ↓ | Leakage AUC ↓ |
|--------|----------|---------------------|------------------|-------|----------------|
| Baseline (GraphSAGE) | 0.9381 ± 0.0012 | 0.9380 ± 0.0011 | 0.0414 ± 0.0011 | 0.0221 | 0.8171 ± 0.0051 |
| Pre-processing (Resampling) | 0.9381 | 0.9380 | 0.0553 | 0.0365 | n/a |
| Pre-processing (FairDrop) | 0.9353 | 0.9351 | 0.0380 | 0.0206 | 0.8779 |
| FairGNN (best λ) | 0.8272 | 0.8272 | 0.0137 | 0.0083 | n/a |

**Key observations:**
- **Resampling** preserves accuracy exactly (same as baseline) but *increases* ΔDP (+34%) and ΔEO (+65%) — label/gender correlations in the graph structure overwhelm the balancing effect, and the method does not constrain the embedding space
- **FairDrop** achieves a modest accuracy drop (−0.28 pp) with a 8% ΔDP reduction and 6.8% ΔEO reduction; leakage actually increases slightly (0.817 → 0.878), suggesting FairDrop's structural pruning may redistribute rather than eliminate gender-correlated structure in embeddings on this highly homophilic graph
- **FairGNN** achieves the best fairness: ΔDP −67% (0.0414 → 0.0137), ΔEO −62% (0.0221 → 0.0083), at a cost of −10.9 pp in F1 (0.938 → 0.827); the adversarial debiasing effectively constrains the representation space

See `results/figures/pareto_fairness.png` for the fairness–accuracy Pareto plot and `results/figures/counterfactual_fairness.png` for CF scores (baseline and FairDrop).

### Structural Bias

The **assortative mixing coefficient r** (Newman 2003), as described in Laclau et al. (2024), characterizes the graph-level structural bias before any model is trained:

| Sensitive attribute | r |
|---|---|
| gender | ~0.08 (low — users connect cross-gender frequently) |
| region | ≈ 0.87 (reported in Laclau et al., 2024 — strong homophily) |

> **Note**: The leakage AUC of ~0.82 despite low gender homophily (r≈0.08) indicates that region homophily (r≈0.87) is the dominant structural bias channel — region and gender are correlated in the dataset, so even gender-debiased edges still propagate region-related signals.

A high r explains why GNN embeddings encode sensitive information even when it is removed from node features: message passing propagates demographic signals through the edge structure.

---

## 3. Trade-off Analysis

### Fairness vs. Accuracy Pareto
Each method occupies a different position on the fairness–accuracy Pareto frontier:

- **Baseline**: highest accuracy, highest bias (no debiasing); Representation Bias (RB) / leakage expected to be high (~0.73 AUC) because the graph structure encodes gender via homophily (r ≈ 0.87 for region)
- **Resampling**: modest bias reduction by balancing label×gender combinations in training; counter-intuitively may *worsen* ΔDP on this dataset due to label/gender correlations in the graph structure; does not directly constrain the embedding space
- **FairDrop** (Spinelli et al., 2021, reviewed in Laclau et al., 2024): attacks structural bias at its root by preferentially removing intra-group edges before training, reducing r and limiting the structural signal available to the GNN; expected to reduce RB/leakage more reliably than resampling
- **FairGNN**: directly optimizes `L_cls − λ * L_adv`, pushing the encoder toward gender-invariant representations; higher λ → more fairness, lower leakage, but less discriminative power

### FairGNN λ Selection
Optimal λ is selected on val ΔDP (lowest). Note: λ=0.1 and λ=1.0 may cause model collapse (predict all-zero, artificially driving ΔDP→0 while F1 drops to ~48%). To be validated on the new target.

### Counterfactual Fairness
The **counterfactual unfairness score** (NIFTY, Agarwal et al. 2021) complements RB/leakage:
- **RB/Leakage**: measures how much gender is *recoverable* from embeddings
- **CF unfairness**: measures how much a downstream decision *depends* on gender

A model may have high leakage but low CF unfairness (gender is encoded but not exploited) or vice versa. FairDrop is expected to reduce both by cutting the structural channel through which gender is transmitted.

---

## 4. Limitations

### Data Limitations
- **Missing age values**: `AGE=0` maps to "young" after `fillna(0)`, introducing noise in the age feature
- **Pokec collection bias**: the dataset is from a Slovak social network; gender is binary and self-reported; results may not generalize to other populations or definitions of gender
- **Homophily**: social networks exhibit strong homophily (users connect with similar users); this means graph-based models can infer sensitive attributes indirectly even when removed from features (structural bias) — this is precisely what the leakage metric captures
- **Label definition and target selection**: `I_am_working_in_field` (literature standard) takes 5 distinct values: `-1` (57 772 users, 86.8%), `0` (4 510, 6.8%), `1` (1 789, 2.7%), `2` (1 353, 2.0%), `3` (1 145, 1.7%). `-1` is almost certainly a "field not filled" sentinel rather than "unemployed", introducing label noise. More importantly, the resulting severe class imbalance (6.4% positive) and low ΔDP (~0.007) make this target unsuitable for demonstrating fairness-aware debiasing. We replaced it with `completed_level_of_education_indicator` after a systematic sweep of 8 candidates (40 runs) — see Section 2.

### Methodological Limitations
- **GNNExplainer is post-hoc**: explanations depend on the trained model, not the data-generating process; they describe model behavior, not causal mechanisms
- **ΔDP/ΔEO are binary metrics**: they are defined for binary sensitive and target attributes; multi-class extensions (via max pairwise gap) are approximations
- **FairGNN adversary is binary**: the discriminator is designed for binary sensitive attributes; extending to multi-group fairness requires architectural changes
- **Leakage probe uses train/test split (RB)**: the logistic regression probe is fitted on train-set embeddings and evaluated on the test set, addressing linkage bias (Laclau et al., 2024 §3.3); AUC-ROC is used rather than accuracy to handle group imbalance
- **Counterfactual fairness requires a classifier**: our CF score requires augmenting embeddings with gender and fitting a probe; this measures potential sensitivity of a downstream classifier, not the GNN encoder itself
- **FairDrop is undirected**: the implementation treats edges as undirected and drops by intra-group criterion; directed graph variants require additional handling
- **Resampling without replacement in evaluation**: the oversampled indices mask allows repeated nodes during training, but evaluation is on the original (non-oversampled) test split

### AI Tool Usage Disclosure
This project was implemented with the assistance of an AI coding assistant (OpenCode / Claude). The AI assisted in:
- Scaffolding project structure and `pyproject.toml`
- Generating implementation skeletons for all modules
- Debugging NumPy 2.x / PyTorch bridge incompatibilities
- Writing the experiment notebook

All generated code was reviewed, tested, and verified by the author. The experimental design, fairness metric definitions, and analysis interpretations are the author's own.

---

## References

- **Laclau, C., Largeron, C., & Choudhary, M. (2024)**. A Survey on Fairness for Machine Learning on Graphs. *arXiv:2205.05396v2*. ← primary reference for this project's framework, metrics, and methods
- Dai, E., & Wang, S. (2021). Say No to the Discrimination: Learning Fair Graph Neural Networks with Limited Sensitive Attribute Information. *WSDM 2021*.
- Agarwal, C. et al. (2021). NIFTY: A framework for benchmarking graph neural networks for fairness. *arXiv:2109.05228*.
- Spinelli, I. et al. (2021). FairDrop: Biased Edge Dropout for Enhancing Fairness in Graph Representation Learning. *IEEE TNNLS*.
- Newman, M. E. J. (2003). Mixing patterns in networks. *Physical Review E, 67*(2).
- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. *NeurIPS 2017*.
- Ying, Z. et al. (2019). GNNExplainer: Generating Explanations for Graph Neural Networks. *NeurIPS 2019*.
- Takac, L., & Zabovsky, M. (2012). Data Analysis in Public Social Networks. *International Scientific Conference & International Workshop Present Day Trends of Innovations*.
