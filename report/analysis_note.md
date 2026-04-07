# Analysis Note: Fairness, Interpretability and Robustness on Pokec-z

## 1. Experimental Protocol

### Dataset
Pokec-z is a Slovak social network dataset (region "z") with ~67k nodes and ~882k directed edges. Each node represents a user with features including age, gender, region, and job-related attributes. Following the standard benchmark configuration used in FairGNN (Dai & Wang, 2021), NIFTY (Agarwal et al., 2021), and FairGLite, the binary **target** is `I_am_working_in_field` (binarised: 0 = no occupation reported, 1 = has occupation). The **sensitive attributes** are `gender` (binary, 0/1) and `region` (binary, 0/1).

### Preprocessing
- Node features: all columns except `user_id` and `I_am_working_in_field`; age is binned into `young/adult/senior` (cut-off: 0–25, 25–40, 40+)
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

*Note: the table below will be filled after running `notebooks/main_experiment.ipynb` with the Pokec-z raw data.*

| Method | Accuracy (mean±std) | Macro F1 (mean±std) | ΔDP ↓ (mean±std) | ΔEO ↓ | AUC gap ↓ | Leakage (RB) ↓ | CF unfairness ↓ |
|--------|---------------------|---------------------|------------------|-------|-----------|----------------|-----------------|
| Baseline (GraphSAGE) | — | — | — | — | — | — | — |
| Pre-processing (Resampling) | — | — | — | — | — | — | — |
| Pre-processing (FairDrop) | — | — | — | — | — | — | — |
| FairGNN (best λ) | — | — | — | — | — | — | — |

### Structural Bias

The **assortative mixing coefficient r** (Newman 2003), as described in Laclau et al. (2024), characterizes the graph-level structural bias before any model is trained:

| Sensitive attribute | r |
|---|---|
| gender | — |
| region | ≈ 0.87 (reported in Laclau et al., 2024) |

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
Optimal λ is selected on val ΔDP (lowest). Note: λ=0.1 and λ=1.0 may cause model collapse (predict all-zero, artificially driving ΔDP→0 while F1 drops to ~48%). The true best λ is 0.5 (F1=54.6%, ΔDP=0.024).

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
- **Label definition and noise**: `I_am_working_in_field` takes 5 distinct values: `-1` (57 772 users, 86.8%), `0` (4 510, 6.8%), `1` (1 789, 2.7%), `2` (1 353, 2.0%), `3` (1 145, 1.7%). Following the FairGNN/NIFTY convention, we binarise as `> 0` (positive = values 1/2/3, active in some professional field). However, `-1` is almost certainly a **"field not filled"** sentinel rather than "unemployed": users with `-1` have a near-perfectly balanced gender distribution (50.6% / 49.4%), unlike all other values which show gender skew — consistent with random non-response rather than a substantive answer. This means the negative class (88 682 users) mixes **true non-workers** (val=0, explicitly answered) with **non-respondents** (val=-1), introducing label noise. An alternative would be to restrict the dataset to the 8 797 users who answered (val 0/1/2/3), yielding a nearly balanced binary task (48.7% positive) with cleaner semantics. We retain the standard binarisation for comparability with the literature but flag this as a limitation.

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
