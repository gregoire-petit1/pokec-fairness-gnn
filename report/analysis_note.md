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
| Leakage | Sensitive attribute leakage: accuracy of a logistic regression probe trained on frozen node embeddings to predict `gender` (higher = more demographic information encoded) |

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

| Method | Accuracy (mean±std) | Macro F1 (mean±std) | ΔDP ↓ (mean±std) | ΔEO ↓ | AUC gap ↓ | Leakage ↓ |
|--------|---------------------|---------------------|------------------|-------|-----------|-----------|
| Baseline (GraphSAGE) | — | — | — | — | — | — |
| Pre-processing (Resampling) | — | — | — | — | — | — |
| FairGNN (best λ) | — | — | — | — | — | — |

*Expected trends based on literature (Dai & Wang, 2021; Agarwal et al., 2021):*
- Resampling reduces ΔDP moderately at mild accuracy cost
- FairGNN at high λ reduces ΔDP more aggressively but at a larger accuracy trade-off; it should also reduce leakage as the adversary explicitly penalizes encoding of gender
- FairGNN at λ=0.5–1.0 typically achieves the best Pareto trade-off

---

## 3. Trade-off Analysis

### Fairness vs. Accuracy Pareto
Each method occupies a different position on the fairness–accuracy Pareto frontier:

- **Baseline**: highest accuracy, highest bias (no debiasing); embedding leakage expected to be high
- **Resampling**: modest bias reduction by balancing label×gender combinations in training; effect is limited since oversampling does not directly constrain the loss or the embedding space; leakage may not decrease significantly
- **FairGNN**: directly optimizes `L_cls − λ * L_adv`, pushing the encoder toward gender-invariant representations; higher λ → more fairness, lower leakage, but less discriminative power

### FairGNN λ Selection
Optimal λ is selected on val ΔDP (lowest). The adversarial loss `L_adv` is a cross-entropy on sensitive attribute prediction; minimizing `−λ * L_adv` penalizes the encoder for encoding gender information — which should simultaneously reduce both ΔDP and embedding leakage.

---

## 4. Limitations

### Data Limitations
- **Missing age values**: `AGE=0` maps to "young" after `fillna(0)`, introducing noise in the age feature
- **Pokec collection bias**: the dataset is from a Slovak social network; gender is binary and self-reported; results may not generalize to other populations or definitions of gender
- **Homophily**: social networks exhibit strong homophily (users connect with similar users); this means graph-based models can infer sensitive attributes indirectly even when removed from features (structural bias) — this is precisely what the leakage metric captures
- **Label definition**: `I_am_working_in_field` is binarised as ">0 = employed in some field"; zero values include both unemployed users and users who left the field blank, introducing label noise

### Methodological Limitations
- **GNNExplainer is post-hoc**: explanations depend on the trained model, not the data-generating process; they describe model behavior, not causal mechanisms
- **ΔDP/ΔEO are binary metrics**: they are defined for binary sensitive and target attributes; multi-class extensions (via max pairwise gap) are approximations
- **FairGNN adversary is binary**: the discriminator is designed for binary sensitive attributes; extending to multi-group fairness requires architectural changes
- **Leakage probe uses training data**: the logistic regression probe is fitted and evaluated on the full embedding set (no held-out split); this overestimates leakage slightly but is consistent with literature practice
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

- Dai, E., & Wang, S. (2021). Say No to the Discrimination: Learning Fair Graph Neural Networks with Limited Sensitive Attribute Information. *WSDM 2021*.
- Agarwal, C. et al. (2021). NIFTY: A framework for benchmarking graph neural networks for fairness. *arXiv:2109.05228*.
- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. *NeurIPS 2017*.
- Ying, Z. et al. (2019). GNNExplainer: Generating Explanations for Graph Neural Networks. *NeurIPS 2019*.
- Takac, L., & Zabovsky, M. (2012). Data Analysis in Public Social Networks. *International Scientific Conference & International Workshop Present Day Trends of Innovations*.
