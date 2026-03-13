# Analysis Note: Fairness, Interpretability and Robustness on Pokec-z

## 1. Experimental Protocol

### Dataset
Pokec-z is a Slovak social network dataset (region "z") with ~67k nodes and ~882k directed edges. Each node represents a user with features including age, gender, and job-related attributes. The binary **target** is `region` (0/1). The **sensitive attribute** is `gender` (binary, 0/1).

### Preprocessing
- Node features: all columns except `user_id` and `region`; age is binned into `young/adult/senior` (cut-off: 0–25, 25–40, 40+)
- `region` is excluded from features to prevent data leakage
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

### Hyperparameters
From `configs/experiment.yaml`:
- Seed: 42
- Model: GraphSAGE, hidden=256, layers=2, dropout=0.5, lr=0.001, epochs=200, patience=20
- FairGNN: λ ∈ {0.1, 0.5, 1.0, 5.0}, adv_hidden=64
- Robustness: noise σ ∈ {0.1, 0.3, 0.5}, edge drop ∈ {0.1, 0.3, 0.5}

---

## 2. Key Results

*Note: the table below will be filled after running `notebooks/main_experiment.ipynb` with the Pokec-z raw data.*

| Method | Accuracy | Macro F1 | ΔDP ↓ | ΔEO ↓ | AUC gap ↓ |
|--------|----------|----------|-------|-------|-----------|
| Baseline (GraphSAGE) | — | — | — | — | — |
| Pre-processing (Resampling) | — | — | — | — | — |
| FairGNN (best λ) | — | — | — | — | — |

*Expected trends based on literature (Dai & Wang, 2021):*
- Resampling reduces ΔDP moderately at mild accuracy cost
- FairGNN at high λ reduces ΔDP more aggressively but at a larger accuracy trade-off
- FairGNN at λ=0.5–1.0 typically achieves the best Pareto trade-off

---

## 3. Trade-off Analysis

### Fairness vs. Accuracy Pareto
Each method occupies a different position on the fairness–accuracy Pareto frontier:

- **Baseline**: highest accuracy, highest bias (no debiasing)
- **Resampling**: modest bias reduction by balancing label×gender combinations in training; effect is limited since oversampling does not directly constrain the loss
- **FairGNN**: directly optimizes `L_cls − λ * L_adv`, pushing the encoder toward gender-invariant representations; higher λ → more fairness but less discriminative power

### FairGNN λ Selection
Optimal λ is selected on val ΔDP (lowest). The adversarial loss `L_adv` is a cross-entropy on sensitive attribute prediction; minimizing `−λ * L_adv` penalizes the encoder for encoding gender information.

---

## 4. Limitations

### Data Limitations
- **Missing age values**: `AGE=0` maps to "young" after `fillna(0)`, introducing noise in the age feature
- **Pokec collection bias**: the dataset is from a Slovak social network; gender is binary and self-reported; results may not generalize to other populations or definitions of gender
- **Homophily**: social networks exhibit strong homophily (users connect with similar users); this means graph-based models can infer sensitive attributes indirectly even when removed from features (structural bias)
- **Label proxy**: `region` is used as the target; it may correlate with other socioeconomic attributes, making fairness enforcement with respect to `gender` only a partial remedy

### Methodological Limitations
- **GNNExplainer is post-hoc**: explanations depend on the trained model, not the data-generating process; they describe model behavior, not causal mechanisms
- **ΔDP/ΔEO are binary metrics**: they are defined for binary sensitive and target attributes; multi-class extensions (via max pairwise gap) are approximations
- **FairGNN adversary is binary**: the discriminator is designed for binary sensitive attributes; extending to multi-group fairness requires architectural changes
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
- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. *NeurIPS 2017*.
- Ying, Z. et al. (2019). GNNExplainer: Generating Explanations for Graph Neural Networks. *NeurIPS 2019*.
- Takac, L., & Zabovsky, M. (2012). Data Analysis in Public Social Networks. *International Scientific Conference & International Workshop Present Day Trends of Innovations*.
