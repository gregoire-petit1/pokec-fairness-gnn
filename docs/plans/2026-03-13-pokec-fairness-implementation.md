# Pokec Fairness GNN Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a reproducible node classification pipeline on Pokec-z with fairness analysis (resampling + FairGNN), GNNExplainer interpretability, and robustness evaluation.

**Architecture:** GraphSAGE baseline + two fairness methods (pre-processing resampling and adversarial FairGNN) + GNNExplainer + perturbation robustness experiments. All experiments run in a single reproducible notebook backed by modular `src/` Python modules.

**Tech Stack:** Python, PyTorch, PyTorch Geometric (PyG), scikit-learn, pandas, numpy, matplotlib/seaborn, PyYAML

---

## Task 1: Project Scaffold & Environment

**Files:**
- Create: `pyproject.toml`
- Create: `configs/experiment.yaml`
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/models/__init__.py`
- Create: `src/fairness/__init__.py`
- Create: `src/interpretability/__init__.py`
- Create: `src/robustness/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "pokec-fairness-gnn"
version = "0.1.0"
description = "Fairness, interpretability and robustness analysis on Pokec-z with GNNs"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.0.0",
    "torch-geometric>=2.3.0",
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

**Step 2: Create configs/experiment.yaml**

```yaml
seed: 42

data:
  dataset: pokec-z
  raw_dir: data/raw/pokec-z
  splits_dir: data/splits
  split_ratios: [0.6, 0.2, 0.2]
  target_col: region
  sensitive_cols: [gender, age_group]

model:
  hidden_dim: 256
  num_layers: 2
  dropout: 0.5
  lr: 0.001
  epochs: 200
  patience: 20

fairgnn:
  lambda_values: [0.1, 0.5, 1.0, 5.0]
  adv_hidden_dim: 64

robustness:
  noise_levels: [0.1, 0.3, 0.5]
  edge_drop_rates: [0.1, 0.3, 0.5]
```

**Step 3: Create all empty `__init__.py` files**

```bash
mkdir -p src/data src/models src/fairness src/interpretability src/robustness
touch src/__init__.py src/data/__init__.py src/models/__init__.py
touch src/fairness/__init__.py src/interpretability/__init__.py src/robustness/__init__.py
mkdir -p data/raw/pokec-z data/splits results/figures results/metrics report notebooks tests
touch tests/__init__.py
```

**Step 4: Install dependencies**

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Note: torch-geometric core installe sans extensions optionnelles. Si des erreurs `torch_scatter`/`torch_sparse` apparaissent à l'exécution, les installer via :
```bash
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
```

**Step 5: Commit**

```bash
git add .
git commit -m "feat: scaffold project structure and config"
```

---

## Task 2: Data Loading & Preprocessing

**Files:**
- Create: `src/data/loader.py`
- Create: `src/data/preprocessing.py`

**Step 1: Write the failing test**

Create `tests/test_data.py`:

```python
import pytest
from src.data.loader import load_pokec_z
from src.data.preprocessing import preprocess

def test_load_returns_pyg_data():
    data = load_pokec_z("data/raw/pokec-z")
    assert hasattr(data, "x")
    assert hasattr(data, "edge_index")
    assert hasattr(data, "y")

def test_sensitive_attrs_not_in_features():
    data = load_pokec_z("data/raw/pokec-z")
    processed = preprocess(data, sensitive_cols=["gender", "age_group"])
    # sensitive attributes must not be in x
    assert processed.x.shape[1] < data.x.shape[1]

def test_age_group_categorization():
    import pandas as pd
    from src.data.preprocessing import categorize_age
    ages = pd.Series([20, 30, 50])
    groups = categorize_age(ages)
    assert list(groups) == ["young", "adult", "senior"]
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_data.py -v
```
Expected: FAIL with ImportError

**Step 3: Implement `src/data/loader.py`**

```python
"""Load Pokec-z dataset into a PyG Data object."""
import os
import pandas as pd
import torch
from torch_geometric.data import Data


def load_pokec_z(raw_dir: str) -> Data:
    """Load Pokec-z from raw CSV files and return a PyG Data object."""
    features_path = os.path.join(raw_dir, "region_job_2.csv")
    edges_path = os.path.join(raw_dir, "region_job_2_relationship.txt")

    df = pd.read_csv(features_path, sep="\t")
    df = df.fillna(0)

    # Build edge index
    edges = pd.read_csv(edges_path, sep="\t", header=None, names=["src", "dst"])
    # Remap node ids to 0-indexed
    node_ids = {nid: idx for idx, nid in enumerate(df["user_id"].values)}
    src = edges["src"].map(node_ids).dropna().astype(int).values
    dst = edges["dst"].map(node_ids).dropna().astype(int).values
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Labels: region (0 or 1 in pokec-z)
    y = torch.tensor(df["region"].values, dtype=torch.long)

    # All columns as features (will be filtered in preprocessing)
    feature_cols = [c for c in df.columns if c not in ["user_id"]]
    x = torch.tensor(df[feature_cols].values, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.feature_cols = feature_cols
    data.raw_df = df
    return data
```

**Step 4: Implement `src/data/preprocessing.py`**

```python
"""Preprocessing: age categorization, feature normalization, sensitive attr removal."""
import pandas as pd
import torch
from torch_geometric.data import Data


def categorize_age(ages: pd.Series) -> pd.Series:
    """Categorize numeric age into young/adult/senior."""
    bins = [0, 25, 40, 200]
    labels = ["young", "adult", "senior"]
    return pd.cut(ages, bins=bins, labels=labels, right=False)


def preprocess(data: Data, sensitive_cols: list[str]) -> Data:
    """Remove sensitive cols from features, normalize remaining features."""
    df = data.raw_df.copy()

    # Categorize age and store as attribute
    if "AGE" in df.columns:
        df["age_group"] = categorize_age(df["AGE"]).cat.codes  # -1,0,1,2
        data.age_group = torch.tensor(df["age_group"].values, dtype=torch.long)

    # Store sensitive attrs separately
    for col in sensitive_cols:
        if col in df.columns:
            setattr(data, col, torch.tensor(df[col].values, dtype=torch.long))

    # Remove sensitive cols from features
    remove_cols = [c for c in sensitive_cols if c in data.feature_cols]
    keep_cols = [c for c in data.feature_cols if c not in remove_cols]
    x = data.x[:, [data.feature_cols.index(c) for c in keep_cols]]

    # Z-score normalization
    mean = x.mean(dim=0)
    std = x.std(dim=0) + 1e-8
    x = (x - mean) / std

    data.x = x
    data.feature_cols = keep_cols
    return data
```

**Step 5: Run tests**

```bash
pytest tests/test_data.py -v
```
Expected: PASS

**Step 6: Commit**

```bash
git add src/data/ tests/test_data.py
git commit -m "feat: add Pokec-z data loader and preprocessing"
```

---

## Task 3: Stratified Splits

**Files:**
- Create: `src/data/splits.py`

**Step 1: Write the failing test**

Add to `tests/test_data.py`:

```python
from src.data.splits import make_splits

def test_splits_sizes():
    import torch
    n = 1000
    y = torch.randint(0, 2, (n,))
    gender = torch.randint(0, 2, (n,))
    train, val, test = make_splits(n, y, gender, ratios=(0.6, 0.2, 0.2), seed=42)
    assert abs(len(train) / n - 0.6) < 0.02
    assert abs(len(val) / n - 0.2) < 0.02
    assert abs(len(test) / n - 0.2) < 0.02

def test_splits_no_overlap():
    import torch
    n = 1000
    y = torch.randint(0, 2, (n,))
    gender = torch.randint(0, 2, (n,))
    train, val, test = make_splits(n, y, gender, ratios=(0.6, 0.2, 0.2), seed=42)
    assert len(set(train.tolist()) & set(test.tolist())) == 0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_data.py::test_splits_sizes -v
```

**Step 3: Implement `src/data/splits.py`**

```python
"""Stratified train/val/test split stratified on label + sensitive attribute."""
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def make_splits(
    n: int,
    y: torch.Tensor,
    sensitive: torch.Tensor,
    ratios: tuple = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return train/val/test index tensors, stratified on y x sensitive."""
    # Combined stratification label
    strat = y.numpy() * 10 + sensitive.numpy()
    idx = np.arange(n)

    train_ratio, val_ratio, _ = ratios
    test_ratio = 1 - train_ratio - val_ratio

    train_idx, temp_idx = train_test_split(
        idx, test_size=(1 - train_ratio), stratify=strat, random_state=seed
    )
    strat_temp = strat[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=strat_temp,
        random_state=seed,
    )
    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx, dtype=torch.long),
        torch.tensor(test_idx, dtype=torch.long),
    )


def save_splits(train: torch.Tensor, val: torch.Tensor, test: torch.Tensor, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    torch.save(train, os.path.join(out_dir, "train.pt"))
    torch.save(val, os.path.join(out_dir, "val.pt"))
    torch.save(test, os.path.join(out_dir, "test.pt"))


def load_splits(out_dir: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train = torch.load(os.path.join(out_dir, "train.pt"))
    val = torch.load(os.path.join(out_dir, "val.pt"))
    test = torch.load(os.path.join(out_dir, "test.pt"))
    return train, val, test
```

**Step 4: Run tests**

```bash
pytest tests/test_data.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add src/data/splits.py tests/test_data.py
git commit -m "feat: add stratified train/val/test splits"
```

---

## Task 4: GraphSAGE Baseline Model

**Files:**
- Create: `src/models/graphsage.py`
- Create: `src/models/trainer.py`

**Step 1: Write the failing test**

Create `tests/test_models.py`:

```python
import torch
from torch_geometric.data import Data
from src.models.graphsage import GraphSAGE

def test_graphsage_forward_shape():
    x = torch.randn(100, 16)
    edge_index = torch.randint(0, 100, (2, 300))
    data = Data(x=x, edge_index=edge_index)
    model = GraphSAGE(in_channels=16, hidden_channels=32, out_channels=4, num_layers=2, dropout=0.5)
    out = model(data.x, data.edge_index)
    assert out.shape == (100, 4)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_models.py -v
```

**Step 3: Implement `src/models/graphsage.py`**

```python
"""GraphSAGE model for node classification."""
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return penultimate layer embeddings."""
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x
```

**Step 4: Implement `src/models/trainer.py`**

```python
"""Training loop with early stopping."""
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def train_epoch(model, data, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=1)
    acc = (pred == data.y[mask]).float().mean().item()
    f1 = f1_score(data.y[mask].cpu(), pred.cpu(), average="macro", zero_division=0)
    return acc, f1


def train(model, data, train_mask, val_mask, lr: float, epochs: int, patience: int):
    """Train with early stopping. Returns best val F1 and training history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        loss = train_epoch(model, data, optimizer, train_mask)
        _, val_f1 = evaluate(model, data, val_mask)
        history.append({"epoch": epoch, "loss": loss, "val_f1": val_f1})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return best_val_f1, history
```

**Step 5: Run tests**

```bash
pytest tests/test_models.py -v
```

**Step 6: Commit**

```bash
git add src/models/ tests/test_models.py
git commit -m "feat: add GraphSAGE model and training loop"
```

---

## Task 5: Fairness Metrics

**Files:**
- Create: `src/fairness/metrics.py`

**Step 1: Write the failing test**

Create `tests/test_fairness.py`:

```python
import torch
from src.fairness.metrics import demographic_parity_diff, equal_opportunity_diff, group_auc_gap

def test_dp_diff_perfect_fairness():
    pred = torch.tensor([1, 1, 0, 0])
    sensitive = torch.tensor([0, 1, 0, 1])
    assert demographic_parity_diff(pred, sensitive) == 0.0

def test_dp_diff_unfair():
    pred = torch.tensor([1, 1, 0, 0])
    sensitive = torch.tensor([0, 0, 1, 1])
    assert demographic_parity_diff(pred, sensitive) == 1.0

def test_eo_diff():
    pred = torch.tensor([1, 1, 0, 0])
    y_true = torch.tensor([1, 1, 1, 1])
    sensitive = torch.tensor([0, 0, 1, 1])
    assert equal_opportunity_diff(pred, y_true, sensitive) == 1.0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_fairness.py -v
```

**Step 3: Implement `src/fairness/metrics.py`**

```python
"""Fairness metrics: ΔDP, ΔEO, Group AUC gap."""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def demographic_parity_diff(pred: torch.Tensor, sensitive: torch.Tensor) -> float:
    """Demographic Parity Difference: |P(y_hat=1|s=0) - P(y_hat=1|s=1)|."""
    groups = sensitive.unique()
    rates = []
    for g in groups:
        mask = sensitive == g
        rates.append(pred[mask].float().mean().item())
    return float(max(rates) - min(rates))


def equal_opportunity_diff(
    pred: torch.Tensor, y_true: torch.Tensor, sensitive: torch.Tensor
) -> float:
    """Equal Opportunity Difference: |TPR_g0 - TPR_g1| for binary y."""
    groups = sensitive.unique()
    tprs = []
    for g in groups:
        mask = (sensitive == g) & (y_true == 1)
        if mask.sum() == 0:
            continue
        tpr = (pred[mask] == 1).float().mean().item()
        tprs.append(tpr)
    if len(tprs) < 2:
        return 0.0
    return float(max(tprs) - min(tprs))


def group_auc_gap(
    proba: np.ndarray, y_true: torch.Tensor, sensitive: torch.Tensor
) -> float:
    """Max AUC difference across sensitive groups."""
    groups = sensitive.unique().tolist()
    aucs = []
    for g in groups:
        mask = sensitive == g
        y_g = y_true[mask].numpy()
        p_g = proba[mask.numpy()]
        if len(np.unique(y_g)) < 2:
            continue
        aucs.append(roc_auc_score(y_g, p_g, multi_class="ovr", average="macro"))
    if len(aucs) < 2:
        return 0.0
    return float(max(aucs) - min(aucs))


def compute_all_fairness_metrics(
    pred: torch.Tensor,
    y_true: torch.Tensor,
    sensitive: torch.Tensor,
    proba: np.ndarray | None = None,
) -> dict:
    """Compute all fairness metrics and return as dict."""
    result = {
        "delta_dp": demographic_parity_diff(pred, sensitive),
        "delta_eo": equal_opportunity_diff(pred, y_true, sensitive),
    }
    if proba is not None:
        result["group_auc_gap"] = group_auc_gap(proba, y_true, sensitive)
    return result
```

**Step 4: Run tests**

```bash
pytest tests/test_fairness.py -v
```

**Step 5: Commit**

```bash
git add src/fairness/metrics.py tests/test_fairness.py
git commit -m "feat: add fairness metrics (ΔDP, ΔEO, group AUC gap)"
```

---

## Task 6: Pre-processing — Resampling

**Files:**
- Create: `src/fairness/resampling.py`

**Step 1: Write the failing test**

Add to `tests/test_fairness.py`:

```python
from src.fairness.resampling import oversample_train_mask
import torch

def test_oversample_increases_minority():
    n = 200
    y = torch.cat([torch.zeros(150), torch.ones(50)]).long()
    gender = torch.randint(0, 2, (n,))
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:160] = True
    new_mask = oversample_train_mask(train_mask, y, gender, seed=42)
    # Should return more indices than original
    assert new_mask.sum() >= train_mask.sum()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_fairness.py::test_oversample_increases_minority -v
```

**Step 3: Implement `src/fairness/resampling.py`**

```python
"""Pre-processing fairness: oversample minority groups in training set."""
import numpy as np
import torch
from sklearn.utils import resample


def oversample_train_mask(
    train_mask: torch.Tensor,
    y: torch.Tensor,
    sensitive: torch.Tensor,
    seed: int = 42,
) -> torch.Tensor:
    """Oversample minority (label x sensitive group) combinations in train set.

    Returns a new boolean mask over all nodes that includes oversampled indices.
    Note: indices can appear more than once — the mask uses index-based access.
    """
    train_idx = train_mask.nonzero(as_tuple=True)[0].numpy()
    y_train = y[train_idx].numpy()
    s_train = sensitive[train_idx].numpy()
    strat = y_train * 10 + s_train

    # Find size of majority group
    unique, counts = np.unique(strat, return_counts=True)
    max_count = counts.max()

    resampled_idx = []
    for group in unique:
        group_idx = train_idx[strat == group]
        if len(group_idx) < max_count:
            group_idx = resample(group_idx, n_samples=max_count, replace=True, random_state=seed)
        resampled_idx.append(group_idx)

    all_idx = np.concatenate(resampled_idx)
    # Return as expanded index tensor (not boolean — use index-based masking downstream)
    return torch.tensor(all_idx, dtype=torch.long)
```

**Step 4: Run tests**

```bash
pytest tests/test_fairness.py -v
```

**Step 5: Commit**

```bash
git add src/fairness/resampling.py tests/test_fairness.py
git commit -m "feat: add pre-processing oversampling for fairness"
```

---

## Task 7: In-training — FairGNN

**Files:**
- Create: `src/models/fairgnn.py`

**Step 1: Write the failing test**

Add to `tests/test_models.py`:

```python
from src.models.fairgnn import FairGNN

def test_fairgnn_forward_returns_pred_and_adv():
    x = torch.randn(100, 16)
    edge_index = torch.randint(0, 100, (2, 300))
    model = FairGNN(in_channels=16, hidden_channels=32, out_channels=2, adv_hidden=16)
    pred_logits, adv_logits = model(x, edge_index)
    assert pred_logits.shape == (100, 2)
    assert adv_logits.shape[0] == 100
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_models.py::test_fairgnn_forward_returns_pred_and_adv -v
```

**Step 3: Implement `src/models/fairgnn.py`**

```python
"""FairGNN: adversarial debiasing GNN (Dai & Wang, 2021)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class FairGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        adv_hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        # Shared encoder
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # Main classifier head
        self.classifier = nn.Linear(hidden_channels, out_channels)

        # Adversarial discriminator head
        self.adversary = nn.Sequential(
            nn.Linear(hidden_channels, adv_hidden),
            nn.ReLU(),
            nn.Linear(adv_hidden, 2),  # binary sensitive attr
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = self.encode(x, edge_index)
        pred_logits = self.classifier(h)
        adv_logits = self.adversary(h)
        return pred_logits, adv_logits


def fairgnn_loss(
    pred_logits: torch.Tensor,
    adv_logits: torch.Tensor,
    y: torch.Tensor,
    sensitive: torch.Tensor,
    mask: torch.Tensor,
    lambda_adv: float,
) -> torch.Tensor:
    """L_total = L_classification - λ * L_adversarial."""
    l_cls = F.cross_entropy(pred_logits[mask], y[mask])
    l_adv = F.cross_entropy(adv_logits[mask], sensitive[mask])
    return l_cls - lambda_adv * l_adv
```

**Step 4: Run tests**

```bash
pytest tests/test_models.py -v
```

**Step 5: Commit**

```bash
git add src/models/fairgnn.py tests/test_models.py
git commit -m "feat: add FairGNN adversarial debiasing model"
```

---

## Task 8: GNNExplainer Wrapper

**Files:**
- Create: `src/interpretability/explainer.py`

**Step 1: Write the failing test**

Create `tests/test_interpretability.py`:

```python
import torch
from torch_geometric.data import Data
from src.models.graphsage import GraphSAGE
from src.interpretability.explainer import explain_node

def test_explain_node_returns_masks():
    x = torch.randn(50, 8)
    edge_index = torch.randint(0, 50, (2, 150))
    data = Data(x=x, edge_index=edge_index)
    model = GraphSAGE(in_channels=8, hidden_channels=16, out_channels=2, num_layers=2)
    edge_mask, feat_mask = explain_node(model, data, node_idx=0, num_hops=2)
    assert edge_mask is not None
    assert feat_mask is not None
    assert edge_mask.shape[0] == edge_index.shape[1]
    assert feat_mask.shape[0] == x.shape[1]
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_interpretability.py -v
```

**Step 3: Implement `src/interpretability/explainer.py`**

```python
"""GNNExplainer wrapper for PyG."""
import torch
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer


def explain_node(
    model: torch.nn.Module,
    data: Data,
    node_idx: int,
    num_hops: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run GNNExplainer on a single node. Returns (edge_mask, feature_mask)."""
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )
    explanation = explainer(data.x, data.edge_index, index=node_idx)
    return explanation.edge_mask, explanation.node_mask[node_idx]


def explain_group(
    model: torch.nn.Module,
    data: Data,
    node_indices: list[int],
    num_hops: int = 2,
) -> dict:
    """Explain a group of nodes, return aggregated feature importance."""
    feat_masks = []
    for idx in node_indices:
        _, feat_mask = explain_node(model, data, idx, num_hops)
        feat_masks.append(feat_mask)
    stacked = torch.stack(feat_masks, dim=0)
    return {
        "mean_feat_importance": stacked.mean(dim=0),
        "std_feat_importance": stacked.std(dim=0),
    }
```

**Step 4: Run tests**

```bash
pytest tests/test_interpretability.py -v
```

**Step 5: Commit**

```bash
git add src/interpretability/explainer.py tests/test_interpretability.py
git commit -m "feat: add GNNExplainer wrapper for node interpretability"
```

---

## Task 9: Robustness Perturbations

**Files:**
- Create: `src/robustness/perturbations.py`

**Step 1: Write the failing test**

Create `tests/test_robustness.py`:

```python
import torch
from src.robustness.perturbations import add_feature_noise, drop_edges

def test_feature_noise_shape_preserved():
    x = torch.ones(100, 16)
    noisy = add_feature_noise(x, sigma=0.3, seed=42)
    assert noisy.shape == x.shape
    assert not torch.allclose(noisy, x)

def test_edge_drop_reduces_edges():
    edge_index = torch.randint(0, 50, (2, 200))
    dropped = drop_edges(edge_index, rate=0.3, seed=42)
    assert dropped.shape[1] < edge_index.shape[1]
    assert abs(dropped.shape[1] / edge_index.shape[1] - 0.7) < 0.05
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_robustness.py -v
```

**Step 3: Implement `src/robustness/perturbations.py`**

```python
"""Controlled perturbations for robustness evaluation."""
import torch
import numpy as np


def add_feature_noise(x: torch.Tensor, sigma: float, seed: int = 42) -> torch.Tensor:
    """Add Gaussian noise N(0, sigma) to all features."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    noise = torch.randn(x.shape, generator=rng) * sigma
    return x + noise


def drop_edges(edge_index: torch.Tensor, rate: float, seed: int = 42) -> torch.Tensor:
    """Randomly drop `rate` fraction of edges."""
    num_edges = edge_index.shape[1]
    num_keep = int(num_edges * (1 - rate))
    rng = np.random.default_rng(seed)
    keep_idx = rng.choice(num_edges, size=num_keep, replace=False)
    keep_idx = np.sort(keep_idx)
    return edge_index[:, keep_idx]
```

**Step 4: Run tests**

```bash
pytest tests/test_robustness.py -v
```

**Step 5: Commit**

```bash
git add src/robustness/perturbations.py tests/test_robustness.py
git commit -m "feat: add feature noise and edge drop perturbations"
```

---

## Task 10: Main Experiment Notebook

**Files:**
- Create: `notebooks/main_experiment.ipynb`

This is the main deliverable. Structure the notebook with the following sections (one cell per section header):

1. **Setup** — seed everything, load config
2. **Data Loading & EDA** — load Pokec-z, show attribute distributions, compute homophily
3. **Baseline GraphSAGE** — train, evaluate perf + fairness metrics
4. **Pre-processing: Resampling** — train with oversampled mask, evaluate
5. **FairGNN λ grid search** — train for each λ, pick best on val ΔDP+F1
6. **GNNExplainer Analysis** — explain 5 nodes per group, compare feature importance
7. **Robustness Experiments** — loop over noise levels + edge drop rates
8. **Synthesis** — Pareto curves (accuracy vs ΔDP), robustness curves, summary table

**Step 1: Create notebook skeleton**

Each section should be a markdown cell followed by code cells that call `src/` modules. Do not put experiment logic directly in notebook cells — import from `src/`.

**Step 2: Run full notebook end-to-end**

```bash
jupyter nbconvert --to notebook --execute notebooks/main_experiment.ipynb \
  --output notebooks/main_experiment_executed.ipynb
```
Expected: no errors, all cells execute successfully.

**Step 3: Commit**

```bash
git add notebooks/main_experiment.ipynb
git commit -m "feat: add main experiment notebook"
```

---

## Task 11: Analysis Note

**Files:**
- Create: `report/analysis_note.md`

Write a 1-2 page analysis note (in English) covering:
1. Experimental protocol (splits, metrics, hyperparameters)
2. Key results (accuracy/ΔDP/ΔEO table per method)
3. Trade-off analysis (FairGNN at optimal λ vs baseline, Pareto discussion)
4. Limitations (homophily, Pokec collection bias, GNNExplainer post-hoc nature, AI tool usage disclosure)

**Step 1: Fill in the note after running experiments**

Wait until Task 10 is complete and results are available.

**Step 2: Commit**

```bash
git add report/analysis_note.md
git commit -m "docs: add analysis note with results and trade-off discussion"
```

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Scaffold + pyproject.toml + config | — |
| 2 | Data loader + preprocessing | `tests/test_data.py` |
| 3 | Stratified splits | `tests/test_data.py` |
| 4 | GraphSAGE + trainer | `tests/test_models.py` |
| 5 | Fairness metrics | `tests/test_fairness.py` |
| 6 | Resampling pre-processing | `tests/test_fairness.py` |
| 7 | FairGNN adversarial | `tests/test_models.py` |
| 8 | GNNExplainer wrapper | `tests/test_interpretability.py` |
| 9 | Robustness perturbations | `tests/test_robustness.py` |
| 10 | Main notebook | End-to-end execution |
| 11 | Analysis note | — |
