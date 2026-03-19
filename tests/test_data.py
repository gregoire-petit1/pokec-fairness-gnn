import os

import pytest
from src.data.loader import load_pokec_z
from src.data.preprocessing import preprocess

_requires_data = pytest.mark.skipif(
    not os.path.exists("data/raw/pokec-z/region_job_2.csv"),
    reason="Raw Pokec-z data not available",
)


@_requires_data
def test_load_returns_pyg_data():
    data = load_pokec_z("data/raw/pokec-z")
    assert hasattr(data, "x")
    assert hasattr(data, "edge_index")
    assert hasattr(data, "y")


@_requires_data
def test_sensitive_attrs_not_in_features():
    data = load_pokec_z("data/raw/pokec-z")
    original_num_features = data.x.shape[1]
    processed = preprocess(data, sensitive_cols=["gender", "region"])
    assert processed.x.shape[1] < original_num_features
    for col in ["gender", "region"]:
        assert col not in processed.feature_cols


def test_age_group_categorization():
    import pandas as pd
    from src.data.preprocessing import categorize_age

    ages = pd.Series([20, 30, 50])
    groups = categorize_age(ages)
    assert list(groups) == ["young", "adult", "senior"]


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
    train_set, val_set, test_set = set(train.tolist()), set(val.tolist()), set(test.tolist())
    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0
