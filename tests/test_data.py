"""Data layer tests — polars-based, no pandas."""

import os

import polars as pl
import pytest
import torch

from src.data.loader import load_pokec_z
from src.data.preprocessing import categorize_age, preprocess
from src.data.splits import make_splits

_requires_data = pytest.mark.skipif(
    not os.path.exists("data/raw/pokec-z/region_job_2.csv"),
    reason="Raw Pokec-z data not available",
)


@pytest.mark.smoke
def test_age_group_categorization():
    ages = pl.Series("AGE", [20, 30, 50])
    codes = categorize_age(ages).to_list()
    assert codes == [0, 1, 2]


@pytest.mark.smoke
def test_age_group_handles_missing():
    """AGE <= 0 maps to -1 (missing) rather than silently being 'young'."""
    ages = pl.Series("AGE", [0, -1, 18, 25, 40])
    codes = categorize_age(ages).to_list()
    assert codes == [-1, -1, 0, 1, 2]


@pytest.mark.smoke
def test_splits_sizes():
    n = 1000
    y = torch.randint(0, 2, (n,))
    gender = torch.randint(0, 2, (n,))
    train, val, test = make_splits(n, y, gender, ratios=(0.6, 0.2, 0.2), seed=42)
    assert abs(len(train) / n - 0.6) < 0.02
    assert abs(len(val) / n - 0.2) < 0.02
    assert abs(len(test) / n - 0.2) < 0.02


@pytest.mark.smoke
def test_splits_no_overlap():
    n = 1000
    y = torch.randint(0, 2, (n,))
    gender = torch.randint(0, 2, (n,))
    train, val, test = make_splits(n, y, gender, ratios=(0.6, 0.2, 0.2), seed=42)
    train_set, val_set, test_set = set(train.tolist()), set(val.tolist()), set(test.tolist())
    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0


@_requires_data
def test_load_returns_pyg_data():
    data = load_pokec_z("data/raw/pokec-z")
    assert hasattr(data, "x")
    assert hasattr(data, "edge_index")
    assert hasattr(data, "y")
    # The raw_df attached to data is a polars DataFrame, not pandas.
    assert isinstance(data.raw_df, pl.DataFrame)


@_requires_data
def test_sensitive_attrs_not_in_features():
    data = load_pokec_z("data/raw/pokec-z")
    original_num_features = data.x.shape[1]
    processed = preprocess(data, sensitive_cols=["gender", "region"])
    assert processed.x.shape[1] < original_num_features
    for col in ["gender", "region"]:
        assert col not in processed.feature_cols
        assert hasattr(processed, col)
    # age_group is automatically computed when AGE column exists
    assert hasattr(processed, "age_group")
    assert hasattr(processed, "age_group_known")
