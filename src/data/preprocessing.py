"""Preprocessing: age categorization, feature normalization, sensitive attr removal."""

import pandas as pd
import torch
from torch_geometric.data import Data


def categorize_age(ages: pd.Series) -> pd.Series:
    """Categorize numeric age into young / adult / senior.

    Bins:
    - young  : [0,  25)
    - adult  : [25, 40)
    - senior : [40, 200)

    Args:
        ages: A :class:`pandas.Series` of numeric age values.

    Returns:
        A :class:`pandas.Categorical` Series with string labels.
    """
    bins = [0, 25, 40, 200]
    labels = ["young", "adult", "senior"]
    return pd.cut(ages, bins=bins, labels=labels, right=False)


def preprocess(data: Data, sensitive_cols: list[str]) -> Data:
    """Remove sensitive columns from features and normalize remaining features.

    The function:
    1. Optionally categorizes the ``AGE`` column into an ``age_group`` attribute.
    2. Stores each sensitive column as a named attribute on *data* (so downstream
       fairness metrics can access them).
    3. Drops sensitive columns from ``data.x``.
    4. Applies z-score normalization to the remaining features.

    Args:
        data: A PyG :class:`~torch_geometric.data.Data` object produced by
            :func:`load_pokec_z`.  Must have ``raw_df`` and ``feature_cols``
            attributes.
        sensitive_cols: Column names that should be removed from ``data.x``
            (e.g. ``["gender", "age_group"]``).

    Returns:
        The mutated *data* object with updated ``x`` and ``feature_cols``.
    """
    df = data.raw_df.copy()

    # Categorize age and store as attribute
    if "AGE" in df.columns:
        df["age_group"] = categorize_age(df["AGE"]).cat.codes  # -1, 0, 1, 2
        data.age_group = torch.tensor(df["age_group"].values, dtype=torch.long)

    # Store sensitive attrs separately so fairness metrics can use them
    for col in sensitive_cols:
        if col in df.columns:
            setattr(data, col, torch.tensor(df[col].values, dtype=torch.long))

    # Remove sensitive cols from feature matrix
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
