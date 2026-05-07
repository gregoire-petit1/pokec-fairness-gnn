"""Preprocessing: age categorisation, sensitive attr extraction, z-score normalisation.

Pure polars + numpy + torch. Sensitive attributes (gender, region, age_group) are
exposed as long tensors on the :class:`Data` object so fairness metrics can read
them — including for multi-attribute and intersectional analyses.
"""

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data


def categorize_age(ages: pl.Series) -> pl.Series:
    """Bin ages into 0=young (<25), 1=adult (25-39), 2=senior (>=40).

    AGE values <= 0 (i.e. originally NA, filled to 0 by the loader) are tagged
    as ``-1`` so downstream fairness analyses can filter them out cleanly. This
    is a behavioural fix vs. the previous pandas implementation which silently
    folded NAs into the ``young`` bucket (~30 % pollution on Pokec-z).

    Args:
        ages: A polars :class:`Series` of numeric ages.

    Returns:
        A polars :class:`Series` of int codes in {-1, 0, 1, 2}.
    """
    return (
        ages.to_frame("age")
        .select(
            pl.when(pl.col("age") <= 0)
            .then(-1)
            .when(pl.col("age") < 25)
            .then(0)
            .when(pl.col("age") < 40)
            .then(1)
            .otherwise(2)
            .cast(pl.Int64)
            .alias("age_group")
        )
        .get_column("age_group")
    )


def preprocess(data: Data, sensitive_cols: list[str]) -> Data:
    """Extract sensitive attributes, drop them from x, z-score the remaining features.

    Mutates *data* in place:
        - For each name in ``sensitive_cols`` that exists as a column, attach it
          as a long tensor attribute (``data.gender``, ``data.region``, ...).
        - If ``AGE`` is present, also attach ``data.age_group`` (long tensor with
          values in {-1, 0, 1, 2}) and ``data.age_group_known`` (bool tensor).
        - Drop those columns from ``x`` and ``feature_cols``.
        - Z-score normalise the surviving features (vectorised, GPU-ready once
          ``data.to(device)`` is called downstream).

    Args:
        data: PyG :class:`Data` produced by :func:`load_pokec_z`. Must have
            ``raw_df`` (polars DataFrame) and ``feature_cols`` attributes.
        sensitive_cols: Column names to extract as sensitive attributes
            (typically ``["gender", "region"]``; ``age_group`` is added
            automatically when ``AGE`` is present).

    Returns:
        The same ``data`` object with updated tensors and attached attributes.
    """
    df: pl.DataFrame = data.raw_df  # polars

    # AGE → age_group (with -1 for NA), exposed as a sensitive attribute.
    if "AGE" in df.columns:
        age_group_series = categorize_age(df.get_column("AGE"))
        ag_np = age_group_series.to_numpy().astype(np.int64)
        data.age_group = torch.from_numpy(ag_np).long()
        data.age_group_known = torch.from_numpy(ag_np >= 0)

    # Extract requested sensitive columns as long tensors on the Data object.
    for col in sensitive_cols:
        if col in df.columns:
            arr = df.get_column(col).cast(pl.Int64).to_numpy().astype(np.int64)
            setattr(data, col, torch.from_numpy(arr).long())

    # Drop sensitive cols from feature matrix (and feature_cols list).
    remove_cols = {c for c in sensitive_cols if c in data.feature_cols}
    keep_cols = [c for c in data.feature_cols if c not in remove_cols]
    if remove_cols:
        keep_indices = torch.tensor(
            [data.feature_cols.index(c) for c in keep_cols], dtype=torch.long
        )
        data.x = data.x.index_select(dim=1, index=keep_indices)
    data.feature_cols = keep_cols

    # Z-score normalisation — vectorised torch ops, runs on whatever device x lives on.
    mean = data.x.mean(dim=0, keepdim=True)
    std = data.x.std(dim=0, keepdim=True).clamp(min=1e-8)
    data.x = (data.x - mean) / std

    return data
