"""Load Pokec-z dataset into a PyG Data object."""

import os

import pandas as pd
import torch
from torch_geometric.data import Data

# Target column selected based on target sweep (40 runs, 8 candidates, 5 seeds).
# completed_level_of_education_indicator: F1=0.939, ΔDP=0.037 — visible bias,
# well-balanced (47.7% positive), and academically interesting for debiasing.
# Backup: nefajcim (F1=0.940 but ΔDP≈0.005, too clean to demonstrate debiasing).
TARGET_COL = "completed_level_of_education_indicator"


def load_pokec_z(raw_dir: str) -> Data:
    """Load Pokec-z from raw CSV files and return a PyG Data object.

    The target is ``completed_level_of_education_indicator``, a binary column
    (0/1) that is already binarised in the raw data. It was selected from a
    sweep of 8 candidate targets because it yields a strong F1 (~0.939) with
    measurable demographic parity gap (ΔDP~0.037), making it suitable for
    demonstrating fairness-aware GNN methods.

    Sensitive attributes ``region`` and ``gender`` are kept in the feature
    matrix so that preprocessing can extract them before building x.

    Args:
        raw_dir: Path to the directory containing ``region_job_2.csv`` and
            ``region_job_2_relationship.txt``.

    Returns:
        A :class:`torch_geometric.data.Data` object with attributes:
        - ``x``: float feature matrix (all columns except ``user_id`` and
          ``completed_level_of_education_indicator``)
        - ``edge_index``: long tensor of shape ``[2, num_edges]``
        - ``y``: long tensor of binary education labels (already 0/1)
        - ``feature_cols``: list of column names corresponding to ``x``
        - ``raw_df``: the raw :class:`pandas.DataFrame` (used by preprocessing)
    """
    features_path = os.path.join(raw_dir, "region_job_2.csv")
    edges_path = os.path.join(raw_dir, "region_job_2_relationship.txt")

    df = pd.read_csv(features_path, sep=",")
    df = df.fillna(0)

    # Build edge index
    edges = pd.read_csv(edges_path, sep="\t", header=None, names=["src", "dst"])
    # Remap node ids to 0-indexed
    node_ids = {nid: idx for idx, nid in enumerate(df["user_id"].values)}
    edges = edges[edges["src"].isin(node_ids) & edges["dst"].isin(node_ids)].copy()
    src = edges["src"].map(node_ids).astype(int).values
    dst = edges["dst"].map(node_ids).astype(int).values
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Labels: completed_level_of_education_indicator is already binary (0/1).
    y = torch.tensor(df[TARGET_COL].values.astype(int), dtype=torch.long)

    # Features: all columns except user_id and the target column.
    # region and gender are kept here so preprocessing can extract them as
    # sensitive attribute vectors before removing them from x.
    feature_cols = [c for c in df.columns if c not in ["user_id", TARGET_COL]]
    x = torch.tensor(df[feature_cols].values, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.feature_cols = feature_cols
    data.raw_df = df
    return data
