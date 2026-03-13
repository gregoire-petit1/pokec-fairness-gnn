"""Load Pokec-z dataset into a PyG Data object."""

import os

import pandas as pd
import torch
from torch_geometric.data import Data


def load_pokec_z(raw_dir: str) -> Data:
    """Load Pokec-z from raw CSV files and return a PyG Data object.

    Args:
        raw_dir: Path to the directory containing ``region_job_2.csv`` and
            ``region_job_2_relationship.txt``.

    Returns:
        A :class:`torch_geometric.data.Data` object with attributes:
        - ``x``: float feature matrix (all columns except ``user_id``)
        - ``edge_index``: long tensor of shape ``[2, num_edges]``
        - ``y``: long tensor of binary region labels
        - ``feature_cols``: list of column names corresponding to ``x``
        - ``raw_df``: the raw :class:`pandas.DataFrame` (used by preprocessing)
    """
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
