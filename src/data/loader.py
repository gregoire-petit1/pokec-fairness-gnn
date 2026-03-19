"""Load Pokec-z dataset into a PyG Data object."""

import os

import pandas as pd
import torch
from torch_geometric.data import Data


def load_pokec_z(raw_dir: str) -> Data:
    """Load Pokec-z from raw CSV files and return a PyG Data object.

    Following the FairGNN/NIFTY standard, the target is ``I_am_working_in_field``
    (binarised: 0 if no occupation reported, 1 otherwise), while ``region`` and
    ``gender`` serve as sensitive attributes and remain available in the feature
    matrix for downstream filtering.

    Args:
        raw_dir: Path to the directory containing ``region_job_2.csv`` and
            ``region_job_2_relationship.txt``.

    Returns:
        A :class:`torch_geometric.data.Data` object with attributes:
        - ``x``: float feature matrix (all columns except ``user_id`` and
          ``I_am_working_in_field``)
        - ``edge_index``: long tensor of shape ``[2, num_edges]``
        - ``y``: long tensor of binary occupation labels (0 vs >0)
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

    # Labels: I_am_working_in_field binarised (0 = no occupation, 1 = has occupation)
    # This is the standard target in FairGNN / NIFTY / FairGLite papers on Pokec-z.
    y = torch.tensor((df["I_am_working_in_field"].values > 0).astype(int), dtype=torch.long)

    # Features: all columns except user_id and the target column.
    # region and gender are kept here so preprocessing can extract them as
    # sensitive attribute vectors before removing them from x.
    feature_cols = [c for c in df.columns if c not in ["user_id", "I_am_working_in_field"]]
    x = torch.tensor(df[feature_cols].values, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.feature_cols = feature_cols
    data.raw_df = df
    return data
