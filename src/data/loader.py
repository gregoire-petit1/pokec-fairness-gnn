"""Load Pokec-z dataset into a PyG Data object (polars-backed, no pandas)."""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data

TARGET_COL = "completed_level_of_education_indicator"


def load_pokec_z(raw_dir: str | Path) -> Data:
    """Load Pokec-z from raw CSV files and return a PyG Data object.

    Pure polars + numpy + torch path: no pandas, no Python loops on rows. The
    raw polars DataFrame is attached to ``data.raw_df`` so downstream
    ``preprocess()`` can extract sensitive attribute columns.

    Args:
        raw_dir: Path to the directory containing ``region_job_2.csv`` and
            ``region_job_2_relationship.txt`` (the FairGNN Pokec-z subset).

    Returns:
        :class:`torch_geometric.data.Data` with ``x``, ``edge_index``, ``y``,
        ``feature_cols`` and ``raw_df`` (polars DataFrame).
    """
    raw_dir = Path(raw_dir)
    features_path = raw_dir / "region_job_2.csv"
    edges_path = raw_dir / "region_job_2_relationship.txt"

    # Features CSV — read, fill nulls, append a 0-indexed node id used to remap edges.
    df = pl.read_csv(features_path).fill_null(0)
    df = df.with_row_index(name="node_idx", offset=0)
    df = df.with_columns(pl.col("node_idx").cast(pl.Int64))

    # Edges (TSV, no header) — read raw, then map both endpoints via inner-joins
    # on user_id. This replaces the dict-comprehension + .isin() + .map() chain
    # with a single declarative pipeline (no Python loops over edges).
    edges = pl.read_csv(
        edges_path,
        separator="\t",
        has_header=False,
        new_columns=["src_user_id", "dst_user_id"],
    )
    id_lookup = df.select(["user_id", "node_idx"])
    edges = edges.join(
        id_lookup.rename({"user_id": "src_user_id", "node_idx": "src"}),
        on="src_user_id",
        how="inner",
    )
    edges = edges.join(
        id_lookup.rename({"user_id": "dst_user_id", "node_idx": "dst"}),
        on="dst_user_id",
        how="inner",
    )

    src_np = edges.get_column("src").to_numpy().astype(np.int64)
    dst_np = edges.get_column("dst").to_numpy().astype(np.int64)
    edge_index = torch.from_numpy(np.stack([src_np, dst_np], axis=0)).long()

    # Labels — already binary 0/1 in this subset.
    y = torch.from_numpy(
        df.get_column(TARGET_COL).cast(pl.Int64).to_numpy().astype(np.int64)
    ).long()

    # Features = everything except user_id, node_idx, and the target.
    # ``region`` and ``gender`` are kept here so preprocessing can extract them
    # as sensitive attribute tensors before removing them from x.
    excluded = {"user_id", "node_idx", TARGET_COL}
    feature_cols = [c for c in df.columns if c not in excluded]
    x = torch.from_numpy(
        df.select(feature_cols).cast(pl.Float32).to_numpy().astype(np.float32)
    ).float()

    data = Data(x=x, edge_index=edge_index, y=y)
    data.feature_cols = feature_cols
    data.raw_df = df  # polars DataFrame
    return data
