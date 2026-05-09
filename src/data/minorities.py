"""Minority speaker attributes — re-extracted from SNAP raw, not in FairGNN curation.

The FairGNN-shipped Pokec subset (``region_job_2.csv``) keeps only 8 spoken
languages and aggregates them into binary indicator columns. The raw SNAP
``soc-pokec-profiles.txt`` file has the full free-text ``spoken_languages``
field where ``madarsky`` (Hungarian), ``cigansky``/``romsky`` (Roma) and
``posunkovu`` (Slovak Sign Language) appear. ``scripts/extract_snap_languages.py``
streams the raw 1.7 GB file once and emits ``data/raw/pokec-z/minority_speakers.csv``
with one row per subset user and binary ``hungarian``/``roma``/``sign`` columns.

This module reads that CSV and attaches the columns as long tensors on a
:class:`torch_geometric.data.Data` object, aligned to the loader's user_id
order. Pure polars + numpy + torch — no pandas, no Python loops.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data

MINORITY_AXES = ("hungarian", "roma", "sign")


def load_minority_speakers(
    raw_dir: str | Path,
    user_ids: torch.Tensor | np.ndarray | pl.Series,
) -> dict[str, torch.Tensor]:
    """Return ``{axis: long tensor of shape (N,)}`` aligned with ``user_ids``.

    Args:
        raw_dir: Directory containing ``minority_speakers.csv``.
        user_ids: Vector of user_ids in the order used by the loader (i.e.
            ``data.raw_df["user_id"]``). Determines tensor row order.

    Returns:
        Dict mapping each axis name in :data:`MINORITY_AXES` to a long tensor.
        Users absent from the CSV (e.g. SNAP raw not available) are
        encoded as ``0`` on every axis.

    Raises:
        FileNotFoundError: If ``minority_speakers.csv`` is missing. Run
            ``scripts/extract_snap_languages.py`` first.
    """
    raw_dir = Path(raw_dir)
    csv_path = raw_dir / "minority_speakers.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found — run scripts/extract_snap_languages.py first."
        )

    # Convert user_ids to a polars Series for the join.
    if isinstance(user_ids, torch.Tensor):
        uid_np = user_ids.detach().cpu().numpy()
    elif isinstance(user_ids, pl.Series):
        uid_np = user_ids.to_numpy()
    else:
        uid_np = np.asarray(user_ids)

    order = pl.DataFrame({"user_id": uid_np.astype(np.int64)}).with_columns(
        pl.int_range(0, pl.len()).alias("__order")
    )
    minority = pl.read_csv(csv_path).with_columns(pl.col("user_id").cast(pl.Int64))

    joined = (
        order.join(minority, on="user_id", how="left")
        .with_columns(
            pl.col("hungarian").fill_null(0).cast(pl.Int64),
            pl.col("roma").fill_null(0).cast(pl.Int64),
            pl.col("sign").fill_null(0).cast(pl.Int64),
        )
        .sort("__order")
    )
    out: dict[str, torch.Tensor] = {}
    for axis in MINORITY_AXES:
        arr = joined.get_column(axis).to_numpy().astype(np.int64)
        out[axis] = torch.from_numpy(arr).long()
    return out


def attach_minorities(data: Data, raw_dir: str | Path) -> Data:
    """Attach ``data.hungarian``, ``data.roma``, ``data.sign`` long tensors.

    Reads ``minority_speakers.csv`` and aligns rows to ``data.raw_df["user_id"]``.
    Mutates and returns ``data``.
    """
    if not hasattr(data, "raw_df"):
        raise AttributeError(
            "data has no `raw_df`; call load_pokec_z first so user_ids are available."
        )
    user_ids = data.raw_df.get_column("user_id")
    tensors = load_minority_speakers(raw_dir, user_ids)
    for axis, t in tensors.items():
        setattr(data, axis, t)
    return data
