"""Smoke tests for the minority fairness pipeline.

Covers the loader extension (``src/data/minorities.py``) and the row-builder
in ``scripts/run_minority_fairness.py``. CPU-only, runs in <1s — meant to gate
the main script before launching the full GPU run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from src.data.minorities import MINORITY_AXES, load_minority_speakers


@pytest.fixture
def synth_minority_csv(tmp_path: Path) -> Path:
    """Tiny CSV with 5 user_ids and known minority flags."""
    df = pl.DataFrame(
        {
            "user_id": [10, 20, 30, 40, 50],
            "hungarian": [1, 0, 1, 0, 0],
            "roma": [0, 0, 0, 1, 0],
            "sign": [0, 0, 0, 0, 1],
        }
    )
    out = tmp_path / "minority_speakers.csv"
    df.write_csv(out)
    return tmp_path


def test_load_minority_speakers_aligned(synth_minority_csv: Path) -> None:
    """Tensor rows must follow the requested user_id order, not the CSV order."""
    user_ids = torch.tensor([30, 10, 50, 99], dtype=torch.long)  # 99 absent → 0s
    out = load_minority_speakers(synth_minority_csv, user_ids)

    assert set(out.keys()) == set(MINORITY_AXES)
    assert out["hungarian"].tolist() == [1, 1, 0, 0]
    assert out["roma"].tolist() == [0, 0, 0, 0]
    assert out["sign"].tolist() == [0, 0, 1, 0]


def test_load_minority_speakers_dtype_long(synth_minority_csv: Path) -> None:
    user_ids = np.array([10, 20], dtype=np.int64)
    out = load_minority_speakers(synth_minority_csv, user_ids)
    for axis in MINORITY_AXES:
        assert out[axis].dtype == torch.long
        assert out[axis].shape == (2,)


def test_load_minority_speakers_missing_csv(tmp_path: Path) -> None:
    user_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    with pytest.raises(FileNotFoundError, match="minority_speakers"):
        load_minority_speakers(tmp_path, user_ids)


def test_load_minority_speakers_polars_series(synth_minority_csv: Path) -> None:
    """Accepts polars Series as user_ids input (matches data.raw_df interface)."""
    uid_series = pl.Series("user_id", [20, 40], dtype=pl.Int64)
    out = load_minority_speakers(synth_minority_csv, uid_series)
    assert out["hungarian"].tolist() == [0, 0]
    assert out["roma"].tolist() == [0, 1]


def test_attach_minorities_uses_raw_df(synth_minority_csv: Path) -> None:
    """attach_minorities reads user_ids from data.raw_df and sets attributes."""
    from src.data.minorities import attach_minorities

    class _StubData:
        pass

    data = _StubData()
    data.raw_df = pl.DataFrame({"user_id": [10, 30, 99]})
    out = attach_minorities(data, synth_minority_csv)
    assert torch.equal(out.hungarian, torch.tensor([1, 1, 0], dtype=torch.long))
    assert torch.equal(out.roma, torch.tensor([0, 0, 0], dtype=torch.long))
    assert torch.equal(out.sign, torch.tensor([0, 0, 0], dtype=torch.long))


def test_attach_minorities_requires_raw_df(synth_minority_csv: Path) -> None:
    from src.data.minorities import attach_minorities

    class _StubData:
        pass

    with pytest.raises(AttributeError, match="raw_df"):
        attach_minorities(_StubData(), synth_minority_csv)
