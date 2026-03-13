"""Stratified train/val/test split stratified on label + sensitive attribute."""

import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def make_splits(
    n: int,
    y: torch.Tensor,
    sensitive: torch.Tensor,
    ratios: tuple = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return train/val/test index tensors, stratified on y x sensitive."""
    assert abs(sum(ratios) - 1.0) < 1e-9, f"ratios must sum to 1.0, got {sum(ratios)}"
    # Combined stratification label
    # Use .tolist() -> np.array() to avoid torch/numpy ABI compatibility issues
    strat = np.array(y.tolist()) * 10 + np.array(sensitive.tolist())
    idx = np.arange(n)

    train_ratio, val_ratio, _ = ratios
    test_ratio = 1 - train_ratio - val_ratio

    train_idx, temp_idx = train_test_split(
        idx, test_size=(1 - train_ratio), stratify=strat, random_state=seed
    )
    strat_temp = strat[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=strat_temp,
        random_state=seed,
    )
    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx, dtype=torch.long),
        torch.tensor(test_idx, dtype=torch.long),
    )


def save_splits(train: torch.Tensor, val: torch.Tensor, test: torch.Tensor, out_dir: str) -> None:
    """Save split index tensors to disk."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(train, os.path.join(out_dir, "train.pt"))
    torch.save(val, os.path.join(out_dir, "val.pt"))
    torch.save(test, os.path.join(out_dir, "test.pt"))


def load_splits(out_dir: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load split index tensors from disk."""
    train = torch.load(os.path.join(out_dir, "train.pt"), weights_only=True)
    val = torch.load(os.path.join(out_dir, "val.pt"), weights_only=True)
    test = torch.load(os.path.join(out_dir, "test.pt"), weights_only=True)
    return train, val, test
