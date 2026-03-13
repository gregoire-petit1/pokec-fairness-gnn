"""Pre-processing fairness: oversample minority groups in training set."""

import numpy as np
import torch
from sklearn.utils import resample


def oversample_train_mask(
    train_mask: torch.Tensor,
    y: torch.Tensor,
    sensitive: torch.Tensor,
    seed: int = 42,
) -> torch.Tensor:
    """Oversample minority (label x sensitive group) combinations in train set.

    Returns a new index tensor over all nodes that includes oversampled indices.
    Note: indices can appear more than once — use index-based masking downstream.
    """
    train_idx = np.array(train_mask.nonzero(as_tuple=True)[0].tolist())
    y_train = np.array(y[train_mask].tolist())
    s_train = np.array(sensitive[train_mask].tolist())
    strat = y_train * 10 + s_train

    # Find size of majority group
    unique, counts = np.unique(strat, return_counts=True)
    max_count = counts.max()

    resampled_idx = []
    for group in unique:
        group_idx = train_idx[strat == group]
        if len(group_idx) < max_count:
            group_idx = resample(group_idx, n_samples=max_count, replace=True, random_state=seed)
        resampled_idx.append(group_idx)

    all_idx = np.concatenate(resampled_idx)
    return torch.tensor(all_idx, dtype=torch.long)
