"""Pre-processing fairness: oversample minority (label × sensitive) groups.

Vectorised: builds repetition counts via :func:`numpy.repeat` and a single
:func:`numpy.random.Generator.choice` for the remainder. No Python loop
iterates over training samples; the only loop is over distinct (label×sensitive)
groups, which has at most :math:`|\\text{labels}| \\times |\\text{groups}|`
iterations (typically ≤ 6 on Pokec-z).
"""

import numpy as np
import torch


def oversample_train_indices(
    train_mask: torch.Tensor,
    y: torch.Tensor,
    sensitive: torch.Tensor,
    seed: int = 42,
) -> torch.Tensor:
    """Return a 1-D long tensor of training **indices** (with duplicates) so
    every (label × sensitive) cell reaches the size of the largest cell.

    Note: returned tensor contains duplicate indices — use index-based selection
    downstream (i.e. ``data.x[train_idx]``), NOT boolean masking. The legacy
    name ``oversample_train_mask`` was a misnomer (the return value is not a
    boolean mask) and is kept as a thin alias for backwards compatibility.
    """
    train_idx = train_mask.detach().cpu().nonzero(as_tuple=True)[0].numpy()
    y_train = y[train_mask].detach().cpu().numpy().astype(np.int64)
    s_train = sensitive[train_mask].detach().cpu().numpy().astype(np.int64)

    # Stratification key per training row.
    n_s = int(s_train.max() + 1) if s_train.size else 1
    strat = y_train * (n_s + 1) + s_train  # +1 to keep keys disjoint
    unique, counts = np.unique(strat, return_counts=True)
    if unique.size == 0:
        return torch.empty(0, dtype=torch.long)
    max_count = int(counts.max())

    # Per-row replication factor (max_count // count_of_its_group), vectorised
    # via a lookup array indexed by the strat key. Range fits in int64 easily.
    factor_lookup = np.zeros(int(unique.max()) + 1, dtype=np.int64)
    factor_lookup[unique] = max_count // counts
    per_row_factor = factor_lookup[strat]

    repeated = np.repeat(train_idx, per_row_factor)

    # For each group, pad up to max_count with random extras (with replacement).
    rng = np.random.default_rng(seed)
    extras_list: list[np.ndarray] = []
    # Loop over k groups (small bounded k ≤ 6 on Pokec-z) — the inner work is vectorised.
    for g, c in zip(unique, counts, strict=True):
        deficit = max_count - (max_count // c) * c
        if deficit > 0:
            g_indices = train_idx[strat == g]
            extras_list.append(rng.choice(g_indices, size=deficit, replace=True))

    all_idx = np.concatenate([repeated, *extras_list]) if extras_list else repeated
    return torch.from_numpy(all_idx).long()


# Legacy alias — keep until callers are migrated. The name was misleading
# (returned indices, not a mask) but is referenced in tests/notebooks.
oversample_train_mask = oversample_train_indices
