"""FairDrop: biased edge dropout for fairness-aware graph pre-processing.

Reference:
    Spinelli, I. et al. (2021). FairDrop: Biased Edge Dropout for Enhancing
    Fairness in Graph Representation Learning. IEEE TNNLS.
    (Reviewed in Laclau et al., 2024 – Survey on Fairness for ML on Graphs.)

Key idea:
    Social graphs exhibit homophily: nodes with the same sensitive attribute
    (e.g. same gender) tend to connect with each other. This structural bias
    is captured by the assortative mixing coefficient r. FairDrop mitigates it
    by preferentially dropping *intra-group* edges (between nodes sharing the
    same sensitive attribute value), reducing r toward 0 (random mixing).

    Unlike random edge dropping used in robustness analysis, FairDrop is a
    *targeted* pre-processing step applied before or during training.
"""

import torch


def fairdrop(
    edge_index: torch.Tensor,
    sensitive: torch.Tensor,
    drop_rate: float = 0.3,
    intra_group_bias: float = 2.0,
    seed: int = 42,
) -> torch.Tensor:
    """Biased edge dropout that preferentially removes intra-group edges.

    Intra-group edges (connecting two nodes with the same sensitive attribute
    value) are dropped with probability ``drop_rate * intra_group_bias``, while
    inter-group edges are dropped with probability ``drop_rate``. Both
    probabilities are clipped to [0, 1].

    Args:
        edge_index: Edge index tensor of shape ``[2, E]``.
        sensitive: Binary sensitive attribute tensor of shape ``[N]``.
        drop_rate: Base edge drop probability applied to inter-group edges.
            Intra-group edges are dropped at ``drop_rate * intra_group_bias``.
        intra_group_bias: Multiplier applied to the drop probability for
            intra-group edges (default: 2.0 → twice as likely to be dropped).
        seed: Random seed for reproducibility.

    Returns:
        Filtered edge index tensor of shape ``[2, E']`` with E' ≤ E.

    Example:
        >>> edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        >>> sensitive = torch.tensor([0, 0, 1, 1])
        >>> # All edges are intra-group here; they will be dropped more often.
        >>> dropped = fairdrop(edge_index, sensitive, drop_rate=0.5, seed=0)
    """
    if edge_index.shape[1] == 0:
        return edge_index

    src, dst = edge_index[0], edge_index[1]

    # True where the edge connects two nodes in the same sensitive group
    intra_mask = sensitive[src] == sensitive[dst]

    # Build per-edge drop probability
    p_inter = torch.full((edge_index.shape[1],), drop_rate, dtype=torch.float)
    p_intra = torch.full(
        (edge_index.shape[1],), drop_rate * intra_group_bias, dtype=torch.float
    ).clamp(max=1.0)
    drop_probs = torch.where(intra_mask, p_intra, p_inter)

    # Sample edges to keep (Bernoulli with p_keep = 1 - drop_prob)
    generator = torch.Generator()
    generator.manual_seed(seed)
    keep_mask = torch.bernoulli(1.0 - drop_probs, generator=generator).bool()

    return edge_index[:, keep_mask]


def fairdrop_stats(
    original_edge_index: torch.Tensor,
    dropped_edge_index: torch.Tensor,
    sensitive: torch.Tensor,
) -> dict:
    """Compute statistics comparing original and FairDrop-filtered graph.

    Returns a dict with:
        - ``n_edges_original``: number of edges before FairDrop
        - ``n_edges_dropped``: number of edges after FairDrop
        - ``intra_fraction_original``: fraction of intra-group edges before
        - ``intra_fraction_dropped``: fraction of intra-group edges after
        - ``drop_rate_actual``: actual fraction of edges removed
    """

    def _intra_fraction(ei: torch.Tensor) -> float:
        if ei.shape[1] == 0:
            return 0.0
        s, d = ei[0], ei[1]
        return float((sensitive[s] == sensitive[d]).float().mean().item())

    n_orig = original_edge_index.shape[1]
    n_drop = dropped_edge_index.shape[1]

    return {
        "n_edges_original": n_orig,
        "n_edges_dropped": n_drop,
        "intra_fraction_original": _intra_fraction(original_edge_index),
        "intra_fraction_dropped": _intra_fraction(dropped_edge_index),
        "drop_rate_actual": float(1.0 - n_drop / max(n_orig, 1)),
    }
