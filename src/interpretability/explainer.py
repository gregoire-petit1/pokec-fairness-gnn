"""GNNExplainer wrapper for PyG."""

import torch
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer


def explain_node(
    model: torch.nn.Module,
    data: Data,
    node_idx: int,
    num_hops: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run GNNExplainer on a single node. Returns (edge_mask, feature_mask).

    edge_mask shape: (num_edges,)
    feature_mask shape: (num_features,)
    """
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )
    explanation = explainer(data.x, data.edge_index, index=node_idx)
    return explanation.edge_mask, explanation.node_mask[node_idx]


def explain_group(
    model: torch.nn.Module,
    data: Data,
    node_indices: list[int],
    num_hops: int = 2,
) -> dict:
    """Explain a group of nodes, return aggregated feature importance."""
    feat_masks = []
    for idx in node_indices:
        _, feat_mask = explain_node(model, data, idx, num_hops)
        feat_masks.append(feat_mask)
    stacked = torch.stack(feat_masks, dim=0)
    return {
        "mean_feat_importance": stacked.mean(dim=0),
        "std_feat_importance": stacked.std(dim=0),
    }


def explain_group_with_edges(
    model: torch.nn.Module,
    data: Data,
    node_indices: list[int],
    num_hops: int = 2,
    edge_intra_attr: torch.Tensor | None = None,
    top_edge_frac: float = 0.10,
) -> dict:
    """Aggregate feature- and edge-importance over a sample of nodes.

    Streaming aggregation (no per-node stack) to keep memory bounded —
    needed because ``edge_mask`` is one float per edge in the full
    graph, and stacking 100 nodes × 700 k edges would burn ~280 MB
    on Pokec-z.

    Args:
        model: trained GNN with ``forward(x, edge_index)`` returning logits.
        data: PyG Data object.
        node_indices: which test nodes to explain (list of ints).
        num_hops: explainer search depth.
        edge_intra_attr: optional bool/int tensor of length ``num_edges``
            equal to 1 when an edge connects two nodes of the same value
            on some sensitive attribute (typically built via
            ``data.region[edge_index[0]] == data.region[edge_index[1]]``).
            Used to compute the fraction of intra-attribute edges among
            the most important edges.
        top_edge_frac: fraction of top-importance edges to inspect for
            the intra-attribute statistic (default top 10 %).

    Returns:
        Dict with :
          ``mean_feat_importance`` : 1-D tensor (num_features,)
          ``mean_edge_importance`` : 1-D tensor (num_edges,)
          ``intra_attr_fraction_global`` : float, baseline rate of
              intra-attribute edges across the full graph.
          ``intra_attr_fraction_top`` : float, rate among the top
              ``top_edge_frac`` of edges by mean importance.
    """
    n_features = data.x.shape[1]
    n_edges = data.edge_index.shape[1]
    sum_feat = torch.zeros(n_features, dtype=torch.float64)
    sum_edge = torch.zeros(n_edges, dtype=torch.float64)
    n_explained = 0

    for idx in node_indices:
        edge_mask, feat_mask = explain_node(model, data, idx, num_hops)
        sum_feat = sum_feat + feat_mask.detach().cpu().double()
        sum_edge = sum_edge + edge_mask.detach().cpu().double()
        n_explained += 1

    if n_explained == 0:
        raise ValueError("node_indices was empty")

    mean_feat = (sum_feat / n_explained).float()
    mean_edge = (sum_edge / n_explained).float()

    out: dict = {
        "mean_feat_importance": mean_feat,
        "mean_edge_importance": mean_edge,
        "n_explained": n_explained,
    }

    if edge_intra_attr is not None:
        intra = edge_intra_attr.detach().cpu().bool()
        if intra.shape[0] != n_edges:
            raise ValueError(f"edge_intra_attr has {intra.shape[0]} entries, expected {n_edges}")
        global_frac = float(intra.float().mean())
        top_k = max(1, int(top_edge_frac * n_edges))
        top_edges = mean_edge.argsort(descending=True)[:top_k]
        top_intra_frac = float(intra[top_edges].float().mean())
        out["intra_attr_fraction_global"] = global_frac
        out["intra_attr_fraction_top"] = top_intra_frac

    return out
