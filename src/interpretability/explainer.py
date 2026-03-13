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
