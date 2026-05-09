import pytest
import torch
from torch_geometric.data import Data

from src.interpretability.explainer import explain_group_with_edges, explain_node
from src.models.graphsage import GraphSAGE


def _toy_data_and_model(seed: int = 0) -> tuple[Data, GraphSAGE]:
    torch.manual_seed(seed)
    x = torch.randn(50, 8)
    edge_index = torch.randint(0, 50, (2, 150))
    data = Data(x=x, edge_index=edge_index)
    model = GraphSAGE(in_channels=8, hidden_channels=16, out_channels=2, num_layers=2)
    return data, model


def test_explain_node_returns_masks():
    data, model = _toy_data_and_model()
    edge_mask, feat_mask = explain_node(model, data, node_idx=0, num_hops=2)
    assert edge_mask.shape[0] == data.edge_index.shape[1]
    assert feat_mask.shape[0] == data.x.shape[1]


@pytest.mark.smoke
def test_explain_group_with_edges_aggregates():
    data, model = _toy_data_and_model()
    out = explain_group_with_edges(
        model=model,
        data=data,
        node_indices=[0, 5, 10],
        num_hops=2,
    )
    assert out["mean_feat_importance"].shape[0] == data.x.shape[1]
    assert out["mean_edge_importance"].shape[0] == data.edge_index.shape[1]
    assert out["n_explained"] == 3
    assert "intra_attr_fraction_global" not in out


@pytest.mark.smoke
def test_explain_group_with_edges_intra_attr_fraction():
    data, model = _toy_data_and_model()
    intra = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
    intra[: data.edge_index.shape[1] // 2] = True
    out = explain_group_with_edges(
        model=model,
        data=data,
        node_indices=[0, 5, 10],
        edge_intra_attr=intra,
        top_edge_frac=0.20,
    )
    assert 0.0 <= out["intra_attr_fraction_global"] <= 1.0
    assert 0.0 <= out["intra_attr_fraction_top"] <= 1.0


@pytest.mark.smoke
def test_explain_group_rejects_empty():
    data, model = _toy_data_and_model()
    with pytest.raises(ValueError, match="empty"):
        explain_group_with_edges(model, data, node_indices=[])


@pytest.mark.smoke
def test_explain_group_rejects_intra_attr_size_mismatch():
    data, model = _toy_data_and_model()
    bad_intra = torch.zeros(10, dtype=torch.bool)
    with pytest.raises(ValueError, match="entries"):
        explain_group_with_edges(model, data, node_indices=[0], edge_intra_attr=bad_intra)
