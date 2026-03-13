import torch
from torch_geometric.data import Data
from src.models.graphsage import GraphSAGE
from src.interpretability.explainer import explain_node


def test_explain_node_returns_masks():
    x = torch.randn(50, 8)
    edge_index = torch.randint(0, 50, (2, 150))
    data = Data(x=x, edge_index=edge_index)
    model = GraphSAGE(in_channels=8, hidden_channels=16, out_channels=2, num_layers=2)
    edge_mask, feat_mask = explain_node(model, data, node_idx=0, num_hops=2)
    assert edge_mask is not None
    assert feat_mask is not None
    assert edge_mask.shape[0] == edge_index.shape[1]
    assert feat_mask.shape[0] == x.shape[1]
