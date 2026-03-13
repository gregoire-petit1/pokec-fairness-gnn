import torch
from torch_geometric.data import Data
from src.models.graphsage import GraphSAGE


def test_graphsage_forward_shape():
    x = torch.randn(100, 16)
    edge_index = torch.randint(0, 100, (2, 300))
    data = Data(x=x, edge_index=edge_index)
    model = GraphSAGE(in_channels=16, hidden_channels=32, out_channels=4, num_layers=2, dropout=0.5)
    out = model(data.x, data.edge_index)
    assert out.shape == (100, 4)


def test_graphsage_get_embeddings_shape():
    x = torch.randn(100, 16)
    edge_index = torch.randint(0, 100, (2, 300))
    model = GraphSAGE(in_channels=16, hidden_channels=32, out_channels=4, num_layers=2, dropout=0.5)
    emb = model.get_embeddings(x, edge_index)
    assert emb.shape == (100, 32)


def test_trainer_evaluate_returns_acc_f1():
    from src.models.trainer import evaluate

    x = torch.randn(50, 8)
    edge_index = torch.randint(0, 50, (2, 100))
    y = torch.randint(0, 2, (50,))
    data = Data(x=x, edge_index=edge_index, y=y)
    model = GraphSAGE(in_channels=8, hidden_channels=16, out_channels=2, num_layers=2, dropout=0.0)
    mask = torch.ones(50, dtype=torch.bool)
    acc, f1 = evaluate(model, data, mask)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0
