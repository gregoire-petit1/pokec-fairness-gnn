"""Training loop with early stopping for node classification."""

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Data


def train_epoch(
    model: torch.nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    train_mask: torch.Tensor,
) -> float:
    """Run one training epoch. Returns cross-entropy loss."""
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: Data,
    mask: torch.Tensor,
) -> tuple[float, float]:
    """Evaluate model on masked nodes. Returns (accuracy, f1_macro)."""
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=1)
    acc = (pred == data.y[mask]).float().mean().item()
    f1 = f1_score(
        data.y[mask].cpu().tolist(), pred.cpu().tolist(), average="macro", zero_division=0
    )
    return acc, float(f1)


def train(
    model: torch.nn.Module,
    data: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    lr: float,
    epochs: int,
    patience: int,
) -> tuple[float, list[dict]]:
    """Train with early stopping on val F1-macro.

    Returns:
        best_val_f1: best validation F1 achieved
        history: list of dicts with epoch, loss, val_f1
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        loss = train_epoch(model, data, optimizer, train_mask)
        _, val_f1 = evaluate(model, data, val_mask)
        history.append({"epoch": epoch, "loss": loss, "val_f1": val_f1})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_f1, history
