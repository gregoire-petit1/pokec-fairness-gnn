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
    sample_weights: torch.Tensor | None = None,
) -> float:
    """Run one training epoch. Returns mean cross-entropy loss.

    When ``sample_weights`` is provided, it must be a 1-D tensor of shape
    ``(N,)`` indexable by ``train_mask`` (typically produced by
    :func:`src.fairness.reweighting.kamiran_calders_weights` and broadcast
    onto the full graph). The weighted loss is ``mean_i w_i * CE_i``.
    """
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    if sample_weights is None:
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    else:
        per_sample = F.cross_entropy(
            out[train_mask], data.y[train_mask], reduction="none"
        )
        loss = (per_sample * sample_weights[train_mask]).mean()
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
    sample_weights: torch.Tensor | None = None,
) -> tuple[float, list[dict]]:
    """Train with early stopping on val F1-macro.

    Args:
        sample_weights: optional ``(N,)`` weights tensor — when provided, the
            cross-entropy loss is reweighted per-row (used by Kamiran-Calders
            pre-processing). Defaults to ``None`` (uniform weights).

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
        loss = train_epoch(model, data, optimizer, train_mask, sample_weights=sample_weights)
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
