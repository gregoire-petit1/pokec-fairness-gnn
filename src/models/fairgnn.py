"""FairGNN: adversarial debiasing GNN (Dai & Wang, 2021)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class FairGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        adv_hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        super().__init__()
        self.dropout = dropout

        # Shared encoder
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # Main classifier head
        self.classifier = nn.Linear(hidden_channels, out_channels)

        # Adversarial discriminator head (binary sensitive attribute)
        self.adversary = nn.Sequential(
            nn.Linear(hidden_channels, adv_hidden),
            nn.ReLU(),
            nn.Linear(adv_hidden, 2),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x, edge_index)
        pred_logits = self.classifier(h)
        adv_logits = self.adversary(h)
        return pred_logits, adv_logits


def fairgnn_loss(
    pred_logits: torch.Tensor,
    adv_logits: torch.Tensor,
    y: torch.Tensor,
    sensitive: torch.Tensor,
    mask: torch.Tensor,
    lambda_adv: float,
) -> torch.Tensor:
    """L_total = L_classification - lambda * L_adversarial."""
    l_cls = F.cross_entropy(pred_logits[mask], y[mask])
    l_adv = F.cross_entropy(adv_logits[mask], sensitive[mask])
    return l_cls - lambda_adv * l_adv
