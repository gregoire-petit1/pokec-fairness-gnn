"""FairGNN: adversarial-debiasing GNN, implemented via Gradient Reversal Layer.

Reference:
    Dai, E. & Wang, S. (2021). Say No to the Discrimination: Learning Fair Graph
    Neural Networks with Limited Sensitive Attribute Information. WSDM 2021.

Implementation note. The original FairGNN paper trains the encoder/classifier
and the adversary in alternating optimisation steps. The mathematically
equivalent Gradient Reversal Layer (GRL) approach (Ganin & Lempitsky, 2015)
collapses the min–max into a single optimiser by inverting the gradient flowing
back from the adversary into the encoder. This module uses the GRL variant
because it integrates cleanly with the existing single-optimiser trainer.

Why this matters here. The previous implementation summed the classification and
adversarial losses with a *negative* coefficient on the adversarial term, but
optimised both heads together. That is **not** adversarial training: the
adversary was rewarded for failing at its own task, which produced the
``F1=0.4834 / ΔDP=0.0`` collapse signature observed at λ=0.1 and λ=1.0 in the
prior notebook results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GradientReversal(torch.autograd.Function):
    """Identity in forward, gradient ×(−λ) in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_adv: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambda_adv = float(lambda_adv)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lambda_adv * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_adv: float) -> torch.Tensor:
    """Helper: apply :class:`GradientReversal` with a Python-float lambda."""
    return GradientReversal.apply(x, lambda_adv)


class FairGNN(nn.Module):
    """SAGE encoder + classifier head + adversarial discriminator (via GRL).

    Args:
        in_channels: input feature dimensionality.
        hidden_channels: encoder hidden dim (= classifier and adversary input dim).
        out_channels: number of classes for the main task.
        adv_hidden: hidden dim of the adversary's MLP.
        num_layers: number of SAGE layers in the encoder.
        dropout: dropout probability between encoder layers.
        lambda_adv: coefficient of the gradient reversal — controls the
            fairness/accuracy trade-off. λ=0 disables debiasing (≈ GraphSAGE);
            larger λ pushes the encoder harder to make sensitive unrecoverable.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        adv_hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
        lambda_adv: float = 1.0,
    ) -> None:
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        super().__init__()
        self.dropout = dropout
        self.lambda_adv = float(lambda_adv)

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.classifier = nn.Linear(hidden_channels, out_channels)

        # Binary sensitive attribute → 2-way discriminator. For multi-valued
        # sensitives (e.g. age_group with 3 values), instantiate with the
        # adversary head sized accordingly via ``set_adversary_outputs``.
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
        # GRL between encoder and adversary: the adversary trains *normally*
        # to predict sensitive (its loss is minimised), but the gradient back
        # to the encoder is reversed × λ → the encoder is pushed to make
        # sensitive unrecoverable from h.
        adv_logits = self.adversary(grad_reverse(h, self.lambda_adv))
        return pred_logits, adv_logits

    @torch.no_grad()
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        h = self.encode(x, edge_index)
        if was_training:
            self.train()
        return h


def fairgnn_loss(
    pred_logits: torch.Tensor,
    adv_logits: torch.Tensor,
    y: torch.Tensor,
    sensitive: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Joint loss = L_cls + L_adv. Both terms are positive: the adversary
    minimises its own classification loss, while the encoder receives a
    sign-flipped gradient through :class:`GradientReversal` and therefore
    maximises that same term as far as the encoder parameters are concerned.

    Note: the lambda is baked into the GRL (see :class:`FairGNN`), so it is
    *not* present in this loss signature any more — passing it here would be a
    bug, since it would compound with the GRL scaling.
    """
    l_cls = F.cross_entropy(pred_logits[mask], y[mask])
    l_adv = F.cross_entropy(adv_logits[mask], sensitive[mask])
    return l_cls + l_adv
