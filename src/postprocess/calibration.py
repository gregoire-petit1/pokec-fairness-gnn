"""Temperature scaling — Guo et al. 2017 post-hoc calibration.

Modern neural networks (including GraphSAGE) produce **over-confident**
softmax outputs: the value 0.99 does not mean « 99 % chance ». TabICL,
in contrast, ensembles 8 in-context rounds and applies a tuned
``softmax_temperature`` so its probabilities are well-calibrated by
construction.

Temperature scaling fixes this asymmetry: a single scalar ``T > 0`` is
fitted on a validation set by minimising the cross-entropy
``NLL(softmax(logits / T), y_true)``. At inference, ``logits / T`` is
softmax'd to give calibrated probabilities. The argmax (= predicted
class) is **unchanged** — only the confidence is rescaled.

Reference:
    Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On
    Calibration of Modern Neural Networks. ICML.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fit_temperature(
    logits_val: torch.Tensor,
    y_val: torch.Tensor,
    max_iter: int = 50,
    lr: float = 0.01,
) -> float:
    """Find the scalar ``T`` that minimises NLL on the val set via L-BFGS.

    Args:
        logits_val: Raw logits on val rows, shape ``(n_val, n_classes)``.
        y_val: Integer labels, shape ``(n_val,)``.
        max_iter: L-BFGS maximum iterations (default 50).
        lr: L-BFGS learning rate (default 0.01).

    Returns:
        The fitted temperature ``T`` as a Python float. Use it as
        ``logits / T`` before softmax to obtain calibrated probabilities.
    """
    device = logits_val.device
    T = torch.nn.Parameter(torch.ones(1, device=device, dtype=logits_val.dtype))
    optimizer = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(logits_val / T.clamp(min=1e-6), y_val)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(T.detach().clamp(min=1e-6).item())


def apply_temperature(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Return calibrated probabilities ``softmax(logits / T)``.

    Vectorised, no Python loops; preserves device.
    """
    return F.softmax(logits / max(temperature, 1e-6), dim=-1)


def calibrate_logits(
    logits_val: torch.Tensor,
    y_val: torch.Tensor,
    logits_to_calibrate: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Convenience: fit T on val, apply to ``logits_to_calibrate``.

    Returns ``(calibrated_proba, T)``.
    """
    T = fit_temperature(logits_val, y_val)
    return apply_temperature(logits_to_calibrate, T), T
