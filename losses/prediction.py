"""
Prediction loss (§18).

Two forms, both logged every step, only one optimised:

    normalized MSE      L = (1/C) sum_c || U_hat_c - U_c ||^2
    channel relative    L = (1/C) sum_c || U_hat_c - U_c ||^2 / (|| U_c ||^2 + eps)

Channels are already z-scored by the frozen normaliser, so plain MSE is not
dominated by temperature the way a raw-units MSE would be.  The relative form is
still worth logging: after normalisation the *variance* is matched but the
per-sample energy is not, and a channel that happens to be near its mean in one
window contributes almost nothing to the absolute loss while dominating the
relative one.  Optimise the absolute form (§18 says so, and it is the stabler
gradient); watch both.

Everything else the proposal lists for later -- physics residuals, spectral
losses, gradient losses -- is deliberately absent (§18, §35).  Adding them now
would make a good result unattributable.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def normalized_mse(pred: torch.Tensor, target: torch.Tensor,
                   weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """pred/target: (B, ..., C).  Mean over batch and space, mean over channels."""
    diff = (pred.float() - target.float()) ** 2
    per_channel = diff.reshape(-1, diff.shape[-1]).mean(dim=0)      # (C,)
    if weights is not None:
        w = weights.to(per_channel.device, per_channel.dtype)
        return (per_channel * w).sum() / w.sum()
    return per_channel.mean()


def channel_relative_l2(pred: torch.Tensor, target: torch.Tensor,
                        eps: float = 1e-8) -> torch.Tensor:
    """Squared relative error per channel, averaged over channels and batch."""
    p = pred.float().reshape(pred.shape[0], -1, pred.shape[-1])
    t = target.float().reshape(target.shape[0], -1, target.shape[-1])
    num = ((p - t) ** 2).sum(dim=1)
    den = (t ** 2).sum(dim=1) + eps
    return (num / den).mean()


def relative_l2(pred: torch.Tensor, target: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
    """Per-sample relative L2 NORM (not squared) — the reported E_field (§25)."""
    p = pred.float().reshape(pred.shape[0], -1)
    t = target.float().reshape(target.shape[0], -1)
    return (torch.norm(p - t, dim=1) / (torch.norm(t, dim=1) + eps)).mean()


class PredictionLoss(torch.nn.Module):
    """Wraps the choice of objective and always reports the alternatives."""

    def __init__(self, kind: str = "nmse", channel_weights=None, eps: float = 1e-8):
        super().__init__()
        self.kind = str(kind)
        self.eps = float(eps)
        # register_buffer refuses a name that is already a plain attribute, so
        # the buffer is always registered -- as None when unweighted.
        self.register_buffer(
            "w",
            None if channel_weights is None
            else torch.as_tensor(channel_weights, dtype=torch.float32))

    def forward(self, pred: torch.Tensor, target: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.kind == "nmse":
            loss = normalized_mse(pred, target, self.w)
        elif self.kind in ("rel", "relative", "channel_relative"):
            loss = channel_relative_l2(pred, target, self.eps)
        else:
            raise ValueError(f"unknown prediction loss '{self.kind}'")

        with torch.no_grad():
            logs = {
                "nmse": float(normalized_mse(pred, target, self.w)),
                "rel_l2_sq": float(channel_relative_l2(pred, target, self.eps)),
                "rel_l2": float(relative_l2(pred, target, self.eps)),
                "mae": float(F.l1_loss(pred.float(), target.float())),
            }
        return loss, logs
