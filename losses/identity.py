"""
Identity constraint (§17).

    Phi_0 = id   =>   L_id = || Phi(z0, 0) - z0 ||^2

Cheap — one extra backbone call on a latent that is already computed — and it
does more than it looks like it should.  A time-conditioned operator trained only
on h >= 1 has no reason to behave sensibly as tau -> 0, and the FiLM parameters
near tau = 0 are then extrapolation.  Pinning the tau = 0 end anchors the whole
conditioning curve, which is what makes the unseen-query-time interpolation in
Experiment B a fair test rather than a test of luck at the short end.

It also interacts with the semigroup term: Phi_{a+b} = Phi_b . Phi_a with a = 0
forces Phi_0 = id anyway, so without L_id the composition loss is being asked to
discover the identity element on its own from horizons that never include zero.

Reported in normalised form so its magnitude is comparable across latent widths.
"""

from __future__ import annotations

import torch


def identity_loss(z0: torch.Tensor, z_id: torch.Tensor,
                  relative: bool = True, eps: float = 1e-8) -> torch.Tensor:
    """|| Phi(z0, 0) - z0 ||^2, optionally normalised by || z0 ||^2."""
    a, b = z0.float(), z_id.float()
    B = a.shape[0]
    num = ((a - b) ** 2).reshape(B, -1).sum(dim=1)
    if not relative:
        return num.mean() / a[0].numel()
    den = (a ** 2).reshape(B, -1).sum(dim=1) + eps
    return (num / den).mean()
