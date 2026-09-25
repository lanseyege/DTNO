"""
Semigroup consistency (§16) and its weight schedule (§19).

    L_SG = || z_direct - z_compose ||^2 / (|| z_direct ||^2 + eps)

with z_direct = Phi(z0, tau_a + tau_b) and z_compose = Phi(Phi(z0, tau_a), tau_b).

The normalisation by || z_direct ||^2 is what the proposal specifies and it is
doing real work: an unnormalised version is minimised most cheaply by shrinking
the latent towards zero, and a collapsed latent satisfies the flow-map identity
perfectly while predicting nothing.  Normalising removes the trivial scale
direction — but not every degenerate one, which is why:

  * `LatentCollapseMonitor` logs the RMS latent norm and the batch-wise latent
    variance every step.  A falling C_SG with a falling latent variance is the
    failure mode, not the result;
  * §31 requires reporting C_SG together with E_test.  Consistency that does not
    come with better forecasting is a warning, not a finding.

`detach_direct` (off by default) is available if the collapse monitor fires: it
stops the gradient on the direct route so the composed route is pulled onto the
direct one rather than the two meeting in the middle.  It changes what the loss
means, so it is a diagnostic switch, not a default.

Weight schedule (§19): lambda_SG stays ~0 for `warmup_epochs`, then ramps
linearly to `lambda_max` in [0.05, 0.2].  Constraining consistency before the
model can predict anything at all just teaches it to be consistently wrong.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def semigroup_loss(z_direct: torch.Tensor, z_compose: torch.Tensor,
                   eps: float = 1e-8, detach_direct: bool = False,
                   per_sample: bool = True,
                   reduction: str = "mean") -> torch.Tensor:
    """Relative squared latent mismatch between the two routes.

    reduction='none' returns one value per sample, which is what lets the task
    mask out unsplittable horizons without building a ragged sub-batch.
    """
    zd = z_direct.detach() if detach_direct else z_direct
    zd, zc = zd.float(), z_compose.float()
    if per_sample or reduction == "none":
        B = zd.shape[0]
        num = ((zd - zc) ** 2).reshape(B, -1).sum(dim=1)
        den = (zd ** 2).reshape(B, -1).sum(dim=1) + eps
        per = num / den
        return per if reduction == "none" else per.mean()
    return ((zd - zc) ** 2).sum() / ((zd ** 2).sum() + eps)


class SemigroupWeight:
    """lambda_SG(epoch): flat ~0 through warm-up, then a linear ramp (§19)."""

    def __init__(self, lambda_max: float = 0.1, warmup_epochs: int = 5,
                 ramp_epochs: int = 10, lambda_init: float = 0.0):
        self.lambda_max = float(lambda_max)
        self.warmup = int(warmup_epochs)
        self.ramp = max(int(ramp_epochs), 1)
        self.lambda_init = float(lambda_init)

    def __call__(self, epoch: int) -> float:
        if epoch < self.warmup:
            return self.lambda_init
        frac = min(1.0, (epoch - self.warmup + 1) / self.ramp)
        return self.lambda_init + frac * (self.lambda_max - self.lambda_init)

    def describe(self) -> str:
        return (f"lambda_SG: {self.lambda_init} for {self.warmup} epochs, "
                f"then ramp to {self.lambda_max} over {self.ramp}")


class LatentCollapseMonitor:
    """Cheap statistics that distinguish 'consistent' from 'degenerate'.

    latent_rms      RMS of z over the batch.  Falling towards 0 => scale collapse.
    latent_var      mean over features of the across-batch variance.  Falling
                    towards 0 => the latent no longer depends on the input,
                    which satisfies any composition rule for free.
    """

    @staticmethod
    @torch.no_grad()
    def stats(z: torch.Tensor, prefix: str = "z") -> Dict[str, float]:
        zf = z.float()
        B = zf.shape[0]
        flat = zf.reshape(B, -1)
        return {
            f"{prefix}_rms": float(flat.pow(2).mean().sqrt()),
            f"{prefix}_var_batch": float(flat.var(dim=0, unbiased=False).mean()),
        }
