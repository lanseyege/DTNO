"""
Field error (§25) and per-variable error (§26).

    E_c(h) = || U_hat_c(t+h) - U_c(t+h) ||_2 / (|| U_c(t+h) ||_2 + eps)
    E_field(h) = (1/C) sum_c E_c(h)

The headline curve is E_field vs h.  It is NOT the headline number: §26 requires
flow / thermodynamics / chemistry / reaction to be reported separately, because
a model can carry velocity beautifully and put the reaction zone in the wrong
place, and the channel mean hides exactly that.  `HorizonMetrics` therefore
accumulates per channel always and aggregates on demand.

Everything is streamed as sums on the device and all-reduced once in `compute`,
so every rank must call `update` and `compute` — the usual DDP trap is a metric
that only ever saw rank 0's shard.

Errors are computed in NORMALISED space by default (that is the space the loss
lives in, and it is what makes channels comparable).  Pass a normaliser to also
get physical-unit RMSE per channel, which is what a combustion audience reads.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed as dist


def _all_reduce_(t: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


class HorizonMetrics:
    """Streaming per-(horizon, channel) accumulator."""

    def __init__(self, horizons: Sequence[int], channel_names: Sequence[str],
                 channel_groups: Optional[Dict[str, List[int]]] = None,
                 device: str | torch.device = "cpu", eps: float = 1e-8):
        self.horizons = [int(h) for h in horizons]
        self.channel_names = list(channel_names)
        self.groups = dict(channel_groups or {})
        self.eps = float(eps)
        self.device = torch.device(device)

        n_h, n_c = len(self.horizons), len(self.channel_names)
        z = lambda: torch.zeros(n_h, n_c, dtype=torch.float64, device=self.device)
        self.rel_sum = z()          # sum over samples of per-sample rel L2
        self.se_sum = z()           # sum of squared error
        self.tgt_sq_sum = z()       # sum of squared target
        self.abs_sum = z()          # sum of absolute error
        self.n_elem = z()           # element count per (h, c)
        self.count = torch.zeros(n_h, dtype=torch.float64, device=self.device)

    # -- update -----------------------------------------------------------
    @torch.no_grad()
    def update(self, h_index: int, pred: torch.Tensor, target: torch.Tensor):
        """pred/target: (B, H, W, C) in normalised units."""
        p = pred.detach().to(self.device, torch.float64)
        t = target.detach().to(self.device, torch.float64)
        B, H, W, C = p.shape
        pf = p.reshape(B, H * W, C)
        tf = t.reshape(B, H * W, C)

        diff2 = (pf - tf) ** 2
        num = diff2.sum(dim=1).sqrt()                       # (B, C)
        den = (tf ** 2).sum(dim=1).sqrt() + self.eps
        self.rel_sum[h_index] += (num / den).sum(dim=0)
        self.se_sum[h_index] += diff2.sum(dim=(0, 1))
        self.tgt_sq_sum[h_index] += (tf ** 2).sum(dim=(0, 1))
        self.abs_sum[h_index] += (pf - tf).abs().sum(dim=(0, 1))
        self.n_elem[h_index] += float(B * H * W)
        self.count[h_index] += float(B)

    # -- reduce -----------------------------------------------------------
    @torch.no_grad()
    def compute(self, distributed: bool = False,
                normalizer=None) -> Dict[str, object]:
        if distributed:
            for t in (self.rel_sum, self.se_sum, self.tgt_sq_sum,
                      self.abs_sum, self.n_elem, self.count):
                _all_reduce_(t)

        cnt = self.count.clamp_min(1.0).unsqueeze(-1)
        rel_c = (self.rel_sum / cnt)                        # (n_h, C)
        rmse_c = (self.se_sum / self.n_elem.clamp_min(1.0)).sqrt()
        mae_c = (self.abs_sum / self.n_elem.clamp_min(1.0))
        nrmse_c = self.se_sum.sqrt() / (self.tgt_sq_sum.sqrt() + self.eps)

        out: Dict[str, object] = {
            "horizons": list(self.horizons),
            "channel_names": list(self.channel_names),
            "n_samples": [int(c) for c in self.count.tolist()],
            "E_field": rel_c.mean(dim=1).tolist(),           # (n_h,)
            "E_channel": rel_c.tolist(),                     # (n_h, C)
            "rmse_channel": rmse_c.tolist(),
            "mae_channel": mae_c.tolist(),
            "nrmse_channel": nrmse_c.tolist(),
        }

        # §26: never let the channel mean be the only number on the page.
        by_group = {}
        for g, idx in self.groups.items():
            if idx:
                sel = torch.tensor(idx, device=rel_c.device)
                by_group[g] = rel_c.index_select(1, sel).mean(dim=1).tolist()
        out["E_group"] = by_group

        if normalizer is not None:
            # RMSE in physical units: undo the z-score only (the nonlinear part
            # of the transform is field-dependent, so a per-channel sigma is the
            # honest scalar conversion and is labelled as such).
            scale = [st.std for st in normalizer.stats]
            out["rmse_channel_transformed_units"] = [
                [v * s for v, s in zip(row, scale)] for row in rmse_c.tolist()]
        return out


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def predictability_horizon(horizons: Sequence[int], errors: Sequence[float],
                           threshold: float = 0.3) -> Optional[float]:
    """T_pred: largest h with E_field(h) < threshold, linearly interpolated (§45).

    Returns None if the curve is already above the threshold at the shortest
    horizon, and the largest evaluated horizon if it never crosses — both cases
    the caller must report honestly rather than as a number.
    """
    hs, es = list(horizons), list(errors)
    if not hs or es[0] >= threshold:
        return None
    last = hs[0]
    for i in range(1, len(hs)):
        if es[i] < threshold:
            last = hs[i]
            continue
        # linear interpolation in (h, E) between the bracketing points
        h0, h1, e0, e1 = hs[i - 1], hs[i], es[i - 1], es[i]
        if e1 == e0:
            return float(h0)
        return float(h0 + (threshold - e0) * (h1 - h0) / (e1 - e0))
    return float(last)


def crossover_horizon(horizons: Sequence[int], err_a: Sequence[float],
                      err_b: Sequence[float]) -> Optional[float]:
    """Smallest h where curve B drops below curve A (H2's h*, §2).

    Called as crossover_horizon(h, E_AR, E_Direct): the horizon past which the
    direct model wins.  None means no crossover inside the evaluated range,
    which is itself a reportable result.
    """
    hs = list(horizons)
    for i in range(len(hs)):
        if err_b[i] < err_a[i]:
            if i == 0:
                return float(hs[0])
            d0 = err_a[i - 1] - err_b[i - 1]
            d1 = err_a[i] - err_b[i]
            if d1 == d0:
                return float(hs[i])
            frac = -d0 / (d1 - d0)
            return float(hs[i - 1] + frac * (hs[i] - hs[i - 1]))
    return None
