"""
Combustion-oriented metrics (§28-30).

These are the metrics that decide whether a low MSE means the model kept the
flame or merely kept the mean.

§28 Gradient error
    E_grad = || grad(X_hat) - grad(X) || / || grad(X) ||
    Flame structure is a gradient object.  A field can match pointwise to a few
    percent and still have a reaction layer twice as thick, which shows up here
    and nowhere else in §25.

§29 Reaction-zone structure
    M(x) = 1[ q_dot(x) > alpha * q_dot_max ]
    IoU, flame area, perimeter, centroid displacement.  The question is the one
    §29 poses directly: when two models have the same MSE, which one put the
    flame in the right place?  Note that the mask threshold is relative to the
    per-sample maximum, so a model that under-predicts peak heat release
    uniformly still gets a fair mask — that is deliberate, and it is why flame
    *area* is reported alongside IoU rather than folded into it.

§30 Global physical quantities
    Q(t) = integral of q_dot over the domain, plus mean/RMS temperature, mean
    velocity and species averages.  Evaluation only — never a loss (§30).

All of these need PHYSICAL units to mean anything, so every entry point takes
the frozen normaliser and inverts before computing.  The threshold alpha and the
mask channel come from the config; `data/channels.py` finds heat release, and
falls back to OH when the dataset has no heat-release channel.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# §28 gradients
# ---------------------------------------------------------------------------

def spatial_gradient(x: torch.Tensor) -> torch.Tensor:
    """x: (B, H, W) -> (B, 2, H, W) central differences with replicate edges."""
    xp = F.pad(x.unsqueeze(1), [1, 1, 1, 1], mode="replicate")
    gy = 0.5 * (xp[:, :, 2:, 1:-1] - xp[:, :, :-2, 1:-1])
    gx = 0.5 * (xp[:, :, 1:-1, 2:] - xp[:, :, 1:-1, :-2])
    return torch.cat([gy, gx], dim=1)


@torch.no_grad()
def gradient_error(pred: torch.Tensor, target: torch.Tensor,
                   channels: Sequence[int], eps: float = 1e-12) -> List[float]:
    """Relative L2 error of the gradient magnitude field, per requested channel."""
    out = []
    for c in channels:
        gp = spatial_gradient(pred[..., c].float())
        gt = spatial_gradient(target[..., c].float())
        num = torch.linalg.vector_norm((gp - gt).reshape(gp.shape[0], -1), dim=1)
        den = torch.linalg.vector_norm(gt.reshape(gt.shape[0], -1), dim=1) + eps
        out.append(float((num / den).mean()))
    return out


# ---------------------------------------------------------------------------
# §29 reaction zone
# ---------------------------------------------------------------------------

def reaction_mask(field: torch.Tensor, alpha: float = 0.2,
                  per_sample: bool = True) -> torch.Tensor:
    """(B, H, W) -> bool mask at alpha * max.

    per_sample=True normalises by each sample's own maximum.  For a lifted or
    unsteady flame the peak heat release varies by an order of magnitude between
    snapshots, so a global threshold would silently score "is the flame strong"
    rather than "is the flame in the right place".
    """
    f = field.float()
    m = f.amax(dim=(-2, -1), keepdim=True) if per_sample else f.amax()
    return f > (alpha * m)


@torch.no_grad()
def reaction_zone_metrics(pred: torch.Tensor, target: torch.Tensor,
                          channel: int, alpha: float = 0.2) -> Dict[str, float]:
    """IoU, flame area ratio, perimeter ratio and centroid displacement."""
    mp = reaction_mask(pred[..., channel], alpha)
    mt = reaction_mask(target[..., channel], alpha)
    B, H, W = mp.shape

    inter = (mp & mt).reshape(B, -1).sum(dim=1).float()
    union = (mp | mt).reshape(B, -1).sum(dim=1).float().clamp_min(1.0)
    area_p = mp.reshape(B, -1).sum(dim=1).float()
    area_t = mt.reshape(B, -1).sum(dim=1).float().clamp_min(1.0)

    def perimeter(m: torch.Tensor) -> torch.Tensor:
        f = m.float().unsqueeze(1)
        k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                         device=m.device).reshape(1, 1, 3, 3)
        e = F.conv2d(F.pad(f, [1, 1, 1, 1], mode="replicate"), k)
        return (e.abs() > 0).reshape(m.shape[0], -1).sum(dim=1).float()

    per_p, per_t = perimeter(mp), perimeter(mt).clamp_min(1.0)

    ys = torch.arange(H, device=mp.device).float().reshape(1, H, 1)
    xs = torch.arange(W, device=mp.device).float().reshape(1, 1, W)

    def centroid(m: torch.Tensor):
        w = m.float()
        s = w.reshape(B, -1).sum(dim=1).clamp_min(1.0)
        return ((w * ys).reshape(B, -1).sum(dim=1) / s,
                (w * xs).reshape(B, -1).sum(dim=1) / s)

    cy_p, cx_p = centroid(mp)
    cy_t, cx_t = centroid(mt)
    disp = torch.sqrt((cy_p - cy_t) ** 2 + (cx_p - cx_t) ** 2)

    return {
        "iou": float((inter / union).mean()),
        "flame_area_ratio": float((area_p / area_t).mean()),
        "flame_perimeter_ratio": float((per_p / per_t).mean()),
        "centroid_displacement_px": float(disp.mean()),
        "flame_area_pred_px": float(area_p.mean()),
        "flame_area_true_px": float(area_t.mean()),
    }


# ---------------------------------------------------------------------------
# §30 global quantities
# ---------------------------------------------------------------------------

@torch.no_grad()
def global_quantities(pred: torch.Tensor, target: torch.Tensor,
                      channel_names: Sequence[str],
                      hrr_channel: int = -1,
                      temp_channel: int = -1) -> Dict[str, float]:
    """Domain-integrated and bulk statistics, in whatever units are passed in."""
    out: Dict[str, float] = {}
    B = pred.shape[0]

    if hrr_channel >= 0:
        qp = pred[..., hrr_channel].float().reshape(B, -1).sum(dim=1)
        qt = target[..., hrr_channel].float().reshape(B, -1).sum(dim=1)
        out["integrated_hrr_rel_error"] = float(
            ((qp - qt).abs() / qt.abs().clamp_min(1e-12)).mean())
        out["integrated_hrr_pred"] = float(qp.mean())
        out["integrated_hrr_true"] = float(qt.mean())

    if temp_channel >= 0:
        tp = pred[..., temp_channel].float().reshape(B, -1)
        tt = target[..., temp_channel].float().reshape(B, -1)
        out["mean_T_pred"] = float(tp.mean())
        out["mean_T_true"] = float(tt.mean())
        out["mean_T_abs_error"] = float((tp.mean(1) - tt.mean(1)).abs().mean())
        out["rms_T_pred"] = float(tp.std(dim=1).mean())
        out["rms_T_true"] = float(tt.std(dim=1).mean())

    for c, name in enumerate(channel_names):
        p = pred[..., c].float().reshape(B, -1).mean(dim=1)
        t = target[..., c].float().reshape(B, -1).mean(dim=1)
        out[f"spatial_mean_abs_err[{name}]"] = float((p - t).abs().mean())
    return out


# ---------------------------------------------------------------------------
# Per-horizon accumulator tying §28-30 together
# ---------------------------------------------------------------------------

class CombustionMetrics:
    """Accumulates §28-30 per horizon, in physical units.

    Kept separate from HorizonMetrics because these need the inverse transform
    and a couple of them (perimeter, centroid) are not cheap; run them on a
    subsample of anchors via `max_batches` when iterating quickly, and on
    everything for the numbers that go in the paper.
    """

    def __init__(self, horizons: Sequence[int], channel_names: Sequence[str],
                 normalizer, alpha: float = 0.2,
                 grad_channels: Optional[Sequence[int]] = None,
                 hrr_channel: int = -1, temp_channel: int = -1,
                 mask_channel: int = -1):
        self.horizons = [int(h) for h in horizons]
        self.channel_names = list(channel_names)
        self.norm = normalizer
        self.alpha = float(alpha)
        self.hrr_channel = int(hrr_channel)
        self.temp_channel = int(temp_channel)
        self.mask_channel = int(mask_channel)
        self.grad_channels = list(grad_channels or [])
        self._acc: Dict[int, Dict[str, float]] = {h: {} for h in self.horizons}
        self._n: Dict[int, int] = {h: 0 for h in self.horizons}

    @torch.no_grad()
    def update(self, h_index: int, pred_norm: torch.Tensor, target_norm: torch.Tensor):
        h = self.horizons[h_index]
        pred = self.norm.inverse_torch(pred_norm.float())
        target = self.norm.inverse_torch(target_norm.float())

        rec: Dict[str, float] = {}
        if self.grad_channels:
            ge = gradient_error(pred, target, self.grad_channels)
            for c, v in zip(self.grad_channels, ge):
                rec[f"E_grad[{self.channel_names[c]}]"] = v
            rec["E_grad"] = sum(ge) / len(ge)
        if self.mask_channel >= 0:
            rec.update(reaction_zone_metrics(pred, target, self.mask_channel,
                                             self.alpha))
        rec.update(global_quantities(pred, target, self.channel_names,
                                     self.hrr_channel, self.temp_channel))

        acc = self._acc[h]
        for k, v in rec.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        self._n[h] += 1

    def compute(self, distributed: bool = False) -> Dict[str, object]:
        keys = sorted({k for h in self.horizons for k in self._acc[h]})
        out: Dict[str, object] = {"horizons": self.horizons}
        if distributed:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                buf = torch.tensor(
                    [[self._acc[h].get(k, 0.0) for k in keys] + [float(self._n[h])]
                     for h in self.horizons], dtype=torch.float64, device=dev)
                dist.all_reduce(buf, op=dist.ReduceOp.SUM)
                rows = buf.tolist()
                for i, h in enumerate(self.horizons):
                    self._n[h] = int(rows[i][-1])
                    self._acc[h] = {k: rows[i][j] for j, k in enumerate(keys)}
        for k in keys:
            out[k] = [self._acc[h].get(k, 0.0) / max(self._n[h], 1)
                      for h in self.horizons]
        out["n_batches"] = [self._n[h] for h in self.horizons]
        return out
