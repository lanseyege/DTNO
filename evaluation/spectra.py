"""
Spectral error (§27).

    E_spec = sum_k | E_pred(k) - E_DNS(k) |  /  sum_k E_DNS(k)

The question this metric exists to answer: does a direct-time operator, asked to
jump 128 frames in one shot, buy its low MSE by smoothing away the high-
wavenumber structure?  A model that predicts a blurred field has a good L2 and a
spectrum that falls off a cliff past the energy-containing range, and §27 is the
only metric in the suite that catches it.

Report the band-resolved ratio too (`spectral_ratio`), not just the scalar: an
E_spec of 0.2 means something entirely different when the deficit sits at
k > 30 (over-smoothing — expected, and quantifiable) than when it sits at k ~ 5
(the model got the large-scale flow wrong, which is a different failure).

`radial_spectrum` is differentiable and fp32-internal, so the same function can
serve a spectral loss later.  It is NOT used as a loss in the MVP (§18, §35).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

_BIN_CACHE: Dict[tuple, torch.Tensor] = {}


def _radial_index(H: int, W: int, device) -> torch.Tensor:
    key = (H, W, str(device))
    idx = _BIN_CACHE.get(key)
    if idx is None:
        ky = torch.fft.fftfreq(H, d=1.0 / H, device=device)
        kx = torch.fft.fftfreq(W, d=1.0 / W, device=device)
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")
        idx = torch.sqrt(KX ** 2 + KY ** 2).round().long().reshape(-1)
        _BIN_CACHE[key] = idx
    return idx


def radial_spectrum(x: torch.Tensor, n_bins: Optional[int] = None) -> torch.Tensor:
    """Isotropic energy spectrum.  x: (..., H, W) -> (..., K)."""
    H, W = x.shape[-2], x.shape[-1]
    lead = x.shape[:-2]
    K = n_bins or (min(H, W) // 2 + 1)
    idx = _radial_index(H, W, x.device).clamp_max(K - 1)

    xf = torch.fft.fft2(x.float())
    power = (xf.real ** 2 + xf.imag ** 2).reshape(-1, H * W)
    out = torch.zeros(power.shape[0], K, device=x.device, dtype=power.dtype)
    out.scatter_add_(1, idx.unsqueeze(0).expand_as(power), power)
    return out.reshape(*lead, K) / float(H * W)


@torch.no_grad()
def spectral_error(pred: torch.Tensor, target: torch.Tensor,
                   channels: Optional[Sequence[int]] = None,
                   fluctuation: bool = True) -> Dict[str, object]:
    """pred/target: (B, H, W, C).

    fluctuation=True removes the per-sample spatial mean before the transform.
    A swirl burner has a strong standing mean field; leaving it in dumps most of
    the energy into k = 0 and the metric stops resolving the turbulent range.
    """
    ch = list(range(pred.shape[-1])) if channels is None else list(channels)
    p = pred[..., ch].permute(0, 3, 1, 2).float()
    t = target[..., ch].permute(0, 3, 1, 2).float()
    if fluctuation:
        p = p - p.mean(dim=(-2, -1), keepdim=True)
        t = t - t.mean(dim=(-2, -1), keepdim=True)

    Ep = radial_spectrum(p).mean(dim=0)            # (C, K)
    Et = radial_spectrum(t).mean(dim=0)
    num = (Ep - Et).abs().sum(dim=-1)
    den = Et.abs().sum(dim=-1).clamp_min(1e-30)
    return {
        "E_spec_per_channel": (num / den).tolist(),
        "E_spec": float((num / den).mean()),
        "spectrum_pred": Ep.tolist(),
        "spectrum_target": Et.tolist(),
        "spectral_ratio": (Ep / Et.clamp_min(1e-30)).tolist(),
        "channels": ch,
    }


class SpectrumAccumulator:
    """Streams mean spectra per horizon so §27 can be plotted, not just scored."""

    def __init__(self, horizons: Sequence[int], n_channels: int,
                 n_bins: int, device="cpu", fluctuation: bool = True):
        self.horizons = [int(h) for h in horizons]
        self.fluct = bool(fluctuation)
        self.device = torch.device(device)
        shape = (len(self.horizons), n_channels, n_bins)
        self.pred = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self.tgt = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self.count = torch.zeros(len(self.horizons), dtype=torch.float64,
                                 device=self.device)

    @torch.no_grad()
    def update(self, h_index: int, pred: torch.Tensor, target: torch.Tensor):
        p = pred.permute(0, 3, 1, 2).float()
        t = target.permute(0, 3, 1, 2).float()
        if self.fluct:
            p = p - p.mean(dim=(-2, -1), keepdim=True)
            t = t - t.mean(dim=(-2, -1), keepdim=True)
        K = self.pred.shape[-1]
        self.pred[h_index] += radial_spectrum(p, K).sum(dim=0).to(self.device, torch.float64)
        self.tgt[h_index] += radial_spectrum(t, K).sum(dim=0).to(self.device, torch.float64)
        self.count[h_index] += float(p.shape[0])

    @torch.no_grad()
    def compute(self, distributed: bool = False) -> Dict[str, object]:
        if distributed:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                for t in (self.pred, self.tgt, self.count):
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
        c = self.count.clamp_min(1.0).reshape(-1, 1, 1)
        Ep, Et = self.pred / c, self.tgt / c
        num = (Ep - Et).abs().sum(dim=-1)                       # (n_h, C)
        den = Et.abs().sum(dim=-1).clamp_min(1e-30)
        return {
            "horizons": self.horizons,
            "E_spec_channel": (num / den).tolist(),
            "E_spec": (num / den).mean(dim=1).tolist(),
            "spectrum_pred": Ep.tolist(),
            "spectrum_target": Et.tolist(),
        }
