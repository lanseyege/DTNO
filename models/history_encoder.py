"""
History encoder (§12.1) and decoder (§15).

Encoder: K frames reshaped channel-wise, coordinates appended, lifted to width d
by a 1x1 (or 3x3) convolution.  Deliberately boring — §12.1 rules out a
Transformer history encoder for the first version, because the MVP is testing
time conditioning and semigroup structure, and a second novel component would
make an ablation table that cannot attribute anything.

    (B, K, H, W, C) --reshape--> (B, H, W, K*C) --+coords--> (B, H, W, K*C+2)
                    --lift-->    (B, d, H, W) = z0

The 3x3 option exists for one reason: the K frames carry finite-difference
tendency information (§10), and a 1x1 lift can only combine them pointwise.  A
3x3 lift can form a spatial gradient of the tendency, which is what an advective
flow map needs.  It is a config flag, and A2 (history length) is the ablation
that decides whether any of this matters.

Decoder: d -> 128 -> C, per §15, applied pointwise.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class HistoryEncoder(nn.Module):
    """K frames + coordinates -> latent field z0 of width d."""

    def __init__(self, n_channels: int, history_len: int, width: int = 64,
                 kernel_size: int = 1, use_coords: bool = True,
                 extra_scalar_dim: int = 0):
        super().__init__()
        self.C = int(n_channels)
        self.K = int(history_len)
        self.width = int(width)
        self.use_coords = bool(use_coords)
        in_dim = self.K * self.C + (2 if use_coords else 0) + int(extra_scalar_dim)
        self.in_dim = in_dim
        pad = kernel_size // 2
        self.lift = nn.Conv2d(in_dim, width, kernel_size=kernel_size, padding=pad)

    def forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None,
                extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, K, H, W, C) channels-last -> z0: (B, d, H, W)."""
        B, K, H, W, C = x.shape
        if K != self.K or C != self.C:
            raise ValueError(f"expected (K={self.K}, C={self.C}), got (K={K}, C={C})")
        # (B, K, H, W, C) -> (B, H, W, K*C): time-major within the channel axis
        z = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, K * C)
        if self.use_coords:
            if grid is None:
                grid = _default_grid(H, W, x.device, x.dtype).expand(B, -1, -1, -1)
            z = torch.cat([z, grid.to(z.dtype)], dim=-1)
        if extra is not None:
            # broadcast a scalar (e.g. tau in the 'scalar' conditioning ablation)
            e = extra.reshape(B, 1, 1, -1).expand(B, H, W, extra.shape[-1])
            z = torch.cat([z, e.to(z.dtype)], dim=-1)
        z = z.permute(0, 3, 1, 2).contiguous()          # (B, in_dim, H, W)
        return self.lift(z)


class Decoder(nn.Module):
    """z -> field increment / field.  d -> hidden -> C, pointwise (§15)."""

    def __init__(self, width: int, n_channels: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(width, hidden, 1), nn.GELU(), nn.Conv2d(hidden, n_channels, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d, H, W) -> (B, H, W, C) channels-last."""
        return self.net(z).permute(0, 2, 3, 1).contiguous()


_GRID_CACHE = {}


def _default_grid(H: int, W: int, device, dtype) -> torch.Tensor:
    key = (H, W, str(device), str(dtype))
    g = _GRID_CACHE.get(key)
    if g is None:
        y = torch.linspace(0, 1, H, device=device, dtype=dtype)
        x = torch.linspace(0, 1, W, device=device, dtype=dtype)
        Y, X = torch.meshgrid(y, x, indexing="ij")
        g = torch.stack([Y, X], dim=-1).unsqueeze(0)
        _GRID_CACHE[key] = g
    return g
