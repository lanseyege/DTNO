"""
The FNO backbone (§15), optionally time-conditioned by FiLM (§14).

    h_{l+1} = sigma( W_l h_l + F^-1 R_l F(h_l) )        [proposal form]
    h_{l+1} = h_l + s_l * sigma( FiLM(W_l h_l + F^-1 R_l F(h_l)) )   [residual]

The same class serves all three models.  AR-FNO instantiates it with cond_dim=0
and DT-FNO / SG-DT-FNO with cond_dim>0; nothing else about the backbone changes,
so the parameter budgets differ only by the FiLM heads (~2 * width * embed_dim
per block, about 3% of the total at width 64).  §3 asks for matched budgets and
`scripts/train.py` prints both counts so the number goes in the paper rather
than being assumed.

Two documented deviations from the literal proposal text, both applied to every
model identically so they cannot confound the AR-vs-direct comparison:

  residual=True     The operator being learned is a finite-time flow map, which
                    at small tau is close to the identity.  A residual stack
                    with LayerScale starts there; a plain stack starts at a
                    random map and has to unlearn it.  This also makes the
                    identity constraint Phi(z, 0) = z (§17) reachable instead of
                    fought for.  Set `residual: false` to recover §15 exactly.
  padding_mode      §15 asks for spatial padding on non-periodic boundaries.
                    A swirl burner is not periodic in either direction, so the
                    default pads by 8 cells in replicate mode and crops after
                    the last block.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral import SpectralConv2d
from .time_embedding import FiLM

_ACTS = {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int,
                 cond_dim: int = 0, residual: bool = True,
                 layer_scale: float = 0.1, act: str = "gelu",
                 norm: bool = False):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.film = FiLM(cond_dim, width) if cond_dim > 0 else None
        self.act = _ACTS[act]()
        self.residual = bool(residual)
        self.norm = nn.GroupNorm(min(8, width), width) if norm else None
        if self.residual:
            self.scale = nn.Parameter(torch.full((1, width, 1, 1), float(layer_scale)))

    def forward(self, h: torch.Tensor, e_t: Optional[torch.Tensor] = None):
        y = self.spectral(h) + self.pointwise(h)
        if self.norm is not None:
            y = self.norm(y)
        if self.film is not None:
            if e_t is None:
                raise ValueError("block is time-conditioned but got e_t=None")
            y = self.film(y, e_t)
        y = self.act(y)
        return h + self.scale * y if self.residual else y


class FNOBackbone(nn.Module):
    """(B, width, H, W) -> (B, width, H, W), constant depth regardless of tau."""

    def __init__(self, width: int = 64, modes1: int = 16, modes2: int = 16,
                 n_layers: int = 4, cond_dim: int = 0, padding: int = 8,
                 padding_mode: str = "replicate", residual: bool = True,
                 layer_scale: float = 0.1, act: str = "gelu",
                 norm: bool = False):
        super().__init__()
        self.width = width
        self.n_layers = n_layers
        self.padding = int(padding)
        self.padding_mode = padding_mode
        self.blocks = nn.ModuleList([
            FNOBlock(width, modes1, modes2, cond_dim=cond_dim,
                     residual=residual, layer_scale=layer_scale, act=act,
                     norm=norm)
            for _ in range(n_layers)])

    def forward(self, z: torch.Tensor, e_t: Optional[torch.Tensor] = None):
        p = self.padding
        if p > 0:
            z = F.pad(z, [p, p, p, p], mode=self.padding_mode)
        for blk in self.blocks:
            z = blk(z, e_t)
        if p > 0:
            z = z[..., p:-p, p:-p]
        return z.contiguous()
