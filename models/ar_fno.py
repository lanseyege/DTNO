"""
AR-FNO (Model A) — the autoregressive baseline, §23.

    F_theta : X_t -> U_hat(t+1),   U_hat(t+h) = F^(h)(X_t),   N_forward(h) = h

Same encoder, same backbone, same decoder as DirectTimeFNO; the only removals
are the time embedding and FiLM.  That is intentional: reviewers discount a
result whose baseline is a different architecture, and §3 asks for a matched
parameter budget so any gap is attributable to the prediction scheme.

Two training variants (§23), selected by config, not by class:

    AR-FNO-1   one-step supervision.  The standard, and the weak one — it never
               sees its own output as input, so rollout error compounds fast.
    AR-FNO-R   short-rollout supervision over r consecutive steps with a random
               r in [1, R].  Costs R times the forward passes per sample and is
               a much harder baseline to beat.  Train it.  A direct-time win
               over AR-FNO-1 alone will not survive review.

`rollout` returns the whole trajectory of length h from one call, and
`n_model_evals(h) == h` is the honest cost the §32 figure plots.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .fno import FNOBackbone
from .unet import UNetBackbone
from .history_encoder import HistoryEncoder, Decoder


def _make_backbone(kind, *, width, modes1, modes2, n_layers, cond_dim, padding,
                   padding_mode, residual, layer_scale, act, norm,
                   unet_base_width=80, unet_depth=3):
    """Shared by ARFNO and DirectTimeFNO so the two arms cannot drift apart."""
    kind = str(kind).lower()
    if kind == "fno":
        return FNOBackbone(width=width, modes1=modes1, modes2=modes2,
                           n_layers=n_layers, cond_dim=cond_dim,
                           padding=padding, padding_mode=padding_mode,
                           residual=residual, layer_scale=layer_scale,
                           act=act, norm=norm)
    if kind == "unet":
        # `norm` defaults to False for the FNO, where LayerScale plus the
        # spectral parameterisation suffices. A U-Net without normalisation
        # trains far less stably, so it is forced on here. That is a difference
        # between the ARCHITECTURES, not between the arms: both the AR and the
        # direct U-Net get it, which is what keeps their crossover meaningful.
        return UNetBackbone(width=width, n_layers=n_layers, cond_dim=cond_dim,
                            base_width=unet_base_width, depth=unet_depth,
                            residual=residual, layer_scale=layer_scale,
                            act=act, norm=True,
                            modes1=modes1, modes2=modes2, padding=padding)
    raise ValueError(f"unknown backbone {kind!r}; expected 'fno' or 'unet'")


class ARFNO(nn.Module):
    def __init__(self,
                 n_channels: int,
                 history_len: int = 4,
                 width: int = 64,
                 modes1: int = 16,
                 modes2: int = 16,
                 n_layers: int = 4,
                 decoder_hidden: int = 128,
                 encoder_kernel: int = 1,
                 padding: int = 8,
                 padding_mode: str = "replicate",
                 residual: bool = True,
                 layer_scale: float = 0.1,
                 predict_delta: bool = True,
                 norm: bool = False,
                 act: str = "gelu",
                 backbone: str = "fno",
                 unet_base_width: int = 80,
                 unet_depth: int = 3):
        super().__init__()
        self.C = int(n_channels)
        self.K = int(history_len)
        self.width = int(width)
        self.predict_delta = bool(predict_delta)

        self.encoder = HistoryEncoder(n_channels, history_len, width,
                                      kernel_size=encoder_kernel)
        # The backbone is the ONLY thing that differs between the FNO and the
        # U-Net arms: same HistoryEncoder, same Decoder, same prediction
        # scheme, same protocol. That is what makes a crossover measured under
        # one comparable with a crossover measured under the other.
        self.backbone = _make_backbone(
            backbone, width=width, modes1=modes1, modes2=modes2,
            n_layers=n_layers, cond_dim=0, padding=padding,
            padding_mode=padding_mode, residual=residual,
            layer_scale=layer_scale, act=act, norm=norm,
            unet_base_width=unet_base_width, unet_depth=unet_depth)
        self.decoder = Decoder(width, n_channels, hidden=decoder_hidden)

    # -- one step ---------------------------------------------------------
    def step(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None):
        """x: (B, K, H, W, C) -> U_hat(t+1): (B, H, W, C)."""
        z = self.backbone(self.encoder(x, grid), None)
        out = self.decoder(z)
        return out + x[:, -1] if self.predict_delta else out

    def forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None,
                n_steps: int = 1) -> torch.Tensor:
        return self.rollout(x, n_steps, grid)

    # -- rollout ----------------------------------------------------------
    def rollout(self, x: torch.Tensor, n_steps: int,
                grid: Optional[torch.Tensor] = None,
                collect: Optional[List[int]] = None) -> torch.Tensor:
        """Feed predictions back in for `n_steps`.

        collect=None      -> (B, n_steps, H, W, C), every step
        collect=[h1, h2]  -> (B, len(collect), H, W, C), only those horizons.
                             Evaluation uses this: one rollout to max(h) yields
                             the AR prediction at every requested horizon, which
                             is the same tensor a separate rollout per horizon
                             would produce, at 1/len(collect) the compute.
                             Timing is measured separately in
                             `evaluation/timing.py` precisely so this shortcut
                             cannot flatter the AR cost curve.
        """
        want = set(collect) if collect is not None else None
        window = x
        outs: List[torch.Tensor] = []
        for step in range(1, int(n_steps) + 1):
            pred = self.step(window, grid)
            if want is None or step in want:
                outs.append(pred.unsqueeze(1))
            window = torch.cat([window[:, 1:], pred.unsqueeze(1)], dim=1)
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, h: int,
                grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.rollout(x, int(h), grid, collect=[int(h)])[:, 0]

    # -- cost accounting --------------------------------------------------
    @staticmethod
    def n_model_evals(h: int) -> int:
        return int(h)

    def summary(self) -> str:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (f"ARFNO | width={self.width} K={self.K} C={self.C} | "
                f"params={n:,}")
