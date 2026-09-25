"""
Time embedding (§13.1) and FiLM conditioning (§14).

The query time enters as tau = h * dt / t_scale, with t_scale chosen so the
longest TRAINING horizon sits at tau = 1.  Horizon extrapolation is then
literally tau > 1, which matters for how the embedding is built.

Three modes, and A3 in the ablation table asks for exactly this comparison:

  scalar      tau broadcast to H x W and concatenated once at the input.  The
              naive baseline the proposal warns against: by the last spectral
              block the network has had many chances to forget it.
  fourier     e(tau) = [tau, sin(2^j pi tau), cos(2^j pi tau)], j = 0..B-1.
              Faithful to §13.1.  Note the failure mode it carries: the highest
              band completes a full cycle every 2^-(B-1) in tau, so at tau = 3
              (h = 384 when trained to 128) the embedding is periodic-aliased
              back onto values seen during training.  Temporal extrapolation
              results from this mode should be read with that in mind.
  fourier_log adds log(tau) and geometric bands in log-tau.  Horizons are
              sampled log-uniformly (§20), so log-tau is the coordinate the
              training distribution is actually uniform in, and it is monotone
              past tau = 1 instead of periodic.

`fourier` is the default because it is what the proposal specifies; the log
variant exists so the extrapolation column of Experiment B can distinguish "the
operator cannot extrapolate" from "the embedding aliased".
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class FourierTimeEmbedding(nn.Module):
    """tau (B,) -> e_t (B, embed_dim)."""

    def __init__(self, embed_dim: int = 128, n_bands: int = 8,
                 mode: str = "fourier", max_freq_log2: float = 7.0):
        super().__init__()
        self.mode = str(mode)
        self.n_bands = int(n_bands)
        self.embed_dim = int(embed_dim)

        if self.mode == "scalar":
            in_dim = 1
        elif self.mode == "fourier":
            in_dim = 1 + 2 * self.n_bands
            self.register_buffer(
                "freqs", math.pi * 2.0 ** torch.arange(self.n_bands).float(),
                persistent=False)
        elif self.mode == "fourier_log":
            in_dim = 2 + 2 * self.n_bands
            self.register_buffer(
                "freqs",
                math.pi * torch.logspace(0.0, max_freq_log2, self.n_bands, base=2.0),
                persistent=False)
        else:
            raise ValueError(f"unknown time embedding mode '{mode}'")

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim))

    def features(self, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.reshape(-1, 1).float()
        if self.mode == "scalar":
            return tau
        if self.mode == "fourier":
            a = tau * self.freqs.reshape(1, -1)
            return torch.cat([tau, torch.sin(a), torch.cos(a)], dim=-1)
        # fourier_log: bands live in log-tau, where the horizon draw is uniform
        ltau = torch.log(tau.clamp_min(1e-8))
        a = ltau * self.freqs.reshape(1, -1)
        return torch.cat([tau, ltau, torch.sin(a), torch.cos(a)], dim=-1)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.features(tau).to(self.mlp[0].weight.dtype))


class FiLM(nn.Module):
    """(gamma, beta) per block from e_t; applied as h' = (1 + gamma) h + beta.

    The (1 + gamma) form with a zero-initialised head means the block is exactly
    unmodulated at initialisation, so training starts from a well-behaved
    unconditioned FNO and the time conditioning is learned on top rather than
    fighting a random multiplicative field on step 0.
    """

    def __init__(self, embed_dim: int, width: int, hidden: int = 0):
        super().__init__()
        hidden = hidden or embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2 * width))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        """h: (B, width, H, W), e_t: (B, embed_dim)."""
        gamma, beta = self.net(e_t.to(h.dtype)).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * h + beta

    def params(self, e_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = self.net(e_t).chunk(2, dim=-1)
        return gamma, beta
