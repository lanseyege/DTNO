"""
2D spectral convolution — the R_l F(.) half of an FNO block.

Kept separate from the blocks because both the AR baseline and the direct-time
operator import the identical layer.  If the two models differed here, every
accuracy gap in §25 would be confounded by architecture (§3), and the MVP's one
job is to isolate the direct-vs-autoregressive question.

fp32 note (inherited from the user's existing models/fno.py, and still true):
cuFFT has no half-precision path for non-power-of-2 transform sizes, so the
layer casts to fp32 internally and casts back on the way out.  Under bf16
autocast that is a genuine cast, not a no-op, and it is the reason `amp: bf16`
is safe on 128 x 128 and would not be on the raw layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    """FFT -> truncated complex linear map -> inverse FFT."""

    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)

        scale = 1.0 / (in_channels * out_channels)
        # Two weight blocks: the positive and negative halves of the first
        # (non-redundant) frequency axis.  rfft2 already folds the second axis.
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels,
                               self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels,
                               self.modes1, self.modes2, dtype=torch.cfloat))

    @staticmethod
    def _mul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, H, W) -> (B, C_out, H, W)."""
        dtype_in = x.dtype
        x = x.float()
        B, _, H, W = x.shape

        x_ft = torch.fft.rfft2(x)
        m1 = min(self.modes1, H // 2)
        m2 = min(self.modes2, W // 2 + 1)

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m1, :m2] = self._mul(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self._mul(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])

        out = torch.fft.irfft2(out_ft, s=(H, W))
        return out.to(dtype_in)
