"""
A FiLM-conditioned U-Net backbone, interface-compatible with `FNOBackbone`.

WHY THIS EXISTS
---------------
Every result in the paper uses one spatial architecture, and the central claim
-- that the direct-time / autoregressive crossover sits at h = 8-20 because
error compounds under composition while a single conditioned query does not --
is a claim about the comparison, not about Fourier layers. With one backbone it
is untestable, and a reviewer is right to read it as possibly an FNO property.

Note what this does NOT do. Adding a standalone non-FNO *direct* model would
show that some other architecture can do finite-time prediction, which nobody
doubts. A crossover is a property of TWO curves, so testing it needs a matched
pair in the second architecture: AR-UNet against DT-UNet, same encoder, same
decoder, same protocol, differing only in the prediction scheme -- exactly the
relationship ARFNO and DirectTimeFNO already have.

INTERFACE CONTRACT
------------------
    forward(z, e_t) : (B, width, H, W) x (B, cond_dim) or None
                   -> (B, width, H, W)

Same as `FNOBackbone`: channel count and spatial size in equal those out, so
`HistoryEncoder` and `Decoder` are reused unchanged and the only difference
between the FNO and U-Net runs is the middle.

TWO THINGS THAT MAKE THE COMPARISON FAIR, AND ARE EASY TO GET WRONG
-------------------------------------------------------------------
1.  **Parameter count.** A U-Net at base width 64 has ~7.7 M parameters against
    the FNO's ~16.9 M at width 64 / 16x16 modes, so a naive swap compares a
    model to one less than half its size and any difference is confounded.
    `base_width` defaults to 80, which lands at 16.6 M against the FNO backbone's
    16.9 M. `describe()` prints the count so the match can be checked rather
    than assumed; if you change depth or width, check it again.

2.  **Divisibility.** Down/up-sampling requires H and W divisible by
    2**depth. The datasets here are 128x128, 128x256 and 160x200; at depth 3
    all of them divide, but 160x200 does not at depth 4. Rather than fail or
    silently crop, the forward pass reflection-pads up to the next multiple and
    crops back, so a new dataset of any size works and the only cost is a few
    wasted columns.

The FNO's `modes1`, `modes2` and `padding` are accepted and ignored: they are
passed by `_shared_kwargs` for every model, and a U-Net has no spectral
truncation and needs no FFT padding. Ignoring them silently would hide a
misconfiguration, so `describe()` reports that they were ignored.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .time_embedding import FiLM

_ACT = {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU}


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with optional FiLM, mirroring FNOBlock's shape."""

    def __init__(self, c_in: int, c_out: int, cond_dim: int = 0,
                 act: str = "gelu", norm: bool = True,
                 layer_scale: float = 0.1, residual: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        # GroupNorm rather than BatchNorm: batch statistics would couple the
        # samples in a batch, and an autoregressive rollout evaluates one step
        # at a time with a batch composition that differs from training.
        # eps above the 1e-5 default: Gray-Scott has quiescent trajectories
        # whose z-scored fields are very nearly constant, so a group's variance
        # can be small enough that 1/sqrt(var + eps) amplifies pure rounding
        # noise. 1e-4 costs nothing on well-conditioned inputs.
        self.norm1 = nn.GroupNorm(8, c_out, eps=1e-4) if norm else nn.Identity()
        self.norm2 = nn.GroupNorm(8, c_out, eps=1e-4) if norm else nn.Identity()
        self.act = _ACT.get(act, nn.GELU)()
        self.film = FiLM(cond_dim, c_out) if cond_dim > 0 else None
        # Same LayerScale convention as FNOBlock, so the two backbones start
        # from comparably conservative residual updates.
        self.residual = residual and c_in == c_out
        self.scale = (nn.Parameter(torch.full((1, c_out, 1, 1),
                                              float(layer_scale)))
                      if self.residual else None)

    def forward(self, h: torch.Tensor, e_t: Optional[torch.Tensor] = None):
        y = self.act(self.norm1(self.conv1(h)))
        y = self.norm2(self.conv2(y))
        if self.film is not None:
            if e_t is None:
                raise ValueError("block is time-conditioned but got e_t=None")
            y = self.film(y, e_t)
        y = self.act(y)
        return h + self.scale * y if self.residual else y


class UNetBackbone(nn.Module):
    """(B, width, H, W) -> (B, width, H, W), constant depth regardless of tau."""

    def __init__(self, width: int = 64, n_layers: int = 4, cond_dim: int = 0,
                 base_width: int = 80, depth: int = 3,
                 mults: Sequence[int] = (1, 2, 4, 8),
                 residual: bool = True, layer_scale: float = 0.1,
                 act: str = "gelu", norm: bool = True,
                 # accepted and ignored; see the module docstring
                 modes1: int = 0, modes2: int = 0, padding: int = 0,
                 padding_mode: str = "replicate"):
        super().__init__()
        self.width = int(width)
        self.depth = int(depth)
        self._ignored = {"modes1": modes1, "modes2": modes2,
                         "padding": padding, "n_layers": n_layers}
        chans = [int(base_width * m) for m in mults[:depth + 1]]

        self.stem = nn.Conv2d(width, chans[0], 1)
        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        for i in range(depth):
            self.down.append(ConvBlock(chans[i], chans[i], cond_dim, act, norm,
                                       layer_scale, residual))
            self.pool.append(nn.Conv2d(chans[i], chans[i + 1], 3, stride=2,
                                       padding=1))
        self.mid = ConvBlock(chans[depth], chans[depth], cond_dim, act, norm,
                             layer_scale, residual)
        self.up = nn.ModuleList()
        self.merge = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up.append(nn.ConvTranspose2d(chans[i + 1], chans[i], 2,
                                              stride=2))
            self.merge.append(nn.Conv2d(2 * chans[i], chans[i], 1))
        self.head = nn.Conv2d(chans[0], width, 1)
        self.blocks_up = nn.ModuleList([
            ConvBlock(chans[i], chans[i], cond_dim, act, norm, layer_scale,
                      residual) for i in reversed(range(depth))])

    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor, e_t: Optional[torch.Tensor] = None):
        """Compute in fp32 regardless of the surrounding autocast region.

        `torch.autocast(enabled=False)` on its own is NOT enough, and that is
        the subtlety worth writing down: disabling autocast stops PyTorch from
        CHOOSING dtypes, but every op still runs in the dtype of its input.
        Under bf16 training the tensor arriving from HistoryEncoder is already
        bf16, so the block goes on computing in bf16 -- and now without
        autocast's own promotion of `group_norm` to fp32, which is strictly
        worse than leaving autocast enabled. The symptom is a rollout that
        trains cleanly for several epochs and then produces a non-finite loss,
        at a different epoch for every seed.

        FNOBackbone never had this problem because SpectralConv2d upcasts
        explicitly:

            dtype_in = x.dtype
            x = x.float()
            ...
            return out.to(dtype_in)

        Doing the same here is also what keeps the two backbones comparable.
        An architecture comparison in which one arm silently runs at a
        different precision from the other is not an architecture comparison.
        """
        dtype_in = z.dtype
        with torch.autocast(device_type=z.device.type, enabled=False):
            out = self._forward_fp32(
                z.float(), None if e_t is None else e_t.float())
        return out.to(dtype_in)

    def _forward_fp32(self, z: torch.Tensor,
                      e_t: Optional[torch.Tensor] = None):
        H, W = z.shape[-2:]
        m = 2 ** self.depth
        ph, pw = (-H) % m, (-W) % m
        if ph or pw:
            # Reflection, not zeros: a zero border is a physical statement
            # about the field that is false on every dataset here.
            z = F.pad(z, [0, pw, 0, ph], mode="reflect")

        h = self.stem(z)
        skips = []
        for blk, pool in zip(self.down, self.pool):
            h = blk(h, e_t)
            skips.append(h)
            h = pool(h)
        h = self.mid(h, e_t)
        for up, merge, blk, skip in zip(self.up, self.merge, self.blocks_up,
                                        reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = merge(torch.cat([h, skip], dim=1))
            h = blk(h, e_t)
        out = self.head(h)
        if ph or pw:
            out = out[..., :H, :W]
        return out.contiguous()

    # ------------------------------------------------------------------
    def describe(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        ig = ", ".join(f"{k}={v}" for k, v in self._ignored.items() if v)
        return (f"UNetBackbone: {n / 1e6:.2f}M parameters, depth {self.depth}"
                + (f"  [ignored FNO kwargs: {ig}]" if ig else "")
                + "\n    Compare against the FNO backbone this replaces "
                  "(~16.9M at width 64, 16x16\n    modes). A large mismatch "
                  "makes the architecture comparison a size comparison;\n"
                  "    tune `unet_base_width` until the two are within a few "
                  "percent.")
