"""
Direct-Time FNO (Model B) and its semigroup-constrained variant (Model C).

    G_theta : (X_t, dT) -> U(t + dT)          one forward pass, any dT

Structurally the model factorises as encode -> propagate -> decode:

    z0    = E(X_t)                 history encoder, §12.1
    z_T   = Phi(z0, tau)           time-conditioned FNO backbone, §13
    U_hat = D(z_T) (+ U_t)         pointwise decoder, §15

Model C is the SAME network — `SG_DT_FNO` is an alias, not a subclass.  The only
difference is that the training task adds L_SG and L_id (§16-17).  Keeping them
one class is what makes the §3 claim checkable: any measured improvement is the
semigroup inductive bias, because the architecture is bit-identical.

Why the semigroup constraint lives in latent space (§16): composing in field
space would require re-encoding, and re-encoding needs K frames of history that
the model has not predicted.  Phi maps the latent to itself, so
Phi(Phi(z0, tau_a), tau_b) is well-typed and exactly the object the flow-map
identity Phi_{t+s} = Phi_s . Phi_t constrains.

`n_model_evals` returns 1 for every horizon.  That is the constant-depth claim
(§1) in code, and `evaluation/timing.py` reads it rather than assuming.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .fno import FNOBackbone
from .ar_fno import _make_backbone
from .history_encoder import HistoryEncoder, Decoder
from .time_embedding import FourierTimeEmbedding


class TimeInjector(nn.Module):
    """Non-FiLM time conditioning (the A3 ablation arm).

    Broadcasts the time features over H x W, concatenates them to the latent at
    the entry of `propagate`, and projects back to width.  Injecting inside
    `propagate` rather than at the encoder is what keeps Phi composable: a model
    that only sees tau at the encoder is not a latent flow map and cannot be
    asked the semigroup question at all.
    """

    def __init__(self, width: int, feat_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(width + feat_dim, width, 1)

    def forward(self, z: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        B, _, H, W = z.shape
        f = feats.reshape(B, -1, 1, 1).expand(B, feats.shape[-1], H, W)
        return self.proj(torch.cat([z, f.to(z.dtype)], dim=1))


class DirectTimeFNO(nn.Module):
    def __init__(self,
                 n_channels: int,
                 history_len: int = 4,
                 width: int = 64,
                 modes1: int = 16,
                 modes2: int = 16,
                 n_layers: int = 4,
                 time_embed_dim: int = 128,
                 time_bands: int = 8,
                 time_embed_mode: str = "fourier",     # fourier | fourier_log | scalar
                 time_cond: str = "film",              # film | concat
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
        self.time_cond = str(time_cond)

        self.time_embed = FourierTimeEmbedding(
            embed_dim=time_embed_dim, n_bands=time_bands, mode=time_embed_mode)

        cond_dim = time_embed_dim if self.time_cond == "film" else 0
        self.encoder = HistoryEncoder(n_channels, history_len, width,
                                      kernel_size=encoder_kernel)
        self.backbone = _make_backbone(
            backbone, width=width, modes1=modes1, modes2=modes2,
            n_layers=n_layers, cond_dim=cond_dim, padding=padding,
            padding_mode=padding_mode, residual=residual,
            layer_scale=layer_scale, act=act, norm=norm,
            unet_base_width=unet_base_width, unet_depth=unet_depth)
        self.decoder = Decoder(width, n_channels, hidden=decoder_hidden)
        self.injector = (None if self.time_cond == "film"
                         else TimeInjector(width, self._feat_dim()))

    def _feat_dim(self) -> int:
        with torch.no_grad():
            probe = torch.zeros(1, device=next(self.time_embed.parameters()).device)
            return int(self.time_embed.features(probe).shape[-1])

    # -- the three stages -------------------------------------------------
    def encode(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None):
        """x: (B, K, H, W, C) -> z0: (B, d, H, W)."""
        return self.encoder(x, grid)

    def propagate(self, z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Phi(z, tau).  Constant depth: tau changes the conditioning, not the
        number of blocks executed."""
        if self.time_cond == "film":
            return self.backbone(z, self.time_embed(tau))
        feats = self.time_embed.features(tau)
        return self.backbone(self.injector(z, feats), None)

    def decode(self, z: torch.Tensor, x: Optional[torch.Tensor] = None):
        """z -> U_hat (B, H, W, C).  With predict_delta the decoder outputs an
        increment on the most recent input frame, so persistence is exactly
        representable and the model starts from a sane floor."""
        out = self.decoder(z)
        if self.predict_delta:
            if x is None:
                raise ValueError("predict_delta=True needs the history tensor")
            out = out + x[:, -1]
        return out

    # -- forward ----------------------------------------------------------
    def forward(self, x: torch.Tensor, tau: torch.Tensor,
                grid: Optional[torch.Tensor] = None,
                mode: str = "predict",
                tau_a: Optional[torch.Tensor] = None,
                tau_b: Optional[torch.Tensor] = None,
                need_identity: bool = False):
        """mode='predict' returns U_hat; mode='train' returns the latents too.

        Everything a training step needs is produced by THIS method, in one
        call.  That is a DistributedDataParallel requirement, not a style
        choice: DDP installs its gradient-reduction hooks when `forward` is
        invoked on the wrapper, so a task that reached past it to call
        `encode` / `propagate` directly would train happily on one GPU and
        silently skip the all-reduce on four.
        """
        z0 = self.encode(x, grid)
        zT = self.propagate(z0, tau)
        pred = self.decode(zT, x)
        if mode == "predict":
            return pred

        out = {"pred": pred, "z0": z0, "zT": zT}
        if tau_a is not None and tau_b is not None:
            # Composed route for the WHOLE batch — no ragged sub-batching.
            # Unsplittable horizons (h = 1) arrive with tau_a = 0, which is the
            # identity-element case of the same law; the task masks them out of
            # L_SG so the reported number keeps the §16 meaning.
            out["z_compose"] = self.propagate(self.propagate(z0, tau_a), tau_b)
        if need_identity:
            out["z_id"] = self.propagate(z0, torch.zeros_like(tau))
        return out

    @torch.no_grad()
    def predict(self, x: torch.Tensor, tau: torch.Tensor,
                grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(x, tau, grid)

    @torch.no_grad()
    def predict_horizons(self, x: torch.Tensor, tau: torch.Tensor,
                         grid: Optional[torch.Tensor] = None,
                         chunk_size: Optional[int] = None) -> torch.Tensor:
        """Predict many target times while encoding the history only once.

        Parameters
        ----------
        x:
            History tensor ``(B, K, H, W, C)``.
        tau:
            Either a shared vector ``(M,)`` of normalized target times, or a
            per-sample matrix ``(B, M)``.
        grid:
            Optional coordinate grid consumed by the history encoder.
        chunk_size:
            Number of horizons evaluated together. ``None`` evaluates all
            horizons in one batched latent propagation. ``1`` is the cached
            sequential path: it still encodes once, but does not increase the
            horizon batch. Smaller chunks are useful when dense trajectory
            queries would otherwise exceed GPU memory.

        Returns
        -------
        Tensor ``(B, M, H, W, C)``.

        This method is deliberately separate from ``forward`` so existing
        training/evaluation numbers are untouched. It is used by the
        multi-horizon amortization benchmark added for the ICLR revision.
        """
        if tau.ndim == 1:
            tau2 = tau.reshape(1, -1).expand(x.shape[0], -1)
        elif tau.ndim == 2 and tau.shape[0] == x.shape[0]:
            tau2 = tau
        else:
            raise ValueError(
                f"tau must have shape (M,) or (B,M); got {tuple(tau.shape)} "
                f"for batch B={x.shape[0]}")

        B, M = tau2.shape
        if M == 0:
            raise ValueError("predict_horizons needs at least one target time")
        chunk = M if chunk_size is None else max(1, int(chunk_size))

        z0 = self.encode(x, grid)
        outs = []
        for j0 in range(0, M, chunk):
            j1 = min(M, j0 + chunk)
            m = j1 - j0
            z = (z0[:, None]
                 .expand(B, m, *z0.shape[1:])
                 .reshape(B * m, *z0.shape[1:]))
            t = tau2[:, j0:j1].reshape(B * m)
            zT = self.propagate(z, t)

            # decode() needs the most recent history frame when predict_delta
            # is enabled. Repeat only a view of x for the current chunk.
            xr = (x[:, None]
                  .expand(B, m, *x.shape[1:])
                  .reshape(B * m, *x.shape[1:]))
            pred = self.decode(zT, xr)
            outs.append(pred.reshape(B, m, *pred.shape[1:]))
        return torch.cat(outs, dim=1)

    # -- semigroup (§16) --------------------------------------------------
    def semigroup_routes(self, x: torch.Tensor, tau_a: torch.Tensor,
                         tau_b: torch.Tensor,
                         grid: Optional[torch.Tensor] = None,
                         z0: Optional[torch.Tensor] = None
                         ) -> Dict[str, torch.Tensor]:
        """Direct and composed latents for the same total interval.

        Returns z_direct = Phi(z0, tau_a + tau_b) and
                z_compose = Phi(Phi(z0, tau_a), tau_b),
        plus the intermediate z_a so a task can decode it for diagnostics.
        """
        if z0 is None:
            z0 = self.encode(x, grid)
        z_direct = self.propagate(z0, tau_a + tau_b)
        z_a = self.propagate(z0, tau_a)
        z_compose = self.propagate(z_a, tau_b)
        return {"z0": z0, "z_a": z_a,
                "z_direct": z_direct, "z_compose": z_compose}

    def identity_latent(self, z0: torch.Tensor) -> torch.Tensor:
        """Phi(z0, 0) — should equal z0 (§17)."""
        zero = torch.zeros(z0.shape[0], device=z0.device, dtype=torch.float32)
        return self.propagate(z0, zero)

    # -- cost accounting --------------------------------------------------
    @staticmethod
    def n_model_evals(h: int) -> int:
        return 1

    def summary(self) -> str:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        film = sum(p.numel() for m in self.modules()
                   if m.__class__.__name__ == "FiLM" for p in m.parameters())
        return (f"DirectTimeFNO | width={self.width} K={self.K} C={self.C} | "
                f"time_cond={self.time_cond} | params={n:,} "
                f"(FiLM heads {film:,}, {100.0 * film / max(n, 1):.1f}%)")


# Model C is Model B plus two loss terms — see training/tasks.py.
SG_DT_FNO = DirectTimeFNO
