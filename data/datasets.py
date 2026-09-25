"""
Datasets (§8).

Three dataset classes, all store-agnostic, all sharing one output contract so
that a direct-time model and an autoregressive model are trained and evaluated
on literally the same bytes.

    DirectPairDataset   online (trajectory, anchor t0, horizon h) sampling.
                        Returns K history frames, tau, and the single target
                        frame at t0 + h.  No pair list is materialised: with
                        30 x 2001 frames and h up to 512 that list would have
                        ~10^7 entries and would pin the horizon distribution at
                        construction time, which is exactly what §20 forbids.

    ARWindowDataset     contiguous window (K history + r consecutive targets)
                        for AR-FNO-1 (r=1) and AR-FNO-R (r>1).

    EvalAnchorDataset   deterministic anchors on a fixed grid, every anchor
                        carrying targets for the FULL horizon list.  Every
                        horizon is therefore scored on the same anchors and the
                        error-vs-horizon curve is a like-for-like comparison
                        rather than a mix of different initial conditions.

Randomness.  DirectPairDataset is map-style with a virtual length, and
`__getitem__(i)` seeds a fresh Generator from (base_seed, epoch, i).  That keeps
online sampling reproducible, makes DistributedSampler shard it correctly across
the 4 GPUs, and means `set_epoch` genuinely changes the draw.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .store import TrajectoryStore
from .transforms import Normalizer
from .sampling import HorizonSampler


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _coord_grid(H: int, W: int) -> np.ndarray:
    """(H, W, 2) normalised coordinates, concatenated onto the history (§12.1)."""
    y = np.linspace(0.0, 1.0, H, dtype=np.float32)
    x = np.linspace(0.0, 1.0, W, dtype=np.float32)
    Y, X = np.meshgrid(y, x, indexing="ij")
    return np.stack([Y, X], axis=-1)


class _Base(Dataset):
    def __init__(self, store: TrajectoryStore, traj_indices: Sequence[int],
                 channel_indices: Sequence[int], normalizer: Normalizer,
                 history_len: int = 4, t_range: Optional[Sequence[int]] = None):
        self.store = store
        self.traj_indices = [int(i) for i in traj_indices]
        self.channel_indices = [int(c) for c in channel_indices]
        self.norm = normalizer
        self.K = int(history_len)
        self.H, self.W = store.H, store.W
        self.C = len(self.channel_indices)
        self.dt = store.dt
        lo, hi = (0, store.T) if t_range is None else (int(t_range[0]), int(t_range[1]))
        self.t_lo, self.t_hi = lo, min(hi, store.T)

    @property
    def channel_names(self) -> List[str]:
        return [self.store.channel_names[c] for c in self.channel_indices]

    def _read(self, traj: int, t_idx: Sequence[int]) -> np.ndarray:
        raw = self.store.read(traj, t_idx, self.channel_indices)
        return self.norm.forward(raw)


# ---------------------------------------------------------------------------
# Direct-time pairs
# ---------------------------------------------------------------------------

class DirectPairDataset(_Base):
    """(X_history, tau, Y_target) with h drawn by a HorizonSampler."""

    def __init__(self, store, traj_indices, channel_indices, normalizer,
                 horizon_sampler: HorizonSampler,
                 history_len: int = 4,
                 samples_per_epoch: int = 20000,
                 t_range: Optional[Sequence[int]] = None,
                 t_scale: Optional[float] = None,
                 base_seed: int = 0,
                 deterministic: bool = False):
        super().__init__(store, traj_indices, channel_indices, normalizer,
                         history_len, t_range)
        self.sampler = horizon_sampler
        self.n = int(samples_per_epoch)
        self.base_seed = int(base_seed)
        self.epoch = 0
        self.deterministic = bool(deterministic)   # validation: frozen draw
        # tau = h * dt / t_scale.  Default puts the longest TRAINING horizon at
        # tau = 1, so horizon extrapolation shows up as tau > 1 and nothing in
        # the time embedding is silently rescaled between datasets.
        self.t_scale = float(t_scale if t_scale is not None
                             else self.sampler.h_max * self.dt)
        self.grid = _coord_grid(self.H, self.W)

        span = self.t_hi - self.t_lo
        if span < self.K + 1:
            raise ValueError(f"time range [{self.t_lo}, {self.t_hi}) is shorter "
                             f"than history {self.K} + 1")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n

    def _rng(self, idx: int) -> np.random.Generator:
        ep = 0 if self.deterministic else self.epoch
        return np.random.default_rng((self.base_seed, ep, idx))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = self._rng(idx)
        traj = self.traj_indices[rng.integers(len(self.traj_indices))]

        # anchor: needs K-1 frames of history behind it and >=1 frame ahead
        t0_lo = self.t_lo + self.K - 1
        t0_hi = self.t_hi - 2                      # inclusive
        t0 = int(rng.integers(t0_lo, t0_hi + 1))
        h = self.sampler.sample(rng, h_limit=self.t_hi - 1 - t0)
        h = max(1, min(h, self.t_hi - 1 - t0))

        t_idx = list(range(t0 - self.K + 1, t0 + 1)) + [t0 + h]
        block = self._read(traj, t_idx)            # (K+1, H, W, C)

        x = torch.from_numpy(np.ascontiguousarray(block[:self.K]))
        y = torch.from_numpy(np.ascontiguousarray(block[self.K]))
        tau = float(h) * self.dt / self.t_scale
        return {
            "x": x,                                        # (K, H, W, C)
            "y": y,                                        # (H, W, C)
            "tau": torch.tensor(tau, dtype=torch.float32),
            "h": torch.tensor(h, dtype=torch.long),
            "grid": torch.from_numpy(self.grid),           # (H, W, 2)
            "traj": torch.tensor(traj, dtype=torch.long),
            "t0": torch.tensor(t0, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Autoregressive windows
# ---------------------------------------------------------------------------

class ARWindowDataset(_Base):
    """K history frames + `rollout` consecutive targets (§23).

    rollout=1 gives AR-FNO-1.  rollout>1 with `random_rollout` gives AR-FNO-R,
    whose training draw is r ~ U{1..R} so a single checkpoint covers the whole
    short-rollout regime instead of overfitting one r.
    """

    def __init__(self, store, traj_indices, channel_indices, normalizer,
                 history_len: int = 4, rollout: int = 1,
                 random_rollout: bool = False,
                 samples_per_epoch: int = 20000,
                 t_range: Optional[Sequence[int]] = None,
                 base_seed: int = 0, deterministic: bool = False):
        super().__init__(store, traj_indices, channel_indices, normalizer,
                         history_len, t_range)
        self.R = int(rollout)
        self.random_rollout = bool(random_rollout)
        self.n = int(samples_per_epoch)
        self.base_seed = int(base_seed)
        self.epoch = 0
        self.deterministic = bool(deterministic)
        self.grid = _coord_grid(self.H, self.W)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep = 0 if self.deterministic else self.epoch
        rng = np.random.default_rng((self.base_seed, ep, idx))
        traj = self.traj_indices[rng.integers(len(self.traj_indices))]

        t0_lo = self.t_lo + self.K - 1
        t0_hi = self.t_hi - 1 - self.R
        if t0_hi < t0_lo:
            raise ValueError("time range too short for the requested rollout")
        t0 = int(rng.integers(t0_lo, t0_hi + 1))

        t_idx = list(range(t0 - self.K + 1, t0 + 1 + self.R))
        block = self._read(traj, t_idx)            # (K+R, H, W, C)

        x = torch.from_numpy(np.ascontiguousarray(block[:self.K]))
        y = torch.from_numpy(np.ascontiguousarray(block[self.K:]))   # (R,H,W,C)

        r_eff = self.R
        if self.random_rollout and self.R > 1:
            r_eff = int(rng.integers(1, self.R + 1))
        return {
            "x": x,
            "y": y,
            "r_eff": torch.tensor(r_eff, dtype=torch.long),
            "grid": torch.from_numpy(self.grid),
            "traj": torch.tensor(traj, dtype=torch.long),
            "t0": torch.tensor(t0, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Evaluation anchors
# ---------------------------------------------------------------------------

class EvalAnchorDataset(_Base):
    """Fixed anchors, each carrying the target frame for EVERY eval horizon.

    Anchors are laid on a stride grid and kept only if t0 + max(horizons) is in
    range, so all horizons share an identical anchor set.  Without that the
    error-vs-horizon curve mixes different initial conditions and its shape
    stops meaning anything.
    """

    def __init__(self, store, traj_indices, channel_indices, normalizer,
                 horizons: Sequence[int], history_len: int = 4,
                 stride: int = 50, t_range: Optional[Sequence[int]] = None,
                 t_scale: float = 1.0, max_anchors_per_traj: Optional[int] = None):
        super().__init__(store, traj_indices, channel_indices, normalizer,
                         history_len, t_range)
        self.horizons = sorted(int(h) for h in horizons)
        self.h_max = self.horizons[-1]
        self.t_scale = float(t_scale)
        self.grid = _coord_grid(self.H, self.W)

        self.anchors: List[tuple] = []
        t0_lo = self.t_lo + self.K - 1
        t0_hi = self.t_hi - 1 - self.h_max
        for traj in self.traj_indices:
            if t0_hi < t0_lo:
                continue
            ts = list(range(t0_lo, t0_hi + 1, int(stride)))
            if max_anchors_per_traj is not None:
                ts = ts[:int(max_anchors_per_traj)]
            self.anchors += [(traj, t) for t in ts]
        if not self.anchors:
            usable = self.t_hi - self.t_lo - self.K
            raise ValueError(
                f"no eval anchors: need {self.K - 1} frames of history plus "
                f"{self.h_max} ahead inside [{self.t_lo}, {self.t_hi}). "
                f"The longest usable horizon here is h = {usable}; drop "
                f"everything above it from `eval_horizons`.")

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj, t0 = self.anchors[idx]
        t_idx = list(range(t0 - self.K + 1, t0 + 1)) + [t0 + h for h in self.horizons]
        block = self._read(traj, t_idx)

        x = torch.from_numpy(np.ascontiguousarray(block[:self.K]))
        y = torch.from_numpy(np.ascontiguousarray(block[self.K:]))   # (n_h,H,W,C)
        taus = torch.tensor([h * self.dt / self.t_scale for h in self.horizons],
                            dtype=torch.float32)
        return {
            "x": x,
            "y": y,
            "taus": taus,
            "horizons": torch.tensor(self.horizons, dtype=torch.long),
            "grid": torch.from_numpy(self.grid),
            "traj": torch.tensor(traj, dtype=torch.long),
            "t0": torch.tensor(t0, dtype=torch.long),
        }
