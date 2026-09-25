"""
Per-channel preprocessing (§11 of the proposal).

Reacting-flow channels do not share a dynamic range: temperature spans ~300-2500
while OH mole fraction spans ~1e-9-1e-2.  A raw MSE over those channels is a
temperature loss with rounding noise attached, so every channel gets its own
transform followed by a z-score.

Transforms
----------
zscore    x' = (x - mu) / sigma
log       x' = log(x + eps), then z-score.  Non-negative, many-decade species.
symlog    x' = sign(x) * log1p(|x| / s), then z-score.  Signed heat release.
log1p     x' = log1p(x / s), then z-score.  Non-negative heat release.
boxcox    x' = ((x + shift)^lam - 1) / lam   (lam != 0)
                log(x + shift)               (lam == 0), then z-score.
          `lam` is fitted once by the audit on a subsample; REALM uses the same
          family, so keeping it here means the two pipelines stay comparable.

Every statistic (mu, sigma, eps, s, lam, shift) is computed on TRAINING
TRAJECTORIES ONLY (§11.1) by `scripts/audit_data.py`, frozen into
`norm_stats.json`, and reloaded verbatim at train and test time.  Nothing here
ever fits on the fly — that is a data-leak surface, and a silent one.

All transforms are invertible; `Normalizer.inverse` is what the physical-units
metrics (§30, integrated heat release) and the qualitative field figures use.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Per-channel spec
# ---------------------------------------------------------------------------

@dataclass
class ChannelStat:
    name: str
    transform: str = "zscore"
    mean: float = 0.0            # mean/std are in TRANSFORMED space
    std: float = 1.0
    eps: float = 1e-12           # log floor
    scale: float = 1.0           # symlog / log1p scale s
    lam: float = 0.0             # box-cox exponent
    shift: float = 0.0           # box-cox shift, keeps the argument positive
    # raw-space diagnostics, written by the audit for the record
    raw_min: float = 0.0
    raw_max: float = 0.0
    raw_mean: float = 0.0
    raw_std: float = 1.0

    # -- forward ---------------------------------------------------------
    def pre(self, x: np.ndarray) -> np.ndarray:
        t = self.transform
        if t == "zscore":
            return x
        if t == "log":
            return np.log(np.maximum(x, self.eps))
        if t == "log1p":
            return np.log1p(np.maximum(x, 0.0) / self.scale)
        if t == "symlog":
            return np.sign(x) * np.log1p(np.abs(x) / self.scale)
        if t == "boxcox":
            z = np.maximum(x + self.shift, self.eps)
            return np.log(z) if abs(self.lam) < 1e-8 else (z ** self.lam - 1.0) / self.lam
        raise ValueError(f"unknown transform '{t}'")

    def post(self, y: np.ndarray) -> np.ndarray:
        t = self.transform
        if t == "zscore":
            return y
        if t == "log":
            return np.exp(y)
        if t == "log1p":
            return np.expm1(y) * self.scale
        if t == "symlog":
            return np.sign(y) * np.expm1(np.abs(y)) * self.scale
        if t == "boxcox":
            z = (np.exp(y) if abs(self.lam) < 1e-8
                 else np.maximum(self.lam * y + 1.0, 1e-12) ** (1.0 / self.lam))
            return z - self.shift
        raise ValueError(f"unknown transform '{t}'")

    def forward(self, x: np.ndarray) -> np.ndarray:
        return (self.pre(x) - self.mean) / self.std

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return self.post(y * self.std + self.mean)


# ---------------------------------------------------------------------------
# Normalizer over a channel selection
# ---------------------------------------------------------------------------

class Normalizer:
    """Vectorised forward/inverse over the last (channel) axis.

    Works on numpy arrays inside the dataset and on torch tensors at eval time
    (`inverse_torch`), which is where physical-unit metrics need it.
    """

    def __init__(self, stats: Sequence[ChannelStat]):
        self.stats: List[ChannelStat] = list(stats)
        self.names = [s.name for s in self.stats]
        self._uniform = all(s.transform == "zscore" for s in self.stats)
        self._mean = np.array([s.mean for s in self.stats], dtype=np.float32)
        self._std = np.array([s.std for s in self.stats], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.stats)

    # -- numpy -----------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (..., C) raw -> normalized."""
        if self._uniform:
            return (x - self._mean) / self._std
        out = np.empty_like(x, dtype=np.float32)
        for c, st in enumerate(self.stats):
            out[..., c] = st.forward(x[..., c])
        return out

    def inverse(self, y: np.ndarray) -> np.ndarray:
        if self._uniform:
            return y * self._std + self._mean
        out = np.empty_like(y, dtype=np.float32)
        for c, st in enumerate(self.stats):
            out[..., c] = st.inverse(y[..., c])
        return out

    # -- torch -----------------------------------------------------------
    def inverse_torch(self, y):
        """Same as `inverse` for a torch tensor with channels last."""
        import torch
        mean = torch.as_tensor(self._mean, device=y.device, dtype=y.dtype)
        std = torch.as_tensor(self._std, device=y.device, dtype=y.dtype)
        z = y * std + mean
        if self._uniform:
            return z
        out = torch.empty_like(z)
        for c, st in enumerate(self.stats):
            t = st.transform
            zc = z[..., c]
            if t == "zscore":
                out[..., c] = zc
            elif t == "log":
                out[..., c] = torch.exp(zc)
            elif t == "log1p":
                out[..., c] = torch.expm1(zc) * st.scale
            elif t == "symlog":
                out[..., c] = torch.sign(zc) * torch.expm1(torch.abs(zc)) * st.scale
            elif t == "boxcox":
                if abs(st.lam) < 1e-8:
                    w = torch.exp(zc)
                else:
                    w = torch.clamp(st.lam * zc + 1.0, min=1e-12) ** (1.0 / st.lam)
                out[..., c] = w - st.shift
            else:
                raise ValueError(t)
        return out

    # -- io --------------------------------------------------------------
    def save(self, path: str, extra: Optional[dict] = None):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        blob = {"channels": [asdict(s) for s in self.stats]}
        if extra:
            blob.update(extra)
        with open(path, "w") as fp:
            json.dump(blob, fp, indent=2)

    @classmethod
    def load(cls, path: str, channel_names: Optional[Sequence[str]] = None) -> "Normalizer":
        with open(path) as fp:
            blob = json.load(fp)
        stats = {d["name"]: ChannelStat(**d) for d in blob["channels"]}
        if channel_names is None:
            return cls(list(stats.values()))
        missing = [n for n in channel_names if n not in stats]
        if missing:
            raise KeyError(f"{path} has no statistics for {missing}. Re-run "
                           f"scripts/audit_data.py with these channels selected.")
        return cls([stats[n] for n in channel_names])

    @staticmethod
    def load_meta(path: str) -> dict:
        with open(path) as fp:
            blob = json.load(fp)
        return {k: v for k, v in blob.items() if k != "channels"}


# ---------------------------------------------------------------------------
# Fitting helpers (audit only)
# ---------------------------------------------------------------------------

def fit_boxcox_lambda(sample: np.ndarray, shift: float,
                      grid: Optional[Sequence[float]] = None) -> float:
    """Pick lambda on a coarse grid by maximising the Box-Cox log-likelihood.

    scipy.stats.boxcox_normmax is the usual tool but it is O(n log n) per
    evaluation and we are handing it millions of points; a 13-point grid over
    [-1, 1] is plenty for a preprocessing decision and is what REALM's
    published range covers.
    """
    grid = np.linspace(-1.0, 1.0, 13) if grid is None else np.asarray(grid)
    x = np.asarray(sample, dtype=np.float64).ravel()
    x = x[np.isfinite(x)] + shift
    x = x[x > 0]
    if x.size < 100:
        return 0.0
    n = x.size
    log_x_sum = np.log(x).sum()
    best_lam, best_ll = 0.0, -np.inf
    for lam in grid:
        y = np.log(x) if abs(lam) < 1e-8 else (x ** lam - 1.0) / lam
        var = y.var()
        if var <= 0 or not np.isfinite(var):
            continue
        ll = -0.5 * n * np.log(var) + (lam - 1.0) * log_x_sum
        if ll > best_ll:
            best_ll, best_lam = ll, float(lam)
    return best_lam
