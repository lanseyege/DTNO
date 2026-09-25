"""
Horizon sampling (§20-21).

The horizon distribution is not a detail — it *is* the experiment.  Uniform
sampling over [1, 128] puts half the supervision above h=64, so the model never
learns the short end and the direct-vs-AR crossover gets manufactured rather
than measured.  Three samplers, one per experiment:

    LogBinnedHorizonSampler   Experiment A (feasibility).  Pick a log bin
                              uniformly, then h uniformly inside it.  Short,
                              medium and long horizons arrive in equal numbers.
    UniformHorizonSampler     the A4 ablation control.
    ExplicitHorizonSampler    Experiment B (unseen query time).  Trains only on
                              H_train = {1,2,4,8,16,32,64,128}; every h in
                              H_interp = {3,6,12,24,48,96} is then genuinely
                              unseen at test time, which is the cleanest
                              evidence that the model is not memorising frame
                              indices.

Also here: `split_horizon` for the semigroup loss.  Given a target h it returns
(a, b) with a + b = h, a, b >= 1, so the composed route Phi(Phi(z, tau_a), tau_b)
lands on exactly the horizon the prediction loss is supervising.  h = 1 cannot
be split; those samples contribute prediction loss only.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_LOG_BINS: List[Tuple[int, int]] = [
    (1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128),
]

# §21 — the Experiment B protocol on RealPDEBench.
EXPB_TRAIN = [1, 2, 4, 8, 16, 32, 64, 128]
EXPB_INTERP = [3, 6, 12, 24, 48, 96]
EXPB_EXTRA = [160, 192, 256, 384]

# §20 — the standard evaluation grid, including horizon extrapolation past 128.
EVAL_HORIZONS = [1, 2, 4, 8, 16, 32, 64, 128, 160, 192, 256, 384, 512]


class HorizonSampler:
    """Draws integer horizons h >= 1.  `h_max` bounds what the loader must read."""

    h_max: int

    def sample(self, rng: np.random.Generator, h_limit: Optional[int] = None) -> int:
        raise NotImplementedError

    def describe(self) -> str:
        return type(self).__name__


class LogBinnedHorizonSampler(HorizonSampler):
    def __init__(self, bins: Optional[Sequence[Tuple[int, int]]] = None,
                 h_max: Optional[int] = None):
        bins = list(DEFAULT_LOG_BINS if bins is None else [tuple(b) for b in bins])
        if h_max is not None:
            bins = [(lo, min(hi, h_max)) for lo, hi in bins if lo <= h_max]
        self.bins = bins
        self.h_max = max(hi for _, hi in self.bins)

    def sample(self, rng, h_limit=None) -> int:
        bins = self.bins
        if h_limit is not None:
            bins = [(lo, min(hi, h_limit)) for lo, hi in bins if lo <= h_limit]
            if not bins:
                return 1
        lo, hi = bins[rng.integers(len(bins))]
        return int(rng.integers(lo, hi + 1))

    def describe(self):
        return f"log-binned {self.bins}"


class UniformHorizonSampler(HorizonSampler):
    def __init__(self, h_min: int = 1, h_max: int = 128):
        self.h_min, self.h_max = int(h_min), int(h_max)

    def sample(self, rng, h_limit=None) -> int:
        hi = self.h_max if h_limit is None else min(self.h_max, h_limit)
        if hi < self.h_min:
            return max(1, hi)
        return int(rng.integers(self.h_min, hi + 1))

    def describe(self):
        return f"uniform [{self.h_min}, {self.h_max}]"


class ExplicitHorizonSampler(HorizonSampler):
    def __init__(self, horizons: Sequence[int]):
        self.horizons = sorted(int(h) for h in horizons)
        self.h_max = self.horizons[-1]

    def sample(self, rng, h_limit=None) -> int:
        pool = (self.horizons if h_limit is None
                else [h for h in self.horizons if h <= h_limit])
        if not pool:
            return self.horizons[0]
        return int(pool[rng.integers(len(pool))])

    def describe(self):
        return f"explicit {self.horizons}"


class FixedHorizonSampler(HorizonSampler):
    """h == 1 always — the AR-FNO-1 training distribution."""

    def __init__(self, h: int = 1):
        self.h = int(h)
        self.h_max = self.h

    def sample(self, rng, h_limit=None) -> int:
        return self.h

    def describe(self):
        return f"fixed h={self.h}"


def build_horizon_sampler(cfg: dict) -> HorizonSampler:
    """cfg: {horizon_sampling: log_binned|uniform|explicit|fixed, ...}"""
    mode = str(cfg.get("horizon_sampling", "log_binned")).lower()
    h_max = int(cfg.get("h_max_train", 128))
    if mode in ("log_binned", "log", "log_balanced"):
        return LogBinnedHorizonSampler(cfg.get("horizon_bins"), h_max=h_max)
    if mode == "uniform":
        return UniformHorizonSampler(int(cfg.get("h_min_train", 1)), h_max)
    if mode == "explicit":
        return ExplicitHorizonSampler(cfg.get("train_horizons", EXPB_TRAIN))
    if mode == "fixed":
        return FixedHorizonSampler(int(cfg.get("fixed_horizon", 1)))
    raise ValueError(f"unknown horizon_sampling '{mode}'")


# ---------------------------------------------------------------------------
# Semigroup split
# ---------------------------------------------------------------------------

def split_horizon(h: int, rng: np.random.Generator,
                  mode: str = "uniform") -> Optional[Tuple[int, int]]:
    """(a, b) with a + b = h and a, b >= 1; None when h < 2.

    mode='uniform'  a ~ U{1..h-1}          — every split equally likely
    mode='log'      a ~ log-uniform        — biases towards unequal splits, so
                                             the constraint is exercised at very
                                             different tau_a/tau_b ratios
    """
    h = int(h)
    if h < 2:
        return None
    if mode == "log":
        lo, hi = np.log(1.0), np.log(float(h - 1))
        a = int(round(float(np.exp(rng.uniform(lo, hi)))))
        a = max(1, min(h - 1, a))
    else:
        a = int(rng.integers(1, h))
    return a, h - a
