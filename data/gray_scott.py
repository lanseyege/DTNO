"""
The Well — Gray-Scott reaction-diffusion.  Experiment A1.

(This is the module the request called `gray_scott_dataset.py`; it is named to
match `data/realpde.py`, `data/lifted_h2.py`, `data/realm.py`, which are the
files it sits beside and behaves like.)

Thin, like `data/lifted_h2.py`, and for the same reason: §41's whole point is
that the architecture, losses, protocol and evaluation do not change between
datasets.  This module sets defaults, publishes the dataset's two traps, and
calls straight into `data/realpde.py`.


WHAT THIS DATASET IS FOR
------------------------
It is not "the easy one".  Its job in the ladder is to **isolate reaction
dynamics**: an autonomous, periodic, two-species reaction-diffusion system with
no advection at all,

    dA/dt = d_A lap A - A B^2 + f (1 - A)
    dB/dt = d_B lap B + A B^2 - (f + k) B

which makes it the cleanest semigroup testbed in the whole study.  Phi is
genuinely time-homogeneous here (no forcing, no absolute-time dependence), so
if the §16-17 latent semigroup constraint is ever going to help, it should help
here.  On both combustion datasets it was satisfiable to 1-2 orders of
magnitude and improved accuracy on neither; a *positive* result here would be
the most interesting single number the extension can produce, and a second null
would be strong evidence that compositional regularization is not the missing
ingredient.  Either way it is a measurement, not a prediction.

Two further properties matter:

  * **1001 frames.**  The longest time axis of any dataset in the study bar
    Cylinder, and unlike Cylinder it is only 128 x 128 x 2, so the full
    h = 1..512 grid is affordable at three seeds.
  * **Periodic boundaries and a uniform grid.**  Set `padding: 0` — the §15
    replicate padding exists for the non-periodic burner and here it would
    *break* the exact periodicity the FNO's FFT already assumes.  The §27
    spectral metric is also physically meaningful on this grid, which it is not
    on Rayleigh-Benard's Chebyshev nodes.


TRAP 1 — STATIONARY TRAJECTORIES
--------------------------------
Some (f, k) settings reach a fixed point.  The Well documents which
trajectories and when: mostly the "Gliders" set (f=0.014, k=0.054) at stored
step ~121-159, plus four "Spirals" (f=0.018, k=0.051) at ~107-113.  That is
roughly 12% of the trajectories in those two parameter sets, and the record is
1001 steps long, so a stationary trajectory is ~87% frames on which
**persistence is exact**.

Leaving them in does not make the numbers noisy, it makes them wrong in a
specific and flattering direction: every method's E(h) flattens, persistence
becomes a strong baseline for the wrong reason, and the crossover between
direct-time and autoregressive prediction moves because AR stops accumulating
error once the field stops moving.  This is the same failure mode §9 of the
handover records for flat error curves generally — a number that looks like
physics and is actually an artefact of the data.

`OFFICIAL_STATIONARY` below is The Well's own table, transcribed.  It is the
*claim*.  `scripts/screen_stationary.py` is the *measurement*, and it also
covers species B, which the official table does not.  Run the screen, compare
the two, and point `data.exclude_trajectories` at the JSON it writes.  Report
the excluded count in the paper.


TRAP 2 — THE 6 PARAMETER SETS ARE 6 DIFFERENT FLOWS
---------------------------------------------------
Gliders, bubbles, maze, worms, spirals and spots are qualitatively different
dynamics, not six samples of one distribution.  Two consequences:

  * The split must be stratified over them, or a held-out set can end up
    measuring one pattern type.  `scripts/prepare_split.py` does this.
  * Holding out a whole parameter set is a genuinely harder generalization
    test — and the one The Well's own documentation names as the interesting
    one.  It is a *different claim* from the headline (unseen initial
    condition, seen dynamics), so it is offered as an optional extra arm
    (`--split-mode param_holdout`), never silently substituted.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

from .realpde import build_data as _build_data, build_eval_loader as _build_eval

# name -> (f, k), from the dataset page.
GRAY_SCOTT_PATTERNS: Dict[str, Tuple[float, float]] = {
    "gliders": (0.014, 0.054),
    "bubbles": (0.098, 0.057),
    "maze":    (0.029, 0.057),
    "worms":   (0.058, 0.065),
    "spirals": (0.018, 0.051),
    "spots":   (0.030, 0.062),
}

# The Well's published stationary-trajectory table, keyed
# (official_split, f, k) -> [realization index].  Species A only; the times at
# which stationarity was reached are 107-159 in every case, i.e. inside the
# first 16% of a 1001-frame record.
OFFICIAL_STATIONARY: Dict[Tuple[str, float, float], List[int]] = {
    ("val", 0.014, 0.054): [7, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19],
    ("val", 0.018, 0.051): [14],
    ("train", 0.014, 0.054): [
        81, 82, 83, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
        100, 101, 102, 103, 105, 107, 108, 110, 111, 112, 113, 114, 115, 116,
        117, 118, 119, 120, 121, 122, 123, 125, 126, 127, 129, 130, 131, 132,
        133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 144, 145, 146, 147,
        148, 149, 150, 151, 152, 153, 154, 155, 156, 158, 159],
    ("train", 0.018, 0.051): [97, 134, 147, 153],
    ("test", 0.014, 0.054): [12, 13, 14, 15, 16, 17, 18, 19],
    ("test", 0.018, 0.051): [11],
}

# Scalar names The Well might use for the feed and kill rates. Discovery reads
# whichever exists; the audit prints what it found.
FEED_KEYS = ("f", "feed_rate", "F", "feed")
KILL_KEYS = ("k", "kill_rate", "K", "kill")


GRAY_SCOTT_DEFAULTS = {
    "backend": "well",
    "file_pattern": "*.hdf5",
    "history_len": 4,
    "param_scalars": (),
    "horizon_sampling": "log_binned",
    "h_max_train": 128,
    "horizon_bins": [[1, 2], [3, 4], [5, 8], [9, 16], [17, 32], [33, 64],
                     [65, 128]],
    # Same grid as RealPDEBench so h is comparable across datasets without a
    # per-dataset rescaling in every figure.
    "eval_horizons": [1, 2, 4, 8, 16, 32, 64, 128, 160, 192, 256, 384, 512],
    "val_horizons": [1, 4, 16, 64, 128],
    "eval_stride": 25,
    "hval_stride": 100,
    "split_path": "artifacts/split_gray_scott.json",
    "norm_stats_path": "artifacts/norm_stats_gray_scott.json",
    "samples_per_epoch": 20000,
    "val_samples": 2000,
    "modes1": 16,
    "modes2": 16,
    "padding": 0,          # periodic domain: see the module docstring
}


def apply_defaults(cfg: dict) -> dict:
    out = dict(GRAY_SCOTT_DEFAULTS)
    out.update(cfg)
    return out


# ---------------------------------------------------------------------------

def pattern_of(params: Dict[str, float], tol: float = 1e-6) -> Optional[str]:
    """Map a file's (f, k) scalars onto one of the six named pattern types."""
    f = next((params[k] for k in FEED_KEYS if k in params), None)
    k_ = next((params[k] for k in KILL_KEYS if k in params), None)
    if f is None or k_ is None:
        return None
    for name, (pf, pk) in GRAY_SCOTT_PATTERNS.items():
        if abs(f - pf) < tol and abs(k_ - pk) < tol:
            return name
    return None


def pattern_from_filename(basename: str) -> Optional[str]:
    """Fallback when a file exposes no (f, k) scalars: match the name."""
    low = basename.lower()
    for name in GRAY_SCOTT_PATTERNS:
        if name in low:
            return name
    return None


def official_stationary_exclusions(store) -> List[Tuple[str, int]]:
    """-> [(file basename, realization index)] for `data.exclude_trajectories`.

    Uses The Well's published table.  This is the *claim*; cross-check it
    against `scripts/screen_stationary.py`, which measures both species and
    will disagree if the file layout is not what this function assumes.
    """
    out: List[Tuple[str, int]] = []
    for t in range(store.n_traj):
        params = store.traj_params(t)
        pat = pattern_of(params) or pattern_from_filename(store.file_of(t))
        if pat is None:
            continue
        f, k = GRAY_SCOTT_PATTERNS[pat]
        key = (store.official_split[t], round(f, 6), round(k, 6))
        if store.realization_of(t) in OFFICIAL_STATIONARY.get(key, ()):
            # store.file_of returns the SPLIT-QUALIFIED key. The Well gives one
            # file name to one operating point in each of train/valid/test, so
            # a bare basename here would exclude the named realization from all
            # three splits at once -- and The Well's table is per-split, which
            # is exactly the distinction that would be lost.
            out.append((store.file_of(t), store.realization_of(t)))
    return out


def strata(store) -> List[str]:
    """Split stratification label: the pattern type, not the file."""
    labels = []
    for t in range(store.n_traj):
        pat = (pattern_of(store.traj_params(t))
               or pattern_from_filename(store.file_of(t))
               or os.path.splitext(store.file_of(t))[0])
        labels.append(pat)
    return labels


# ---------------------------------------------------------------------------

def build_data(cfg: dict, model_kind: str = "direct",
               distributed: bool = False, seed: int = 0) -> Dict:
    return _build_data(apply_defaults(cfg), model_kind, distributed, seed)


def build_eval_loader(cfg: dict, subset: str = "test",
                      horizons: Optional[Sequence[int]] = None,
                      distributed: bool = False):
    return _build_eval(apply_defaults(cfg), subset, horizons, distributed)
