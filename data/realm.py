"""
REALM IgnitHIT / EvolveJet — cross-dataset verification (§5, §22, Phase 5).

Phase 5's whole point is that the code does NOT change: same architecture, same
training protocol, same losses.  Only three things move — channel count, data
loader, horizon range — so this module is deliberately thin.  It sets REALM's
defaults and then calls straight into `data/realpde.py`.

What actually differs:

  * 36 trajectories x 30 timesteps (IgnitHIT), 12 variables.  30 frames is 67x
    shorter than RealPDEBench, so the horizon protocol shrinks to
    H_train = {1,2,4,8,12}, H_interp = {3,6,10}, H_extra = {14,16,20} (§22).
    With K = 4 history frames the last usable anchor is t0 = 25, so h = 20
    already needs t0 <= 5 — anchors are scarce and `eval_stride` must be small.
  * dt is dataset-specific; set it in the config so tau stays physical.
  * REALM ships as HDF5/npy rather than one big zarr.  Convert once with
    `scripts/convert_realm.py`, which writes the per-trajectory .npy layout the
    NPYStore expects, then point `data_path` at that directory.

The value here is not longer horizons.  It is whether the trend reproduces on a
second high-fidelity reacting-flow benchmark whose official protocol is
short-horizon AR training plus full-horizon AR rollout — i.e. the setting our
method is arguing against.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from .realpde import build_data as _build_data, build_eval_loader as _build_eval

# §22 protocol for a 30-frame trajectory.
IGNITHIT_TRAIN_HORIZONS = [1, 2, 4, 8, 12]
IGNITHIT_INTERP = [3, 6, 10]
IGNITHIT_EXTRA = [14, 16, 20]
IGNITHIT_EVAL = sorted(set(IGNITHIT_TRAIN_HORIZONS + IGNITHIT_INTERP + IGNITHIT_EXTRA))

REALM_DEFAULTS = {
    "backend": "npy",
    "history_len": 4,
    "horizon_sampling": "log_binned",
    "horizon_bins": [[1, 2], [3, 4], [5, 8], [9, 12]],
    "h_max_train": 12,
    "eval_horizons": IGNITHIT_EVAL,
    "eval_stride": 1,
    "n_val_traj": 6,
    "n_test_traj": 6,
    "split_path": "artifacts/split_realm_ignithit.json",
    "norm_stats_path": "artifacts/norm_stats_realm_ignithit.json",
    "samples_per_epoch": 8000,
    "val_samples": 1000,
}


def apply_defaults(cfg: dict) -> dict:
    """Fill REALM defaults without overriding anything the config states."""
    out = dict(REALM_DEFAULTS)
    out.update(cfg)
    return out


def build_data(cfg: dict, model_kind: str = "direct",
               distributed: bool = False, seed: int = 0) -> Dict:
    return _build_data(apply_defaults(cfg), model_kind, distributed, seed)


def build_eval_loader(cfg: dict, subset: str = "test",
                      horizons: Optional[Sequence[int]] = None,
                      distributed: bool = False):
    return _build_eval(apply_defaults(cfg), subset, horizons, distributed)
