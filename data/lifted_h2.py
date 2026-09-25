"""
BLASTNet Lifted Hydrogen Jet Flame — Phase 5 cross-dataset check.

Thin, like `data/realm.py`, and for the same reason: §41's whole point is that
the architecture, losses, protocol and evaluation do not change between
datasets. This module only sets defaults and then calls straight into
`data/realpde.py`.

Registered separately from `realpde_combustion` rather than reusing that name
with an npy backend. The functional behaviour is identical either way, but a
config that says `dataset_name: realpde_combustion` while pointing at a
hydrogen jet is a lie that will eventually be read by someone as the truth --
including by whoever writes the paper's data section.

The user's `lifted_h2_dataset.py` is NOT imported here. It is used once, by
`scripts/convert_lifted_h2.py`, to turn the raw .dat files into the per-
trajectory .npy layout that NPYStore reads. After conversion, this dataset goes
through exactly the same code path as RealPDEBench.

What differs from RealPDEBench, and it is worth keeping in view:

  * 8 trajectories (Re = 5000..11000), against 30. With 1 val + 1 test the
    held-out sets are a single Reynolds number each, so there are no error bars
    over flow realizations -- only over seeds. Trend check, not measurement.
  * 181 usable frames after id-alignment, against 2001. Horizons past ~128
    leave too few anchors on one test trajectory to mean anything.
  * 160 x 200, anisotropic and non-periodic in both directions.
  * 5 channels (UX, UY, T, YH2O, YOH) and NO heat-release channel, so the §29
    reaction-zone mask falls back to OH. An OH iso-contour and a heat-release
    iso-contour are not interchangeable; say which one a figure used.
  * The generalization axis is Reynolds number, not fuel composition. Re = 7500
    is interpolated between two training cases, which is a genuinely different
    (and easier) test than RealPDEBench's held-out fuel blends.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from .realpde import build_data as _build_data, build_eval_loader as _build_eval

# Re, in the trajectory order written by scripts/convert_lifted_h2.py.
RE_ORDER = [5000, 6000, 7000, 7500, 8000, 9000, 10000, 11000]
DESIGNED_TEST_RE = 7500

LIFTED_H2_DEFAULTS = {
    "backend": "npy",
    "file_pattern": "*.npy",
    "history_len": 4,
    "horizon_sampling": "log_binned",
    "horizon_bins": [[1, 2], [3, 4], [5, 8], [9, 16], [17, 32], [33, 64]],
    "h_max_train": 64,
    "eval_horizons": [1, 2, 4, 8, 16, 32, 64, 96, 128],
    "val_horizons": [1, 4, 16, 64],
    "eval_stride": 2,
    "hval_stride": 8,
    "n_val_traj": 1,
    "n_test_traj": 1,
    "split_path": "artifacts/split_lifted_h2.json",
    "norm_stats_path": "artifacts/norm_stats_lifted_h2.json",
    "samples_per_epoch": 12000,
    "val_samples": 1500,
    "modes1": 16,
    "modes2": 20,
}


def apply_defaults(cfg: dict) -> dict:
    out = dict(LIFTED_H2_DEFAULTS)
    out.update(cfg)
    return out


def build_data(cfg: dict, model_kind: str = "direct",
               distributed: bool = False, seed: int = 0) -> Dict:
    return _build_data(apply_defaults(cfg), model_kind, distributed, seed)


def build_eval_loader(cfg: dict, subset: str = "test",
                      horizons: Optional[Sequence[int]] = None,
                      distributed: bool = False):
    return _build_eval(apply_defaults(cfg), subset, horizons, distributed)
