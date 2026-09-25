"""
The Well — Rayleigh-Benard convection.  Experiment A3.

(The module the request called `rayleigh_benard_dataset.py`; named to match its
neighbours `realpde.py` / `lifted_h2.py`.)

Its job in the ladder is to **isolate thermal convection**: buoyancy-driven
transport with a real temperature-like field and a real pressure field, but no
chemistry.  It is the closest non-reacting analogue of what the combustion
datasets do, which makes it the bridge between Experiment A and Experiment B —
if the direct-time advantage survives advection (Cylinder) and survives
buoyancy-driven transport (here), then whatever changes in the reacting flows
is attributable to chemistry rather than to "harder fluid dynamics".

    1750 simulations (35 parameter settings x 50 initial conditions)
    200 frames, 512 x 128, horizontally periodic, walls top and bottom
    4 channels: buoyancy, pressure, velocity_x, velocity_y


THREE THINGS TO SETTLE BEFORE TRAINING
--------------------------------------

1.  **The vertical grid is not uniform.**  The Well's `rayleigh_benard` is
    sampled at Chebyshev nodes along z; `rayleigh_benard_uniform` is the same
    data resampled onto a uniform grid.  An FNO's FFT and the §27 spectral
    metric both assume uniform spacing.  On the Chebyshev grid the model still
    trains and E_field is still exact — it is a pointwise error on the points
    the data has — but the *vertical* wavenumber axis of E_spec is not a
    physical wavenumber axis, and Figure 7's spectral claim cannot be made
    along z.

    Recommendation: download `rayleigh_benard_uniform` and point `data_path`
    at it.  If you use the Chebyshev version, set
    `spectral_axes_note: chebyshev_z` in the config and say so in the caption.
    This module does not silently resample: an interpolation inserted below the
    normalisation statistics would change every number in the paper and leave
    no trace in the results JSON.

2.  **200 frames is the binding constraint.**  With K = 4 history frames and
    all horizons sharing one anchor set (§8), the longest horizon sets how many
    anchors exist:

        T = 200, K = 4         anchors/traj = len(range(3, 199 - h, stride))
          h_max = 128 stride=1  ->   68
          h_max =  64 stride=1  ->  132
          h_max =  32 stride=1  ->  164

    Unlike Lifted H2 this is not fatal, because there are hundreds of
    trajectories rather than one, so 68 x n_test is a real sample.  But
    h > 128 does not exist here and must not appear in `eval_horizons`.
    Defaults cap training at 64 and evaluation at 128, which is the same
    ceiling Lifted H2 uses and therefore keeps the two "short-record" datasets
    directly comparable.

3.  **512 x 128 is cropped to 128 x 128 by default** (`spatial_stride: [4, 1]`,
    i.e. every 4th point along the periodic horizontal axis, native
    resolution vertically).  Two reasons, one good and one merely practical:
    it keeps the grid identical to Gray-Scott and RealPDEBench so *resolution*
    is not a second uncontrolled difference between datasets, and it keeps the
    FNO's cost comparable.  Striding a uniform periodic axis by an exact
    divisor preserves periodicity, so nothing is broken by it; it is a
    low-pass filter, and if the paper claims anything about small horizontal
    scales it must be re-run at native resolution
    (`--set data.spatial_stride=1 --set data.spatial_crop=null`, with
    `modes2` raised accordingly).


BOUNDARY CONDITIONS
-------------------
Periodic in x, no-slip walls in z.  `models/fno.py` pads both axes with the
same width and mode, so there is no way to say "periodic in x, replicate in z".
The default keeps `padding: 8` with replicate, which is correct for z and
mildly wasteful in x.  This is the same compromise the burner geometry already
makes, applied identically to all three models, so it cannot confound the
AR-versus-direct comparison.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .realpde import build_data as _build_data, build_eval_loader as _build_eval

# The two scalars that vary between files and define the operating point.
REGIME_PARAMS = ("Rayleigh", "Prandtl")

RAYLEIGH_BENARD_DEFAULTS = {
    "backend": "well",
    "file_pattern": "*.hdf5",
    "history_len": 4,
    "param_scalars": REGIME_PARAMS,
    "spatial_crop": [128, 128],
    "spatial_stride": [4, 1],
    "channel_rename": {},        # buoyancy / pressure / velocity_x / velocity_y
    "horizon_sampling": "log_binned",
    "h_max_train": 64,
    "horizon_bins": [[1, 2], [3, 4], [5, 8], [9, 16], [17, 32], [33, 64]],
    "eval_horizons": [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128],
    "val_horizons": [1, 4, 16, 64],
    "eval_stride": 8,
    "hval_stride": 32,
    "split_path": "artifacts/split_rayleigh_benard.json",
    "norm_stats_path": "artifacts/norm_stats_rayleigh_benard.json",
    "samples_per_epoch": 20000,
    "val_samples": 2000,
    "modes1": 16,
    "modes2": 16,
    "padding": 8,
    "padding_mode": "replicate",
}


def apply_defaults(cfg: dict) -> dict:
    out = dict(RAYLEIGH_BENARD_DEFAULTS)
    out.update(cfg)
    return out


def strata(store) -> List[str]:
    """Stratify the split on the (Rayleigh, Prandtl) operating point.

    Rayleigh spans orders of magnitude, so it is bucketed on log10 before being
    used as a label; otherwise every file is its own stratum and stratification
    degenerates into a shuffle.
    """
    import numpy as np
    labels = []
    for t in range(store.n_traj):
        p = store.traj_params(t)
        ra = p.get("Rayleigh")
        pr = p.get("Prandtl")
        if ra is None or pr is None:
            labels.append(store.file_of(t))
            continue
        labels.append(f"Ra1e{np.log10(max(ra, 1e-30)):.1f}|Pr{pr:g}")
    return labels


def build_data(cfg: dict, model_kind: str = "direct",
               distributed: bool = False, seed: int = 0) -> Dict:
    return _build_data(apply_defaults(cfg), model_kind, distributed, seed)


def build_eval_loader(cfg: dict, subset: str = "test",
                      horizons: Optional[Sequence[int]] = None,
                      distributed: bool = False):
    return _build_eval(apply_defaults(cfg), subset, horizons, distributed)
