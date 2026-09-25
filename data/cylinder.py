"""
RealPDEBench Cylinder (numerical branch).  Experiment A2.

(The module the request called `cylinder_dataset.py`.)

    92 trajectories, 3990 frames, dt = 2.5e-3 s (400 Hz PIV for 20 s)
    64 x 128, channels u, v, p
    Reynolds 1800-12000, cylinder diameter D = 30 mm

Its job in the ladder is to **isolate nonlinear fluid transport**: a canonical
wake with vortex shedding, no chemistry, no buoyancy.  It is also the single
most valuable addition to the paper for a reason that has nothing to do with
generality, and everything to do with §8 of the handover:

    "What is actually missing is a chaotic flow sampled finely enough that
     h = 32-128 sits inside the predictable window."

Both current datasets fail that test in opposite ways.  RealPDEBench
combustion has a predictability horizon of 4-6 frames, so h >= 8 is already in
the climatology-dominated regime, and Lifted H2's entire 181-frame record is
shorter than that horizon.  Cylinder has 3990 frames of a *quasi-periodic*
flow: the shedding period is tens to hundreds of frames, so h = 128 and beyond
plausibly sit inside a window where the field is still predictable.

That is a hypothesis, not a fact, and it is cheap to test before spending any
GPU time.  Run

    python scripts/probe_timescales.py --config configs/cylinder.yaml

first.  It measures the shedding period and the horizon at which a persistence
forecast reaches E = 0.3 and E = 1.0, per trajectory, from data alone.  If the
predictable window really does extend past h ~ 128, this dataset is where the
paper's central claim can be made without the climatology caveat attached to
it — and `eval_horizons` should then be extended to 1024, which 3990 frames
comfortably supports.  If it does not, that is also a result, and it says the
horizon ceiling is a property of the models rather than of combustion.


THE SPLIT IS NOT THE OFFICIAL ONE, ON PURPOSE
---------------------------------------------
RealPDEBench ships `{train,val,test}_index_numerical.json`, which index
(sim_id, time_id) *windows*.  All 92 trajectories appear in train; 43 of them
also appear in val and in test.  The (sim_id, time_id) pairs are disjoint, so
there is no literal duplicate — but windows starting 20 frames apart in one
trajectory are near-duplicates of each other, which is exactly the leak §9 of
this project exists to prevent, quoted here in full because it was written
before anyone had seen this file:

    "windows (U_100:104 -> U_120) and (U_105:109 -> U_121) come from the same
     trajectory and are almost the same sample.  Splitting windows at random
     puts near-duplicates on both sides of the wall and every model looks
     excellent."

So the primary protocol here is trajectory-held-out, stratified over Reynolds
number, like every other dataset in this study.  `scripts/prepare_split.py`
builds it.  The official window split remains available for a comparability
table (`--split-mode official_windows`) and should be reported as such, never
as the headline: numbers from the two protocols are not comparable, and the
official one is the flattering direction.

Note also that our numbers are not comparable to the RealPDEBench leaderboard
for a second reason: their loader trains at 32 x 64 (`sub_s_numerical = 2`) and
randomly zeroes the pressure channel (`mask_prob = 0.5`).  We do neither.  See
`data/arrow_store.py` for why.


GENERALIZATION AXIS
-------------------
Reynolds number, read from `sim_id` (the file stem "10031.h5" is Re = 10031).
This is the same axis Lifted H2 uses, which is a small but real convenience:
the two datasets' held-out claims are of the same kind, so the paper does not
have to argue about two different notions of "unseen condition".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .realpde import build_data as _build_data, build_eval_loader as _build_eval

CYLINDER_DEFAULTS = {
    "backend": "arrow",
    "arrow_fields": ["u", "v", "p"],
    # `p` alone does not match data/channels.py's `_PRESSURE` regex, and `u`,
    # `v` do match `_FLOW`. Renaming here is what keeps the §26 per-variable
    # table meaningful; check the grouping the audit prints.
    "all_channel_names": ["u", "v", "pressure"],
    "dt": 2.5e-3,
    "history_len": 4,
    "horizon_sampling": "log_binned",
    "h_max_train": 128,
    "horizon_bins": [[1, 2], [3, 4], [5, 8], [9, 16], [17, 32], [33, 64],
                     [65, 128]],
    "eval_horizons": [1, 2, 4, 8, 16, 32, 64, 128, 160, 192, 256, 384, 512],
    "val_horizons": [1, 4, 16, 64, 128],
    "eval_stride": 100,
    "hval_stride": 400,
    "split_path": "artifacts/split_cylinder.json",
    "norm_stats_path": "artifacts/norm_stats_cylinder.json",
    "n_val_traj": 10,
    "n_test_traj": 10,
    "split_seed": 42,
    "samples_per_epoch": 20000,
    "val_samples": 2000,
    # 64 x 128: Nyquist is 32 and 64. The wake is elongated streamwise, so the
    # long axis gets more modes -- the same reasoning as Lifted H2's 16/20.
    "modes1": 16,
    "modes2": 32,
    "padding": 8,
    "padding_mode": "replicate",   # inlet / outlet / walls, non-periodic
    "climatology_stride": 20,
}


def apply_defaults(cfg: dict) -> dict:
    out = dict(CYLINDER_DEFAULTS)
    out.update(cfg)
    return out


def reynolds(store, traj: int) -> Optional[float]:
    return store.sim_param(traj)


def strata(store, n_bins: int = 5) -> List[str]:
    """Reynolds-number bins, as equal-count as the sample allows.

    Quantile bins rather than equal-width bins: Re is not uniformly sampled
    over 1800-12000, and equal-width bins would leave the extremes with one
    trajectory each, which `stratified_split` would then be forced to give
    entirely to training.
    """
    import numpy as np
    re = np.array([store.sim_param(t) or np.nan for t in range(store.n_traj)],
                  dtype=float)
    if np.isnan(re).any():
        raise ValueError(
            "some sim_ids do not parse as a Reynolds number; adjust "
            "`data.sim_id_regex` or stratify on something else. "
            f"Example ids: {[store.sim_id(t) for t in range(min(5, store.n_traj))]}")
    edges = np.quantile(re, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1.0
    idx = np.clip(np.searchsorted(edges, re, side="left") - 1, 0, n_bins - 1)
    return [f"Re_bin{int(i)}" for i in idx]


def build_data(cfg: dict, model_kind: str = "direct",
               distributed: bool = False, seed: int = 0) -> Dict:
    return _build_data(apply_defaults(cfg), model_kind, distributed, seed)


def build_eval_loader(cfg: dict, subset: str = "test",
                      horizons: Optional[Sequence[int]] = None,
                      distributed: bool = False):
    return _build_eval(apply_defaults(cfg), subset, horizons, distributed)
