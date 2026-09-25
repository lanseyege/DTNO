"""Data pipeline: stores, transforms, splits, horizon samplers, datasets.

Five datasets now, arranged as a complexity ladder rather than a list.  The
ordering is the argument: each rung adds exactly one physical mechanism, so a
change in the direct-time-versus-autoregressive result can be attributed to
that mechanism instead of to "a different dataset".

    Experiment A — general PDE / CFD dynamics
      gray_scott          reaction, no advection      1001 frames, 128x128, 2 ch
      cylinder            advection / wake            3990 frames,  64x128, 3 ch
      rayleigh_benard     buoyancy-driven transport    200 frames, 512x128, 4 ch

    Experiment B — reacting flows
      lifted_h2           transport + chemistry        181 frames, 160x200, 5 ch
      realpde_combustion  strongly coupled multiscale 2001 frames, 128x128, 13 ch

Every entry exposes the same two functions and differs only in defaults; the
architecture, losses, protocol and evaluation are identical across all five
(§41).  `realm_ignithit` is kept registered but unused — its 30-50 frame
trajectories cannot support a horizon axis.
"""

from . import realpde, realm, lifted_h2, gray_scott, rayleigh_benard, cylinder
from .store import build_store, TrajectoryStore, ZarrStore, NPYStore
from .transforms import Normalizer, ChannelStat
from .splits import Split, load_or_create_split, stratified_split
from .sampling import (build_horizon_sampler, split_horizon, EVAL_HORIZONS,
                       EXPB_TRAIN, EXPB_INTERP, EXPB_EXTRA)
from .datasets import DirectPairDataset, ARWindowDataset, EvalAnchorDataset
from .channels import group_channels, channel_group, reaction_zone_channel

DATASET_REGISTRY = {
    # Experiment A
    "gray_scott": gray_scott,
    "cylinder": cylinder,
    "rayleigh_benard": rayleigh_benard,
    # Experiment B
    "lifted_h2": lifted_h2,
    "realpde_combustion": realpde,
    # retained, not used: see docs/HANDOVER.md §8
    "realm_ignithit": realm,
}

# Which experiment a dataset belongs to. `scripts/make_figures.py` and the
# results tables read this so that the A/B grouping lives in one place instead
# of being re-typed into every plotting command.
EXPERIMENT_GROUP = {
    "gray_scott": "A",
    "cylinder": "A",
    "rayleigh_benard": "A",
    "lifted_h2": "B",
    "realpde_combustion": "B",
}

# Where each dataset sits on the ladder, for table and figure ordering.
LADDER_ORDER = ["gray_scott", "cylinder", "rayleigh_benard",
                "lifted_h2", "realpde_combustion"]


def _module(name: str):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"unknown dataset '{name}'. "
                         f"Available: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name]


def build_data(dataset_name: str, cfg: dict, model_kind: str = "direct",
               distributed: bool = False, seed: int = 0):
    return _module(dataset_name).build_data(cfg, model_kind, distributed, seed)


def build_eval_loader(dataset_name: str, cfg: dict, subset: str = "test",
                      horizons=None, distributed: bool = False):
    return _module(dataset_name).build_eval_loader(cfg, subset, horizons,
                                                   distributed)


def apply_dataset_defaults(dataset_name: str, cfg: dict) -> dict:
    """Flat config with the dataset's defaults filled in.

    Scripts that build a store directly (the audit, the climatology baselines,
    prepare_split, the screens) need the same defaults `build_data` applies,
    or they will read a different store from the one training reads -- a
    different spatial crop, a different frame stride, a different dt.
    """
    mod = _module(dataset_name)
    fn = getattr(mod, "apply_defaults", None)
    return fn(cfg) if fn is not None else dict(cfg)


def experiment_group(dataset_name: str) -> str:
    return EXPERIMENT_GROUP.get(dataset_name, "?")


def list_datasets():
    return list(DATASET_REGISTRY)
