"""
RealPDEBench combustion (numerical branch) — the MVP's primary dataset (§4).

30 numerical trajectories, 2001 frames each at dt = 2.5e-4 s, 128 x 128, 15
reacting-flow channels.  What makes it the right first dataset is not the
physics but the time axis: 2001 frames per trajectory is what lets us sample a
rich (t0, dT, t0 + dT) product, and arbitrary query time is the whole
hypothesis.

This module is the wiring: store -> frozen split -> frozen normalizer ->
datasets -> loaders, plus a `data_info` dict that every downstream component
reads instead of re-deriving shapes.  It is generic over the backend, so
`data/realm.py` reuses all of it and only changes defaults.

Config keys (all under `data:` in the YAML)
    backend, data_path, dt, channels, history_len
    split_path, norm_stats_path, n_val_traj, n_test_traj, split_seed
    horizon_sampling, h_max_train, horizon_bins, train_horizons
    samples_per_epoch, val_samples, eval_horizons, eval_stride
    batch_size, batch_size_eval, num_workers
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from .store import build_store, TrajectoryStore
from .transforms import Normalizer
from .splits import Split, load_or_create_split
from .sampling import build_horizon_sampler, EVAL_HORIZONS
from .datasets import DirectPairDataset, ARWindowDataset, EvalAnchorDataset
from .channels import group_channels


DEFAULT_CHANNELS_15 = None      # None = use every channel the store exposes


def resolve_channels(store: TrajectoryStore,
                     requested: Optional[Sequence]) -> Tuple[List[int], List[str]]:
    """Map names and/or integer indices onto store channel indices."""
    names = store.channel_names
    if requested is None:
        return list(range(len(names))), list(names)
    idx, sel = [], []
    for c in requested:
        if isinstance(c, int):
            if not 0 <= c < len(names):
                raise ValueError(f"channel index {c} outside 0..{len(names)-1}")
            idx.append(c)
        else:
            if c not in names:
                raise ValueError(f"channel '{c}' not in store. Available: {names}")
            idx.append(names.index(c))
        sel.append(names[idx[-1]])
    return idx, sel


def _worker_init(worker_id: int):
    """Pin every DataLoader worker to a single compute thread.

    Without this the loader saturates the machine and starves the GPU. Two
    libraries are the culprits and both default to "use all cores":

      * numcodecs/blosc spawns its own decompression thread pool per process,
        so 12 workers x 8 blosc threads = 96 threads fighting over the same
        cores. Observed in practice as ~400% CPU per pt_data_worker with the
        GPUs mostly idle.
      * torch's intra-op pool does the same for the normalisation arithmetic.

    Workers are already the unit of parallelism -- one thread each is what makes
    them scale. The env vars are set too because BLAS pools are usually created
    at import time, before this hook runs, in libraries the workers import
    lazily.
    """
    import os
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "BLOSC_NTHREADS"):
        os.environ[var] = "1"
    try:
        import torch as _t
        _t.set_num_threads(1)
    except Exception:
        pass
    for mod in ("numcodecs.blosc", "blosc"):
        try:
            __import__(mod)
            import sys
            sys.modules[mod].set_nthreads(1)
        except Exception:
            pass


def _t_scale(cfg: dict, sampler, store) -> float:
    """tau = h * dt / t_scale.

    Default puts the longest TRAINING horizon at tau = 1, so horizon
    extrapolation reads as tau > 1 and the time embedding is never silently
    rescaled between datasets. `t_scale: null` in the YAML means "use the
    default", which is not the same as "absent".
    """
    v = cfg.get("t_scale")
    return float(v) if v else float(sampler.h_max * store.dt)


def build_data(cfg: dict, model_kind: str = "direct",
               distributed: bool = False, seed: int = 0) -> Dict:
    """Build train/val loaders + data_info.

    model_kind: 'direct' (DT-FNO / SG-DT-FNO) or 'ar' (AR-FNO-1 / AR-FNO-R).
    """
    store = build_store(cfg)
    ch_idx, ch_names = resolve_channels(store, cfg.get("channels"))

    # ---- frozen split -------------------------------------------------
    split_path = cfg.get("split_path", "artifacts/split_realpde.json")
    split = load_or_create_split(split_path, store.n_traj, cfg)
    split.check_disjoint()

    # ---- frozen normalisation (training trajectories only, §11.1) -----
    stats_path = cfg.get("norm_stats_path", "artifacts/norm_stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"{stats_path} not found. Run Phase 0 first:\n"
            f"    python scripts/audit_data.py --config <your config>\n"
            f"Statistics must be computed on training trajectories only; "
            f"deriving them here would leak the test set.")
    norm = Normalizer.load(stats_path, ch_names)

    # Statistics are fitted on the training trajectories of a particular split.
    # If the split on disk has since changed, those statistics were fitted on
    # data that is now partly test -- a leak, and a silent one, because nothing
    # else would ever notice.
    fp = Normalizer.load_meta(stats_path).get("split_fingerprint")
    if fp and list(fp.get("train", [])) != list(split.train):
        raise RuntimeError(
            f"{stats_path} was fitted on train={fp['train']}\n"
            f"but {split_path} now says train={split.train}.\n"
            f"The split changed after the statistics were frozen, so these "
            f"statistics leak held-out data. Recompute with:\n"
            f"    python scripts/audit_data.py --config <config> --force"
            + (f" --strata artifacts/strata.json" if True else ""))

    K = int(cfg.get("history_len", 4))
    t_range = None
    if split.protocol == "within_trajectory" and split.time_cut:
        t_range = split.time_cut

    sampler = build_horizon_sampler(cfg)
    t_scale = _t_scale(cfg, sampler, store)

    common = dict(store=store, channel_indices=ch_idx, normalizer=norm,
                  history_len=K)

    if model_kind == "direct":
        train_ds = DirectPairDataset(
            traj_indices=split.train, horizon_sampler=sampler,
            samples_per_epoch=int(cfg.get("samples_per_epoch", 20000)),
            t_range=(t_range or {}).get("train"), t_scale=t_scale,
            base_seed=seed, **common)
        val_ds = DirectPairDataset(
            traj_indices=split.val, horizon_sampler=sampler,
            samples_per_epoch=int(cfg.get("val_samples", 2000)),
            t_range=(t_range or {}).get("val"), t_scale=t_scale,
            base_seed=seed + 9999, deterministic=True, **common)
    elif model_kind == "ar":
        R = int(cfg.get("ar_rollout", 1))
        train_ds = ARWindowDataset(
            traj_indices=split.train, rollout=R,
            random_rollout=bool(cfg.get("ar_random_rollout", R > 1)),
            samples_per_epoch=int(cfg.get("samples_per_epoch", 20000)),
            t_range=(t_range or {}).get("train"), base_seed=seed, **common)
        val_ds = ARWindowDataset(
            traj_indices=split.val, rollout=R,
            random_rollout=False,
            samples_per_epoch=int(cfg.get("val_samples", 2000)),
            t_range=(t_range or {}).get("val"), base_seed=seed + 9999,
            deterministic=True, **common)
    else:
        raise ValueError(f"model_kind must be 'direct' or 'ar', got {model_kind}")

    train_sampler = val_sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)

    # Three loaders exist per rank -- train, val, and the §24 horizon
    # validation that scripts/train.py builds through build_eval_loader -- and
    # each forks its own workers. `num_workers: 6` on 4 GPUs is therefore
    # 4 * (6 + 3) = 36 persistent worker processes, plus 4 * 6 = 24 more
    # whenever the horizon validation runs. `num_workers_val` and
    # `num_workers_eval` let the three be sized separately; omitting them
    # reproduces the previous behaviour exactly.
    #
    # Worker counts cannot change a number: every sample is seeded from
    # (base_seed, epoch, index), never from the worker id, so this is a pure
    # throughput knob and results are bit-identical across settings.
    nw = int(cfg.get("num_workers", 4))
    # `max(1, nw // 2)` used to fork one un-pinned validation worker even at
    # nw = 0, so `num_workers: 0` never gave a genuinely in-process run --
    # which is exactly when you want one, because a traceback from inside a
    # worker is much harder to read.
    nw_val = int(cfg.get("num_workers_val", 0 if nw == 0 else max(1, nw // 2)))
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg.get("batch_size", 8)),
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=nw > 0,
        prefetch_factor=int(cfg.get("prefetch_factor", 4)) if nw > 0 else None,
        worker_init_fn=_worker_init if nw > 0 else None)
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg.get("batch_size_eval", 8)),
        shuffle=False, sampler=val_sampler,
        num_workers=nw_val, pin_memory=True,
        persistent_workers=nw_val > 0,
        worker_init_fn=_worker_init if nw_val > 0 else None)

    data_info = {
        "H": store.H, "W": store.W, "C": len(ch_idx), "K": K,
        "dt": store.dt, "t_scale": t_scale,
        "channel_names": ch_names,
        "channel_indices": ch_idx,
        "channel_groups": group_channels(ch_names),
        "normalizer": norm,
        "split": split,
        "store_summary": store.summary(),
        "horizon_sampler": sampler.describe(),
        "h_max_train": sampler.h_max,
    }
    return {"train_loader": train_loader, "val_loader": val_loader,
            "data_info": data_info}


def build_eval_loader(cfg: dict, subset: str = "test",
                      horizons: Optional[Sequence[int]] = None,
                      distributed: bool = False) -> Tuple[DataLoader, Dict]:
    """Deterministic anchor loader for `scripts/evaluate_horizon.py`."""
    store = build_store(cfg)
    ch_idx, ch_names = resolve_channels(store, cfg.get("channels"))
    split = Split.load(cfg.get("split_path", "artifacts/split_realpde.json"))
    norm = Normalizer.load(cfg.get("norm_stats_path", "artifacts/norm_stats.json"),
                           ch_names)

    traj = {"train": split.train, "val": split.val, "test": split.test}[subset]
    horizons = list(horizons or cfg.get("eval_horizons", EVAL_HORIZONS))
    K = int(cfg.get("history_len", 4))
    sampler = build_horizon_sampler(cfg)
    t_scale = _t_scale(cfg, sampler, store)

    t_range = None
    if split.protocol == "within_trajectory" and split.time_cut:
        t_range = split.time_cut.get(subset)

    ds = EvalAnchorDataset(
        store=store, traj_indices=traj, channel_indices=ch_idx, normalizer=norm,
        horizons=horizons, history_len=K,
        stride=int(cfg.get("eval_stride", 50)), t_range=t_range,
        t_scale=t_scale,
        max_anchors_per_traj=cfg.get("max_anchors_per_traj"))

    sampler_ddp = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler_ddp = DistributedSampler(ds, shuffle=False, drop_last=False)

    nw_eval = int(cfg.get("num_workers_eval", cfg.get("num_workers", 4)))
    loader = DataLoader(ds, batch_size=int(cfg.get("batch_size_eval", 4)),
                        shuffle=False, sampler=sampler_ddp,
                        num_workers=nw_eval, pin_memory=True,
                        worker_init_fn=_worker_init if nw_eval > 0 else None)
    info = {
        "H": store.H, "W": store.W, "C": len(ch_idx), "K": K,
        "dt": store.dt, "t_scale": t_scale,
        "channel_names": ch_names, "channel_groups": group_channels(ch_names),
        "normalizer": norm, "horizons": horizons,
        "n_anchors": len(ds), "subset": subset,
        "traj_indices": traj,
    }
    return loader, info
