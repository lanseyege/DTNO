#!/usr/bin/env python
"""
Apply the three small edits the new datasets need to files that already exist.

Everything else in this extension is additive -- new modules, new configs, new
scripts.  Three existing files have to change, and only three:

  data/store.py      `build_store` learns two backends: 'well' and 'arrow'.
  data/channels.py   two regexes learn three names: a bare `p`, and `buoyancy`.
  data/__init__.py   the registry gains three datasets.  Shipped whole; this
                     script only checks it.

The script is idempotent and verifies before it writes: if an anchor is
missing it says which one and changes nothing, and if the edit is already
present it reports that and moves on.  Run it from the repository root.

    python scripts/apply_patches.py --repo . --dry-run
    python scripts/apply_patches.py --repo .

Then run the smoke test from §2 of the handover before anything else:

    python scripts/make_synthetic_data.py --out /tmp/smoke_data
    python scripts/audit_data.py       --config configs/smoke.yaml --force
    python scripts/train.py            --config configs/smoke.yaml
    python scripts/evaluate_horizon.py --config configs/smoke.yaml \
        --checkpoint checkpoints/smoke/best_model.pth

It exercises every path including DDP and takes about two minutes on CPU.  The
edits below touch the store factory, which is on the path of every dataset in
the project, so "it works on Gray-Scott" is not evidence that RealPDEBench
still works.
"""

from __future__ import annotations

import argparse
import os
import sys

# --------------------------------------------------------------------------
# 1. data/store.py — two new backends
# --------------------------------------------------------------------------

STORE_OLD = '''def build_store(cfg: dict) -> TrajectoryStore:
    """cfg keys: backend ('zarr'|'npy'), data_path, dt, time_stride, channel_names?"""
    backend = str(cfg.get("backend", "zarr")).lower()
    path = cfg["data_path"]
    dt = float(cfg.get("dt", 2.5e-4))
    names = cfg.get("all_channel_names", None)
    if backend == "zarr":
        store = ZarrStore(path, dt=dt, channel_names=names)
    elif backend in ("npy", "numpy"):
        store = NPYStore(path, dt=dt, channel_names=names,
                         pattern=cfg.get("file_pattern", "*.npy"))
    else:
        raise ValueError(f"unknown backend '{backend}' (expected zarr | npy)")

    stride = int(cfg.get("time_stride", 1))
    return store if stride == 1 else StridedStore(store, stride)'''

STORE_NEW = '''def build_store(cfg: dict) -> TrajectoryStore:
    """cfg: backend ('zarr'|'npy'|'well'|'arrow'), data_path, dt, time_stride, ...

    'well'   The Well HDF5 (gray_scott, rayleigh_benard) -- data/well.py
    'arrow'  RealPDEBench HF/Arrow (cylinder)            -- data/arrow_store.py

    Both are imported lazily, so h5py and pyarrow stay optional dependencies of
    the people who actually read those formats.
    """
    backend = str(cfg.get("backend", "zarr")).lower()
    path = cfg["data_path"]
    # `dt: null` is legal for backends that can read the time axis off the file
    # (The Well carries /dimensions/time, RealPDEBench Arrow carries a `t`
    # column), so dt is resolved per backend rather than eagerly: float(None)
    # would raise here and the failure would look like a config typo.
    _dt = cfg.get("dt", 2.5e-4)
    dt = float(_dt) if _dt is not None else None
    names = cfg.get("all_channel_names", None)
    if backend == "zarr":
        store = ZarrStore(path, dt=dt if dt is not None else 2.5e-4,
                          channel_names=names)
    elif backend in ("npy", "numpy"):
        store = NPYStore(path, dt=dt if dt is not None else 1.0,
                         channel_names=names,
                         pattern=cfg.get("file_pattern", "*.npy"))
    elif backend == "well":
        from .well import WellStore
        store = WellStore.from_config(cfg)
    elif backend in ("arrow", "hf_arrow"):
        from .arrow_store import ArrowTrajectoryStore
        store = ArrowTrajectoryStore.from_config(cfg)
    else:
        raise ValueError(f"unknown backend '{backend}' "
                         f"(expected zarr | npy | well | arrow)")

    stride = int(cfg.get("time_stride", 1))
    return store if stride == 1 else StridedStore(store, stride)'''

# --------------------------------------------------------------------------
# 2. data/channels.py — three names the existing regexes miss
# --------------------------------------------------------------------------
#
# The handover's §9 lesson, verbatim: "Regex channel matching breaks silently
# across datasets."  Two concrete misses on the new datasets:
#
#   `p`         Cylinder spells pressure with one letter. `_PRESSURE` is the
#               literal word "pressure", so `p` fell through every pattern to
#               the conservative `thermo` fallback -- reported, but in the wrong
#               group, which is how the §26 per-variable table stops meaning
#               anything.
#   `buoyancy`  Rayleigh-Benard's temperature-analogue. Same fallback.
#
# Gray-Scott's `A` and `B` are handled the other way, by renaming them
# `concentration_A` / `concentration_B` in the config: `_SPECIES` already
# matches "concentration", and a one-letter channel name is not something a
# regex should be taught to interpret.

CHANNELS_EDITS = [
    ('_PRESSURE = re.compile(r"pressure", re.I)',
     '_PRESSURE = re.compile(r"pressure|^p$|_p$|\\bpres\\b", re.I)'),
    ('_THERMO = re.compile(r"temperature|^t$|density|rho|enthalpy|entropy", re.I)',
     '_THERMO = re.compile(r"temperature|^t$|density|rho|enthalpy|entropy|'
     'buoyancy", re.I)'),
]


# --------------------------------------------------------------------------
# 3. data/realpde.py — size the three DataLoaders separately
# --------------------------------------------------------------------------
#
# `scripts/train.py` builds THREE loaders per rank, not one:
#
#     train_loader   num_workers = nw                 persistent
#     val_loader     num_workers = max(1, nw // 2)    persistent
#     hval_loader    num_workers = nw                 respawned every
#                    (build_eval_loader)              `hval_every` epochs
#
# so `num_workers: 6` with `--nproc_per_node 4` is 4*(6+3) = 36 persistent
# worker processes, rising to 60 whenever the §24 horizon validation runs.
# One key controlling all three means the only way to relieve the val and
# hval loaders is to starve the train loader too.
#
# The defaults below reproduce the current behaviour exactly for nw > 0. The
# one behavioural change is at nw = 0: `max(1, 0)` silently forked one
# un-pinned validation worker, so `num_workers: 0` never actually gave you an
# in-process run -- which is precisely when you want one, because a traceback
# from inside a worker is much harder to read.
#
# None of this can change a number. `DirectPairDataset` and `ARWindowDataset`
# seed every sample from `(base_seed, epoch, index)` and never from the worker
# id, and `EvalAnchorDataset` is deterministic, so worker counts are a pure
# throughput knob and results are bit-identical across them.

REALPDE_OLD = '''    nw = int(cfg.get("num_workers", 4))
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
        num_workers=max(1, nw // 2), pin_memory=True,
        persistent_workers=nw > 0,
        worker_init_fn=_worker_init if nw > 0 else None)'''

REALPDE_NEW = '''    # Three loaders exist per rank -- train, val, and the §24 horizon
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
        worker_init_fn=_worker_init if nw_val > 0 else None)'''

REALPDE_EVAL_OLD = '''    nw_eval = int(cfg.get("num_workers", 4))'''
REALPDE_EVAL_NEW = ('''    nw_eval = int(cfg.get("num_workers_eval", cfg.get("num_workers", 4)))'''
                    )


# --------------------------------------------------------------------------
# 4. evaluation/runner.py — the bounded score must survive NaN, not just 1e20
# --------------------------------------------------------------------------
#
# §24's bounded score exists because a diverged rollout makes the plain mean
# meaningless: one E = 1e20 point drags it to 1e19 and it stops ranking
# anything. `min(v, 1.0)` fixes that. It does not fix NaN -- `min(nan, 1.0)`
# is nan in Python and `np.minimum(nan, 1.0)` is nan in numpy, so one
# non-finite horizon turns the whole summary into nan and the run becomes
# unreportable rather than merely bad.
#
# That is not hypothetical. Gray-Scott AR-FNO-R at seed 1 produced non-finite
# fields at h >= 256 and reported `Eval (§24, bounded) = nan`, while seed 0 of
# the same configuration never diverged at all and seed 2 diverged at six
# horizons with a finite 1.1e2. Losing seed 1 to a nan would have left exactly
# the impression the seeds were run to dispel.
#
# A non-finite field is not a large error, it is a failed forecast, and the
# bounded score's own convention already has a slot for that: 1.0, "no better
# than predicting the training mean". Scoring it there is the least generous
# defensible choice, and the count is preserved separately in
# `nonfinite_horizons` so the failure is reported rather than absorbed.

RUNNER_OLD = '''    E = [float(v) for v in out["E_field"]]
    n = max(len(horizons), 1)
    out["eval_score"] = float(sum(E) / n)
    out["eval_score_bounded"] = float(sum(min(v, 1.0) for v in E) / n)
    out["diverged_horizons"] = [int(h) for h, v in zip(horizons, E) if v > 3.0]'''

RUNNER_NEW = '''    E = [float(v) for v in out["E_field"]]
    n = max(len(horizons), 1)
    out["eval_score"] = float(sum(E) / n)
    # A non-finite E is a FAILED forecast, not a large error, and `min(nan, 1)`
    # is nan -- so without this one horizon turns the whole bounded score into
    # nan and an otherwise reportable run becomes unreportable. Score it at the
    # bound (1.0, "no better than predicting the training mean"), which is the
    # least generous defensible value, and keep the count so the failure is
    # reported rather than absorbed.
    def _bounded(v):
        return 1.0 if not (v == v and abs(v) != float("inf")) else min(v, 1.0)
    out["eval_score_bounded"] = float(sum(_bounded(v) for v in E) / n)
    out["nonfinite_horizons"] = [
        int(h) for h, v in zip(horizons, E)
        if not (v == v and abs(v) != float("inf"))]
    out["diverged_horizons"] = [
        int(h) for h, v in zip(horizons, E)
        if (v == v and abs(v) != float("inf") and v > 3.0)
        or not (v == v and abs(v) != float("inf"))]'''


# --------------------------------------------------------------------------

def patch_file(path: str, edits, dry_run: bool) -> int:
    """-> 0 applied / already applied, 1 anchor missing."""
    if not os.path.exists(path):
        print(f"  [!] {path} does not exist")
        return 1
    with open(path) as fp:
        text = fp.read()
    changed = False
    for old, new in edits:
        if new in text:
            print(f"  ok   already applied: {old.splitlines()[0][:60]}...")
            continue
        if old not in text:
            print(f"  [!] anchor NOT FOUND in {path}:")
            print(f"      {old.splitlines()[0][:100]}")
            print("      The file has changed since this patch was written. "
                  "Apply it by hand;\n      the intended result is in "
                  "docs/INTEGRATION.md.")
            return 1
        if text.count(old) != 1:
            print(f"  [!] anchor appears {text.count(old)} times in {path}; "
                  f"refusing to guess")
            return 1
        text = text.replace(old, new)
        changed = True
        print(f"  edit {old.splitlines()[0][:60]}...")
    if changed and not dry_run:
        with open(path, "w") as fp:
            fp.write(text)
        print(f"  wrote {path}")
    elif changed:
        print(f"  (dry run) would write {path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Patch store.py and channels.py")
    ap.add_argument("--repo", default=".", help="repository root")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rc = 0
    print("data/store.py — build_store learns 'well' and 'arrow'")
    rc |= patch_file(os.path.join(args.repo, "data", "store.py"),
                     [(STORE_OLD, STORE_NEW)], args.dry_run)

    print("\ndata/channels.py — bare 'p' is pressure, 'buoyancy' is thermo")
    rc |= patch_file(os.path.join(args.repo, "data", "channels.py"),
                     CHANNELS_EDITS, args.dry_run)

    print("\ndata/realpde.py — size the train / val / hval loaders separately")
    rc |= patch_file(os.path.join(args.repo, "data", "realpde.py"),
                     [(REALPDE_OLD, REALPDE_NEW),
                      (REALPDE_EVAL_OLD, REALPDE_EVAL_NEW)], args.dry_run)

    print("\nevaluation/runner.py — bounded score survives NaN")
    rc |= patch_file(os.path.join(args.repo, "evaluation", "runner.py"),
                     [(RUNNER_OLD, RUNNER_NEW)], args.dry_run)

    print("\ndata/__init__.py — registry")
    init = os.path.join(args.repo, "data", "__init__.py")
    with open(init) as fp:
        txt = fp.read()
    if "gray_scott" in txt and "cylinder" in txt:
        print("  ok   registry already lists the new datasets")
    else:
        print("  [!] copy the shipped data/__init__.py over this file; it is "
              "a whole-file\n      replacement, not a patch")
        rc |= 1

    print("\n" + ("=" * 60))
    if rc:
        print("INCOMPLETE — see the messages above.")
    else:
        print("Done. Now run the smoke test before anything else:")
        print("  python scripts/make_synthetic_data.py --out /tmp/smoke_data")
        print("  python scripts/audit_data.py       --config configs/smoke.yaml --force")
        print("  python scripts/train.py            --config configs/smoke.yaml")
        print("  python scripts/evaluate_horizon.py --config configs/smoke.yaml \\")
        print("      --checkpoint checkpoints/smoke/best_model.pth")
        print("\nbuild_store is on the path of EVERY dataset, so 'Gray-Scott "
              "works' is not\nevidence that RealPDEBench still does.")
    sys.exit(rc)


if __name__ == "__main__":
    main()
