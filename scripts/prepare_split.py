#!/usr/bin/env python
"""
Freeze the trajectory split for a dataset — run this BEFORE the Phase 0 audit.

Why this exists as a separate script
------------------------------------
`load_or_create_split` creates a split on first call if the JSON is absent,
falling back to a seeded shuffle when no strata are supplied.  On RealPDEBench
combustion that fallback was acceptable because `derive_strata.py` supplied the
fuel-composition labels.  On the three new datasets it is not, and for three
different reasons:

  * The Well ships its **own** train / valid / test directory split.  Ignoring
    it and reshuffling makes our numbers incomparable with every published
    result on those datasets, for no gain.
  * Gray-Scott's six (f, k) settings are six qualitatively different flows.  A
    seeded shuffle can hand a held-out set a pattern type the model barely
    trained on -- the same failure `stratified_split`'s docstring records for
    the 9/21 fuel split, where population 30% pure-CH4 became validation 60%.
  * Cylinder's generalization axis is Reynolds number, which is a continuous
    coordinate; the split has to cover its range on both sides or "held-out
    Re" means "held-out end of the range".

Freezing the split first also removes a whole class of accident: the audit's
`--force` recomputes statistics but never redraws the split (that needs
`--redraw-split`), so once this script has run, statistics and split can no
longer drift apart.

Modes
-----
  official           The Well's own file-level train/valid/test assignment,
                     subsampled to a workable number of held-out trajectories.
                     Default for gray_scott and rayleigh_benard.
  stratified         Trajectory-held-out, proportionally stratified over the
                     dataset's own labels (Re bins, pattern type, Ra/Pr).
                     Default for cylinder.
  param_holdout      Hold out one or more whole parameter settings as test.
                     A strictly harder claim -- unseen dynamics, not unseen
                     initial condition -- and a separate experiment, never a
                     silent substitute for the headline.
  within_trajectory  Same trajectories, disjoint time ranges.  The project's
                     secondary protocol (§9): reported for comparability with
                     benchmark numbers, never used to select a model.

Usage
-----
    python scripts/prepare_split.py --config configs/gray_scott.yaml
    python scripts/prepare_split.py --config configs/cylinder.yaml --mode stratified
    python scripts/prepare_split.py --config configs/gray_scott.yaml \
        --mode param_holdout --holdout spirals worms --tag gs_paramholdout
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from common import base_parser, resolve, json_default          # noqa: E402

from data import apply_dataset_defaults, DATASET_REGISTRY       # noqa: E402
from data.store import build_store                             # noqa: E402
from data.splits import Split, stratified_split, within_trajectory_split  # noqa: E402


DEFAULT_MODE = {
    "gray_scott": "official",
    "rayleigh_benard": "official",
    "cylinder": "stratified",
    "lifted_h2": "stratified",
    "realpde_combustion": "stratified",
}


# ---------------------------------------------------------------------------

def dataset_strata(dataset_name: str, store) -> Optional[List[str]]:
    """Per-trajectory stratification labels, from the dataset module."""
    mod = DATASET_REGISTRY.get(dataset_name)
    fn = getattr(mod, "strata", None)
    if fn is None:
        return None
    return list(fn(store))


def _balanced_take(pool: Sequence[int], labels: Sequence[str], n: int,
                   rng: np.random.Generator) -> List[int]:
    """Take `n` from `pool`, round-robin over labels, shuffled inside each.

    Round-robin is the right rule *here* and the wrong rule inside
    `stratified_split` -- the difference is what is being balanced.  There we
    are apportioning a fixed population and proportional allocation is what
    keeps the held-out sets representative.  Here we are subsampling an
    already-assigned official split purely to make evaluation affordable, and
    what we want is coverage: every pattern type present in the test set even
    if one of them is rare.
    """
    by: Dict[str, List[int]] = defaultdict(list)
    for i in pool:
        by[labels[i]].append(i)
    for k in by:
        by[k] = rng.permutation(by[k]).tolist()
    keys = sorted(by)
    out: List[int] = []
    while len(out) < n and any(by[k] for k in keys):
        for k in keys:
            if by[k] and len(out) < n:
                out.append(by[k].pop())
    return sorted(out)


def split_official(store, labels, n_val, n_test, max_train, seed) -> Split:
    off = store.official_split
    if set(off) == {"unknown"}:
        raise ValueError(
            "the store found no train/valid/test directories, so there is no "
            "official split to honour. Use --mode stratified.")
    rng = np.random.default_rng(seed)
    pools = {s: [i for i, o in enumerate(off) if o == s]
             for s in ("train", "val", "test")}
    for s in ("val", "test"):
        if not pools[s]:
            raise ValueError(f"official split has no '{s}' trajectories")

    val = _balanced_take(pools["val"], labels, n_val, rng)
    test = _balanced_take(pools["test"], labels, n_test, rng)
    train = pools["train"]
    if max_train and len(train) > max_train:
        train = _balanced_take(train, labels, max_train, rng)

    sp = Split(train=sorted(train), val=val, test=test, seed=seed,
               note=(f"The Well official file-level split; val/test subsampled "
                     f"to {n_val}/{n_test} balanced over "
                     f"{len(set(labels))} parameter groups"
                     + (f"; train capped at {max_train}" if max_train else "")))
    sp.check_disjoint()
    return sp


def split_param_holdout(store, labels, holdout, n_val, seed) -> Split:
    groups = sorted(set(labels))
    unknown = [h for h in holdout if h not in groups]
    if unknown:
        raise ValueError(f"--holdout {unknown} not among the dataset's "
                         f"parameter groups {groups}")
    test = [i for i, l in enumerate(labels) if l in holdout]
    rest = [i for i, l in enumerate(labels) if l not in holdout]
    if not rest:
        raise ValueError("--holdout consumed every parameter group")
    rng = np.random.default_rng(seed)
    val = _balanced_take(rest, labels, n_val, rng)
    train = sorted(set(rest) - set(val))
    sp = Split(train=train, val=val, test=sorted(test), seed=seed,
               note=(f"parameter holdout: test = {sorted(holdout)}, entirely "
                     f"unseen dynamics. NOT comparable with the headline "
                     f"split, which holds out initial conditions only."))
    sp.check_disjoint()
    return sp


# ---------------------------------------------------------------------------

def main():
    ap = base_parser("Freeze the trajectory split (run before the audit)")
    ap.add_argument("--mode", default=None,
                    choices=["official", "stratified", "param_holdout",
                             "within_trajectory"])
    ap.add_argument("--n_val", type=int, default=None)
    ap.add_argument("--n_test", type=int, default=None)
    ap.add_argument("--max_train", type=int, default=None,
                    help="cap the number of training trajectories; useful on "
                         "Gray-Scott, where 1200 realizations is far more "
                         "than samples_per_epoch will ever touch")
    ap.add_argument("--holdout", nargs="+", default=None,
                    help="param_holdout mode: parameter group name(s) to "
                         "reserve entirely for test")
    ap.add_argument("--train_frac", type=float, default=0.7,
                    help="within_trajectory mode only")
    ap.add_argument("--val_frac", type=float, default=0.15,
                    help="within_trajectory mode only")
    ap.add_argument("--out", default=None,
                    help="override data.split_path")
    ap.add_argument("--force", action="store_true",
                    help="DESTRUCTIVE. Overwrite an existing split. Every "
                         "result produced with the previous split becomes "
                         "incomparable, and the frozen normalisation "
                         "statistics become a leak, so the audit must then be "
                         "re-run with --force.")
    args = ap.parse_args()

    cfg = resolve(args)
    dataset_name = cfg.get("dataset_name", "realpde_combustion")
    cfg = apply_dataset_defaults(dataset_name, cfg)
    mode = args.mode or DEFAULT_MODE.get(dataset_name, "stratified")
    path = args.out or cfg.get("split_path")
    if not path:
        raise ValueError("no data.split_path in the config and no --out")

    print("=" * 72)
    print(f"PREPARE SPLIT — {dataset_name}  |  mode={mode}")
    print("=" * 72)

    if os.path.exists(path) and not args.force:
        sp = Split.load(path)
        print(f"\n{path} already exists; refusing to overwrite.\n"
              f"  {sp.summary()}\n"
              f"  {sp.note}\n"
              f"Pass --force only if you are prepared to re-run the audit with "
              f"--force as well; otherwise the frozen normalisation statistics "
              f"were fitted on trajectories that are about to become test.")
        return

    store = build_store(cfg)
    print(f"\n[1] Store\n    {store.summary()}")

    labels = dataset_strata(dataset_name, store)
    if labels is None:
        labels = ["all"] * store.n_traj
        print("\n[2] No strata function for this dataset; every trajectory is "
              "one group.")
    else:
        c = Counter(labels)
        print(f"\n[2] Stratification labels ({len(c)} groups)")
        for k in sorted(c):
            print(f"    {k:<28} {c[k]}")

    n_val = args.n_val if args.n_val is not None else int(cfg.get("n_val_traj", 5))
    n_test = args.n_test if args.n_test is not None else int(cfg.get("n_test_traj", 5))
    seed = int(cfg.get("split_seed", 42))

    if mode == "official":
        sp = split_official(store, labels, n_val, n_test, args.max_train, seed)
    elif mode == "stratified":
        sp = stratified_split(store.n_traj, n_val=n_val, n_test=n_test,
                              seed=seed, strata=labels)
    elif mode == "param_holdout":
        if not args.holdout:
            raise ValueError("--mode param_holdout needs --holdout GROUP [...]")
        sp = split_param_holdout(store, labels, args.holdout, n_val, seed)
    elif mode == "within_trajectory":
        sp = within_trajectory_split(store.n_traj, store.T,
                                     train_frac=args.train_frac,
                                     val_frac=args.val_frac)
    else:
        raise ValueError(mode)

    print(f"\n[3] Split\n    {sp.summary()}")
    print(f"    {sp.note}")
    for subset, ts in (("train", sp.train), ("val", sp.val), ("test", sp.test)):
        c = Counter(labels[t] for t in ts)
        print(f"    {subset:<6} n={len(ts):<5} {dict(c)}")
    missing = set(labels) - {labels[t] for t in sp.test}
    if missing and mode != "param_holdout":
        print(f"    [!] groups absent from TEST: {sorted(missing)}")
        print( "        the test set does not measure those operating points")

    sp.save(path)

    # Manifest: what each split index actually refers to on disk. Without it a
    # split JSON is a list of integers whose meaning depends on the glob order
    # of a directory, which is not a reproducible artefact.
    man = {
        "dataset": dataset_name, "mode": mode, "split_path": path,
        "n_traj": store.n_traj, "T": store.T, "H": store.H, "W": store.W,
        "dt": store.dt, "channel_names": list(store.channel_names),
        "labels": labels,
        "trajectories": [
            {"index": t, "label": labels[t],
             **({"file": store.file_of(t),
                 "realization": store.realization_of(t),
                 "official_split": store.official_split[t]}
                if hasattr(store, "file_of") else {}),
             **({"sim_id": store.sim_id(t)} if hasattr(store, "sim_id") else {}),
             }
            for t in range(store.n_traj)],
    }
    man_path = os.path.splitext(path)[0] + "_manifest.json"
    with open(man_path, "w") as fp:
        json.dump(man, fp, indent=2, default=json_default)

    # Also emit the labels in the shape scripts/derive_strata.py produces, so
    # the audit can be given --strata and print the real per-subset breakdown.
    # Without it the audit prints "[!] Unstratified ... run derive_strata.py",
    # which is a false alarm here -- the split on disk IS stratified, the audit
    # just has no way to know that. A warning that is wrong in the normal case
    # is a warning people learn to ignore.
    strata_path = os.path.splitext(path)[0] + "_strata.json"
    with open(strata_path, "w") as fp:
        json.dump({"strata": labels, "dataset": dataset_name,
                   "source": "scripts/prepare_split.py", "mode": mode},
                  fp, indent=2, default=json_default)

    print(f"\n[4] Written\n    {path}\n    {man_path}\n    {strata_path}")
    print("\nNext:\n"
          f"    python scripts/audit_data.py --config {args.config} "
          f"--strata {strata_path}\n"
          "Read the transform table and the channel GROUPING it prints before "
          "training. A channel in the wrong group silently disables the §26 "
          "per-variable reporting.")


if __name__ == "__main__":
    main()
