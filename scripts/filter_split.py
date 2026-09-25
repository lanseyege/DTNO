#!/usr/bin/env python
"""
Write a new split whose TEST list is a subset of an existing one, leaving
`train` and `val` untouched.

WHY THIS IS SAFE, AND WHY IT IS THE RIGHT TOOL HERE
---------------------------------------------------
`data/realpde.build_data` refuses to run when the frozen normalisation
statistics were fitted on a different `train` list -- statistics fitted on
trajectories that are now test are a leak, and a silent one. It compares
`train` only, and correctly so: which trajectories you *score* on changes what
the number means, not whether the number is contaminated.

So a test-only filter needs **no retraining and no re-audit**. The same
checkpoints, the same statistics, a different reporting population. That is
what makes "report dynamic and stationary trajectories separately" cheap enough
to actually do, instead of an argument in the discussion section.

It is a REPORTING split. It is not a way to make a number look better by
dropping the trajectories a model does badly on: use it to split the test set
along a property of the DATA that was decided before any model was scored
(quiescent vs dynamic, one parameter regime vs another), report every subset
you produce, and say how many trajectories are in each.

Usage
-----
    # everything the stationarity screen flagged, removed from test
    python scripts/filter_split.py \
        --split artifacts/split_gray_scott.json \
        --exclude artifacts/stationary_gray_scott.json \
        --out artifacts/split_gray_scott_dynamic.json --config configs/gray_scott.yaml

    # the complement: score ONLY the quiescent ones
    python scripts/filter_split.py ... --keep-only-excluded \
        --out artifacts/split_gray_scott_stationary.json

    # then, no retraining:
    python scripts/evaluate_horizon.py --config configs/gray_scott.yaml \
        --set data.split_path=artifacts/split_gray_scott_dynamic.json \
        --checkpoint ./checkpoints/gs_dt_fno_s0/best_model.pth \
        --set experiment.exp_name=gs_dt_fno_dynamic_s0
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Set

from common import base_parser, resolve, json_default          # noqa: E402

from data import apply_dataset_defaults                        # noqa: E402
from data.splits import Split                                  # noqa: E402


def excluded_indices(path: str, store=None) -> Set[int]:
    """Read a screen's output as a set of trajectory indices.

    Accepts either `exclude_indices` (already resolved against this store) or
    `exclude` (file/realization pairs), and prefers the pairs when a store is
    available -- indices are only meaningful for the exact store that produced
    them, and a config change between the screen and now would silently shift
    them.
    """
    with open(path) as fp:
        payload = json.load(fp)
    pairs = payload.get("exclude") or []
    if pairs and store is not None and hasattr(store, "file_of"):
        want = {(str(f), int(r)) for f, r in pairs}
        return {t for t in range(store.n_traj)
                if (store.file_of(t), store.realization_of(t)) in want}
    idx = payload.get("exclude_indices")
    if idx is None:
        raise KeyError(f"{path} has neither 'exclude' nor 'exclude_indices'")
    return {int(i) for i in idx}


def verify_store_matches_split(split_path: str, store, sp: Split) -> None:
    """Refuse to run when the store is not the one the split was written for.

    A trajectory index means nothing on its own -- it is a position in the
    store's enumeration, and anything that changes that enumeration silently
    redefines every index in every frozen split.  `data.exclude_trajectories`
    is the usual culprit, and it is a trap laid by the stationarity screen's
    own closing advice ("point the config at it"): correct if you then rebuild
    the split, catastrophic if you do not.

    The crash is the SAFE half.  Excluding 894 of 1200 trajectories leaves
    indices 0..305, so a split holding 1081..1196 raises IndexError and you
    find out.  The indices that still fall in range do not raise -- they point
    at DIFFERENT trajectories.  Measured on a 36-trajectory fixture with 2
    excluded:

        index 27  ->  test/gliders#0   became  test/gliders#2
        index 28  ->  test/gliders#1   became  test/maze#0
        index 31  ->  test/maze#1      became  test/spirals#0

    Nothing in the pipeline would notice: the split-fingerprint check in
    `build_data` compares the train LIST, which is unchanged, not the
    trajectories it now names.  So this check is exact rather than a range
    test -- it compares the (file, realization) behind each index against the
    manifest `prepare_split.py` wrote beside the split.
    """
    man_path = os.path.splitext(split_path)[0] + "_manifest.json"
    idx = sorted(set(sp.train) | set(sp.val) | set(sp.test))

    if not os.path.exists(man_path):
        if idx and idx[-1] >= store.n_traj:
            raise SystemExit(
                f"\n{split_path} refers to trajectory {idx[-1]}, but the store "
                f"built from this config has only {store.n_traj}.\n"
                f"{_cause_hint()}")
        print(f"[filter_split] no manifest at {man_path}; only a range check "
              f"was possible.\n    Indices could still have shifted without "
              f"going out of range. Re-run\n    scripts/prepare_split.py to "
              f"get a manifest if this split predates it.")
        return

    with open(man_path) as fp:
        man = json.load(fp)
    if int(man.get("n_traj", -1)) != store.n_traj:
        raise SystemExit(
            f"\n{split_path} was built against a store of "
            f"{man.get('n_traj')} trajectories.\n"
            f"The store built from this config has {store.n_traj}.\n"
            f"{_cause_hint()}")

    if not hasattr(store, "file_of"):
        return
    rows = {int(r["index"]): r for r in man.get("trajectories", [])}
    for t in idx[:: max(1, len(idx) // 64)]:
        r = rows.get(t)
        if r is None or "file" not in r:
            continue
        now = (store.file_of(t), store.realization_of(t))
        was = (r["file"], int(r["realization"]))
        if now != was:
            raise SystemExit(
                f"\nIndex {t} named {was[0]}#{was[1]} when {split_path} was "
                f"written,\nbut names {now[0]}#{now[1]} in the store built "
                f"from this config.\n"
                f"The enumeration shifted, so EVERY index in that split now "
                f"refers to a\ndifferent trajectory -- including the ones that "
                f"did not go out of range.\n{_cause_hint()}")


def _cause_hint() -> str:
    return (
        "The usual cause is `data.exclude_trajectories` being set AFTER the "
        "split was\nfrozen. Excluding trajectories renumbers the store, and a "
        "frozen split is a\nlist of numbers.\n"
        "\n"
        "You almost certainly do not want the exclusion here. Two different "
        "operations:\n"
        "\n"
        "  exclude_trajectories   removes trajectories from the STORE. Decide "
        "it BEFORE\n"
        "                         prepare_split, and re-run prepare_split and "
        "the audit\n"
        "                         after changing it. It invalidates every "
        "checkpoint.\n"
        "  filter_split.py        subsets the TEST list for REPORTING. Store "
        "unchanged,\n"
        "                         train/val unchanged, statistics still valid, "
        "same\n"
        "                         checkpoints. This is what you want to report "
        "dynamic and\n"
        "                         stationary trajectories separately.\n"
        "\n"
        "Fix: set `data.exclude_trajectories: null` in the config, re-run the "
        "screen so\nits indices refer to the full store, then re-run this "
        "script.")


def main():
    ap = base_parser("Filter the TEST list of a split; train/val untouched")
    ap.add_argument("--split", required=True, help="the frozen split to filter")
    ap.add_argument("--exclude", required=True,
                    help="JSON from screen_stationary.py (or any file with "
                         "'exclude' / 'exclude_indices')")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-only-excluded", action="store_true",
                    help="invert: keep ONLY the flagged trajectories, so the "
                         "two halves can be reported side by side")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force to overwrite")

    store = None
    try:
        cfg = apply_dataset_defaults(
            resolve(args).get("dataset_name", "realpde_combustion"),
            resolve(args))
        from data.store import build_store
        cfg["store_verbose"] = False
        store = build_store(cfg)
    except Exception as e:                                      # noqa: BLE001
        print(f"[filter_split] no store ({type(e).__name__}: {e}); falling "
              f"back to 'exclude_indices'. Pass --config to resolve by "
              f"file/realization instead, which is safer.")

    sp = Split.load(args.split)
    if store is not None:
        verify_store_matches_split(args.split, store, sp)
    drop = excluded_indices(args.exclude, store)
    if not drop:
        print(f"\n[!] {args.exclude} flagged nothing in this store.\n"
              f"    Either the screen genuinely found no quiescent "
              f"trajectories at its lag and\n    threshold -- check its [3] "
              f"line -- or it was run against a DIFFERENT store\n    than this "
              f"one, in which case its (file, realization) pairs match "
              f"nothing here.\n    Nothing to filter; not writing an output.")
        return

    old = list(sp.test)
    if args.keep_only_excluded:
        new: List[int] = [t for t in old if t in drop]
        label = "flagged only"
    else:
        new = [t for t in old if t not in drop]
        label = "flagged removed"

    print(f"split      {args.split}")
    print(f"exclude    {args.exclude}  ({len(drop)} trajectories flagged "
          f"across the whole store)")
    print(f"test       {len(old)} -> {len(new)}   ({label})")
    if store is not None and hasattr(store, "file_of"):
        for t in old:
            mark = "flagged" if t in drop else "kept   "
            print(f"    {mark}  {t:>6}  {store.file_of(t)}"
                  f"#{store.realization_of(t)}")
    if not new:
        raise SystemExit(
            "the filter emptied the test set. Nothing to score -- widen the "
            "screen's threshold, or report the other half instead.")
    if len(new) < 4:
        print(f"\n[!] {len(new)} test trajectories is a very small population. "
              f"Report the count\n    next to every number you take from it, "
              f"and do not compare it against a\n    figure computed on the "
              f"full test set without saying so.")

    out = Split(train=list(sp.train), val=list(sp.val), test=new,
                seed=sp.seed,
                note=(f"REPORTING SUBSET of {os.path.basename(args.split)}: "
                      f"test filtered by {os.path.basename(args.exclude)} "
                      f"({label}), {len(old)} -> {len(new)}. train and val are "
                      f"unchanged, so the frozen normalisation statistics "
                      f"remain valid and no retraining or re-audit is needed."))
    out.check_disjoint()
    out.save(args.out)
    print(f"\n-> {args.out}")
    print("\nRe-score with the SAME checkpoints:")
    print(f"    python scripts/evaluate_horizon.py --config {args.config} \\")
    print(f"        --set data.split_path={args.out} \\")
    print( "        --checkpoint ./checkpoints/<tag>/best_model.pth \\")
    print( "        --set experiment.exp_name=<tag>_<subset>")
    print("\nReport BOTH halves. A subset chosen after seeing the scores is "
          "not a result.")


if __name__ == "__main__":
    main()
