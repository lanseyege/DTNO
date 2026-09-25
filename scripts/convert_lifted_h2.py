#!/usr/bin/env python
"""
Convert BLASTNet Lifted Hydrogen Jet Flame into the NPYStore layout.

Reuses the user's existing `lifted_h2_dataset.py` for the raw .dat assembly and
downsampling, then writes what `data/store.NPYStore` expects:

    <out>/traj_000.npy      (T, H, W, 5) float32
    ...
    <out>/channels.json     {"channel_names": [...], "dt": ...}
    <out>/manifest.json     Re per trajectory index, frame id range, source

Why go through .npy rather than adding a Lifted-H2 branch to the pipeline: the
whole point of Phase 5 (§41) is that the architecture, losses, protocol and
evaluation do not change between datasets. Converting into the same on-disk
contract keeps "the code did not change" a true statement instead of a claim.
It also sidesteps the I/O problem entirely -- 8 x 181 x 160 x 200 x 5 x 4 B is
about 0.9 GB, so the whole dataset is memory-mappable and a frame gather costs
nothing.

TWO THINGS TO SETTLE BEFORE TRAINING, both flagged by this script:

1. dt. The cases ship snapshot ids, not physical times. The horizon axis of
   every figure is in FRAMES, which only becomes comparable across datasets once
   dt is known -- and comparing the predictability horizon between RealPDEBench
   (dt = 2.5e-4 s) and this dataset is the single most valuable thing Phase 5
   can do here. Pass --dt once you have it from info.json or the paper; the
   script records it and warns if left at the placeholder.

2. Trajectory length. Re=11000 ships ids 20..200 (181 frames) while the others
   ship 0..200 (201). NPYStore assumes a common T. `--align id` truncates every
   case to the shared id window 20..200, which lines the cases up by physical
   time rather than by array position -- the right choice for a developing jet,
   where frame 0 of one case is not the same flow state as frame 0 of another.

Usage:
    python scripts/convert_lifted_h2.py \\
        --loader /path/to/lifted_h2_dataset.py \\
        --data_root /mnt/sdb/yuanye/datasets/blastnet/lifted_hydrogen_jet_flame \\
        --out /mnt/sdb/yuanye/datasets/lifted_h2_npy \\
        --dt <seconds per snapshot>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

from common import REPO_ROOT  # noqa: F401

ALL_RE = [5000, 6000, 7000, 7500, 8000, 9000, 10000, 11000]


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("lifted_h2_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lifted_h2_dataset"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description="Lifted H2 -> NPYStore")
    ap.add_argument("--loader", required=True,
                    help="path to the existing lifted_h2_dataset.py")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--re", type=int, nargs="+", default=ALL_RE,
                    help="Reynolds numbers to convert, in trajectory-index order")
    ap.add_argument("--x_num", type=int, default=160)
    ap.add_argument("--y_num", type=int, default=200)
    ap.add_argument("--n_snapshots", type=int, default=201)
    ap.add_argument("--dt", type=float, default=None,
                    help="seconds per snapshot; REQUIRED for cross-dataset "
                         "comparison of the predictability horizon")
    ap.add_argument("--align", default="id", choices=["id", "head", "none"],
                    help="id: truncate all cases to the shared frame-id window "
                         "(recommended). head: keep the first T_min frames. "
                         "none: leave ragged (NPYStore will reject it)")
    ap.add_argument("--test_re", type=int, default=7500,
                    help="Re held out for testing. 7500 is the designed choice: "
                         "it is interpolated between two training cases, which "
                         "is the generalization claim this dataset supports.")
    ap.add_argument("--val_re", type=int, default=9000,
                    help="Re held out for model selection (§24)")
    ap.add_argument("--split_out", default="artifacts/split_lifted_h2.json")
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()

    mod = load_module(args.loader)
    names = list(mod.CHANNEL_NAMES)
    os.makedirs(args.out, exist_ok=True)

    print("=" * 74)
    print("CONVERT LIFTED H2 -> NPYStore")
    print("=" * 74)
    print(f"  Re cases : {args.re}")
    print(f"  channels : {names}")

    arrays, spans = [], []
    for re in args.re:
        case_dir = os.path.join(args.data_root, f"hydrogen-jet-{re}")
        if not os.path.isdir(case_dir):
            raise FileNotFoundError(case_dir)
        a = mod._load_one_case(case_dir, args.n_snapshots, args.x_num, args.y_num)
        # recover the frame-id window this case actually covers
        with open(os.path.join(case_dir, "info.json")) as fp:
            info = json.load(fp)
        ids = [l.get("id", l.get("time step", i))
               for i, l in enumerate(info["local"][: args.n_snapshots])]
        ids = ids[len(ids) - a.shape[0]:]
        arrays.append(a)
        spans.append((int(ids[0]), int(ids[-1])))
        print(f"  Re={re:>6}: {a.shape}  ids {ids[0]}..{ids[-1]}")

    # -- align --------------------------------------------------------
    if args.align == "id":
        lo = max(s[0] for s in spans)
        hi = min(s[1] for s in spans)
        print(f"\n  shared id window: {lo}..{hi}  ({hi - lo + 1} frames)")
        cut = []
        for a, (s0, s1) in zip(arrays, spans):
            i0 = lo - s0
            i1 = i0 + (hi - lo + 1)
            cut.append(a[i0:i1])
        arrays = cut
    elif args.align == "head":
        T = min(a.shape[0] for a in arrays)
        print(f"\n  truncating all cases to the first {T} frames")
        arrays = [a[:T] for a in arrays]

    Ts = {a.shape[0] for a in arrays}
    if len(Ts) != 1 and args.align != "none":
        raise RuntimeError(f"alignment failed, lengths still differ: {Ts}")

    # -- write --------------------------------------------------------
    for i, (re, a) in enumerate(zip(args.re, arrays)):
        dst = os.path.join(args.out, f"traj_{i:03d}.npy")
        np.save(dst, np.ascontiguousarray(a, dtype=np.float32))
        print(f"  -> {os.path.basename(dst)}  {a.shape}  Re={re}")

    dt = args.dt if args.dt is not None else 1.0
    with open(os.path.join(args.out, "channels.json"), "w") as fp:
        json.dump({"channel_names": names, "dt": dt}, fp, indent=2)
    with open(os.path.join(args.out, "manifest.json"), "w") as fp:
        json.dump({"source": args.data_root,
                   "re_per_trajectory": {str(i): re
                                         for i, re in enumerate(args.re)},
                   "id_spans_before_align": spans,
                   "align": args.align,
                   "resolution": [args.x_num, args.y_num],
                   "dt": dt}, fp, indent=2)

    total = sum(a.nbytes for a in arrays) / 1e9
    print(f"\n  wrote {len(arrays)} trajectories, {total:.2f} GB, to {args.out}")

    # -- split ---------------------------------------------------------
    # Written here rather than left to load_or_create_split's seeded shuffle.
    # This dataset has a DESIGNED held-out case; a random draw would land on it
    # only by luck, and 5 of the 8 possible draws would silently answer a
    # different question (extrapolation past Re = 11000, say, instead of
    # interpolation at 7500).
    if args.test_re not in args.re:
        raise SystemExit(f"--test_re {args.test_re} not in --re {args.re}")
    if args.val_re not in args.re or args.val_re == args.test_re:
        raise SystemExit(f"--val_re {args.val_re} must be in --re and differ "
                         f"from --test_re")
    idx = {re: i for i, re in enumerate(args.re)}
    test = [idx[args.test_re]]
    val = [idx[args.val_re]]
    train = [i for i in range(len(args.re)) if i not in test + val]

    sys.path.insert(0, REPO_ROOT)
    from data.splits import Split
    sp = Split(train=train, val=val, test=test,
               protocol="trajectory_held_out", seed=0,
               note=(f"Re held out by design: test={args.test_re} "
                     f"(interpolated), val={args.val_re}. "
                     f"Re per index: {dict(enumerate(args.re))}"))
    sp.check_disjoint()
    sp.save(args.split_out)
    print(f"\n  split -> {args.split_out}")
    print(f"    train Re {[args.re[i] for i in train]}")
    print(f"    val   Re {[args.re[i] for i in val]}")
    print(f"    test  Re {[args.re[i] for i in test]}  (interpolated)")

    if args.dt is None:
        print("\n  [!] dt was not supplied, so channels.json records dt = 1.0.")
        print("      Everything still trains -- tau is h * dt / t_scale and the")
        print("      scale cancels -- but the predictability horizon will only")
        print("      be expressible in FRAMES, and the comparison against")
        print("      RealPDEBench (dt = 2.5e-4 s) is the main reason to run this")
        print("      dataset at all. Find dt in info.json or Sharma et al. 2024")
        print("      and re-run with --dt, or edit channels.json in place.")

    print("\nNext:")
    print("  python scripts/audit_data.py --config configs/lifted_h2.yaml")
    print()
    print("  NOT derive_strata: it detects regime markers from species")
    print("  presence/absence, and every channel is present in every case here")
    print("  -- the cases differ by Reynolds number, a continuous parameter that")
    print("  lives in manifest.json, not in the fields. The split is already")
    print("  written above, so audit_data.py will load it rather than draw one.")
    print("  Run audit WITHOUT --force unless you mean to redraw the split.")


if __name__ == "__main__":
    main()
