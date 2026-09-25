#!/usr/bin/env python
"""
Convert REALM (IgnitHIT / EvolveJet) into the per-trajectory .npy layout that
`data/store.NPYStore` reads.

    <out>/traj_000.npy      (T, H, W, C) float32
    <out>/traj_001.npy
    ...
    <out>/channels.json     {"channel_names": [...], "dt": ...}

REALM ships as HDF5 with a layout that varies by dataset, so this script asks
you to confirm the mapping instead of guessing: run it with `--inspect` first,
read the tree it prints, then pass `--variables` and `--layout`.

    python scripts/convert_realm.py --src /data/realm/ignithit --inspect
    python scripts/convert_realm.py --src /data/realm/ignithit \
        --out /data/realm_ignithit_npy --dt 1e-6 --layout TCHW

Phase 5 (§41) is a portability test.  Converting into the same on-disk contract
as RealPDEBench is what makes "the code did not change" a true statement rather
than a claim, so resist the urge to special-case anything downstream of here.
"""

from __future__ import annotations

import argparse
import json
import os
import glob

import numpy as np

from common import REPO_ROOT  # noqa: F401


def inspect(path: str, max_depth: int = 3):
    import h5py
    files = sorted(glob.glob(os.path.join(path, "*.h5"))
                   + glob.glob(os.path.join(path, "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"no .h5/.hdf5 under {path}")
    print(f"{len(files)} file(s); showing the structure of {files[0]}\n")

    def walk(name, obj):
        depth = name.count("/")
        if depth > max_depth:
            return
        pad = "  " * depth
        if hasattr(obj, "shape"):
            print(f"{pad}{name}  shape={obj.shape} dtype={obj.dtype}")
        else:
            print(f"{pad}{name}/")

    with h5py.File(files[0], "r") as f:
        f.visititems(walk)
        if f.attrs:
            print("\nattrs:", dict(f.attrs))
    print("\nPick the variable datasets with --variables a b c ... and state "
          "their axis order with --layout (THWC | TCHW | CTHW).")


def to_thwc(arr: np.ndarray, layout: str) -> np.ndarray:
    layout = layout.upper()
    if layout == "THWC":
        return arr
    if layout == "TCHW":
        return np.transpose(arr, (0, 2, 3, 1))
    if layout == "CTHW":
        return np.transpose(arr, (1, 2, 3, 0))
    if layout == "HWCT":
        return np.transpose(arr, (3, 0, 1, 2))
    raise ValueError(f"unknown layout '{layout}'")


def main():
    ap = argparse.ArgumentParser(description="REALM -> NPYStore converter")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--variables", nargs="+", default=None,
                    help="HDF5 dataset paths, one per channel; omit to take "
                         "every 3D/4D dataset at the top level, sorted by name")
    ap.add_argument("--layout", default="THWC",
                    choices=["THWC", "TCHW", "CTHW", "HWCT"],
                    help="axis order of each variable dataset")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--pattern", default="*.h5")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.src)
        return
    if not args.out:
        raise SystemExit("--out is required unless --inspect")

    import h5py
    files = sorted(glob.glob(os.path.join(args.src, args.pattern)))
    if not files:
        raise FileNotFoundError(f"no files matching {args.pattern} in {args.src}")
    os.makedirs(args.out, exist_ok=True)

    names = None
    for i, path in enumerate(files):
        with h5py.File(path, "r") as f:
            if args.variables:
                keys = list(args.variables)
            else:
                keys = sorted(k for k in f.keys()
                              if hasattr(f[k], "shape") and f[k].ndim in (3, 4))
            if names is None:
                names = keys
                print(f"channels ({len(names)}): {names}")
            elif keys != names:
                raise ValueError(f"{path} has variables {keys}, expected {names}")

            chans = []
            for k in keys:
                a = np.asarray(f[k][...], dtype=np.float32)
                if a.ndim == 3:                    # (T, H, W) — one channel
                    chans.append(a[..., None])
                else:
                    chans.append(to_thwc(a, args.layout))
            traj = np.concatenate(chans, axis=-1)

        dst = os.path.join(args.out, f"traj_{i:03d}.npy")
        np.save(dst, traj)
        print(f"  {os.path.basename(path)} -> {os.path.basename(dst)}  "
              f"{traj.shape}")

    with open(os.path.join(args.out, "channels.json"), "w") as fp:
        json.dump({"channel_names": names, "dt": args.dt}, fp, indent=2)
    print(f"\nWrote {len(files)} trajectories to {args.out}")
    print("Next: point configs/realm_ignithit.yaml at it and run\n"
          "    python scripts/audit_data.py --config configs/realm_ignithit.yaml")


if __name__ == "__main__":
    main()
