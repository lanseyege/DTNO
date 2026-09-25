#!/usr/bin/env python
"""
Optional: materialise RealPDEBench Cylinder from Arrow into the per-trajectory
`.npy` layout `NPYStore` reads.

You do not need this.  `data/arrow_store.py` reads the Arrow shards in place,
zero-copy, and produces identical numbers.  Convert only if one of these
applies:

  * the Arrow blobs turn out not to be raw little-endian float32 (the store
    raises a specific error saying so), so the zero-copy view is impossible;
  * the Arrow files live on a slow or network filesystem where the page-cache
    behaviour of a memory map is poor, and you would rather pay ~36 GB of local
    disk once;
  * you want to subsample in time or space and freeze that decision as an
    artefact rather than as a config key that someone can change later.

Cost at native resolution: 92 x 3990 x 64 x 128 x 3 x 4 B = 36.1 GB.

The conversion is deliberately dumb -- decode, stack, write -- with one
exception worth stating.  Trajectories can differ in length; the DTNO store
contract has a single scalar `T`, and a ragged time axis would let a sampler
draw a frame past the end of a short trajectory.  So every trajectory is
truncated to the common minimum and the script prints exactly how many frames
that discarded.  If the histogram it prints is lopsided, drop the short
trajectories with `--min_frames` instead of truncating all 92 down to the
worst one.

Usage
-----
    python scripts/convert_cylinder.py \
        --src /mnt/sdb/yuanye/datasets/realpdebench/cylinder/hf_dataset/numerical \
        --out /mnt/sdb/yuanye/datasets/cylinder_npy
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from typing import List

import numpy as np

DEFAULT_FIELDS = ("u", "v", "p")
CHANNEL_NAMES = ("u", "v", "pressure")


def _tables(src: str):
    import pyarrow as pa
    for path in sorted(glob.glob(os.path.join(src, "*.arrow"))):
        source = pa.memory_map(path, "rb")
        try:
            reader = pa.ipc.open_stream(source)
        except pa.ArrowInvalid:
            source.seek(0)
            reader = pa.ipc.open_file(source)
        yield path, reader.read_all()


def main():
    ap = argparse.ArgumentParser(description="RealPDEBench Arrow -> npy")
    ap.add_argument("--src", required=True,
                    help="<root>/cylinder/hf_dataset/numerical")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fields", nargs="+", default=list(DEFAULT_FIELDS))
    ap.add_argument("--names", nargs="+", default=list(CHANNEL_NAMES))
    ap.add_argument("--dt", type=float, default=2.5e-3)
    ap.add_argument("--sub_t", type=int, default=1)
    ap.add_argument("--sub_s", type=int, default=1)
    ap.add_argument("--min_frames", type=int, default=None,
                    help="skip trajectories shorter than this instead of "
                         "truncating every trajectory to the global minimum")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if len(args.names) != len(args.fields):
        raise SystemExit(f"{len(args.names)} names for {len(args.fields)} fields")

    # ---- pass 1: shapes --------------------------------------------------
    meta: List[dict] = []
    for path, tb in _tables(args.src):
        missing = [f for f in args.fields if f not in tb.column_names]
        if missing:
            raise SystemExit(f"{os.path.basename(path)} lacks {missing}; "
                             f"has {tb.column_names}")
        for r in range(tb.num_rows):
            meta.append({
                "path": path, "row": r,
                "sim_id": str(tb.column("sim_id")[r].as_py()),
                "T": int(tb.column("shape_t")[r].as_py()),
                "H": int(tb.column("shape_h")[r].as_py()),
                "W": int(tb.column("shape_w")[r].as_py()),
            })
    if not meta:
        raise SystemExit(f"no rows found under {args.src}")

    lens = Counter(m["T"] for m in meta)
    print(f"{len(meta)} trajectories; frame-count histogram: {dict(lens)}")
    if args.min_frames:
        keep = [m for m in meta if m["T"] >= args.min_frames]
        print(f"  --min_frames {args.min_frames}: keeping {len(keep)}/{len(meta)}")
        meta = keep
    T_common = min(m["T"] for m in meta)
    lost = sum(m["T"] - T_common for m in meta)
    T_out = len(range(0, T_common, args.sub_t))
    H = int(np.ceil(meta[0]["H"] / args.sub_s))
    W = int(np.ceil(meta[0]["W"] / args.sub_s))
    C = len(args.fields)
    nbytes = len(meta) * T_out * H * W * C * 4
    print(f"  common T = {T_common} (truncation discards {lost} frames total)")
    print(f"  writing {len(meta)} x ({T_out}, {H}, {W}, {C}) float32 "
          f"= {nbytes / 1e9:.1f} GB")
    print(f"  dt {args.dt} -> {args.dt * args.sub_t} after sub_t={args.sub_t}")
    if args.dry_run:
        return

    os.makedirs(args.out, exist_ok=True)
    order = sorted(range(len(meta)), key=lambda i: meta[i]["sim_id"])
    t_idx = np.arange(0, T_common, args.sub_t)

    cache = {}
    for out_i, i in enumerate(order):
        m = meta[i]
        tb = cache.get(m["path"])
        if tb is None:
            cache.clear()                     # one mapped table at a time
            import pyarrow as pa
            src = pa.memory_map(m["path"], "rb")
            try:
                tb = pa.ipc.open_stream(src).read_all()
            except pa.ArrowInvalid:
                src.seek(0)
                tb = pa.ipc.open_file(src).read_all()
            cache[m["path"]] = tb
        cols = []
        for f in args.fields:
            buf = tb.column(f)[m["row"]].as_buffer()
            arr = np.frombuffer(memoryview(buf), dtype=np.float32)
            need = m["T"] * m["H"] * m["W"]
            if arr.size != need:
                raise SystemExit(
                    f"{m['sim_id']} field '{f}': {arr.size} values for "
                    f"declared shape ({m['T']}, {m['H']}, {m['W']}) = {need}. "
                    f"The blob is not raw float32.")
            v = arr.reshape(m["T"], m["H"], m["W"])[t_idx]
            if args.sub_s > 1:
                v = v[:, ::args.sub_s, ::args.sub_s]
            cols.append(np.ascontiguousarray(v, dtype=np.float32))
        block = np.stack(cols, axis=-1)
        name = f"traj_{out_i:03d}_{os.path.splitext(m['sim_id'])[0]}.npy"
        np.save(os.path.join(args.out, name), block)
        print(f"  [{out_i + 1:>3}/{len(order)}] {name}  {block.shape}")

    sidecar = {
        "channel_names": list(args.names),
        "dt": args.dt * args.sub_t,
        "source": os.path.abspath(args.src),
        "sub_t": args.sub_t, "sub_s": args.sub_s,
        "T_common": T_common, "frames_discarded_by_truncation": lost,
        # sim_id is the Reynolds number; keeping the mapping here is what lets
        # prepare_split stratify an npy store the same way it stratifies the
        # Arrow one.
        "sim_ids": [meta[i]["sim_id"] for i in order],
        "reynolds": [float(os.path.splitext(meta[i]["sim_id"])[0])
                     for i in order],
    }
    with open(os.path.join(args.out, "channels.json"), "w") as fp:
        json.dump(sidecar, fp, indent=2)
    print(f"\nWrote {len(order)} files + channels.json to {args.out}")
    print("Point the config at it with:\n"
          "  data:\n    backend: npy\n"
          f"    data_path: {os.path.abspath(args.out)}\n"
          f"    dt: {args.dt * args.sub_t}")


if __name__ == "__main__":
    main()
