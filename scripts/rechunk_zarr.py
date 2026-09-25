#!/usr/bin/env python
"""
Rechunk the combustion zarr for frame-gather access.

The store as shipped is chunked (1, 64, 128, 128, 1): 64 frames on the time
axis and ONE channel per chunk. That layout is right for "stream a whole
trajectory of one variable" and wrong for what direct-time training does, which
is gather K+1 scattered frames across all channels.

Measured on the real store:

    single sample = 4+1 frames x 13 channels =  4.26 MB of useful data
    chunks touched                           = 13 (one per channel)
    decompressed                             = 13 x 4.2 MB = 54.6 MB
    amplification                            = ~12.8x
    store.read                               = 85.3 ms  (50 MB/s effective)
    one worker                               = 10.5 samples/s

At 4 workers that was 25 samples/s and 13 minutes of data time per epoch per
rank, with the GPUs idle and the CPU saturated -- the loader, not the model, was
the experiment's throughput.

Target layout (1, 4, 128, 128, C): a few frames, all channels together. A 5
frame gather then touches 2-3 chunks instead of 13, and the channel selection
comes free out of a chunk already in memory.

    python scripts/rechunk_zarr.py \\
        --src /mnt/sdb/yuanye/datasets/realpdebench_combustion.zarr \\
        --dst /mnt/sdb/yuanye/datasets/realpdebench_combustion_t4.zarr

Then point `data.data_path` at the new store. Nothing else changes: the split
and the normalisation statistics are keyed by trajectory index and channel name,
both of which are preserved, so they stay valid and must NOT be recomputed.

One pass over ~59 GB. Run it once, verify, then delete the original if space is
tight -- `--verify` re-reads random gathers from both stores and compares them
exactly before you commit to that.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np

from common import REPO_ROOT  # noqa: F401


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def main():
    ap = argparse.ArgumentParser(description="Rechunk a combustion zarr store")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--time_chunk", type=int, default=4,
                    help="frames per chunk on the time axis. 4 suits K=4 "
                         "history + 1 target; raise it if you also stream long "
                         "contiguous windows (the AR rollout evaluation does)")
    ap.add_argument("--block", type=int, default=128,
                    help="frames copied per read/write step (memory knob only)")
    ap.add_argument("--compressor", default="zstd",
                    choices=["zstd", "lz4", "none"])
    ap.add_argument("--clevel", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", type=int, default=20,
                    help="random gathers compared against the source afterwards")
    args = ap.parse_args()

    import zarr

    if os.path.exists(args.dst):
        if not args.overwrite:
            raise SystemExit(f"{args.dst} exists; pass --overwrite to replace it")
        shutil.rmtree(args.dst)

    src_root = zarr.open(args.src, mode="r")
    src = src_root["data"]
    N, T, H, W, C = src.shape
    new_chunks = (1, int(args.time_chunk), H, W, C)

    print("=" * 74)
    print("RECHUNK")
    print("=" * 74)
    print(f"  src     {args.src}")
    print(f"          shape={src.shape} chunks={src.chunks} dtype={src.dtype}")
    print(f"  dst     {args.dst}")
    print(f"          chunks={new_chunks} "
          f"({human(int(np.prod(new_chunks)) * src.dtype.itemsize)} each)")

    old_per_sample = len(range(C)) * int(np.prod(src.chunks)) * src.dtype.itemsize
    new_touch = (args.time_chunk and 3) or 1
    new_per_sample = new_touch * int(np.prod(new_chunks)) * src.dtype.itemsize
    print(f"\n  decompressed per 5-frame gather: {human(old_per_sample)} -> "
          f"~{human(new_per_sample)}  ({old_per_sample/new_per_sample:.1f}x less)")

    # -- compressor ------------------------------------------------------
    compressor = None
    if args.compressor != "none":
        try:
            from numcodecs import Blosc
            compressor = Blosc(cname=args.compressor, clevel=args.clevel,
                               shuffle=Blosc.SHUFFLE)
        except Exception as e:
            print(f"  [warn] numcodecs unavailable ({e}); writing uncompressed")

    # zarr 2 and 3 disagree on both the method name and the compressor
    # argument, and zarr 3 removed create_dataset entirely rather than
    # deprecating it, so feature-detect instead of catching TypeError.
    '''
    dst_root = zarr.open(args.dst, mode="w")
    if hasattr(dst_root, "create_dataset"):          # zarr 2.x
        dst = dst_root.create_dataset("data", shape=src.shape,
                                      chunks=new_chunks, dtype=src.dtype,
                                      compressor=compressor)
    else:                                            # zarr 3.x
        codecs = None
        if args.compressor != "none":
            from zarr.codecs import BloscCodec
            codecs = [BloscCodec(cname=args.compressor, clevel=args.clevel)]
        dst = dst_root.create_array("data", shape=src.shape, chunks=new_chunks,
                                    dtype=src.dtype, compressors=codecs)
    '''

    dst_root = zarr.open(args.dst, mode="w")
    import zarr
    if zarr.__version__.startswith('2.'):   # zarr 2.x
        dst = dst_root.create_dataset("data", shape=src.shape,
                                      chunks=new_chunks, dtype=src.dtype,
                                      compressor=compressor)
    else:                                   # zarr 3.x
        codecs = None
        if args.compressor != "none":
            from zarr.codecs import BloscCodec
            codecs = [BloscCodec(cname=args.compressor, clevel=args.clevel)]
        dst = dst_root.create_array("data", shape=src.shape, chunks=new_chunks,
                                    dtype=src.dtype, compressors=codecs)
    # -- copy ------------------------------------------------------------
    t0 = time.perf_counter()
    total = N * T
    done = 0
    for i in range(N):
        for lo in range(0, T, args.block):
            hi = min(lo + args.block, T)
            dst[i, lo:hi] = np.asarray(src[i, lo:hi])
            done += hi - lo
        el = time.perf_counter() - t0
        rate = done / max(el, 1e-9)
        eta = (total - done) / max(rate, 1e-9)
        print(f"  traj {i+1:>3}/{N}  {done}/{total} frames  "
              f"{rate:.0f} frames/s  ETA {eta/60:.1f} min", flush=True)

    # -- metadata --------------------------------------------------------
    side = os.path.join(args.src, "channels.json")
    if os.path.exists(side):
        shutil.copy(side, os.path.join(args.dst, "channels.json"))
        print(f"\n  copied channels.json")
    try:
        for k, v in dict(src_root.attrs).items():
            dst_root.attrs[k] = v
    except Exception:
        pass

    print(f"\n  wrote {args.dst} in {(time.perf_counter()-t0)/60:.1f} min")

    # -- verify ----------------------------------------------------------
    if args.verify > 0:
        print(f"\n  verifying {args.verify} random gathers...")
        rng = np.random.default_rng(0)
        bad = 0
        t_src = t_dst = 0.0
        for _ in range(args.verify):
            i = int(rng.integers(N))
            t = int(rng.integers(4, T - 200))
            idx = list(range(t - 3, t + 1)) + [t + 128]
            ch = list(range(C))
            a = time.perf_counter()
            ref = np.stack([np.asarray(src[i, j, :, :, ch]) for j in idx])
            b = time.perf_counter()
            got = np.stack([np.asarray(dst[i, j, :, :, ch]) for j in idx])
            c = time.perf_counter()
            t_src += b - a
            t_dst += c - b
            if not np.array_equal(ref, got):
                bad += 1
        print(f"    {args.verify - bad}/{args.verify} gathers identical"
              + ("  [!] MISMATCH — do not delete the source" if bad else ""))
        print(f"    read time  src {1000*t_src/args.verify:.1f} ms  ->  "
              f"dst {1000*t_dst/args.verify:.1f} ms  "
              f"({t_src/max(t_dst,1e-9):.1f}x faster)")

    print("\nNext:")
    print(f"    --set data.data_path={args.dst}")
    print("    python scripts/bench_io.py configs/dt_fno.yaml")
    print("  The frozen split and norm_stats stay valid: they key on trajectory")
    print("  index and channel name, and rechunking changes neither.")


if __name__ == "__main__":
    main()
