#!/usr/bin/env python
"""
Measure where the input pipeline's time actually goes, in minutes, on one GPU
or none — instead of inferring it from two epoch timings taken hours apart on a
shared machine.

WHY THIS EXISTS
---------------
An epoch time is a single number produced by GPU compute, host-side loading,
storage, and whatever else is running on the box, all at once.  Tuning against
it means changing one thing, waiting 20 minutes, and attributing the difference
to the thing you changed — on a machine where two other jobs are also moving.
That is how a worker-count change and a page-cache eviction get confused for
each other.

Three phases, each isolating one layer:

  [1] RAW STORE     one process, no DataLoader, no torch. Times the exact
                    (K+1)-frame gather the sampler asks for. Reports cold and
                    warm separately, because the gap between them IS the
                    page-cache story. If warm is fast and cold is slow, the
                    working set does not fit in RAM and no number of workers
                    will fix it.
  [2] WORKER SWEEP  the real DataLoader at several worker counts, fitting
                    T(nw) = A/nw + B. `A` is loading work that parallelises,
                    `B` is the floor that does not. If B is small, you are
                    input-bound and should raise workers; if A/nw is already
                    below B, more workers buy nothing.
  [3] PREDICTION    epoch time at each worker count, so the choice is made
                    from a curve rather than from one more 20-minute trial.

Run it while the other jobs on the box are running.  A benchmark taken on an
idle machine answers a question you do not have.

    python scripts/bench_io.py --config configs/cylinder.yaml
    python scripts/bench_io.py --config configs/cylinder.yaml \
        --workers 4 8 12 16 --n_batches 60
    python scripts/bench_io.py --config configs/cylinder.yaml --raw_only
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# [1] raw store — deliberately free of torch, so it runs anywhere
# ---------------------------------------------------------------------------

def bench_raw(store, K: int, h_max: int, n: int, trajs: Sequence[int],
              seed: int = 0):
    """Time the (K+1)-frame gather: K history frames plus one target at +h.

    This is the access pattern the affordability argument rests on, and it is
    not a sequential read: the history block is contiguous, the target is up to
    h_max frames away, and each channel is a separate region in the file. On
    Cylinder that is ~6 distinct regions per sample.
    """
    rng = np.random.default_rng(seed)
    C = store.n_channels
    bytes_per = (K + 1) * store.H * store.W * C * 4

    def one_pass(tag, idx_list):
        t0 = time.perf_counter()
        for tr, t_idx in idx_list:
            store.read(tr, t_idx)
        dt = time.perf_counter() - t0
        print(f"    {tag:<22} {1000 * dt / len(idx_list):8.2f} ms/sample   "
              f"{len(idx_list) * bytes_per / dt / 1e6:7.1f} MB/s   "
              f"({dt:.1f}s for {len(idx_list)})")
        return dt

    # Draw once, replay twice: identical requests, so the only difference
    # between the two passes is whether the pages are already resident.
    idx_list = []
    for _ in range(n):
        tr = int(trajs[rng.integers(len(trajs))])
        t0 = int(rng.integers(K - 1, store.T - 1 - h_max))
        h = int(rng.integers(1, h_max + 1))
        idx_list.append((tr, list(range(t0 - K + 1, t0 + 1)) + [t0 + h]))

    print(f"\n[1] RAW STORE  ({K}+1 frames, {bytes_per / 1024:.0f} KB/sample, "
          f"{n} samples, 1 process)")
    cold = one_pass("cold (first touch)", idx_list)
    warm = one_pass("warm (same reads)", idx_list)
    ratio = cold / max(warm, 1e-9)
    print(f"    cold / warm            {ratio:8.1f}x")
    if ratio > 3:
        print("    -> The same reads are much cheaper the second time, so the "
              "first pass was\n       waiting on STORAGE, not on CPU. Over a "
              "full epoch the working set is\n       re-read from disk because "
              "it does not stay in page cache. More workers\n       raise "
              "queue depth and help some; shrinking the working set or moving "
              "it to\n       faster storage is the actual fix.")
    elif ratio < 1.3:
        print("    -> Cold and warm are alike, so the data was already "
              "resident or the device\n       is fast. The cost is CPU per "
              "sample; more workers should scale nearly\n       linearly until "
              "you run out of cores.")
    return {"bytes_per_sample": bytes_per,
            "cold_ms": 1000 * cold / n, "warm_ms": 1000 * warm / n,
            "cold_MBps": n * bytes_per / cold / 1e6,
            "warm_MBps": n * bytes_per / warm / 1e6,
            "cold_over_warm": ratio}


# ---------------------------------------------------------------------------
# [2] worker sweep
# ---------------------------------------------------------------------------

def bench_loader(cfg: dict, dataset_name: str, nw: int, n_batches: int,
                 warmup: int = 5):
    from data import build_data
    c = dict(cfg)
    c["num_workers"] = nw
    c["num_workers_val"] = 0
    c["val_samples"] = 8            # the val loader is not what we are timing
    c["samples_per_epoch"] = max(n_batches + warmup, 16) * int(
        c.get("batch_size", 8))
    bundle = build_data(dataset_name, c, model_kind="direct",
                        distributed=False, seed=int(c.get("seed", 42)))
    loader = bundle["train_loader"]
    it = iter(loader)
    for _ in range(warmup):         # fork, fill the prefetch queues
        next(it)
    t0 = time.perf_counter()
    got = 0
    for _ in range(n_batches):
        try:
            next(it)
        except StopIteration:
            break
        got += 1
    dt = time.perf_counter() - t0
    del it, loader, bundle
    bs = int(c.get("batch_size", 8))
    return {"workers": nw, "batches": got, "seconds": dt,
            "samples_per_s": got * bs / dt}


def fit_amdahl(points):
    """Least-squares fit of T = A/nw + B to (workers, seconds-per-sample)."""
    x = np.array([1.0 / p["workers"] if p["workers"] else 1.0 for p in points])
    y = np.array([1.0 / p["samples_per_s"] for p in points])
    if len(points) < 2:
        return None, None
    A, B = np.polyfit(x, y, 1)
    return float(A), float(B)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Input-pipeline benchmark")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--n_raw", type=int, default=120,
                    help="samples for the raw-store phase")
    ap.add_argument("--workers", type=int, nargs="+",
                    default=[2, 4, 6, 8, 12],
                    help="worker counts to sweep")
    ap.add_argument("--n_batches", type=int, default=40)
    ap.add_argument("--raw_only", action="store_true",
                    help="skip the DataLoader sweep (no torch needed)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import load_config, flatten, apply_overrides
    cfg = flatten(load_config(args.config))
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)

    from data import apply_dataset_defaults
    from data.store import build_store
    from data.splits import Split

    dataset_name = cfg.get("dataset_name", "realpde_combustion")
    cfg = apply_dataset_defaults(dataset_name, cfg)
    store = build_store(cfg)

    print("=" * 72)
    print(f"IO BENCHMARK — {dataset_name}")
    print("=" * 72)
    print(f"    {store.summary()}")
    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:
        ncpu = os.cpu_count()
    print(f"    {ncpu} CPUs visible to this process")

    sp_path = cfg.get("split_path")
    trajs = list(range(store.n_traj))
    if sp_path and os.path.exists(sp_path):
        sp = Split.load(sp_path)
        trajs = sp.train
        gb = len(trajs) * store.T * store.H * store.W * store.n_channels * 4 / 1e9
        print(f"    {len(trajs)} training trajectories, working set "
              f"{gb:.1f} GB uncompressed")
        print(f"    -> compare against free RAM (`free -g`). A working set "
              f"larger than the\n       page cache is re-read from disk every "
              f"epoch, and that is a storage\n       problem that worker counts "
              f"can only partly hide.")

    K = int(cfg.get("history_len", 4))
    h_max = int(cfg.get("h_max_train", 128))
    raw = bench_raw(store, K, h_max, args.n_raw, trajs,
                    seed=int(cfg.get("seed", 42)))

    payload = {"dataset": dataset_name, "store": store.summary(),
               "n_cpu": ncpu, "n_train_traj": len(trajs), "raw": raw}

    if not args.raw_only:
        print(f"\n[2] WORKER SWEEP  (real DataLoader, {args.n_batches} batches "
              f"after a {5}-batch warm-up)")
        print(f"    {'workers':>8}{'samples/s':>12}{'seconds':>10}")
        pts = []
        for nw in args.workers:
            try:
                p = bench_loader(cfg, dataset_name, nw, args.n_batches)
            except Exception as e:                              # noqa: BLE001
                print(f"    {nw:>8}   failed: {type(e).__name__}: {e}")
                continue
            pts.append(p)
            print(f"    {nw:>8}{p['samples_per_s']:>12.1f}{p['seconds']:>10.1f}")
        payload["sweep"] = pts

        if len(pts) >= 2:
            A, B = fit_amdahl(pts)
            spe = int(cfg.get("samples_per_epoch", 20000))
            world = int(os.environ.get("BENCH_WORLD", "4"))
            print(f"\n[3] PREDICTED EPOCH TIME  "
                  f"(samples_per_epoch={spe}, world={world}; "
                  f"set BENCH_WORLD to change)")
            print(f"    fitted per-sample cost = {A * 1000:.2f} ms / workers "
                  f"+ {B * 1000:.2f} ms")
            if B > 0 and A > 0:
                knee = A / B
                print(f"    the parallel part equals the floor at nw ~ "
                      f"{knee:.0f}; past that, extra\n    workers buy less than "
                      f"they cost")
            print()
            print(f"    {'workers':>8}{'loader s/epoch':>17}"
                  f"{'vs 6 workers':>15}")
            base = None
            for nw in sorted(set(list(args.workers) + [16, 20])):
                t = (A / nw + B) * spe / world
                if nw == 6:
                    base = t
                pct = f"{100 * t / base:.0f}%" if base else "-"
                flag = "  <- more workers than CPUs" if nw * world > ncpu else ""
                print(f"    {nw:>8}{t:>17.0f}{pct:>15}{flag}")
            print("\n    This is the LOADER's contribution only. GPU compute "
                  "runs concurrently,\n    so the real epoch time is roughly "
                  "max(loader, compute) once the queues are\n    full -- which "
                  "is why the payoff flattens once loading drops below "
                  "compute.")
            payload["fit"] = {"A_s": A, "B_s": B}

    out = args.out or os.path.join(cfg.get("results_dir", "./results"),
                                   "probe", f"io_{dataset_name}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fp:
        json.dump(payload, fp, indent=2, default=float)
    print(f"\n    -> {out}")
    print("\nWhile this runs, in another shell:")
    print("    iostat -x 5 3            # %util and r_await on the data device")
    print("    free -g                  # is the working set in page cache?")
    print("    nvidia-smi dmon -s u     # are the GPUs idle while you wait?")


if __name__ == "__main__":
    main()
