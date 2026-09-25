#!/usr/bin/env python
"""
Find trajectories that stop moving, and write the exclusion list.

THE FAILURE THIS PREVENTS
-------------------------
A trajectory that reaches a fixed point is a trajectory on which persistence is
exact.  It does not add noise to the results; it adds a *bias*, in the
flattering direction, to every quantity the paper reports:

  * E(h) flattens for every method, which is the signature this project already
    learned to distrust (§9: "a flat error curve means nothing without a
    climatology baseline").
  * The autoregressive baseline stops accumulating error, because there is
    nothing left to accumulate.  The direct-time-versus-AR crossover -- the
    paper's headline -- moves for a reason that has nothing to do with either
    method.
  * The climatology baselines become *unbeatable*, because on a stationary
    record the climatological mean IS the state.

Gray-Scott is the acute case: The Well documents ~12% of its "gliders" and
"spirals" trajectories reaching equilibrium at stored step 107-159, out of a
1001-step record.  That is ~87% of those trajectories' frames on which the
answer is "the input".  But the same screen is worth running on any new
dataset, including for the opposite problem: Rayleigh-Benard starts from rest,
so its *opening* frames are spin-up transient rather than dynamics, and
`traj_cut` exists to drop them.

MEASUREMENT, NOT TABLE LOOKUP
-----------------------------
`data/gray_scott.py` carries The Well's published stationary table.  That table
is the claim; this script is the measurement, and the two are cross-checked
when they can be.  Three reasons not to rely on the table alone:

  * it covers species A only, and B is the species with structure;
  * it is indexed by (split, f, k, realization), so it depends on this store's
    file enumeration matching The Well's -- an assumption worth testing rather
    than trusting;
  * it says nothing about *near*-stationary trajectories, which cause the same
    bias in weaker form.

The statistic is deliberately the simplest thing that cannot be gamed:

    r(t) = || z(u(t+L)) - z(u(t)) ||_2 / median_t || z(u(t)) - mean_t z(u) ||_2

the change over a lag L, measured against the trajectory's own fluctuation
amplitude.  A trajectory is called stationary from the first probe time after
which r stays below `--threshold` for the rest of the record.


CHOOSE L TO MATCH THE HORIZON YOU SCORE — THIS IS THE WHOLE DESIGN
------------------------------------------------------------------
`--lag 1` asks "is this field quiet between consecutive stored frames".  That
is almost never the question.  The question is "is persistence exact at the
horizons I evaluate at", and the two differ by however many steps separate
them: at r = 0.02 per step, 512 steps accumulate to 0.45 if the changes are
uncorrelated and 10.2 if they are coherent.  Both are enormous; neither is
"stationary".

This is not hypothetical either.  Run at `--lag 1 --threshold 0.02`,
Gray-Scott flagged 894 of 1200 trajectories — every single bubbles, maze,
spots and worms realization — against The Well's published count of 95.  The
measured persistence baseline on that same test set was E = 0.70 at h = 512,
which a set of frozen trajectories cannot produce.  The screen was right that
those patterns evolve slowly; it was wrong to call slow evolution stationary,
because slow evolution over 512 steps is precisely the long-horizon signal the
study is about.

So `--lag` now defaults to `data.h_max_train` from the config rather than to 1,
and the threshold is read on that scale.  Run it at more than one lag if you
want the picture: `--lag 8` finds what makes short horizons trivial, `--lag 512`
finds what makes the long tail trivial, and the difference between the two
counts is itself worth reporting.

Usage
-----
    python scripts/screen_stationary.py --config configs/gray_scott.yaml
    python scripts/screen_stationary.py --config configs/gray_scott.yaml \
        --threshold 0.02 --min_tail_frac 0.5 --out artifacts/stationary_gs.json

Then point the config at the result:

    data:
      exclude_trajectories: artifacts/stationary_gs.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np

from common import base_parser, resolve, json_default          # noqa: E402

from data import apply_dataset_defaults                        # noqa: E402
from data.store import build_store                             # noqa: E402


def trajectory_activity(store, traj: int, n_probe: int, mu, sd,
                        lag: int = 1) -> tuple:
    """-> (probe_times, r(t)), the one-step change over the record."""
    hi = store.T - 1 - lag
    ts = np.unique(np.linspace(0, hi, n_probe).astype(int))
    want = sorted(set(ts.tolist()) | set((ts + lag).tolist()))
    pos = {t: i for i, t in enumerate(want)}
    block = (store.read(traj, want) - mu) / sd                 # (n,H,W,C)
    flat = block.reshape(len(want), -1)
    ref = flat.mean(axis=0, keepdims=True)
    amp = np.linalg.norm(flat - ref, axis=1)
    scale = float(np.median(amp)) + 1e-12
    r = np.array([np.linalg.norm(flat[pos[int(t) + lag]] - flat[pos[int(t)]])
                  / scale for t in ts])
    return ts, r


def first_quiescent(ts, r, threshold: float, min_tail_frac: float
                    ) -> Optional[int]:
    """First probe time after which r stays below threshold to the end.

    `min_tail_frac` guards against calling a trajectory stationary on the
    strength of the last two probe points, which is noise, not equilibrium.
    """
    below = r < threshold
    n = len(ts)
    for i in range(n):
        if below[i:].all():
            if (n - i) / n >= min_tail_frac:
                return int(ts[i])
            return None
    return None


def main():
    ap = base_parser("Screen trajectories for stationarity")
    ap.add_argument("--n_probe", type=int, default=64,
                    help="probe times per trajectory")
    ap.add_argument("--lag", type=int, default=None,
                    help="frames between the two states compared. Defaults to "
                         "data.h_max_train -- the horizon you actually score "
                         "at. --lag 1 asks a different and usually wrong "
                         "question; see the module docstring.")
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="r(t) below this counts as quiescent, ON THE SCALE OF "
                         "--lag. At lag 1 use ~0.02; at a horizon-scale lag "
                         "~0.1 means the field moved less than a tenth of its "
                         "own amplitude over the whole horizon.")
    ap.add_argument("--min_tail_frac", type=float, default=0.4,
                    help="fraction of the record that must stay quiescent")
    ap.add_argument("--max_traj", type=int, default=None)
    ap.add_argument("--n_stat_frames", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = resolve(args)
    dataset_name = cfg.get("dataset_name", "realpde_combustion")
    cfg = apply_dataset_defaults(dataset_name, cfg)

    store = build_store(cfg)
    lag = args.lag if args.lag is not None else int(cfg.get("h_max_train", 1))
    lag = max(1, min(lag, store.T - 2))
    print("=" * 72)
    print(f"STATIONARITY SCREEN — {dataset_name}")
    print("=" * 72)
    print(f"\n[1] Store\n    {store.summary()}")
    print(f"    r(t) = ||z(u(t+{lag})) - z(u(t))|| / median_t "
          f"||z(u(t)) - mean||")
    if args.lag is None:
        print(f"    lag = data.h_max_train = {lag}. This asks whether "
              f"persistence is exact at\n    the horizon you score at, which "
              f"is the question. Pass --lag 1 only if you\n    want "
              f"instantaneous quiescence instead, and lower --threshold to "
              f"~0.02 with it.")
    print(f"    quiescent if r < {args.threshold} for the last "
          f">= {100 * args.min_tail_frac:.0f}% of the record")

    # statistics from a handful of trajectories; only the scale matters here
    from probe_timescales import sample_stats
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    probe_trajs = list(range(0, store.n_traj,
                             max(1, store.n_traj // 8)))[:8]
    mu, sd = sample_stats(store, probe_trajs, args.n_stat_frames, rng)

    n = store.n_traj if args.max_traj is None else min(store.n_traj,
                                                       args.max_traj)
    records: List[Dict] = []
    print(f"\n[2] Screening {n} trajectories")
    for t in range(n):
        ts, r = trajectory_activity(store, t, args.n_probe, mu, sd, lag)
        t_stat = first_quiescent(ts, r, args.threshold, args.min_tail_frac)
        rec = {
            "index": t,
            "r_median": float(np.median(r)),
            "r_last": float(r[-1]),
            "r_first": float(r[0]),
            "t_stationary": t_stat,
        }
        if hasattr(store, "file_of"):
            rec["file"] = store.file_of(t)
            rec["realization"] = store.realization_of(t)
            rec["official_split"] = store.official_split[t]
        if hasattr(store, "sim_id"):
            rec["sim_id"] = store.sim_id(t)
        records.append(rec)
        if (t + 1) % max(1, n // 10) == 0:
            done = sum(1 for x in records if x["t_stationary"] is not None)
            print(f"    {t + 1:>5}/{n}   stationary so far: {done}")

    stat = [r for r in records if r["t_stationary"] is not None]
    print(f"\n[3] Result: {len(stat)}/{n} trajectories go quiescent")
    if stat:
        onsets = np.array([r["t_stationary"] for r in stat])
        print(f"    onset frame: min {onsets.min()}  median "
              f"{int(np.median(onsets))}  max {onsets.max()}   of T={store.T}")
        frac = 1.0 - float(np.median(onsets)) / store.T
        print(f"    a median flagged trajectory is {100 * frac:.0f}% "
              f"frames on which persistence is exact at lag {lag}")
        if len(stat) > 0.4 * n:
            print(f"\n    [!] {100 * len(stat) / n:.0f}% of the store is "
                  f"flagged. Before excluding anything, check it\n        "
                  f"against the persistence baseline you already measured: if "
                  f"E_persistence(h)\n        is well above 0 at h ~ {lag}, "
                  f"these trajectories are evolving SLOWLY, not\n        "
                  f"frozen, and slow evolution at long h is the signal, not "
                  f"an artefact.\n        Tighten --threshold, or accept that "
                  f"this dataset is mostly slow.")
        by_file = Counter(r.get("file", "?") for r in stat)
        for k in sorted(by_file):
            print(f"    {k:<48} {by_file[k]}")

    # ---- cross-check against The Well's published table ------------------
    if dataset_name == "gray_scott" and hasattr(store, "file_of"):
        from data.gray_scott import official_stationary_exclusions
        try:
            official = set(official_stationary_exclusions(store))
        except Exception as e:                                  # noqa: BLE001
            official = None
            print(f"\n[4] Could not evaluate the official table: {e}")
        if official is not None:
            measured = {(r["file"], r["realization"]) for r in stat}
            print(f"\n[4] Cross-check against The Well's published table")
            print(f"    official {len(official)}   measured {len(measured)}   "
                  f"agree {len(official & measured)}")
            only_o = sorted(official - measured)[:8]
            only_m = sorted(measured - official)[:8]
            if only_o:
                print(f"    official only (first 8): {only_o}")
                print("      -> either the threshold is too strict, or this "
                      "store's file/realization\n         enumeration does not "
                      "match The Well's. Check before excluding.")
            if only_m:
                print(f"    measured only (first 8): {only_m}")
                print("      -> the official table covers species A only; "
                      "these may be genuine\n         B-quiescent "
                      "trajectories the table does not list.")
            if not only_o and not only_m:
                print("    exact agreement. Use either list; they are the "
                      "same set.")

    payload = {
        "dataset": dataset_name,
        "store": store.summary(),
        "criterion": {"lag": lag, "threshold": args.threshold,
                      "min_tail_frac": args.min_tail_frac,
                      "n_probe": args.n_probe},
        "n_screened": n,
        "n_stationary": len(stat),
        "records": records,
        # the field data/well.py's `exclude` and the configs read
        "exclude": [[r["file"], r["realization"]] for r in stat
                    if "file" in r],
        "exclude_indices": [r["index"] for r in stat],
    }
    out = args.out or f"artifacts/stationary_{dataset_name}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fp:
        json.dump(payload, fp, indent=2, default=json_default)

    print(f"\n[5] Written -> {out}")
    print("""
    TWO WAYS TO USE THIS, AND THEY ARE NOT INTERCHANGEABLE.

    (a) REPORTING SUBSET -- almost always what you want, costs nothing:

          python scripts/filter_split.py --config <config> \\
              --split <the frozen split> --exclude %s \\
              --out <split>_dynamic.json

        Subsets the TEST list only. Store, train, val and the frozen
        statistics are all unchanged, so the SAME checkpoints are re-scored
        with no retraining and no re-audit. Run it twice with and without
        --keep-only-excluded and report both halves.

    (b) REMOVE FROM THE STORE -- only if you decide it BEFORE the split
        exists:

          data:
            exclude_trajectories: %s

        This changes n_traj, which RENUMBERS every trajectory. A split frozen
        against the old store then refers to different trajectories -- the
        out-of-range ones raise, and the in-range ones do not, which is worse.
        So it invalidates the split, the statistics and every checkpoint, and
        prepare_split and the audit must both be re-run.

    Do NOT set (b) and then run (a). scripts/filter_split.py checks for it and
    refuses, but the check only exists because this is easy to do by accident.
""" % (out, out))


if __name__ == "__main__":
    main()
