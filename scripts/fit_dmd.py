#!/usr/bin/env python
"""
Fit the POD-DMD baseline (§3).

Run this in Phase 1, before any neural training finishes.  It is cheap, it takes
minutes, and it gives the project a floor that is not a neural network: POD-DMD
is itself a direct finite-time model (a_{t+h} = A^h a_t, evaluated through the
eigendecomposition at h-independent cost), so it tests the FORMULATION.  §3 is
blunt about the implication — if the direct operator cannot beat POD-DMD, the
problem is the formulation or the data pipeline, and no amount of architecture
will fix it.

Memory: the snapshot matrix is n_snapshots x (H*W*C) in float32.  At 128x128x15
that is 0.98 MB per snapshot, so the default 1500 snapshots is ~1.5 GB and the
Gram matrix is 1500^2 doubles (18 MB).  Raise --n_snapshots if you have the RAM;
the POD spectrum saturates well before it matters.

Usage:
    python scripts/fit_dmd.py --config configs/dt_fno.yaml
    python scripts/fit_dmd.py --config configs/dt_fno.yaml --rank 256 \
        --n_snapshots 3000 --out artifacts/pod_dmd_r256.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from common import base_parser, resolve, set_seed                     # noqa: E402

from data.store import build_store                                     # noqa: E402
from data.splits import Split                                          # noqa: E402
from data.transforms import Normalizer                                 # noqa: E402
from data.realpde import resolve_channels                              # noqa: E402
from models.baselines import PODDMD                                    # noqa: E402


def main():
    ap = base_parser("Fit POD-DMD (§3)")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--n_snapshots", type=int, default=1500,
                    help="total snapshots across all training trajectories")
    ap.add_argument("--max_traj", type=int, default=10)
    ap.add_argument("--no_clip_eigs", action="store_true",
                    help="keep |lambda| > 1; see the docstring in baselines.py")
    ap.add_argument("--rank_sweep", type=int, nargs="+", default=None,
                    help="report captured POD energy for several ranks before "
                         "fitting, so an under-resolved basis is visible rather "
                         "than inferred from a bad error curve")
    ap.add_argument("--out", default="artifacts/pod_dmd.npz")
    args = ap.parse_args()
    cfg = resolve(args)
    set_seed(int(cfg["seed"]))

    store = build_store(cfg)
    ch_idx, ch_names = resolve_channels(store, cfg.get("channels"))
    split = Split.load(cfg.get("split_path", "artifacts/split_realpde.json"))
    norm = Normalizer.load(cfg.get("norm_stats_path", "artifacts/norm_stats.json"),
                           ch_names)

    trajs = split.train[: args.max_traj]
    per_traj = max(8, args.n_snapshots // len(trajs))
    # DMD needs CONSECUTIVE pairs, so snapshots are contiguous runs, not a
    # random scatter: a subsampled-in-time sequence would fit A for a stride of
    # k frames while the evaluation asks for stride 1.
    print("=" * 72)
    print("FIT POD-DMD")
    print("=" * 72)
    print(f"  trajectories : {trajs}")
    print(f"  {per_traj} consecutive frames each -> "
          f"{per_traj * len(trajs)} snapshots")

    rng = np.random.default_rng(int(cfg["seed"]))
    snapshots = []
    for t in trajs:
        lo = int(rng.integers(0, max(1, store.T - per_traj)))
        idx = list(range(lo, min(lo + per_traj, store.T)))
        raw = store.read(t, idx, ch_idx)
        snapshots.append(norm.forward(raw))
        print(f"    traj {t:>3}: frames [{idx[0]}, {idx[-1]}]")

    if args.rank_sweep:
        # POD energy is a property of the snapshots, not of the fit, so one
        # eigendecomposition answers every rank at once. E(h=1) for POD-DMD is
        # dominated by PROJECTION error, and relative L2 from truncation is
        # about sqrt(1 - captured); 0.95 error implies ~10% captured energy,
        # which is a basis problem and no amount of DMD tuning will fix it.
        F = int(np.prod(snapshots[0].shape[1:]))
        X = np.concatenate([s.reshape(s.shape[0], F) for s in snapshots], axis=0)
        X = X - X.mean(axis=0)
        w = np.linalg.eigvalsh((X @ X.T).astype(np.float64))[::-1]
        w = np.clip(w, 0, None)
        tot = w.sum()
        print("\n  POD energy vs rank (on the fitting snapshots)")
        print(f"    {'rank':>6}{'captured':>11}{'implied rel-L2 floor':>24}")
        for r in sorted(args.rank_sweep):
            r = min(r, len(w))
            cap = float(w[:r].sum() / max(tot, 1e-30))
            print(f"    {r:>6}{100*cap:>10.2f}%{np.sqrt(max(1-cap,0)):>24.3f}")
        print()

    model = PODDMD(rank=args.rank, subtract_mean=True,
                   clip_eigs=not args.no_clip_eigs)
    model.fit(snapshots, verbose=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    model.save(args.out)
    print(f"\n  -> {args.out}")
    print("\nNext: evaluate it on the same anchors as the neural models —")
    print(f"    python scripts/evaluate_horizon.py --config {args.config} "
          f"--model pod_dmd --dmd {args.out}")


if __name__ == "__main__":
    main()
