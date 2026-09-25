#!/usr/bin/env python
"""
Synthetic reacting-flow-shaped data for the smoke test.

This is plumbing, not physics.  It exists so the whole pipeline — audit, split,
normalisation, both dataset classes, all three models, the semigroup losses, the
horizon evaluator and the figures — can be exercised end to end on a laptop in
about two minutes, before anything is queued on the A800s.

What it does reproduce, deliberately, are the properties the pipeline has to
cope with:

  * channel names in RealPDEBench's style, so `data/channels.py` grouping and
    the transform heuristics are actually tested;
  * dynamic ranges spanning decades (OH ~ 1e-6, temperature ~ 1e3), so the
    per-channel transforms are exercised rather than bypassed;
  * a drifting, rotating front with a decorrelating turbulent component, so
    error genuinely grows with horizon and the curves are not flat.

Usage:
    python scripts/make_synthetic_data.py --out /tmp/smoke_data
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from common import REPO_ROOT  # noqa: F401  (path bootstrap)

# Deliberately includes the pathologies the real RealPDEBench audit turned up,
# so the audit's fixes are verified rather than assumed:
#   * Pressure = Absolute_Pressure - 101325  (affine duplicate; a z-score erases
#     the offset and the two become the same channel)
#   * Velocity_Magnitude = sqrt(u^2+v^2)     (deterministic function of channels
#     already present)
#   * Mole_Fraction_of_NH2                   (sparse: confined to a thin layer,
#     so median(|x|) sits in the noise floor -- the case that produced a
#     sigma = 0.027 dead channel under the old symlog scaling)
#   * negative mole fractions and sub-zero temperature (resampling undershoot),
#     which rule out log and Box-Cox
CHANNELS = [
    "Absolute_Pressure",
    "Chemistry_Heat_Release_Rate",
    "Mole_Fraction_of_CH4",
    "Mole_Fraction_of_H2O",
    "Mole_Fraction_of_NH2",
    "Mole_Fraction_of_OH",
    "Pressure",
    "Temperature",
    "Velocity[i]",
    "Velocity[j]",
    "Velocity_Magnitude",
]


def make_trajectory(T: int, H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    y, x = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")

    # trajectory-specific "operating point" — this is what makes a held-out
    # trajectory genuinely held out rather than a time shift of a seen one
    swirl = rng.uniform(0.6, 1.4)
    drift = rng.uniform(0.1, 0.3)
    phi = rng.uniform(0.0, 2 * np.pi)
    peak_T = rng.uniform(1800.0, 2400.0)

    # a few random turbulent modes with their own decorrelation rates
    n_modes = 6
    kx = rng.integers(2, 9, n_modes)
    ky = rng.integers(2, 9, n_modes)
    ph = rng.uniform(0, 2 * np.pi, n_modes)
    om = rng.uniform(0.3, 1.6, n_modes)
    amp = rng.uniform(0.02, 0.08, n_modes)

    out = np.zeros((T, H, W, len(CHANNELS)), dtype=np.float32)
    for t in range(T):
        s = t / 12.0
        cx = 0.5 + drift * 0.25 * np.cos(swirl * s + phi)
        cy = 0.5 + drift * 0.25 * np.sin(swirl * s + phi)
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        turb = np.zeros_like(r)
        for m in range(n_modes):
            turb += amp[m] * np.sin(2 * np.pi * (kx[m] * x + ky[m] * y)
                                    + ph[m] + om[m] * s)

        front = 0.18 + 0.03 * np.sin(2.0 * swirl * s + phi) + 0.35 * turb
        flame = np.exp(-((r - front) ** 2) / (2 * 0.05 ** 2))

        # resampling undershoot near steep fronts: mole fractions and
        # temperature both dip below zero, exactly as the real data does
        jitter = 0.004 * rng.standard_normal(flame.shape)

        temperature = 300.0 + peak_T * flame + 60.0 * turb - 400.0 * jitter
        hrr = 1.0e7 * flame ** 2 * (1.0 + 0.3 * turb) - 2.0e5 * jitter
        oh = 3.0e-3 * flame ** 1.5 + 3.0e-5 * jitter
        ch4 = 9.0e-2 * np.clip(1.0 - flame, 0.0, None) + 9.0e-4 * jitter
        h2o = 1.2e-1 * flame + 1.2e-3 * jitter
        # NH2 lives only in a thin layer around the front: sparse, signed
        nh2 = 4.7e-4 * np.exp(-((r - front) ** 2) / (2 * 0.012 ** 2)) \
            + 4.0e-6 * jitter

        abs_p = 1.01325e5 + 250.0 * turb
        pressure = abs_p - 101325.0          # affine duplicate, by construction
        u = -swirl * (y - cy) + 0.4 * turb
        v = swirl * (x - cx) + 0.4 * turb
        vmag = np.sqrt(u ** 2 + v ** 2)      # derived, by construction

        out[t] = np.stack([abs_p, hrr, ch4, h2o, nh2, oh, pressure,
                           temperature, u, v, vmag],
                          axis=-1).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/smoke_data")
    ap.add_argument("--n_traj", type=int, default=8)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--dt", type=float, default=2.5e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.n_traj):
        traj = make_trajectory(args.T, args.H, args.W, rng)
        np.save(os.path.join(args.out, f"traj_{i:03d}.npy"), traj)
        print(f"  traj_{i:03d}.npy  {traj.shape}  "
              f"[{traj.min():.3e}, {traj.max():.3e}]")

    with open(os.path.join(args.out, "channels.json"), "w") as fp:
        json.dump({"channel_names": CHANNELS, "dt": args.dt}, fp, indent=2)
    print(f"\nWrote {args.n_traj} trajectories to {args.out}")


if __name__ == "__main__":
    main()
