#!/usr/bin/env python
"""
Measure a dataset's predictable window and its dominant period — no GPU, no
training, no checkpoint.

WHY THIS RUNS FIRST
-------------------
The single most expensive mistake this project has already paid for is
interpreting an error-vs-horizon curve without knowing where the data stops
carrying information.  §9 of the handover puts it plainly: "a flat error curve
means nothing without a climatology baseline".  That lesson has a cheaper
corollary — most of what a climatology baseline eventually tells you is already
visible in the data, before a single model is trained.

    E_pers(h) = || z(u(t+h)) - z(u(t)) ||_2 / || z(u(t+h)) ||_2

is exactly the persistence baseline of §25, computed on z-scored fields, and it
is the *upper* envelope of what any forecaster faces: it is small while the
flow still remembers t, and it saturates near 1 once u(t+h) is uncorrelated
with u(t), because in z-space a target of unit norm and an uncorrelated
predictor of unit norm differ by about sqrt(2)... which is why the number to
watch is where it crosses 0.3, not where it reaches 1.

What that buys, concretely:

  * **Whether `eval_horizons` is honest.**  If E_pers is already 0.9 at h = 8,
    a grid running to 512 is thirteen columns of climatology and one column of
    physics.  RealPDEBench combustion is that case (T_pred = 4-6 frames), and
    the paper had to spend a whole subsection explaining it after the fact.
  * **Whether Cylinder does what §8 of the handover hopes.**  "What is actually
    missing is a chaotic flow sampled finely enough that h = 32-128 sits inside
    the predictable window."  This script answers that in minutes, for the cost
    of reading a few hundred frames.  If it does, extend `eval_horizons` to
    1024 -- 3990 frames support it -- and the headline claim can be made
    without the climatology caveat.  If it does not, the horizon ceiling is a
    property of the models, and that is a bigger finding than the dataset.
  * **A sanity check on `dt`.**  The dominant period is reported in frames and
    in seconds.  If the seconds are physically absurd, `data.dt` is wrong, and
    every "T_pred in milliseconds" number in the paper is wrong with it.

Nothing here is a substitute for the trained baselines.  Persistence is not
climatology, and the two answer different questions.  This is a
cheap-and-early instrument, not evidence for the paper.

Usage
-----
    python scripts/probe_timescales.py --config configs/cylinder.yaml
    python scripts/probe_timescales.py --config configs/gray_scott.yaml \
        --n_traj 12 --n_anchors 40 --out results/probe/gray_scott.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from common import base_parser, resolve, json_default          # noqa: E402

from data import apply_dataset_defaults                        # noqa: E402
from data.store import build_store                             # noqa: E402


DEFAULT_LAGS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256,
                384, 512, 768, 1024]


# ---------------------------------------------------------------------------

def sample_stats(store, trajs: Sequence[int], n_frames: int, rng
                 ) -> tuple:
    """Per-channel (mean, std) from frames spread over `trajs`."""
    s = np.zeros(store.n_channels, np.float64)
    ss = np.zeros(store.n_channels, np.float64)
    n = 0
    per = max(1, n_frames // max(len(trajs), 1))
    for t in trajs:
        idx = np.unique(np.linspace(0, store.T - 1, per).astype(int))
        block = store.read(t, idx.tolist())                    # (n,H,W,C)
        flat = block.reshape(-1, store.n_channels).astype(np.float64)
        s += flat.sum(0)
        ss += (flat ** 2).sum(0)
        n += flat.shape[0]
    mu = s / n
    sd = np.sqrt(np.maximum(ss / n - mu ** 2, 1e-12))
    return mu.astype(np.float32), sd.astype(np.float32)


def persistence_curve(store, trajs, lags, n_anchors, mu, sd, rng
                      ) -> Dict[int, np.ndarray]:
    """E_pers per lag, per channel, averaged over anchors."""
    lags = [h for h in lags if h < store.T - 1]
    acc = {h: [] for h in lags}
    eps = 1e-8
    for t in trajs:
        h_max = max(lags)
        hi = store.T - 1 - h_max
        if hi <= 0:
            continue
        anchors = np.unique(np.linspace(0, hi, n_anchors).astype(int))
        for t0 in anchors:
            want = [int(t0)] + [int(t0) + h for h in lags]
            block = store.read(t, want)
            z = (block - mu) / sd
            a = z[0].reshape(-1, store.n_channels)
            for i, h in enumerate(lags, start=1):
                b = z[i].reshape(-1, store.n_channels)
                num = np.linalg.norm(a - b, axis=0)
                den = np.linalg.norm(b, axis=0) + eps
                acc[h].append(num / den)
    return {h: np.mean(np.stack(v), axis=0) for h, v in acc.items() if v}


def crossing(lags: Sequence[int], vals: Sequence[float], level: float
             ) -> Optional[float]:
    """First h where the curve reaches `level`, linearly interpolated."""
    lags, vals = list(lags), list(vals)
    if vals[0] >= level:
        return float(lags[0])
    for i in range(1, len(lags)):
        if vals[i] >= level:
            v0, v1 = vals[i - 1], vals[i]
            h0, h1 = lags[i - 1], lags[i]
            if v1 == v0:
                return float(h1)
            return float(h0 + (h1 - h0) * (level - v0) / (v1 - v0))
    return None


def amplitude_profile(store, trajs, mu, sd, n_probe: int = 40):
    """||z(u(t))|| per channel as a function of t, normalised by its own median.

    THE FAILURE THIS CATCHES
    ------------------------
    E_field is a RELATIVE error, ||pred - target|| / ||target||, so a target
    whose normalised norm is near zero sends it to infinity no matter how good
    the prediction is.  A dataset that starts from rest has exactly that: for
    the first N frames the velocity IS the global mean, z(u) is ~0, and every
    method scores badly -- including the climatology baselines, which is the
    tell, because a climatology cannot "diverge".

    That is not hypothetical.  The first Rayleigh-Benard run reported
    E_field(h=1) = 13.7 for AR, 23.2 for the direct model and 437 for the
    climatology, while the buoyancy channel alone sat at a perfectly healthy
    0.12 rising smoothly to 1.36.  All of it came from velocity and pressure
    being ~0 over the spin-up, combined with T = 200 and h_max = 128 leaving
    t0 in [3, 71) -- so every evaluation anchor landed inside the transient.

    Read the ratio this prints.  Where a channel is far below 1 the relative
    error is not measuring the model, and `traj_cut` should drop those frames.
    """
    n_probe = min(n_probe, store.T)
    ts = np.unique(np.linspace(0, store.T - 1, n_probe).astype(int))
    block = (store.read(int(trajs[0]), ts.tolist()) - mu) / sd
    amp = np.linalg.norm(block.reshape(len(ts), -1, store.n_channels), axis=1)
    for t in trajs[1:4]:
        b = (store.read(int(t), ts.tolist()) - mu) / sd
        amp = amp + np.linalg.norm(b.reshape(len(ts), -1, store.n_channels),
                                   axis=1)
    amp /= min(len(trajs), 4)
    med = np.median(amp, axis=0) + 1e-12
    rel = amp / med
    return ts, rel


def report_amplitude(store, ts, rel, K: int, h_max: int, floor: float = 0.5):
    print(f"\n[2b] FIELD AMPLITUDE vs TIME   ||z(u(t))|| / median_t ||z(u(t))||")
    print( "     E_field divides by this. Where it is << 1, a relative error "
           "is not\n     measuring the model -- it is dividing by nothing.")
    names = [n[:11] for n in store.channel_names]
    print("     " + "".join(f"{'t':>6}") + "".join(f"{n:>13}" for n in names))
    step = max(1, len(ts) // 16)
    for i in range(0, len(ts), step):
        row = "".join(f"{rel[i, c]:>13.2f}" for c in range(store.n_channels))
        print(f"     {ts[i]:>6}{row}")

    worst = rel.min(axis=0)
    bad = [c for c in range(store.n_channels) if worst[c] < floor]
    if not bad:
        print(f"\n     No channel drops below {floor:g}x its median. "
              f"Nothing to cut.")
        return None
    # last frame at which ANY channel is still below the floor
    below = (rel < floor).any(axis=1)
    idx = np.where(below)[0]
    cut = int(ts[idx.max()]) + 1 if len(idx) else 0
    head = idx.max() < len(ts) // 2
    print(f"\n     [!] {[store.channel_names[c] for c in bad]} fall below "
          f"{floor:g}x the median.")
    if head:
        print(f"     This is a SPIN-UP TRANSIENT in the first {cut} frames "
              f"of {store.T}.")
        print(f"     Set  data.traj_cut: {cut}  and re-run prepare_split and "
              f"the audit.")
        t_left = store.T - cut
        print(f"     That leaves T = {t_left}. Anchors then run over "
              f"[{K - 1}, {t_left - 1} - h_max), so keep")
        print(f"     max(eval_horizons) <= {max(8, (t_left - K) // 2)} or the "
              f"anchors bunch back into the head")
        print(f"     of what remains -- which is the other half of the same "
              f"mistake.")
    else:
        print(f"     The low-amplitude frames are NOT at the head "
              f"(last at t = {cut - 1} of {store.T}), so\n     `traj_cut` is "
              f"not the fix. Check whether the field genuinely goes quiescent "
              f"(run\n     scripts/screen_stationary.py) or whether one "
              f"channel is a gauge with no scale.")
    return cut


def dominant_period(store, trajs, mu, sd, rng, n_probe: int = 24,
                    n_frames: int = 1024) -> Dict[str, float]:
    """Peak of the temporal power spectrum at fixed spatial probe points."""
    n_frames = min(int(n_frames), store.T)
    idx = list(range(n_frames))
    peaks, means = [], []
    for t in trajs[:4]:
        block = store.read(t, idx)                             # (T,H,W,C)
        z = (block - mu) / sd
        T, H, W, C = z.shape
        ys = rng.integers(0, H, n_probe)
        xs = rng.integers(0, W, n_probe)
        sig = z[:, ys, xs, :].reshape(T, -1)                   # (T, n_probe*C)
        sig = sig - sig.mean(0, keepdims=True)
        # Hann window: the record is not periodic, and leakage from the
        # rectangular window puts a spurious peak at the lowest bin, which is
        # exactly where a slow drift would also sit.
        w = np.hanning(T)[:, None]
        P = np.abs(np.fft.rfft(sig * w, axis=0)) ** 2
        f = np.fft.rfftfreq(T, d=1.0)                          # cycles / frame
        P[0] = 0.0                                             # drop DC
        k = int(np.argmax(P.sum(axis=1)))
        if f[k] > 0:
            peaks.append(1.0 / f[k])
        # spectral centroid as a robustness check on a broadband flow
        w_ = P.sum(axis=1)
        if w_.sum() > 0 and (f * w_).sum() > 0:
            means.append(1.0 / ((f * w_).sum() / w_.sum()))
    if not peaks:
        return {}
    return {"period_frames_peak": float(np.median(peaks)),
            "period_frames_centroid": float(np.median(means)) if means else None}


# ---------------------------------------------------------------------------

def main():
    ap = base_parser("Measure the predictable window and dominant period")
    ap.add_argument("--n_traj", type=int, default=8,
                    help="trajectories to probe (spread over the split)")
    ap.add_argument("--n_anchors", type=int, default=24,
                    help="anchors per trajectory")
    ap.add_argument("--n_stat_frames", type=int, default=96)
    ap.add_argument("--lags", type=int, nargs="+", default=None)
    ap.add_argument("--subset", default="train",
                    choices=["train", "val", "test", "all"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = resolve(args)
    dataset_name = cfg.get("dataset_name", "realpde_combustion")
    cfg = apply_dataset_defaults(dataset_name, cfg)
    rng = np.random.default_rng(int(cfg.get("seed", 42)))

    store = build_store(cfg)
    print("=" * 72)
    print(f"TIMESCALE PROBE — {dataset_name}")
    print("=" * 72)
    print(f"\n[1] Store\n    {store.summary()}")

    pool = list(range(store.n_traj))
    sp_path = cfg.get("split_path")
    if args.subset != "all" and sp_path and os.path.exists(sp_path):
        from data.splits import Split
        sp = Split.load(sp_path)
        pool = {"train": sp.train, "val": sp.val, "test": sp.test}[args.subset]
        print(f"    probing the '{args.subset}' split ({len(pool)} "
              f"trajectories)")
    elif args.subset != "all":
        print(f"    no split at {sp_path}; probing all trajectories")
    trajs = [pool[i] for i in
             np.unique(np.linspace(0, len(pool) - 1,
                                   min(args.n_traj, len(pool))).astype(int))]

    mu, sd = sample_stats(store, trajs, args.n_stat_frames, rng)
    print(f"\n[2] Per-channel statistics from {args.n_stat_frames} frames")
    for c, nm in enumerate(store.channel_names):
        print(f"    {nm:<24} mean={mu[c]:+.4e}  std={sd[c]:.4e}")

    ts_amp, rel_amp = amplitude_profile(store, trajs, mu, sd)
    cut = report_amplitude(store, ts_amp, rel_amp,
                           int(cfg.get("history_len", 4)),
                           int(cfg.get("h_max_train", 128)))

    lags = [h for h in (args.lags or DEFAULT_LAGS) if h < store.T - 1]
    curve = persistence_curve(store, trajs, lags, args.n_anchors, mu, sd, rng)
    lags = sorted(curve)
    mean_curve = [float(curve[h].mean()) for h in lags]

    print(f"\n[3] Persistence error E_pers(h), z-scored, "
          f"{len(trajs)} trajectories x {args.n_anchors} anchors")
    print(f"    {'h':>6}{'seconds':>12}{'E_pers':>10}   per channel")
    for h, v in zip(lags, mean_curve):
        per = "  ".join(f"{store.channel_names[c][:10]}={curve[h][c]:.2f}"
                        for c in range(store.n_channels))
        print(f"    {h:>6}{h * store.dt:>12.4g}{v:>10.3f}   {per}")

    cross = {f"h_at_E{lv:g}": crossing(lags, mean_curve, lv)
             for lv in (0.1, 0.3, 0.5, 0.9)}
    print(f"\n[4] Predictable window (persistence, threshold = the §45 "
          f"E = 0.3 used for T_pred)")
    for k, v in cross.items():
        if v is None:
            print(f"    {k:<12} not reached inside h <= {lags[-1]}")
        else:
            print(f"    {k:<12} h = {v:8.2f} frames = {v * store.dt:.4g} s")

    per = dominant_period(store, trajs, mu, sd, rng)
    if per:
        print(f"\n[5] Dominant temporal period")
        p = per["period_frames_peak"]
        print(f"    spectral peak      {p:8.1f} frames = {p * store.dt:.4g} s")
        if per.get("period_frames_centroid"):
            pc = per["period_frames_centroid"]
            print(f"    spectral centroid  {pc:8.1f} frames = "
                  f"{pc * store.dt:.4g} s")
        print("    A period much LONGER than h_max means the horizon axis "
              "lives inside\n    one cycle -- the regime where direct-time "
              "prediction has something to\n    predict. A period SHORTER "
              "than h_max means the long-horizon columns\n    are measuring "
              "phase, and phase is what saturates first.")

    # ---- recommendation -------------------------------------------------
    h03 = cross["h_at_E0.3"]
    h09 = cross["h_at_E0.9"]
    print(f"\n[6] Reading it")
    if h03 is None:
        print(f"    E_pers never reaches 0.3 within h <= {lags[-1]}. The flow "
              f"is far more\n    predictable than any horizon on the grid; "
              f"EXTEND eval_horizons (this\n    dataset can carry longer "
              f"ones) and check for a stationary or\n    near-stationary "
              f"record before celebrating.")
    elif h09 is not None and h09 <= 8:
        print(f"    E_pers reaches 0.9 by h = {h09:.1f}. Everything past that "
              f"is the\n    climatology-dominated regime described in §5 of "
              f"docs/RESULTS.md.\n    Keep the long horizons -- they are an "
              f"honest measurement -- but do\n    not expect them to "
              f"discriminate between models, and make sure both\n    "
              f"climatology baselines are on every figure.")
    else:
        hi = int(h09 if h09 is not None else lags[-1])
        print(f"    E_pers crosses 0.3 at h = {h03:.1f} and 0.9 at "
              f"{h09 if h09 else float('nan'):.1f}.\n    The discriminative "
              f"range is roughly h = 1 .. {hi}; put most of the\n    "
              f"evaluation grid inside it, and set h_max_train to at least "
              f"{hi}\n    (A5 measured that larger is better and that models "
              f"do not\n    extrapolate past their training range).")

    payload = {
        "dataset": dataset_name,
        "store": store.summary(),
        "dt": store.dt, "T": store.T, "n_traj": store.n_traj,
        "trajectories_probed": [int(t) for t in trajs],
        "channel_names": list(store.channel_names),
        "mean": mu, "std": sd,
        "lags": lags,
        "E_pers_mean": mean_curve,
        "E_pers_per_channel": {int(h): curve[h] for h in lags},
        "crossings_frames": cross,
        "crossings_seconds": {k: (v * store.dt if v is not None else None)
                              for k, v in cross.items()},
        "dominant_period": per,
        "amplitude_t": ts_amp, "amplitude_rel": rel_amp,
        "suggested_traj_cut": cut,
    }
    out = args.out or os.path.join(
        cfg.get("results_dir", "./results"), "probe",
        f"timescales_{dataset_name}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fp:
        json.dump(payload, fp, indent=2, default=json_default)
    print(f"\n    -> {out}")


if __name__ == "__main__":
    main()
