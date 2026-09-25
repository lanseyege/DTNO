#!/usr/bin/env python
"""
Phase 0 — Data Audit (§36).

Everything downstream depends on this script having been run, and `build_data`
refuses to start without its output.  That is intentional: the two ways this
project can produce a beautiful, meaningless result are a leaked split and
normalisation statistics fitted on the test set, and both are prevented here or
not at all.

What it does:

  1. verifies the store opens and reports (N, T, H, W, C) — for RealPDEBench
     combustion numerical that should be (30, 2001, 128, 128, 15) (§36.3);
  2. per channel: min / max / mean / std / dynamic range / quantiles /
     fraction of exact zeros / negativity, on TRAINING TRAJECTORIES ONLY;
  3. assigns each channel a group (§26) and a transform (§11), showing its
     reasoning so you can override anything that looks wrong;
  4. freezes the 20/5/5 trajectory-held-out split to JSON (§9);
  5. computes and freezes the normalisation statistics to JSON (§11.1);
  6. optionally plots per-channel histograms before and after the transform.

The single most useful thing it prints is the transform table.  Read it before
launching anything: a species channel that the heuristic left on `zscore`
because its name did not match will quietly dominate the loss, and a
heat-release channel that turns out to be strictly non-negative should be
`log1p`, not `symlog` (§11.4).

Usage:
    python scripts/audit_data.py --config configs/dt_fno.yaml
    python scripts/audit_data.py --config configs/dt_fno.yaml --plots --force
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np

from common import (REPO_ROOT, base_parser, resolve, json_default)  # noqa: E402

from data.store import build_store                                   # noqa: E402
from data.splits import load_or_create_split                          # noqa: E402
from data.channels import (channel_group, default_transform,          # noqa: E402
                           group_channels, reaction_zone_channel,
                           temperature_channel, find_channel)
from data.transforms import ChannelStat, Normalizer, fit_boxcox_lambda  # noqa: E402
from data.realpde import resolve_channels                             # noqa: E402


# ---------------------------------------------------------------------------

def scan_channel(store, trajs, c_idx: int, t_stride: int,
                 max_traj: int) -> Dict[str, float]:
    """Streaming statistics for one channel over the training trajectories."""
    n = 0
    s1 = 0.0
    s2 = 0.0
    cmin, cmax = np.inf, -np.inf
    n_zero = 0
    n_neg = 0
    sample: List[np.ndarray] = []

    for traj in trajs[:max_traj]:
        block = store.read_traj_channel(traj, c_idx, t_stride=t_stride)
        b = block.astype(np.float64).ravel()
        n += b.size
        s1 += b.sum()
        s2 += (b ** 2).sum()
        cmin = min(cmin, float(b.min()))
        cmax = max(cmax, float(b.max()))
        n_zero += int((b == 0).sum())
        n_neg += int((b < 0).sum())
        step = max(1, b.size // 20000)
        sample.append(b[::step])

    mean = s1 / n
    var = max(s2 / n - mean ** 2, 0.0)
    sm = np.concatenate(sample)
    q = np.quantile(sm, [0.001, 0.01, 0.5, 0.99, 0.999])

    # Dynamic range, measured robustly.
    #
    # The naive max(x>0)/min(x>0) is worse than useless on this data: a single
    # 1e-19 interpolation jitter value in the product region sets the
    # denominator and the channel reports a range of 1e18. That number then
    # drives the transform choice, so a bad statistic silently destroys a
    # channel. Quantile ratios of |x| cannot be moved by one outlier.
    absx = np.abs(sm)
    a50, a90, a99, a999 = np.quantile(absx, [0.5, 0.9, 0.99, 0.999])
    tiny = max(a999 * 1e-12, 1e-300)
    dyn_robust = float(a999 / max(a50, tiny))
    dyn_bulk = float(a99 / max(a90, tiny))
    # sparsity: fraction of the domain that is essentially empty. A sparse
    # channel (a species living only in a thin reaction layer) has a meaningless
    # q999/q50 because q50 sits in the noise floor, not in the signal.
    frac_tiny = float((absx < 1e-3 * a999).mean())
    return {
        "min": cmin, "max": cmax, "mean": mean, "std": float(np.sqrt(var)),
        "q001": float(q[0]), "q01": float(q[1]), "median": float(q[2]),
        "q99": float(q[3]), "q999": float(q[4]),
        "abs_q50": float(a50), "abs_q99": float(a99), "abs_q999": float(a999),
        "frac_zero": n_zero / n, "frac_negative": n_neg / n,
        "frac_tiny": frac_tiny,
        "robust_dynamic_range": dyn_robust, "bulk_dynamic_range": dyn_bulk,
        "n_values": int(n),
        "_sample": sm,
    }


def choose_transform(name: str, st: Dict[str, float], override: str = "") -> str:
    """Conservative, measurement-driven default. The config has the last word.

    Deliberately much less clever than it used to be. No automatic rule can
    separate "this species genuinely spans six decades" from "this species is
    sparse and q50 sits in the interpolation noise", and getting it wrong costs
    a channel: a badly scaled symlog compressed NH3 down to sigma = 0.027, i.e.
    a channel the model cannot learn at all.

    So: z-score unless the channel is strictly non-negative AND robustly
    multi-decade, which is the one case where log is unambiguously right. The
    audit prints every diagnostic the decision would need, and anything else is
    an explicit `data.channel_transforms` entry in the config -- a recorded
    decision rather than an emergent one.

    Note for this dataset in particular: resampling onto the 128x128 grid puts
    Gibbs undershoot next to steep fronts, so mole fractions and temperature
    both go negative. log and Box-Cox are therefore off the table regardless of
    what REALM's pipeline does (SS11.3), and that is a property of the data, not
    a preference.
    """
    if override:
        return override
    if st["frac_negative"] == 0.0 and st["robust_dynamic_range"] > 1e3:
        return "log"
    return "zscore"


def build_stat(name: str, st: Dict[str, float], transform: str,
               scale_quantile: float = 0.5) -> ChannelStat:
    sample = st["_sample"]
    cs = ChannelStat(name=name, transform=transform,
                     raw_min=st["min"], raw_max=st["max"],
                     raw_mean=st["mean"], raw_std=st["std"])
    pos = sample[sample > 0]
    if transform == "log":
        cs.eps = float(max(pos.min() * 1e-3, 1e-30)) if pos.size else 1e-12
    elif transform in ("log1p", "symlog"):
        # y = sign(x) * log1p(|x| / s): linear while |x| << s, logarithmic above.
        # So `s` decides which part of the field is compressed, and the choice is
        # a genuine trade-off rather than a bug to be fixed. Measured on species
        # data with a 3-decade operating-point spread between trajectories:
        #
        #   s = median(|x|)  (q=0.5, default)
        #       s lands in the noise floor, so the transform is effectively `log`
        #       over the whole signal. Multiplicative spread between trajectories
        #       is compressed: per-trajectory sigma stayed in [0.40, 1.52].
        #       Cost: the near-zero noise floor is stretched to O(1) alongside
        #       the flame signal.
        #   s = q99(|x|)
        #       Puts the bulk in the linear regime and compresses only the peak,
        #       which is the textbook symlog intent. But `s` is then set by the
        #       hottest trajectory, and weaker ones collapse into the linear
        #       regime: per-trajectory sigma spread to [0.001, 1.93] -- dead
        #       channels for the low-magnitude operating points.
        #
        # The median default is therefore kept. Raise `symlog_scale_quantile`
        # only for a channel whose magnitude is comparable across trajectories.
        absx = np.abs(sample)
        q = float(scale_quantile)
        if q >= 0.999:
            base = float(st.get("abs_q999", np.quantile(absx, 0.999)))
        elif q >= 0.99:
            base = float(st.get("abs_q99", np.quantile(absx, 0.99)))
        else:
            base = float(np.quantile(absx, q))
        cs.scale = max(base, 1e-30)
    elif transform == "boxcox":
        cs.shift = float(max(0.0, -sample.min() + 1e-6))
        cs.lam = fit_boxcox_lambda(sample, cs.shift)

    y = cs.pre(sample.astype(np.float64))
    cs.mean = float(np.mean(y))
    cs.std = float(np.std(y))
    if not np.isfinite(cs.std) or cs.std < 1e-12:
        print(f"    [warn] {name}: sigma = {cs.std:.3e} after '{transform}'. "
              f"This channel is effectively constant; it will contribute noise "
              f"to the loss. Consider dropping it via data.channels.")
        cs.std = 1.0
    return cs


# ---------------------------------------------------------------------------
# Redundancy detection
# ---------------------------------------------------------------------------

def detect_redundancy(store, trajs, ch_idx, names, n_frames: int = 40,
                      n_traj: int = 3, thresh: float = 0.999) -> Dict[str, list]:
    """Find channels that are affine copies or exact functions of others.

    This exists because the first audit shipped a duplicated pressure channel
    and the only reason it was noticed is that two rows of the post-transform
    table happened to print identical numbers. That is luck, not a process.

    Two checks:

    1. Pairwise Pearson correlation on raw values. |r| > 0.999 means one channel
       is an affine function of the other, and a z-score erases the offset and
       the scale -- so after normalisation they are the SAME input, and the
       model sees the same field twice. The cost is not wasted capacity: the
       duplicated quantity gets double weight in the loss and double weight in
       the channel-averaged E_field of SS25, and its group is diluted in SS26.

    2. Velocity magnitude: is `Velocity_Magnitude` really sqrt(u^2+v^2+w^2)? If
       so it is a deterministic function of channels already present and belongs
       out of the channel set. Worth measuring rather than assuming -- a
       negative minimum, as this dataset reports, is not what a modulus does.
    """
    rng = np.random.default_rng(0)
    cols = []
    for t in trajs[:n_traj]:
        idx = sorted(rng.choice(store.T, size=min(n_frames, store.T),
                                replace=False).tolist())
        blk = store.read(int(t), idx, ch_idx)                # (n, H, W, C)
        cols.append(blk.reshape(-1, len(ch_idx)))
    X = np.concatenate(cols, axis=0).astype(np.float64)
    step = max(1, X.shape[0] // 200000)                      # cap the cost
    X = X[::step]

    sd = X.std(axis=0)
    ok = sd > 0
    R = np.full((len(names), len(names)), np.nan)
    if ok.sum() > 1:
        R[np.ix_(ok, ok)] = np.corrcoef(X[:, ok], rowvar=False)

    dupes = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = R[i, j]
            if np.isfinite(r) and abs(r) > thresh:
                # y = a x + b, so the offset that a z-score would erase is explicit
                a = float(np.polyfit(X[:, i], X[:, j], 1)[0])
                b = float(np.polyfit(X[:, i], X[:, j], 1)[1])
                dupes.append({"a": names[i], "b": names[j], "r": float(r),
                              "slope": a, "offset": b})

    derived = []
    comps = [find_channel(names, r"Velocity\[i\]", r"^u$"),
             find_channel(names, r"Velocity\[j\]", r"^v$"),
             find_channel(names, r"Velocity\[k\]", r"^w$")]
    mag = find_channel(names, r"Velocity_Magnitude", r"velocity.*mag")
    comps = [c for c in comps if c >= 0]
    if mag >= 0 and len(comps) >= 2:
        pred = np.sqrt(sum(X[:, c] ** 2 for c in comps))
        num = np.linalg.norm(X[:, mag] - pred)
        den = np.linalg.norm(X[:, mag]) + 1e-30
        derived.append({"channel": names[mag],
                        "definition": f"sqrt(sum of {[names[c] for c in comps]}^2)",
                        "rel_error": float(num / den),
                        "n_components": len(comps),
                        "frac_negative": float((X[:, mag] < 0).mean())})
    return {"duplicates": dupes, "derived": derived}


def main():
    ap = base_parser("Phase 0 data audit (§36)")
    ap.add_argument("--force", action="store_true",
                    help="recompute the normalisation statistics. Does NOT "
                         "touch the frozen split -- see --redraw-split.")
    ap.add_argument("--redraw-split", dest="redraw_split", action="store_true",
                    help="DESTRUCTIVE. Delete and redraw the frozen split. Every "
                         "result produced before this point becomes "
                         "incomparable, so it is a separate flag from --force "
                         "on purpose.")
    ap.add_argument("--t_stride", type=int, default=10,
                    help="temporal subsampling for the statistics scan")
    ap.add_argument("--max_traj", type=int, default=12,
                    help="cap on training trajectories used for statistics")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--strata", default=None,
                    help="strata JSON from scripts/derive_strata.py; without it "
                         "the split is a seeded shuffle, which §9 does not ask for")
    args = ap.parse_args()
    cfg = resolve(args)

    print("=" * 72)
    print("PHASE 0 — DATA AUDIT")
    print("=" * 72)

    store = build_store(cfg)
    print(f"\n[1] Store\n    {store.summary()}")
    expect = cfg.get("expect_shape")
    got = (store.n_traj, store.T, store.H, store.W, store.n_channels)
    if expect and list(expect) != list(got):
        print(f"    [warn] expected {tuple(expect)}, found {got}")
    if store.n_traj == 30 and store.T == 2001 and store.n_channels == 15:
        print("    matches the RealPDEBench combustion numerical branch of §4")

    ch_idx, ch_names = resolve_channels(store, cfg.get("channels"))
    print(f"\n[2] Channels ({len(ch_names)} selected of {store.n_channels})")
    groups = group_channels(ch_names)
    for g, idx in groups.items():
        print(f"    {g:<10} {[ch_names[i] for i in idx]}")
    hrr = reaction_zone_channel(ch_names)
    tmp = temperature_channel(ch_names)
    print(f"    reaction-zone mask channel (§29): "
          f"{ch_names[hrr] if hrr >= 0 else 'NONE FOUND — §29 metrics disabled'}")
    print(f"    temperature channel (§30):        "
          f"{ch_names[tmp] if tmp >= 0 else 'NONE FOUND'}")

    # ---- redundancy (§26 weighting depends on this) ---------------------
    print("\n[2b] Redundancy check")
    red = detect_redundancy(store, list(range(store.n_traj)), ch_idx, ch_names)
    if red["duplicates"]:
        for d in red["duplicates"]:
            print(f"    [!] {d['a']} and {d['b']} are affine copies "
                  f"(r = {d['r']:.6f}, {d['b']} ~ {d['slope']:.4f} * {d['a']} "
                  f"{d['offset']:+.4g})")
        print("        A z-score erases the offset and the scale, so after")
        print("        normalisation these are the SAME channel. Drop one via")
        print("        data.channels, or that quantity gets double weight in the")
        print("        loss and in the channel-averaged E_field of §25.")
    else:
        print("    no affine duplicate pairs (|r| > 0.999)")
    for d in red["derived"]:
        v = d["rel_error"]
        verdict = ("IS the modulus -> drop it, it is a function of channels you"
                   " already have" if v < 0.01 else
                   "is NOT the modulus -> keep it, and find out what it is")
        print(f"    {d['channel']}: ||x - {d['definition']}|| / ||x|| = {v:.4f} "
              f"({100*d['frac_negative']:.1f}% negative) -> {verdict}")

    # ---- split ---------------------------------------------------------
    split_path = cfg.get("split_path", "artifacts/split_realpde.json")

    # --force used to delete the split as well as the statistics. That is how a
    # sampling-rate sweep, which legitimately needs fresh statistics for each
    # stride, silently redrew the trajectory split underneath itself -- and
    # redrew it WITHOUT the strata, because the sweep script had no reason to
    # pass --strata. Four strides then trained on a different, unstratified
    # split from every earlier experiment, and the frozen file on disk was
    # replaced, so subsequent runs would have inherited the wrong split too.
    #
    # Recomputing statistics and redrawing the split are unrelated operations
    # with wildly different blast radii. They get separate flags.
    if args.redraw_split and os.path.exists(split_path):
        print(f"    [!] --redraw-split: deleting {split_path}. Results produced "
              f"with the previous split are no longer comparable.")
        os.remove(split_path)
    strata = None
    if args.strata:
        with open(args.strata) as fp:
            strata = json.load(fp)["strata"]
        if len(strata) != store.n_traj:
            raise ValueError(f"{args.strata} has {len(strata)} labels for "
                             f"{store.n_traj} trajectories")
    split = load_or_create_split(split_path, store.n_traj, cfg, strata=strata)
    print(f"\n[3] Split (§9) -> {split_path}\n    {split.summary()}")
    if strata:
        from collections import Counter
        for subset, ts in (("train", split.train), ("val", split.val),
                           ("test", split.test)):
            c = Counter(strata[t] for t in ts)
            print(f"    {subset:<6} {dict(c)}")
        missing = set(strata) - set(strata[t] for t in split.test)
        if missing:
            print(f"    [!] regimes absent from TEST: {sorted(missing)}")
            print( "        the test set does not measure those operating points")
    elif not os.path.exists(args.strata or ""):
        print("    [!] Unstratified. §9 asks for stratification by fuel "
              "composition /\n        equivalence ratio. Run "
              "scripts/derive_strata.py and re-run with --strata\n"
              "        before training, because the split cannot be changed "
              "afterwards.")
    print(f"    {split.note}")
    print("    Test trajectories appear in NO training window, at any time "
          "offset.")

    # ---- statistics ----------------------------------------------------
    stats_path = cfg.get("norm_stats_path", "artifacts/norm_stats.json")
    if os.path.exists(stats_path) and not args.force:
        print(f"\n[4] {stats_path} exists; pass --force to recompute.")
        return

    print(f"\n[4] Per-channel statistics on TRAINING trajectories only "
          f"({min(len(split.train), args.max_traj)} of {len(split.train)}, "
          f"every {args.t_stride}th frame)")
    overrides = cfg.get("channel_transforms", {}) or {}
    scale_q = float(cfg.get("symlog_scale_quantile", 0.5))
    raw: Dict[str, Dict] = {}
    stats: List[ChannelStat] = []

    header = (f"    {'channel':<32}{'group':<11}{'min':>11}{'max':>11}"
              f"{'dyn.rng':>10}{'neg%':>7}{'sparse%':>9}  transform")
    print(header)
    print("    " + "-" * (len(header) - 4))
    for pos, c in enumerate(ch_idx):
        name = ch_names[pos]
        st = scan_channel(store, split.train, c, args.t_stride, args.max_traj)
        raw[name] = {k: v for k, v in st.items() if not k.startswith("_")}
        tr = choose_transform(name, st, overrides.get(name, ""))
        cs = build_stat(name, st, tr, scale_q)
        stats.append(cs)
        dyn = st["robust_dynamic_range"]
        dyn_s = "inf" if not np.isfinite(dyn) else f"{dyn:.1e}"
        print(f"    {name:<32}{channel_group(name):<11}{st['min']:>11.3e}"
              f"{st['max']:>11.3e}{dyn_s:>10}{100*st['frac_negative']:>6.1f}%"
              f"{100*st['frac_tiny']:>8.1f}%  {tr}")

    norm = Normalizer(stats)
    norm.save(stats_path, extra={
        "source": cfg.get("data_path"),
        "train_trajectories": split.train,
        "split_fingerprint": {"train": split.train, "val": split.val,
                              "test": split.test, "protocol": split.protocol},
        "t_stride": args.t_stride,
        "max_traj": args.max_traj,
        "dt": store.dt,
        "raw_statistics": raw,
        "redundancy": red,
    })
    print(f"\n    -> {stats_path}")

    # ---- sanity: normalised channels should be ~N(0,1) ------------------
    print("\n[5] Post-transform check across ALL training trajectories")
    print("    Reported per channel: sigma on each training trajectory, as")
    print("    min..max. One trajectory alone cannot tell 'the transform")
    print("    crushed this channel' apart from 'this operating point simply")
    print("    contains little of this species' -- the first shows low sigma")
    print("    EVERYWHERE, the second shows a wide spread across trajectories.")
    print("    The first is a bug to fix here; the second is physics, and the")
    print("    channel is still learnable from the trajectories that have it.")
    probe_t = list(range(0, min(store.T, 400), max(1, store.T // 40)))
    sig_tbl = {}
    for traj in split.train[:min(len(split.train), args.max_traj)]:
        z = norm.forward(store.read(traj, probe_t, ch_idx))
        for i_c, name in enumerate(ch_names):
            sig_tbl.setdefault(name, []).append(float(z[..., i_c].std()))

    dead, uneven, constant_on = [], [], {}
    print(f"\n    {'channel':<32}{'sigma min':>11}{'sigma max':>11}"
          f"{'spread':>9}  verdict")
    print("    " + "-" * 74)
    for name in ch_names:
        v = np.array(sig_tbl[name])
        lo, hi = float(v.min()), float(v.max())
        spread = hi / max(lo, 1e-12)
        if hi < 0.3:
            verdict, _ = "DEAD everywhere -- fix the transform", dead.append(name)
        elif lo < 0.05:
            # sigma ~ 0 on SOME trajectories and healthy on others: the channel
            # is identically zero there. That is a regime marker, not noise.
            zero_on = [int(t) for t, sv in
                       zip(split.train[:len(v)], v.tolist()) if sv < 0.05]
            constant_on[name] = zero_on
            verdict = f"CONSTANT on {len(zero_on)}/{len(v)} trajectories"
            uneven.append(name)
        elif lo < 0.3:
            verdict, _ = "uneven across operating points", uneven.append(name)
        elif hi > 3.0:
            verdict = "heavy-tailed -- check the transform"
        else:
            verdict = "ok"
        print(f"    {name:<32}{lo:>11.3f}{hi:>11.3f}{spread:>9.1f}x  {verdict}")

    if dead:
        print(f"\n    [!] DEAD: {dead}")
        print( "        Low sigma on every training trajectory means the")
        print( "        transform, not the physics. Set an explicit")
        print( "        data.channel_transforms entry (zscore is the safe")
        print( "        choice) and re-run with --force.")
    if constant_on:
        print("\n    [!] REGIME MARKERS -- identically zero on some trajectories:")
        for name, ts in constant_on.items():
            print(f"        {name}: zero on training trajectories {ts}")
        print( "        This is not a normalisation problem. A species that is")
        print( "        numerically absent in one trajectory and present in")
        print( "        another means the 30 trajectories span more than one")
        print( "        FUEL, and that is exactly the stratification variable")
        print( "        §9 asks the split to balance. Run:")
        print( "            python scripts/derive_strata.py --config <config>")
        print( "        and re-freeze the split with --strata before training.")
    other = [u for u in uneven if u not in constant_on]
    if other:
        print(f"\n    [i] UNEVEN: {other}")
        print( "        Varies more between trajectories than within one.")
        print( "        Usually different operating points rather than a bug --")
        print( "        but the same stratification warning applies.")

    if args.plots:
        _plots(store, split, ch_idx, ch_names, norm, cfg)

    print("\nPhase 0 complete. Next:")
    print("    python scripts/fit_dmd.py --config <config>       # POD-DMD floor")
    print("    python scripts/train.py   --config configs/ar_fno_r.yaml")


def _plots(store, split, ch_idx, ch_names, norm, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join(cfg.get("results_dir", "./results"), "audit")
    os.makedirs(out_dir, exist_ok=True)
    t_idx = list(range(0, min(store.T, 400), 8))
    block = store.read(split.train[0], t_idx, ch_idx)
    z = norm.forward(block)

    n = len(ch_names)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.2))
    axes = np.atleast_2d(axes)
    for i, name in enumerate(ch_names):
        axes[0, i].hist(block[..., i].ravel(), bins=80, color="0.4")
        axes[0, i].set_title(name, fontsize=7)
        axes[0, i].set_yscale("log")
        axes[1, i].hist(z[..., i].ravel(), bins=80, color="tab:blue")
        axes[1, i].set_yscale("log")
    axes[0, 0].set_ylabel("raw")
    axes[1, 0].set_ylabel("transformed")
    fig.tight_layout()
    p = os.path.join(out_dir, "channel_histograms.png")
    fig.savefig(p, dpi=130)
    print(f"    plots -> {p}")

    fig, axes = plt.subplots(1, len(ch_names), figsize=(2.4 * len(ch_names), 2.6))
    for i, name in enumerate(np.atleast_1d(ch_names)):
        axes[i].imshow(block[len(t_idx) // 2, :, :, i], cmap="magma")
        axes[i].set_title(name, fontsize=7)
        axes[i].axis("off")
    fig.tight_layout()
    p = os.path.join(out_dir, "example_fields.png")
    fig.savefig(p, dpi=130)
    print(f"    plots -> {p}")


if __name__ == "__main__":
    main()
