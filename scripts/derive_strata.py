#!/usr/bin/env python
"""
Derive stratification labels from the data itself (§9).

§9 asks for the split to be stratified by fuel composition / equivalence ratio
"after inspecting the metadata". RealPDEBench's published split files are empty
and its metadata does not expose those fields, so the labels are recovered from
the fields instead. That is not a workaround for missing metadata; a species
that is identically zero in one trajectory and reaches 3.5e-2 in another IS the
fuel composition, measured directly.

Method: presence fingerprinting rather than clustering.

    For each channel, compute a per-trajectory magnitude (mean of |x| over a
    subsample of frames). A channel is a REGIME MARKER if its across-trajectory
    magnitude ratio exceeds `--presence_ratio` (default 1e3) -- that is, some
    trajectories contain it and others contain numerically nothing. Each
    trajectory then gets a binary present/absent fingerprint over the marker
    channels, and trajectories sharing a fingerprint form one stratum.

Chosen over k-means deliberately: no k to pick, no distance metric to defend,
and the output is human-readable ("CH4+ NH3-") rather than "cluster 2". When
the underlying difference really is a fuel switch, presence/absence is the
signal, and a continuous clustering would blur it with turbulence intensity.

Leakage: strata are computed over ALL trajectories including held-out ones.
That is the same thing stratified k-fold does with labels, and it is what §9
describes -- coarse per-trajectory descriptors used to DESIGN the split. No
normalisation statistic and no model input is derived here; those still come
from training trajectories only (§11.1).

Usage:
    python scripts/derive_strata.py --config configs/dt_fno.yaml
    python scripts/audit_data.py    --config configs/dt_fno.yaml --force \
        --strata artifacts/strata.json
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

from common import base_parser, resolve, json_default   # noqa: E402

from data.store import build_store                       # noqa: E402
from data.realpde import resolve_channels                # noqa: E402
from data.channels import find_channel                   # noqa: E402


def short(name: str) -> str:
    """'Mole_Fraction_of_NH3' -> 'NH3', for readable fingerprints."""
    for pre in ("Mole_Fraction_of_", "Mass_Fraction_of_", "Chemistry_"):
        if name.startswith(pre):
            return name[len(pre):]
    return name


def kmeans_1d(x: np.ndarray, k: int, n_iter: int = 100) -> np.ndarray:
    """Lloyd's algorithm on a scalar, seeded at quantiles. Returns labels 0..k-1
    ordered by increasing centre, so the label IS the ordinal level."""
    x = np.asarray(x, dtype=np.float64)
    k = min(k, len(np.unique(x)))
    c = np.quantile(x, np.linspace(0, 1, k + 2)[1:-1]) if k > 1 else np.array([x.mean()])
    for _ in range(n_iter):
        lab = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
        new = np.array([x[lab == j].mean() if (lab == j).any() else c[j]
                        for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    order = np.argsort(c)
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    return remap[np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)]


def equivalence_proxy(mag: np.ndarray, names: List[str]) -> np.ndarray:
    """A monotone stand-in for the equivalence ratio, or NaN if unavailable.

    Richer mixtures leave unburnt fuel and more CO relative to CO2, so
    log10(CO / CO2) rises with phi. It is a proxy, not a measurement: it
    conflates phi with fuel composition, since an NH3 blend has less carbon to
    begin with. Use it to check that a fuel stratum spans a range of phi, not
    to assign phi values.
    """
    def col(*pats):
        i = find_channel(names, *pats)
        return mag[:, i] if i >= 0 else None
    co, co2 = col(r"of_CO$", r"\bCO\b"), col(r"of_CO2$", r"\bCO2\b")
    if co is None or co2 is None:
        return np.full(mag.shape[0], np.nan)
    return np.log10(np.maximum(co, 1e-300) / np.maximum(co2, 1e-300))


def main():
    ap = base_parser("Derive stratification labels (§9)")
    ap.add_argument("--n_frames", type=int, default=24,
                    help="frames sampled per trajectory")
    ap.add_argument("--presence_ratio", type=float, default=1e3,
                    help="across-trajectory magnitude ratio above which a "
                         "channel counts as a regime marker")
    ap.add_argument("--levels", type=int, default=0,
                    help="Number of ORDINAL levels to resolve on the strongest "
                         "marker channel (0 = binary present/absent only). "
                         "RealPDEBench combustion sweeps CH4 at "
                         "{100,80,60,40,20}%%, so --levels 5 recovers the "
                         "design instead of collapsing four blends into one "
                         "group.")
    ap.add_argument("--out", default="artifacts/strata.json")
    args = ap.parse_args()
    cfg = resolve(args)

    store = build_store(cfg)
    ch_idx, ch_names = resolve_channels(store, cfg.get("channels"))
    rng = np.random.default_rng(0)

    print("=" * 78)
    print("DERIVE STRATA (§9)")
    print("=" * 78)
    print(f"  {store.summary()}")
    print(f"  {args.n_frames} frames per trajectory, {store.n_traj} trajectories\n")

    # ---- per-trajectory magnitudes -------------------------------------
    mag = np.zeros((store.n_traj, len(ch_idx)))
    for t in range(store.n_traj):
        idx = sorted(rng.choice(store.T, size=min(args.n_frames, store.T),
                                replace=False).tolist())
        blk = store.read(t, idx, ch_idx).astype(np.float64)
        mag[t] = np.abs(blk).mean(axis=(0, 1, 2))

    # ---- which channels separate regimes -------------------------------
    floor = np.maximum(mag.max(axis=0) * 1e-12, 1e-300)
    ratio = mag.max(axis=0) / np.maximum(mag.min(axis=0), floor)
    markers = [i for i in range(len(ch_names)) if ratio[i] > args.presence_ratio]

    print("  Across-trajectory magnitude spread (mean of |x| per trajectory)")
    print(f"    {'channel':<30}{'min':>12}{'max':>12}{'ratio':>12}  marker")
    print("    " + "-" * 70)
    for i, n in enumerate(ch_names):
        print(f"    {n:<30}{mag[:, i].min():>12.3e}{mag[:, i].max():>12.3e}"
              f"{ratio[i]:>12.1e}  {'YES' if i in markers else ''}")

    if not markers:
        print("\n  No regime markers found: every channel is present in every "
              "trajectory at a comparable level.")
        print("  A seeded random split is then defensible. Nothing to write.")
        return

    print(f"\n  Regime markers: {[short(ch_names[i]) for i in markers]}")
    print("  These channels are numerically absent in some trajectories and "
          "present in others,\n  which is a fuel-composition difference "
          "measured directly from the fields.")

    # ---- fingerprint per trajectory ------------------------------------
    labels: List[str] = []
    for t in range(store.n_traj):
        parts = []
        for i in markers:
            thresh = np.sqrt(max(mag[:, i].max(), 1e-300)
                             * max(mag[:, i].min(), floor[i]))  # geometric mean
            parts.append(f"{short(ch_names[i])}{'+' if mag[t, i] > thresh else '-'}")
        labels.append(" ".join(parts))

    # ---- optional ordinal refinement -----------------------------------
    if args.levels > 1:
        primary = max(markers, key=lambda i: ratio[i])
        pname = short(ch_names[primary])
        v = mag[:, primary]
        present = v > np.sqrt(max(v.max(), 1e-300) * max(v.min(), floor[primary]))
        lab_ord = np.zeros(store.n_traj, dtype=int)      # level 0 = absent
        if present.sum() >= args.levels - 1:
            lab_ord[present] = 1 + kmeans_1d(np.log10(v[present]),
                                             args.levels - 1)
        labels = [f"{pname}_L{int(l)}" for l in lab_ord]
        print(f"\n  Ordinal levels on {pname} (log-magnitude, "
              f"{args.levels} levels incl. absent):")
        for L in range(args.levels):
            ts = [t for t in range(store.n_traj) if lab_ord[t] == L]
            if ts:
                vv = v[ts]
                print(f"    L{L}: {len(ts):>2} traj  "
                      f"mean|{pname}| in [{vv.min():.3e}, {vv.max():.3e}]  {ts}")

    groups: Dict[str, List[int]] = defaultdict(list)
    for t, lab in enumerate(labels):
        groups[lab].append(t)

    # ---- equivalence-ratio spread inside each stratum -------------------
    phi = equivalence_proxy(mag, ch_names)
    if np.isfinite(phi).all():
        print("\n  Equivalence-ratio proxy  log10(mean|CO| / mean|CO2|)")
        print("  A stratum whose proxy spans a wide range contains several")
        print("  equivalence ratios; §9 wants those on both sides of the split")
        print("  too, but with 30 trajectories you cannot stratify on fuel AND")
        print("  phi at once. Stratify on fuel, then check this column.")
        print(f"    {'stratum':<22}{'n':>4}{'proxy min':>12}{'proxy max':>12}"
              f"{'spread':>9}")
        for lab, ts in sorted(groups.items()):
            pv = phi[ts]
            print(f"    {lab:<22}{len(ts):>4}{pv.min():>12.3f}{pv.max():>12.3f}"
                  f"{pv.max() - pv.min():>9.3f}")

    print(f"\n  {len(groups)} stratum/strata:")
    for lab, ts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"    [{len(ts):>2} traj] {lab:<28} {ts}")

    counts = Counter(labels)
    n_val = int(cfg.get("n_val_traj", 5))
    n_test = int(cfg.get("n_test_traj", 5))
    small = [lab for lab, c in counts.items() if c < 3]
    if small:
        print(f"\n  [!] Strata with fewer than 3 trajectories: {small}")
        print("      A 20/5/5 split cannot represent these on all three sides.")
        print("      Either accept that the regime appears in training only, or")
        print("      shrink val/test, or treat it as a separate experiment.")
    if len(groups) > (n_val + n_test):
        print(f"\n  [!] {len(groups)} strata but only {n_val}+{n_test} held-out "
              f"slots; some regimes cannot appear in both val and test.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump({
            "strata": labels,
            "groups": {k: v for k, v in groups.items()},
            "marker_channels": [ch_names[i] for i in markers],
            "presence_ratio": args.presence_ratio,
            "per_trajectory_magnitude": {
                ch_names[i]: mag[:, i].tolist() for i in range(len(ch_names))},
            "equivalence_proxy_log10_CO_over_CO2": phi.tolist(),
            "levels": args.levels,
            "note": "Derived from field magnitudes; §9 stratification input only. "
                    "No normalisation statistic or model input comes from here.",
        }, fp, indent=2, default=json_default)
    print(f"\n  -> {args.out}")
    print("\n  Next, freeze a stratified split:")
    print(f"    python scripts/audit_data.py --config {args.config} "
          f"--force --strata {args.out}")


if __name__ == "__main__":
    main()
