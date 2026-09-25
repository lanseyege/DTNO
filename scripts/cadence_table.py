#!/usr/bin/env python
"""
Build the cadence-intervention table from results that already exist.

WHY THIS NEEDS NO GPU
---------------------
`run/29_cadence.sh` was written to train RealPDEBench at several temporal
strides. It does not need to: `results/realpde_str{1,2,4,8}_{ar_fno_r,dt_fno}`
are already on disk, from the sampling-rate sweep of the original study, and the
matching `artifacts/norm_stats_realpde_stride*.json` are dated alongside them.
The experiment a reviewer asked for as the way to turn the crossover result from
a correlation across five datasets into an intervention on one has, in the
relevant respect, already been run.

WHAT THE TABLE ANSWERS
----------------------
Table 3 of the paper shows the crossover at h = 8-20 FRAMES across flows whose
physical timesteps span 5 us to 10 s. Across datasets that is observational.
Striding one dataset holds the flow, the architecture, the protocol and the
split fixed and changes only the cadence, so the two readings make opposite
predictions:

    composition     h* stays 8-20 frames;  t* = h* * s * dt grows with stride
    physical time   t* stays ~2 ms;        h* falls roughly as 1/s

THE CAVEAT THAT MUST TRAVEL WITH THE NUMBERS
--------------------------------------------
Two of them, and the script prints both rather than leaving them to a footnote.

1.  Those runs carry `loss_channel_weights: {Absolute_Pressure: 0.0}`, the flag
    the handover records as contaminating a whole block of experiments. All four
    strides carry it equally, so the stride-to-stride comparison this table
    exists to make is internally valid; only the absolute error level is
    displaced. Do not quote the Eval* values beside the paper's uncontaminated
    ones.

2.  Striding is not a single-variable intervention. A 4x coarser step is a
    harder step, so the per-step autoregressive error rises with stride and the
    predictability horizon measured in frames shrinks. A crossover that stays
    at 8-20 frames is therefore strong evidence for the composition reading,
    but a crossover that MOVES is not automatically evidence against it -- it
    could be the per-step error moving. The table prints E_AR(h=1) next to the
    crossover so the two can be read together, and refuses to draw a conclusion
    when both move.

    python scripts/cadence_table.py --results "results/*/horizon_metrics.json"
    python scripts/cadence_table.py --results "..." --dt 2.5e-4 --latex
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict


def crossover(h, a, b):
    """First horizon where curve `a` rises above `b`, log-linearly interpolated.

    Same convention as make_figures.fig1_crossover, so the number is comparable
    with the paper's Table 3 rather than merely similar to it.
    """
    import math
    prev = None
    for i, hv in enumerate(h):
        if a[i] != a[i] or b[i] != b[i]:
            continue
        d = a[i] - b[i]
        if prev is not None and prev[1] < 0 <= d:
            h0, d0 = prev
            if d != d0:
                lg = math.log(h0) + (math.log(hv) - math.log(h0)) * (-d0) / (d - d0)
                return math.exp(lg)
            return float(hv)
        prev = (hv, d)
    return None


def _recompute(r, drop, tag):
    """E_field over the channels that remain after dropping `drop`."""
    ec = r.get("E_channel")
    names = (r.get("channel_names")
             or r.get("meta", {}).get("channel_names"))
    if ec is None or not names:
        print(f"  {tag}: no E_channel/channel_names in the JSON, using the "
              f"stored E_field")
        return None, None
    keep = [i for i, n in enumerate(names) if n not in set(drop)]
    missing = [d for d in drop if d not in names]
    if missing:
        print(f"  {tag}: --exclude-channels {missing} not among {names}")
        return None, None
    return [sum(row[i] for i in keep) / len(keep) for row in ec], \
           [names[i] for i in keep]


def main():
    ap = argparse.ArgumentParser(description="Crossover versus temporal stride")
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--prefix", default="realpde_str",
                    help="tag prefix; runs are <prefix><S>_<model>")
    ap.add_argument("--ar", default="ar_fno_r")
    ap.add_argument("--dt", dest="dt_model", default="dt_fno")
    ap.add_argument("--timestep", type=float, default=2.5e-4,
                    help="physical dt of the UNSTRIDED record, in seconds")
    ap.add_argument("--exclude-channels", nargs="+", default=None,
                    metavar="NAME",
                    help="recompute E_field from E_channel over the remaining "
                         "channels. On the archived stride sweep the runs were "
                         "trained with Absolute_Pressure zeroed IN THE LOSS, so "
                         "the channel-averaged E_field they report is dominated "
                         "by a channel the model was never asked to learn. "
                         "Excluding it from the METRIC is a reporting choice and "
                         "a legitimate one; excluding it from the loss was the "
                         "mistake.")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    paths = []
    for pat in args.results:
        paths += glob.glob(pat)

    runs = defaultdict(dict)
    for p in sorted(set(paths)):
        tag = os.path.basename(os.path.dirname(p))
        m = re.match(rf"^{re.escape(args.prefix)}(\d+)_(.+?)(?:_s\d+)?$", tag)
        if not m:
            continue
        s, model = int(m.group(1)), m.group(2)
        with open(p) as fp:
            r = json.load(fp)
        E = [float(v) for v in r["E_field"]]
        if args.exclude_channels:
            E2, names = _recompute(r, args.exclude_channels, tag)
            if E2 is not None:
                E = E2
        runs[s][model] = (r["horizons"], E)

    if not runs:
        raise SystemExit(
            f"no runs matching '{args.prefix}<S>_<model>'. Looked in "
            f"{args.results}.\n  If the sweep was tagged differently, pass "
            f"--prefix; if it was never run, use run/29_cadence.sh.")

    rows = []
    for s in sorted(runs):
        got = runs[s]
        if args.ar not in got or args.dt_model not in got:
            print(f"  stride {s}: missing "
                  f"{[m for m in (args.ar, args.dt_model) if m not in got]}, "
                  f"skipping")
            continue
        h_ar, e_ar = got[args.ar]
        h_dt, e_dt = got[args.dt_model]
        if h_ar != h_dt:
            print(f"  stride {s}: AR and DT were evaluated on different "
                  f"horizon grids; skipping")
            continue
        # DT beats AR from here on: cross where (e_ar - e_dt) turns positive
        x = crossover(h_ar, e_ar, e_dt)
        # `crossover` looks for a sign CHANGE, so it returns None both when AR
        # never loses and when AR was already losing at the first horizon.
        # Those are opposite situations and must not print the same word: if
        # AR is already worse at h = 1, the ordering the whole comparison rests
        # on -- AR is the better one-step predictor -- does not hold in that
        # run, and the crossover is not "never" but below the grid.
        note = ""
        if x is None:
            note = ("AR worse already at h=1" if e_ar[0] > e_dt[0]
                    else "AR still better at h_max")
        rows.append({"note": note, "stride": s, "dt": args.timestep * s,
                     "ar1": e_ar[0] if e_ar else float("nan"),
                     "dt1": e_dt[0] if e_dt else float("nan"),
                     "h": x, "t": (x * args.timestep * s) if x else None})

    if args.latex:
        print("\\begin{tabular}{rrrrr}\n\\toprule")
        print("stride & $\\Delta t$ & AR $\\Efield(1)$ & crossover $h^{*}$ "
              "& $t^{*}$ \\\\\n\\midrule")
        for r in rows:
            print(f"{r['stride']} & ${r['dt']*1e3:.3g}$\\,ms & ${r['ar1']:.4f}$ & "
                  + (f"${r['h']:.1f}$ & ${r['t']*1e3:.2f}$\\,ms \\\\"
                     if r["h"] else "--- & --- \\\\"))
        print("\\bottomrule\n\\end{tabular}")
        return

    print(f"{'stride':>7}{'dt':>12}{'AR E(h=1)':>12}{'DT E(h=1)':>12}"
          f"{'crossover h*':>14}{'t* = h* dt':>13}")
    print("-" * 70)
    for r in rows:
        print(f"{r['stride']:>7}{r['dt']*1e3:>10.4g}ms{r['ar1']:>12.4f}"
              f"{r['dt1']:>12.4f}"
              + (f"{r['h']:>14.1f}{r['t']*1e3:>11.3g}ms" if r["h"]
                 else f"{'--':>14}{'  ' + r['note']:>13}"))

    bad = [r for r in rows if not r["h"] and "h=1" in r["note"]]
    if bad:
        print(f"\n  [!] On {len(bad)} of {len(rows)} strides the autoregressive "
              f"model is ALREADY worse\n      than the direct model at h = 1. "
              f"Every other experiment in this study has\n      AR as the "
              f"better one-step predictor, so these runs do not reproduce the\n"
              f"      ordering the crossover is defined by, and no crossover "
              f"can be read from\n      them.\n\n      The likely cause is the "
              f"loss-weight contamination: with a channel zeroed\n      in the "
              f"loss but still averaged into E_field, the reported error is "
              f"dominated\n      by a channel the model was never asked to "
              f"learn. Try\n\n        --exclude-channels Absolute_Pressure\n\n"
              f"      which recomputes E_field from E_channel over the "
              f"remaining channels. If\n      the ordering still does not "
              f"recover, these runs cannot answer the cadence\n      question "
              f"and run/29_cadence.sh has to be run properly.")

    hs = [r["h"] for r in rows if r["h"]]
    a1 = [r["ar1"] for r in rows if r["h"]]
    if len(hs) >= 2:
        h_ratio = max(hs) / min(hs)
        e_ratio = max(a1) / min(a1)
        ts = [r["t"] for r in rows if r["h"]]
        t_ratio = max(ts) / min(ts)
        print(f"\n  h* varies by {h_ratio:.2f}x,  t* by {t_ratio:.2f}x,  "
              f"AR E(h=1) by {e_ratio:.2f}x")
        print("\n  Reading it:")
        if h_ratio < 1.6 and t_ratio > 2.0:
            print("    h* is roughly CONSTANT in frames while t* grows with the "
                  "stride.\n    That is the composition reading, and it is the "
                  "intervention the\n    cross-dataset table could not provide.")
        elif t_ratio < 1.6 and h_ratio > 2.0:
            print("    t* is roughly constant while h* falls with the stride.\n"
                  "    That is the PHYSICAL-TIME reading, and Section 5.4 of the "
                  "paper\n    must be rewritten: the crossover would be a "
                  "property of the flow's\n    timescale, not of composing a "
                  "learned map.")
        else:
            print("    Neither h* nor t* is clearly invariant.")
        if e_ratio > 1.5:
            print(f"\n    NOTE: AR E(h=1) also varies by {e_ratio:.2f}x across "
                  f"strides, so this is\n    not a single-variable "
                  f"intervention. If h* moved as well, the experiment\n    "
                  f"cannot separate the two hypotheses and should be reported "
                  f"as\n    inconclusive rather than read in whichever "
                  f"direction is convenient.")
    print("\n  These runs carry loss_channel_weights {Absolute_Pressure: 0.0}. "
          "All strides\n  carry it equally, so the stride-to-stride comparison "
          "is internally valid;\n  do not quote their absolute Eval* beside "
          "the paper's other numbers.")


if __name__ == "__main__":
    main()
