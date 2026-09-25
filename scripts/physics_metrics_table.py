#!/usr/bin/env python
"""
Tabulate the reacting-flow metrics that `evaluate_horizon.py` already computes.

This is not an experiment. `evaluation/combustion_metrics.py` computes flame
IoU, gradient error and integrated heat-release error on every non-`--light`
evaluation, and the results are written to
`results/<tag>/horizon_metrics.json`. They were never pulled into a table,
which is why the paper said they were "implemented but not reported". For a
combustion reader they are the metrics that matter: a channel-averaged relative
error cannot tell a model that tracks the velocity field while misplacing the
flame from one that does not, and the qualitative figure makes exactly that
claim visually without quantifying it.

WHERE THEY LIVE -- READ THIS BEFORE CHANGING ANYTHING
-----------------------------------------------------
Nested, not at the top level:

    results["combustion"]["horizons"]   [1, 2, 4, ...]
    results["combustion"]["iou"]        one value per horizon
    results["combustion"]["E_grad"]     ...

`runner.run_horizon_eval` writes them to `out["combustion"]`, parallel to
`out["spectral"]`, and only when `light=False`. The first version of this
script looked for `results["iou"]` at the TOP level, found nothing in any run,
and reported "no run carried any reacting-flow metric" -- a statement that was
true of the place it looked and false of the files. `--inspect` prints the keys
each JSON actually has, so the next person does not have to guess.

    python scripts/physics_metrics_table.py --results "results/*/horizon_metrics.json"
    python scripts/physics_metrics_table.py --results "..." --inspect
    python scripts/physics_metrics_table.py --results "..." \
        --include ar_fno_r dt_fno sg_dt_fno climatology --latex
"""

from __future__ import annotations

import argparse
import glob
import signal
import json
import os
import re
from collections import defaultdict

# key inside results["combustion"] -> (header, lower-is-better)
METRICS = [
    ("iou",                      "flame IoU",      False),
    ("E_grad",                   "grad. error",    True),
    ("integrated_hrr_rel_error", "int. HRR error", True),
    ("centroid_disp",            "centroid disp.", True),
    ("mean_T_abs_error",         "mean dT [K]",    True),
]

MODEL_LABEL = {"ar_fno_r": "AR-FNO-R", "ar_fno": "AR-FNO-1", "dt_fno": "DT-FNO",
               "sg_dt_fno": "SG-DT-FNO", "persistence": "Persistence",
               "climatology": "Climatology",
               "nearest_climatology": "Nearest clim.", "pod_dmd": "POD-DMD"}
DATASET_PREFIX = ("gray_scott", "rayleigh_benard", "cylinder", "lifted_h2",
                  "gs", "cyl", "rb", "lh2", "comb")


def group_key(tag):
    return re.sub(r"_s\d+$", "", tag)


def pretty(g):
    core = g
    for pre in sorted(DATASET_PREFIX, key=len, reverse=True):
        if core.startswith(pre + "_"):
            core = core[len(pre) + 1:]
            break
    return MODEL_LABEL.get(core, g)


def classify(r):
    """-> ('ok'|'light'|'non_reacting', combustion dict or None)."""
    c = r.get("combustion")
    if c is None:
        return "light", None
    if any(k in c for k, _, _ in METRICS):
        return "ok", c
    return "non_reacting", c


def main():
    # `--inspect | head` is the obvious way to use this and closes the pipe
    # early; without this it ends in a BrokenPipeError traceback that looks
    # like the script failed.
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="Reacting-flow metrics table")
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 8, 32, 128])
    ap.add_argument("--include", nargs="+", default=None,
                    help="exact group names. A glob can pull in a contaminated "
                         "run; exact names cannot.")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--inspect", action="store_true",
                    help="print the keys every JSON actually contains and exit, "
                         "instead of guessing where a metric lives")
    args = ap.parse_args()

    paths = []
    for pat in args.results:
        paths += glob.glob(pat)
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("no files matched %s" % (args.results,))

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    status = {}
    for p in paths:
        with open(p) as fp:
            r = json.load(fp)
        tag = (r.get("meta", {}).get("exp_name")
               or os.path.basename(os.path.dirname(p)))
        g = group_key(tag)
        st, c = classify(r)
        status.setdefault(g, st)
        if args.inspect:
            print("%-34s %-13s blocks: %s" % (
                tag, st, sorted(k for k in r if isinstance(r[k], dict))))
            if c:
                print("%-34s %-13s combustion: %s" % (
                    "", "", sorted(k for k in c
                                   if k not in ("horizons", "n_batches"))))
            continue
        if st != "ok":
            continue
        hs = c.get("horizons", r.get("horizons", []))
        for key, _, _ in METRICS:
            vals = c.get(key)
            if vals is None:
                continue
            if not isinstance(vals, list):
                vals = [vals] * len(hs)
            for h, v in zip(hs, vals):
                if v is not None and v == v:
                    data[g][key][int(h)].append(float(v))
    if args.inspect:
        return

    if args.include:
        data = {k: v for k, v in data.items() if k in set(args.include)}

    if not data:
        n_light = sum(1 for v in status.values() if v == "light")
        n_nr = sum(1 for v in status.values() if v == "non_reacting")
        print("No run in this selection carries reacting-flow metrics.\n")
        print("  %4d run(s) on a NON-REACTING dataset. The metrics correctly do "
              "not exist\n       there: Gray-Scott, Cylinder and "
              "Rayleigh-Benard have no flame." % n_nr)
        print("  %4d run(s) have no `combustion` block at all, which means "
              "evaluate_horizon.py\n       was run with --light. Re-scoring is "
              "cheap and needs no retraining:\n"
              "         python scripts/evaluate_horizon.py --config configs/base.yaml \\\n"
              "             --set meta.model_variant=dt_fno \\\n"
              "             --checkpoint ./checkpoints/dt_fno_s0/best_model.pth \\\n"
              "             --set experiment.exp_name=dt_fno_s0" % n_light)
        print("\n  Run with --inspect to see what each file actually contains.")
        return

    present = [(k, lab, lo) for k, lab, lo in METRICS
               if any(k in v for v in data.values())]
    avail = {h for v in data.values() for k, _, _ in present for h in v.get(k, {})}
    hs = sorted(avail & set(args.horizons)) or sorted(avail)[:4]

    def cell(g, key, h):
        v = data[g].get(key, {}).get(h)
        return (None, 0) if not v else (sum(v) / len(v), len(v))

    if args.latex:
        print("\\begin{tabular}{l" + "r" * (len(present) * len(hs)) + "}")
        print("\\toprule")
        print(" & " + " & ".join("\\multicolumn{%d}{c}{%s}" % (len(hs), lab)
                                 for _, lab, _ in present) + " \\\\")
        print(" & " + " & ".join("$h{=}%d$" % h for _ in present for h in hs)
              + " \\\\\n\\midrule")
        for g in sorted(data):
            cells = []
            for key, _, _ in present:
                for h in hs:
                    m, _ = cell(g, key, h)
                    cells.append("---" if m is None else "$%.3f$" % m)
            print("%s & %s \\\\" % (pretty(g), " & ".join(cells)))
        print("\\bottomrule\n\\end{tabular}")
        return

    head = "%-16s" % "model"
    for _, lab, _ in present:
        head += "".join("%13s" % (lab.split()[0][:7] + "@" + str(h)) for h in hs)
    print(head)
    print("-" * len(head))
    for g in sorted(data):
        row = "%-16s" % pretty(g)
        for key, _, _ in present:
            for h in hs:
                m, n = cell(g, key, h)
                row += "%13s" % ("---" if m is None else
                                 ("%.4f" % m) + (("(%d)" % n) if n > 1 else ""))
        print(row)

    print("\n  columns: " + ", ".join(
        "%s (%s is better)" % (lab, "higher" if lo is False else "lower")
        for _, lab, lo in present))
    print("  (n) is the seed count where more than one run exists.")
    skipped = {g: st for g, st in status.items() if st != "ok"}
    nr = sorted(g for g, st in skipped.items() if st == "non_reacting")
    lt = sorted(g for g, st in skipped.items() if st == "light")
    if nr:
        print("\n  non-reacting, no metrics by construction: %s%s"
              % (nr[:8], " ..." if len(nr) > 8 else ""))
    if lt:
        print("  evaluated with --light, no combustion block: %s%s"
              % (lt[:8], " ..." if len(lt) > 8 else ""))
    print("\n  Read the IoU column against Eval* in the main table: the claim "
          "the qualitative\n  figure makes---that two models with similar field "
          "error place the flame\n  differently---is either supported here or "
          "it is not.")


if __name__ == "__main__":
    main()
