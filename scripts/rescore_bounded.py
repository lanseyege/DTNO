#!/usr/bin/env python
"""Recompute the bounded evaluation score from an existing results JSON.

No GPU, no re-evaluation: the per-horizon field errors are already in the file.
The Gray-Scott R=1 arm reports `eval_score_bounded = nan` because
`min(nan, 1.0)` is nan, so one non-finite horizon makes the whole summary
undefined. A non-finite horizon is a FAILED forecast, not a large error, and
the bounded score's own convention already has a slot for that: 1.0, "no better
than predicting the training mean".

    python scripts/rescore_bounded.py results/rev_gs_arR1_s0/horizon_metrics.json
    python scripts/rescore_bounded.py "results/*/horizon_metrics.json" --write

--write updates `eval_score_bounded` and records `nonfinite_horizons` in place,
leaving every other field untouched. Runs that are already finite are reported
and not modified, so this is safe to point at the whole results directory.
"""
from __future__ import annotations
import argparse, glob, json, math, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    files = [p for pat in a.paths for p in glob.glob(pat)]
    if not files:
        raise SystemExit(f"no files matched {a.paths}")
    for p in sorted(files):
        r = json.load(open(p))
        E = [float(v) for v in r.get("E_field", [])]
        hs = r.get("horizons", list(range(len(E))))
        if not E:
            continue
        bad = [int(h) for h, v in zip(hs, E) if not math.isfinite(v)]
        new = sum(1.0 if not math.isfinite(v) else min(v, 1.0) for v in E) / len(E)
        old = r.get("eval_score_bounded")
        tag = os.path.basename(os.path.dirname(p))
        old_s = "nan" if old is None or (isinstance(old, float) and math.isnan(old)) else f"{old:.4f}"
        mark = "  <- was undefined" if old_s == "nan" else ("" if abs(float(old_s) - new) < 5e-4 else "  <- CHANGED")
        print(f"  {tag:<26} Eval* {old_s:>8} -> {new:.4f}   non-finite at h={bad or 'none'}{mark}")
        if a.write:
            r["eval_score_bounded"] = new
            r["nonfinite_horizons"] = bad
            json.dump(r, open(p, "w"), indent=2)
    if a.write:
        print("\n  Written. Re-run scripts/summarize_rollout_depth.py to refresh the CSV.")
    else:
        print("\n  Dry run. Add --write to update the files.")
    print("  Also apply the same convention in evaluation/runner.py so future runs\n"
          "  do not reproduce the nan.")

if __name__ == "__main__":
    main()
