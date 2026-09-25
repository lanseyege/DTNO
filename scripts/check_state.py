#!/usr/bin/env python
"""
Read-only. Reports what state the repo is in and which checkpoint to resume
from. Writes nothing, imports nothing from the project, needs no GPU.

    python scripts/check_state.py
    python scripts/check_state.py --save-dir checkpoints/cyl_dt_fno_s0

Two questions it answers.

1. WHICH PATCHES ARE IN.  It looks for the *effect* of each edit rather than
   for the exact text `apply_patches.py` would insert, so a hand-applied edit
   worded differently still reads as applied.  That is the difference that
   matters if you patched by hand: `apply_patches.py` matches literal anchors
   and will report "anchor NOT FOUND" on a hand edit that is perfectly correct.
   That report is a false alarm about wording, not a problem with your code --
   and it changes nothing, because the patcher only writes a file when every
   edit for that file resolved.

2. WHERE TO RESUME.  `--resume auto` in scripts/train.py considers only
   `checkpoint_epoch*.pth` and `final_model.pth`.  It ignores `best_model.pth`,
   which is written on every hval improvement and carries full optimizer,
   scheduler and scaler state like any other checkpoint -- so with
   `save_every: 25` it is often 10-20 epochs newer than anything `auto` will
   find.  This prints the highest-epoch checkpoint in the directory and the
   exact command to resume from it.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

# (file, description, probe) -- probe is a predicate on the file text that
# tests for the EFFECT of the edit, not for its literal wording.


def _pattern_matches(text: str, var: str, sample: str) -> bool:
    """Pull `var = re.compile(r"...")` out of the source and actually run it.

    Testing the wording would fail on a correct hand edit that phrased the
    regex differently -- `buoy` and `buoyancy` both match "buoyancy", and only
    one of them would survive a string comparison. What matters is whether the
    channel name lands in the right group, so compile the pattern and ask it.
    """
    m = re.search(var + r'\s*=\s*re\.compile\(\s*r"((?:[^"\\]|\\.)*)"', text)
    if not m:
        return False
    try:
        return bool(re.search(m.group(1), sample, re.I))
    except re.error:
        return False


PROBES = [
    ("data/store.py", "build_store knows the 'well' backend",
     lambda t: "WellStore" in t and 'backend == "well"' in t),
    ("data/store.py", "build_store knows the 'arrow' backend",
     lambda t: "ArrowTrajectoryStore" in t and '"arrow"' in t),
    ("data/store.py", "dt: null is tolerated (no bare float(cfg.get('dt')))",
     lambda t: 'dt = float(cfg.get("dt", 2.5e-4))' not in t),
    ("data/channels.py", "a bare 'p' matches pressure (Cylinder)",
     lambda t: _pattern_matches(t, "_PRESSURE", "p")),
    ("data/channels.py", "'buoyancy' matches thermo (Rayleigh-Benard)",
     lambda t: _pattern_matches(t, "_THERMO", "buoyancy")),
    ("data/channels.py", "existing names unmoved: 'Absolute_Pressure'",
     lambda t: _pattern_matches(t, "_PRESSURE", "Absolute_Pressure")),
    ("data/channels.py", "existing names unmoved: 'Temperature'",
     lambda t: _pattern_matches(t, "_THERMO", "Temperature")),
    ("data/realpde.py", "num_workers_val exists (val loader sized separately)",
     lambda t: "num_workers_val" in t),
    ("data/realpde.py", "num_workers_eval exists (hval / eval sized separately)",
     lambda t: "num_workers_eval" in t),
    ("data/realpde.py", "num_workers: 0 no longer forks a val worker",
     lambda t: "max(1, nw // 2)" not in t or "0 if nw == 0" in t),
    ("data/__init__.py", "registry lists the three new datasets",
     lambda t: all(k in t for k in ("gray_scott", "cylinder",
                                    "rayleigh_benard"))),
    ("data/well.py", "HDF5 handles are keyed on pid (fork-safe)",
     lambda t: "os.getpid()" in t),
    ("data/arrow_store.py", "Arrow store present",
     lambda t: "class ArrowTrajectoryStore" in t),
]


def check_patches(repo: str) -> int:
    print("PATCH STATE")
    print("-" * 72)
    missing = 0
    cache = {}
    for path, what, probe in PROBES:
        full = os.path.join(repo, path)
        if full not in cache:
            cache[full] = open(full).read() if os.path.exists(full) else None
        text = cache[full]
        if text is None:
            print(f"  MISSING FILE  {path}")
            missing += 1
            continue
        ok = probe(text)
        print(f"  {'in ' if ok else 'OUT'}  {path:<22} {what}")
        missing += 0 if ok else 1
    print()
    if missing:
        print(f"  {missing} item(s) not applied. `python scripts/apply_patches.py "
              f"--repo . --dry-run`\n  will show what it would do; it writes a "
              f"file only when EVERY edit for that\n  file resolves, so a "
              f"mismatch is a no-op, never a partial edit.")
    else:
        print("  All applied. Nothing to run.")
    return missing


def _epoch_of(path: str):
    """Epoch recorded inside a checkpoint, without importing torch."""
    try:
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return int(ck.get("epoch", -1)), ck.get("best_metric")
    except Exception:
        m = re.search(r"epoch(\d+)", os.path.basename(path))
        return (int(m.group(1)) if m else -1), None


def check_resume(save_dir: str):
    print()
    print(f"CHECKPOINTS IN {save_dir}")
    print("-" * 72)
    if not os.path.isdir(save_dir):
        print("  directory does not exist")
        return
    files = sorted(glob.glob(os.path.join(save_dir, "*.pth")))
    if not files:
        print("  no .pth files -- nothing to resume from; start fresh")
        return
    rows = []
    for f in files:
        ep, best = _epoch_of(f)
        rows.append((ep, f, best))
        b = f"  best_metric={best:.5f}" if isinstance(best, float) else ""
        seen = "" if os.path.basename(f).startswith(("checkpoint_epoch",
                                                     "final_model")) \
            else "   <- ignored by --resume auto"
        print(f"  epoch {ep:>4}  {os.path.basename(f):<28}{b}{seen}")
    rows.sort(key=lambda r: r[0])
    latest = rows[-1]
    auto = [r for r in rows
            if os.path.basename(r[1]).startswith(("checkpoint_epoch",
                                                  "final_model"))]
    print()
    print(f"  newest state          : epoch {latest[0]}  "
          f"({os.path.basename(latest[1])})")
    if auto:
        print(f"  what --resume auto finds: epoch {max(auto)[0]}  "
              f"({os.path.basename(max(auto)[1])})")
        if max(auto)[0] < latest[0]:
            print(f"  --resume auto would discard {latest[0] - max(auto)[0]} "
                  f"epoch(s). Name the file instead:")
    else:
        print("  --resume auto would find NOTHING (no checkpoint_epoch*/"
              "final_model). Name the file:")
    print()
    print(f"    --resume {latest[1]}")
    print()
    print("  Resuming restores model, optimizer, scheduler, AMP scaler, epoch,")
    print("  global_step and best_metric. It does NOT restore the global torch")
    print("  RNG, so dropout draws after the restart differ from an")
    print("  uninterrupted run -- a legitimate resume, not a bit-identical one.")
    print("  The DATA is unaffected: every sample is seeded from")
    print("  (base_seed, epoch, index) and epoch is restored, so the sample")
    print("  stream from here on is exactly what it would have been.")


def main():
    ap = argparse.ArgumentParser(description="Read-only repo state check")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--save-dir", default=None,
                    help="a checkpoints/<tag> directory to inspect")
    args = ap.parse_args()
    check_patches(args.repo)
    if args.save_dir:
        check_resume(args.save_dir)


if __name__ == "__main__":
    main()
