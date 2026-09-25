#!/usr/bin/env python
"""Collect rollout-depth runs into one CSV suitable for plotting/table work."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics


def _load(path):
    with open(path) as fp:
        return json.load(fp)


def _median(xs):
    xs = [float(x) for x in xs]
    return statistics.median(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_root", default="checkpoints")
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--prefixes", nargs="*", default=["rev_gs", "rev_rp", "rev_rb"])
    ap.add_argument("--out", default="results/revision/rollout_depth_summary.csv")
    args = ap.parse_args()

    rows = []
    for prefix in args.prefixes:
        pat = os.path.join(args.checkpoint_root, f"{prefix}_arR*_s*", "history.json")
        for hp in sorted(glob.glob(pat)):
            tag = os.path.basename(os.path.dirname(hp))
            m = re.search(r"_arR(\d+)_s(\d+)$", tag)
            if not m:
                continue
            R, seed = int(m.group(1)), int(m.group(2))
            hist = _load(hp)
            # Drop epoch 0 for steady-state timing if possible: CUDA allocator
            # and Adam state initialization make it systematically different.
            steady = hist[1:] if len(hist) > 1 else hist
            train_times = [r.get("time_train_s") for r in steady if r.get("time_train_s") is not None]
            throughputs = [r.get("train/examples_per_s") for r in steady
                           if r.get("train/examples_per_s") is not None]
            peaks = [r.get("train/peak_allocated_GB") for r in hist
                     if r.get("train/peak_allocated_GB") is not None]
            reserved = [r.get("train/peak_reserved_GB") for r in hist
                        if r.get("train/peak_reserved_GB") is not None]

            rp = os.path.join(args.results_root, tag, "horizon_metrics.json")
            result = _load(rp) if os.path.isfile(rp) else {}
            hs = result.get("horizons", [])
            es = result.get("E_field", [])
            emap = {int(h): float(e) for h, e in zip(hs, es)}
            cfg = result.get("meta", {}).get("train_config", {})

            row = {
                "tag": tag,
                "dataset_prefix": prefix,
                "R": R,
                "seed": seed,
                "batch_size": cfg.get("batch_size"),
                "epochs": len(hist),
                "total_train_s": sum(float(r.get("time_train_s", 0.0)) for r in hist),
                "median_epoch_train_s": _median(train_times),
                "median_examples_per_s": _median(throughputs),
                "peak_allocated_GB": max(peaks) if peaks else "",
                "peak_reserved_GB": max(reserved) if reserved else "",
                "eval_score": result.get("eval_score", ""),
                "eval_score_bounded": result.get("eval_score_bounded", ""),
                "T_pred": result.get("T_pred", ""),
                "diverged_horizons": ";".join(map(str, result.get("diverged_horizons", []))),
            }
            for h in [1, 4, 8, 16, 32, 64, 128]:
                row[f"E_h{h}"] = emap.get(h, "")
            rows.append(row)

    fields = [
        "tag", "dataset_prefix", "R", "seed", "batch_size", "epochs",
        "total_train_s", "median_epoch_train_s", "median_examples_per_s",
        "peak_allocated_GB", "peak_reserved_GB", "eval_score",
        "eval_score_bounded", "T_pred", "diverged_horizons",
        "E_h1", "E_h4", "E_h8", "E_h16", "E_h32", "E_h64", "E_h128",
    ]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
