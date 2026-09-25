#!/usr/bin/env python
"""Microbenchmark training depth: AR unrolling vs direct finite-time supervision.

This benchmark isolates the forward/backward/optimizer graph from data-loader I/O.
It answers the revision's central systems question:

    How do step time and peak training memory scale with AR rollout depth R,
    while a direct-time pair keeps fixed network depth even at a long horizon h?

The accuracy side of the story comes from ``run/32_ar_rollout_depth.sh``.  This
script is only the compute/memory side, measured on one already-loaded batch.

Example (paper run, on an idle GPU):

    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_training_cost.py \
      --config configs/gray_scott.yaml --rollouts 1 4 8 16 32 \
      --direct_horizons 1 128 --batch_size 2 --n_warmup 3 --n_repeat 10

Use the same GPU, precision and micro-batch for every arm.  The script reports
both total allocated memory and the incremental peak above the warmed-up
model+optimizer baseline.  The latter is the cleaner activation-memory measure.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from contextlib import nullcontext
from typing import Dict, List

import torch
import torch.nn as nn

from common import base_parser, resolve, set_seed, json_default  # noqa: E402
from data import build_data                                      # noqa: E402
from models import build_model                                   # noqa: E402
from training import build_task                                  # noqa: E402
from evaluation.timing import gpu_state, warm_gpu                # noqa: E402


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def _median_iqr(xs: List[float]) -> Dict[str, float]:
    ys = sorted(float(x) for x in xs)
    lo = ys[: len(ys) // 2]
    hi = ys[(len(ys) + 1) // 2:]
    q1 = statistics.median(lo) if lo else ys[0]
    q3 = statistics.median(hi) if hi else ys[-1]
    return {
        "median": statistics.median(ys),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": ys[0],
        "max": ys[-1],
    }


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def _arm_cfg(base: dict, kind: str, depth: int, batch_size: int) -> dict:
    cfg = dict(base)
    cfg.update({
        "batch_size": int(batch_size),
        "batch_size_eval": int(batch_size),
        "samples_per_epoch": max(int(batch_size) * 4, 8),
        "val_samples": max(int(batch_size) * 2, 4),
        "num_workers": 0,
        "num_workers_val": 0,
        "num_workers_eval": 0,
        "tensorboard": False,
        "use_semigroup": False,
        "use_identity": False,
    })
    if kind == "ar":
        cfg.update({
            "model_name": "ar_fno",
            "ar_rollout": int(depth),
            "ar_random_rollout": False,  # exact graph depth R
            "horizon_sampling": "fixed",
            "fixed_horizon": 1,
        })
    else:
        cfg.update({
            "model_name": "dt_fno",
            "horizon_sampling": "fixed",
            "fixed_horizon": int(depth),
            "h_max_train": max(int(depth), 1),
        })
    return cfg


def _run_arm(base_cfg: dict, dataset_name: str, kind: str, depth: int,
             batch_size: int, precision: str, n_warmup: int, n_repeat: int,
             device: torch.device, max_grad_norm: float) -> dict:
    cfg = _arm_cfg(base_cfg, kind, depth, batch_size)
    bundle = build_data(dataset_name, cfg, model_kind=kind,
                        distributed=False, seed=int(cfg.get("seed", 0)))
    batch = _to_device(next(iter(bundle["train_loader"])), device)
    info = bundle["data_info"]
    name = cfg["model_name"]
    model = build_model(name, cfg, info).to(device).train()
    task = build_task(name, cfg, info)
    task.set_epoch(0)

    # lr=0 keeps weights fixed across timing repeats while still exercising the
    # real AdamW state allocation and optimizer kernels.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0,
                                  weight_decay=float(cfg.get("weight_decay", 1e-4)))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and precision == "fp16"))

    def one_step():
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, precision):
            loss, _ = task.training_step(model, batch)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        return loss

    # Warm-up creates Adam state and stabilizes kernels/clocks before the peak
    # counters are interpreted.
    for _ in range(max(0, n_warmup)):
        one_step()
    _sync(device)

    base_alloc = (torch.cuda.memory_allocated(device) / 1e9
                  if device.type == "cuda" else 0.0)
    times_ms: List[float] = []
    peaks_GB: List[float] = []
    increments_GB: List[float] = []

    for _ in range(n_repeat):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        t0 = time.perf_counter()
        one_step()
        _sync(device)
        times_ms.append((time.perf_counter() - t0) * 1e3)
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            peaks_GB.append(peak)
            increments_GB.append(max(0.0, peak - base_alloc))

    time_stats = _median_iqr(times_ms)
    rec = {
        "arm": f"ar_R{depth}" if kind == "ar" else f"dt_h{depth}",
        "kind": kind,
        "temporal_depth": int(depth) if kind == "ar" else 1,
        "supervised_horizon": int(depth),
        "batch_size": int(batch_size),
        "params": int(sum(p.numel() for p in model.parameters())),
        "time_median_ms": time_stats["median"],
        "time_iqr_ms": time_stats["iqr"],
        "time_min_ms": time_stats["min"],
        "time_max_ms": time_stats["max"],
        "precision": precision,
        "n_repeat": int(n_repeat),
    }
    if peaks_GB:
        rec.update({
            "baseline_allocated_GB": base_alloc,
            "peak_allocated_GB": max(peaks_GB),
            "peak_increment_median_GB": _median_iqr(increments_GB)["median"],
            "peak_increment_max_GB": max(increments_GB),
        })

    del optimizer, task, model, batch, bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()
        _sync(device)
    return rec


def main():
    ap = base_parser("Training compute/memory scaling benchmark")
    ap.add_argument("--rollouts", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    ap.add_argument("--direct_horizons", nargs="+", type=int, default=None,
                    help="default: 1 and config h_max_train")
    ap.add_argument("--batch_size", type=int, default=2,
                    help="same per-GPU micro-batch for every arm")
    ap.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"],
                    default="auto")
    ap.add_argument("--n_warmup", type=int, default=3)
    ap.add_argument("--n_repeat", type=int, default=10)
    ap.add_argument("--warmup_seconds", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = resolve(args)
    set_seed(int(cfg["seed"]))
    dataset_name = cfg.get("dataset_name", "realpde_combustion")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = args.precision
    if precision == "auto":
        precision = (str(cfg.get("amp_dtype", "bf16")).lower()
                     if bool(cfg.get("amp", True)) and device.type == "cuda"
                     else "fp32")
        if precision not in ("fp32", "bf16", "fp16"):
            precision = "fp32"

    hmax = int(cfg.get("h_max_train", 128))
    direct_h = args.direct_horizons or sorted(set([1, hmax]))
    state = gpu_state(device)
    if args.warmup_seconds > 0:
        warm_gpu(device, args.warmup_seconds)

    print("=" * 78)
    print("TRAINING DEPTH / MEMORY MICROBENCHMARK")
    print("=" * 78)
    print(f"dataset={dataset_name} device={device} precision={precision} "
          f"batch={args.batch_size}")
    for k, v in state.items():
        print(f"  {k}: {v}")

    rows = []
    for R in args.rollouts:
        print(f"\n  AR fixed rollout R={R}")
        try:
            rec = _run_arm(cfg, dataset_name, "ar", R, args.batch_size,
                           precision, args.n_warmup, args.n_repeat, device,
                           float(cfg.get("max_grad_norm", 1.0)))
        except torch.cuda.OutOfMemoryError as e:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            rec = {"arm": f"ar_R{R}", "kind": "ar", "temporal_depth": R,
                   "supervised_horizon": R, "batch_size": args.batch_size,
                   "oom": True, "error": str(e)}
        rows.append(rec)
        print("   ", rec)

    for h in direct_h:
        print(f"\n  DT fixed target h={h}")
        try:
            rec = _run_arm(cfg, dataset_name, "direct", h, args.batch_size,
                           precision, args.n_warmup, args.n_repeat, device,
                           float(cfg.get("max_grad_norm", 1.0)))
        except torch.cuda.OutOfMemoryError as e:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            rec = {"arm": f"dt_h{h}", "kind": "direct", "temporal_depth": 1,
                   "supervised_horizon": h, "batch_size": args.batch_size,
                   "oom": True, "error": str(e)}
        rows.append(rec)
        print("   ", rec)

    result = {
        "dataset": dataset_name,
        "gpu": state,
        "precision": precision,
        "batch_size": args.batch_size,
        "n_warmup": args.n_warmup,
        "n_repeat": args.n_repeat,
        "rows": rows,
    }
    out = args.out or os.path.join(cfg.get("results_dir", "./results"),
                                   "training_cost",
                                   f"{dataset_name}_training_depth.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fp:
        json.dump(result, fp, indent=2, default=json_default)
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
