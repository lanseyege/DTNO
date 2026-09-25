#!/usr/bin/env python
"""Dense multi-horizon inference benchmark for the DTNO factorization.

Compares three ways to query the same trained direct model:

  naive          M complete forwards: encode + propagate + decode, repeated Mx
  cached_seq     encode once, then M sequential propagate/decode calls
  cached_batch   encode once, query several target times in one horizon batch

Optionally also times the matched AR checkpoint, which naturally emits all
intermediate states in one rollout.  This is the honest benchmark for the
reviewer question "CFD users usually need the whole trajectory, not only h=128".

Example:

  CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_multi_horizon.py \
    --config configs/gray_scott.yaml \
    --dt checkpoints/gs_dt_fno_s0/best_model.pth \
    --ar checkpoints/gs_ar_fno_r_s0/best_model.pth \
    --query_counts 1 4 8 16 32 64 128 --chunks 4 8 16 32 0

``chunks=0`` means "all requested horizons in one batch".  Smaller chunks bound
memory while still amortizing the history encoder.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from contextlib import nullcontext
from typing import Callable, Dict, List

import torch

from common import base_parser, resolve, set_seed, json_default  # noqa: E402
from data import build_eval_loader                              # noqa: E402
from models import build_model                                  # noqa: E402
from evaluation.timing import gpu_state, warm_gpu               # noqa: E402


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _ctx(device, precision):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def _stats(xs: List[float]) -> Dict[str, float]:
    ys = sorted(xs)
    lo, hi = ys[:len(ys)//2], ys[(len(ys)+1)//2:]
    q1 = statistics.median(lo) if lo else ys[0]
    q3 = statistics.median(hi) if hi else ys[-1]
    return {"median_ms": statistics.median(ys), "iqr_ms": q3-q1,
            "min_ms": ys[0], "max_ms": ys[-1]}


@torch.no_grad()
def _measure(fn: Callable[[], torch.Tensor], device, precision: str,
             n_warmup: int, n_repeat: int) -> Dict[str, float]:
    for _ in range(n_warmup):
        with _ctx(device, precision):
            fn()
    _sync(device)

    ts, peaks = [], []
    for _ in range(n_repeat):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        t0 = time.perf_counter()
        with _ctx(device, precision):
            out = fn()
        _sync(device)
        ts.append((time.perf_counter() - t0) * 1e3)
        if device.type == "cuda":
            peaks.append(torch.cuda.max_memory_allocated(device) / 1e9)
        # Keep the output alive through synchronization, then release it.
        del out
    rec = _stats(ts)
    if peaks:
        rec["peak_allocated_GB"] = max(peaks)
    return rec


def _load(cfg, info, name, ckpt, device):
    model = build_model(name, cfg, {"C": info["C"], "K": info["K"]})
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state), strict=False)
    return model.to(device).eval()


def main():
    ap = base_parser("Amortized multi-horizon inference benchmark")
    ap.add_argument("--dt", required=True, help="DT-FNO checkpoint")
    ap.add_argument("--ar", default=None, help="optional matched AR-FNO checkpoint")
    ap.add_argument("--query_counts", nargs="+", type=int,
                    default=[1, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--chunks", nargs="+", type=int, default=[4, 8, 16, 32, 0],
                    help="0 = all queried horizons in one batch")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--precision", choices=["fp32", "bf16", "fp16"],
                    default="fp32")
    ap.add_argument("--n_warmup", type=int, default=5)
    ap.add_argument("--n_repeat", type=int, default=20)
    ap.add_argument("--warmup_seconds", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = resolve(args)
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = cfg.get("dataset_name", "realpde_combustion")
    hmax = int(cfg.get("h_max_train", 128))
    max_q = min(max(args.query_counts), hmax)

    loader, info = build_eval_loader(dataset_name, cfg, subset="test", horizons=[1])
    info["K"] = int(cfg.get("history_len", 4))
    batch = next(iter(loader))
    x = batch["x"][:args.batch_size].to(device)
    grid = batch.get("grid")
    grid = grid[:args.batch_size].to(device) if grid is not None else None

    dt = _load(cfg, info, "dt_fno", args.dt, device)
    ar = _load(cfg, info, "ar_fno", args.ar, device) if args.ar else None
    state = gpu_state(device)
    if args.warmup_seconds > 0:
        warm_gpu(device, args.warmup_seconds)

    # Sanity check before timing: caching/batching must be numerically identical
    # (up to normal floating point reordering) to repeated full forwards.
    check_h = list(range(1, min(8, max_q) + 1))
    tau_check = torch.tensor([h * info["dt"] / info["t_scale"] for h in check_h],
                             device=device, dtype=torch.float32)
    with torch.no_grad(), _ctx(device, args.precision):
        naive_check = torch.stack([
            dt(x, torch.full((x.shape[0],), float(t), device=device), grid)
            for t in tau_check
        ], dim=1)
        cached_check = dt.predict_horizons(x, tau_check, grid, chunk_size=4)
    max_abs = float((naive_check.float() - cached_check.float()).abs().max().cpu())
    rel = float(torch.linalg.vector_norm((naive_check-cached_check).float()).cpu()
                / (torch.linalg.vector_norm(naive_check.float()).cpu() + 1e-12))
    del naive_check, cached_check

    print("=" * 78)
    print("AMORTIZED MULTI-HORIZON INFERENCE")
    print("=" * 78)
    print(f"dataset={dataset_name} input={tuple(x.shape)} precision={args.precision}")
    print(f"equivalence check: max_abs={max_abs:.3e}, relative={rel:.3e}")

    rows = []
    for M0 in args.query_counts:
        M = min(int(M0), hmax)
        if M < 1:
            continue
        hs = list(range(1, M + 1))
        taus = torch.tensor([h * info["dt"] / info["t_scale"] for h in hs],
                            device=device, dtype=torch.float32)

        def naive():
            return torch.stack([
                dt(x, torch.full((x.shape[0],), float(t), device=device), grid)
                for t in taus
            ], dim=1)

        rec = {"n_queries": M, "max_horizon": M, "methods": {}}
        rec["methods"]["naive"] = _measure(
            naive, device, args.precision, args.n_warmup, args.n_repeat)
        rec["methods"]["cached_seq"] = _measure(
            lambda: dt.predict_horizons(x, taus, grid, chunk_size=1),
            device, args.precision, args.n_warmup, args.n_repeat)

        for c0 in args.chunks:
            c = M if int(c0) == 0 else min(int(c0), M)
            key = f"cached_batch_{c}"
            if key in rec["methods"]:
                continue
            try:
                rec["methods"][key] = _measure(
                    lambda c=c: dt.predict_horizons(x, taus, grid, chunk_size=c),
                    device, args.precision, args.n_warmup, args.n_repeat)
            except torch.cuda.OutOfMemoryError as e:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                rec["methods"][key] = {"oom": True, "error": str(e)}

        if ar is not None:
            rec["methods"]["ar_dense_rollout"] = _measure(
                lambda: ar.rollout(x, M, grid, collect=hs),
                device, args.precision, max(2, args.n_warmup // 2),
                max(5, args.n_repeat // 2))

        naive_ms = rec["methods"]["naive"]["median_ms"]
        for k, v in rec["methods"].items():
            if "median_ms" in v:
                v["speedup_vs_naive"] = naive_ms / max(v["median_ms"], 1e-12)
                v["ms_per_query"] = v["median_ms"] / M
        rows.append(rec)

        print(f"\n  M={M} dense horizons 1..{M}")
        for k, v in rec["methods"].items():
            if v.get("oom"):
                print(f"    {k:>20}: OOM")
            else:
                mem = f" peak={v.get('peak_allocated_GB', float('nan')):.2f}GB" \
                      if device.type == "cuda" else ""
                print(f"    {k:>20}: {v['median_ms']:9.3f} ms  "
                      f"{v['speedup_vs_naive']:6.2f}x vs naive{mem}")

    result = {
        "dataset": dataset_name,
        "checkpoint_dt": args.dt,
        "checkpoint_ar": args.ar,
        "gpu": state,
        "precision": args.precision,
        "batch_size": args.batch_size,
        "equivalence": {"max_abs": max_abs, "relative": rel},
        "rows": rows,
    }
    out = args.out or os.path.join(cfg.get("results_dir", "./results"),
                                   "timing",
                                   f"multi_horizon_{dataset_name}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fp:
        json.dump(result, fp, indent=2, default=json_default)
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
