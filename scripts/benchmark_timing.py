#!/usr/bin/env python
"""
Inference cost benchmark (§32) — "one of the most important figures".

Protocol, exactly as §32 specifies and for the reasons it gives:
    batch size 1, one GPU, fixed precision, GPU warm-up,
    torch.cuda.synchronize() around every measurement, repeated trials,
    median + IQR, model loading excluded.

Run this on an IDLE GPU.  The script records `nvidia-smi` utilisation at start
and stamps a warning into the output when the card is already busy — on a shared
node that is the difference between a figure and an artefact.  Pin the device:

    CUDA_VISIBLE_DEVICES=3 python scripts/benchmark_timing.py \
        --config configs/sg_dt_fno.yaml \
        --ar checkpoints/ar_fno_r/best_model.pth \
        --dt checkpoints/dt_fno/best_model.pth \
        --sg checkpoints/sg_dt_fno/best_model.pth

N_model_evals is reported alongside wall-clock and is the number that survives
methodological argument: it is exact, hardware-independent, and unaffected by
whatever else is sharing the node.

By default AR rollouts are timed up to h = 128 and linearly extrapolated beyond
(flagged in the JSON).  Pass --max_timed_horizon 512 for the paper run; timing a
512-step rollout ten times is minutes of GPU for a number that is linear by
construction, which is fine once and wasteful while iterating.
"""

from __future__ import annotations

import json
import os

import torch

from common import base_parser, resolve, set_seed, json_default        # noqa: E402

from data import build_eval_loader                                      # noqa: E402
from models import build_model                                          # noqa: E402
from evaluation.timing import (benchmark_direct, benchmark_ar,          # noqa: E402
                               gpu_state, time_callable, warm_gpu)


def _load(cfg, info, name, ckpt, device):
    model = build_model(name, cfg, {"C": info["C"], "K": info["K"]})
    if ckpt:
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(st.get("model", st), strict=False)
    return model.to(device).eval()


def main():
    ap = base_parser("Inference cost benchmark (§32)")
    ap.add_argument("--ar", default=None, help="AR-FNO checkpoint")
    ap.add_argument("--dt", default=None, help="DT-FNO checkpoint")
    ap.add_argument("--sg", default=None, help="SG-DT-FNO checkpoint")
    ap.add_argument("--random_weights", action="store_true",
                    help="time untrained models — timing does not depend on "
                         "weights, and this lets you produce the figure before "
                         "training finishes")
    ap.add_argument("--horizons", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--max_timed_horizon", type=int, default=128)
    ap.add_argument("--n_repeat", type=int, default=30)
    ap.add_argument("--n_warmup", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--warmup_seconds", type=float, default=3.0,
                    help="sustained GPU load before any timing, to reach steady "
                         "clocks; 0 disables (and reintroduces the clock-ramp "
                         "artifact described in evaluation/timing.warm_gpu)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = resolve(args)
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["amp"] = False                       # §32: identical, fixed precision

    dataset_name = cfg.get("dataset_name", "realpde_combustion")
    loader, info = build_eval_loader(dataset_name, cfg, subset="test",
                                     horizons=[1])
    info["K"] = int(cfg.get("history_len", 4))

    batch = next(iter(loader))
    x = batch["x"][: args.batch_size].to(device)
    grid = batch["grid"][: args.batch_size].to(device)

    # Record the card's state BEFORE warming it. Reading it afterwards means
    # measuring our own warm-up: the 3 s matmul loop leaves utilisation at 100%,
    # gpu_state() then stamps a "GPU was already busy" warning into the output,
    # and a clean run looks contaminated.
    state = gpu_state(device)

    if args.warmup_seconds > 0:
        print(f"  warming the GPU for {args.warmup_seconds:.0f}s to reach "
              f"steady clocks...")
        warm_gpu(device, args.warmup_seconds)
    print("=" * 72)
    print("INFERENCE COST BENCHMARK (§32)")
    print("=" * 72)
    for k, v in state.items():
        print(f"  {k}: {v}")
    print(f"  input: {tuple(x.shape)}  batch_size={args.batch_size}  "
          f"precision=fp32  repeats={args.n_repeat}")
    print()

    results = {"gpu": state, "warmup_seconds": args.warmup_seconds,
               "horizons": args.horizons,
               "batch_size": args.batch_size, "precision": "fp32",
               "n_repeat": args.n_repeat, "models": {}}

    specs = [("ar_fno", args.ar), ("dt_fno", args.dt), ("sg_dt_fno", args.sg)]
    for name, ckpt in specs:
        if ckpt is None and not args.random_weights:
            continue
        model = _load(cfg, info, name, ckpt, device)
        n_par = sum(p.numel() for p in model.parameters())
        if name == "ar_fno":
            rows = benchmark_ar(model, x, args.horizons, grid,
                                n_warmup=max(5, args.n_warmup // 2),
                                n_repeat=max(8, args.n_repeat // 3),
                                max_timed_horizon=args.max_timed_horizon)
        else:
            rows = benchmark_direct(model, x, args.horizons, info["dt"],
                                    info["t_scale"], grid,
                                    n_warmup=args.n_warmup,
                                    n_repeat=args.n_repeat)
        results["models"][name] = {"params": n_par, "checkpoint": ckpt,
                                   "rows": rows}

        print(f"  {name}  ({n_par:,} params)")
        print(f"    {'h':>6}{'N_eval':>8}{'median_ms':>12}{'IQR_ms':>10}  note")
        for r in rows:
            note = "extrapolated" if r.get("extrapolated") else ""
            print(f"    {r['h']:>6}{r['n_model_evals']:>8}"
                  f"{r['median_ms']:>12.3f}{r['iqr_ms']:>10.3f}  {note}")
        print()

    # speedup table — the headline of Figure 2
    if "ar_fno" in results["models"] and "dt_fno" in results["models"]:
        ar = {r["h"]: r for r in results["models"]["ar_fno"]["rows"]}
        dt = {r["h"]: r for r in results["models"]["dt_fno"]["rows"]}
        speed = []
        print(f"  {'h':>6}{'AR ms':>10}{'DT ms':>10}{'speedup':>10}"
              f"{'eval ratio':>12}")
        for h in args.horizons:
            if h in ar and h in dt:
                s = ar[h]["median_ms"] / max(dt[h]["median_ms"], 1e-9)
                e = ar[h]["n_model_evals"] / max(dt[h]["n_model_evals"], 1)
                speed.append({"h": h, "wallclock_speedup": s, "eval_ratio": e})
                print(f"  {h:>6}{ar[h]['median_ms']:>10.2f}"
                      f"{dt[h]['median_ms']:>10.2f}{s:>9.1f}x{e:>11.0f}x")
        results["speedup"] = speed

    # A second reading at the end catches contention that started mid-run --
    # the failure mode the first reading cannot see.
    results["gpu_after"] = gpu_state(device)
    u0 = state.get("gpu_util_percent_at_start")
    u1 = results["gpu_after"].get("gpu_util_percent_at_start")
    m1 = results["gpu_after"].get("gpu_mem_used_MB_at_start")
    m0 = state.get("gpu_mem_used_MB_at_start")
    if None not in (m0, m1) and m1 - m0 > 2000:
        print(f"\n  [!] GPU memory in use grew {m0} -> {m1} MiB during the run: "
              f"another job\n      landed on this card mid-benchmark. Re-run "
              f"on a quiet GPU before using these numbers.")

    out = args.out or os.path.join(cfg.get("results_dir", "./results"),
                                   "timing", "inference_cost.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fp:
        json.dump(results, fp, indent=2, default=json_default)
    print(f"\n  -> {out}")
    if "warning" in state:
        print(f"\n  [!] {state['warning']}")


if __name__ == "__main__":
    main()
