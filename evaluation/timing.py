"""
Inference cost (§32) — "the most important figure of the whole project".

The measurement protocol is the interesting part, because wall-clock on a shared
node is easy to get wrong in the direction that flatters us.  §32 fixes it:

    batch size = 1
    same GPU for every model
    identical precision setting
    GPU warm-up before timing
    torch.cuda.synchronize() around every measurement
    repeated trials, report MEDIAN + IQR (not mean +- std: a co-tenant job
        landing mid-run produces one 5x outlier that a mean cannot survive and
        a median ignores)
    model loading excluded

And, independently of the clock: N_model_evals(h).  That number is exact,
hardware-independent, unaffected by whatever else is running on the 4 A800s, and
is the honest complexity statement — AR is h, direct is 1.  When a reviewer
disputes the timing methodology, this is the column that survives.

`assert_gpu_exclusive` is a courtesy check: it reads current GPU utilisation and
warns when the device is already busy, because a timing figure produced while
three other jobs share the card is not a timing figure.
"""

from __future__ import annotations

import statistics
import time
from typing import Callable, Dict, List, Optional, Sequence

import torch


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def warm_gpu(device: torch.device, seconds: float = 3.0):
    """Drive the GPU to its sustained clock before ANY measurement.

    An idle A800 sits in a low P-state and ramps only under sustained load, so
    per-iteration warm-up is not enough when the iterations are short. Measured
    on an otherwise idle card: the AR baseline came out at 2.87 ms/step for
    h <= 32 and 1.95 ms/step for h >= 64 -- a clean 32% step, perfectly constant
    within each regime, because the short rollouts finished before the clocks
    came up and the long ones did not.

    That artifact flatters us: it makes AR look slow exactly where AR is
    competitive (small h). One global warm-up before the whole benchmark puts
    every model on the same clock state.
    """
    if device.type != "cuda":
        return
    a = torch.randn(4096, 4096, device=device)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a = torch.mm(a, a).clamp_(-1, 1)
    _sync(device)


@torch.no_grad()
def time_callable(fn: Callable[[], object], device: torch.device,
                  n_warmup: int = 10, n_repeat: int = 30) -> Dict[str, float]:
    """Median + IQR wall-clock, in milliseconds."""
    for _ in range(n_warmup):
        fn()
    _sync(device)

    samples: List[float] = []
    for _ in range(n_repeat):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1e3)

    samples.sort()
    q1 = statistics.median(samples[: len(samples) // 2])
    q3 = statistics.median(samples[(len(samples) + 1) // 2:])
    return {
        "median_ms": statistics.median(samples),
        "iqr_ms": q3 - q1,
        "q1_ms": q1, "q3_ms": q3,
        "min_ms": samples[0], "max_ms": samples[-1],
        "n_repeat": len(samples),
    }


@torch.no_grad()
def benchmark_direct(model, x: torch.Tensor, horizons: Sequence[int],
                     dt: float, t_scale: float, grid=None,
                     n_warmup: int = 10, n_repeat: int = 30) -> List[Dict]:
    """One forward per horizon, regardless of how large the horizon is."""
    device = x.device
    out = []
    for h in horizons:
        tau = torch.full((x.shape[0],), h * dt / t_scale, device=device)
        rec = time_callable(lambda: model(x, tau, grid), device, n_warmup, n_repeat)
        rec.update({"h": int(h), "n_model_evals": int(model.n_model_evals(h))})
        out.append(rec)
    return out


@torch.no_grad()
def benchmark_ar(model, x: torch.Tensor, horizons: Sequence[int], grid=None,
                 n_warmup: int = 3, n_repeat: int = 10,
                 max_timed_horizon: Optional[int] = None) -> List[Dict]:
    """h sequential steps per horizon.

    `max_timed_horizon` caps what is actually measured; beyond it the time is
    EXTRAPOLATED linearly from the largest measured point and flagged
    `extrapolated: true`.  Timing a genuine 512-step rollout 10 times over is
    minutes of GPU for a number that is linear by construction.  Anything used
    in a figure should be measured; use the cap for iteration, not for the paper.
    """
    device = x.device
    out: List[Dict] = []
    ref: Optional[Dict] = None
    for h in horizons:
        if max_timed_horizon is not None and h > max_timed_horizon and ref:
            scale = h / ref["h"]
            out.append({"h": int(h),
                        "median_ms": ref["median_ms"] * scale,
                        "iqr_ms": ref["iqr_ms"] * scale,
                        "q1_ms": ref["q1_ms"] * scale, "q3_ms": ref["q3_ms"] * scale,
                        "min_ms": ref["min_ms"] * scale, "max_ms": ref["max_ms"] * scale,
                        "n_repeat": 0, "extrapolated": True,
                        "n_model_evals": int(model.n_model_evals(h))})
            continue
        rec = time_callable(lambda: model.rollout(x, int(h), grid, collect=[int(h)]),
                            device, n_warmup, n_repeat)
        rec.update({"h": int(h), "n_model_evals": int(model.n_model_evals(h)),
                    "extrapolated": False})
        out.append(rec)
        ref = rec
    return out


def gpu_state(device: torch.device) -> Dict[str, object]:
    """Recorded alongside every timing run so the numbers stay interpretable."""
    if device.type != "cuda":
        return {"device": "cpu"}
    idx = device.index or 0
    props = torch.cuda.get_device_properties(idx)
    info = {
        "device": props.name,
        "total_memory_GB": round(props.total_memory / 1e9, 2),
        "capability": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "allocated_GB": round(torch.cuda.memory_allocated(idx) / 1e9, 3),
    }
    try:                                    # nvidia-smi is not always present
        import subprocess
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "-i", str(idx)],
            capture_output=True, text=True, timeout=5)
        if q.returncode == 0:
            util, mem = (v.strip() for v in q.stdout.strip().split(","))
            info["gpu_util_percent_at_start"] = int(util)
            info["gpu_mem_used_MB_at_start"] = int(mem)
            if int(util) > 10:
                info["warning"] = (
                    f"GPU {idx} was already {util}% busy when timing started; "
                    f"wall-clock numbers from this run are contaminated. "
                    f"N_model_evals is unaffected.")
    except Exception:
        pass
    return info
