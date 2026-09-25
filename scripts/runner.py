"""
One evaluation path for every model.

`scripts/train.py` (model selection, §24) and `scripts/evaluate_horizon.py` (the
paper numbers, §25-31) must not have two implementations of "error at horizon h"
that can drift apart.  They share this module.

A `Predictor` adapts each model to one call:

    predict(x, grid, horizons) -> (B, n_h, H, W, C)

which is where the AR-vs-direct asymmetry is handled honestly:

  * DirectPredictor issues one forward per horizon (that IS the model);
  * ARPredictor rolls out once to max(h) and snapshots the requested horizons.
    Snapshotting is exactly what a per-horizon rollout would produce, so the
    ACCURACY is unaffected, and the COST is measured separately by
    `evaluation/timing.py` where the rollout is actually paid for.  Reporting AR
    cost from this loop would understate it by len(horizons)x.

`run_horizon_eval` streams anchors through the predictor and fills the §25-30
accumulators.  Set `light=True` for the in-training call: field metrics only,
which is the model-selection criterion and nothing more.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from .field_metrics import HorizonMetrics, predictability_horizon
from .spectra import SpectrumAccumulator
from .combustion_metrics import CombustionMetrics
from data.channels import (reaction_zone_channel, temperature_channel,
                           find_channel)


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

class Predictor:
    name = "predictor"
    is_direct = True

    def predict(self, x, grid, horizons) -> torch.Tensor:
        raise NotImplementedError

    def n_model_evals(self, h: int) -> int:
        return 1


class DirectPredictor(Predictor):
    def __init__(self, model, dt: float, t_scale: float, name: str = "dt_fno"):
        self.model, self.dt, self.t_scale, self.name = model, dt, t_scale, name

    @torch.no_grad()
    def predict(self, x, grid, horizons):
        outs = []
        for h in horizons:
            tau = torch.full((x.shape[0],), h * self.dt / self.t_scale,
                             device=x.device, dtype=torch.float32)
            outs.append(self.model(x, tau, grid).unsqueeze(1))
        return torch.cat(outs, dim=1)

    def n_model_evals(self, h):
        return 1


class ARPredictor(Predictor):
    is_direct = False

    def __init__(self, model, name: str = "ar_fno"):
        self.model, self.name = model, name

    @torch.no_grad()
    def predict(self, x, grid, horizons):
        hs = [int(h) for h in horizons]
        out = self.model.rollout(x, max(hs), grid, collect=hs)
        # rollout returns the collected horizons in ascending step order
        order = sorted(range(len(hs)), key=lambda i: hs[i])
        inv = [0] * len(hs)
        for pos, i in enumerate(order):
            inv[i] = pos
        return out[:, inv]

    def n_model_evals(self, h):
        return int(h)


class BaselinePredictor(Predictor):
    """Wraps Persistence / POD-DMD / climatology: they take (x, h) -> one frame."""

    def __init__(self, baseline, name: Optional[str] = None):
        self.baseline = baseline
        self.name = name or getattr(baseline, "name", "baseline")

    def set_context(self, batch):
        # the climatology oracle needs to know which trajectory each anchor
        # came from; everything else ignores this
        fn = getattr(self.baseline, "set_context", None)
        if fn is not None and "traj" in batch:
            fn(batch["traj"].tolist())

    @torch.no_grad()
    def predict(self, x, grid, horizons):
        return torch.cat([self.baseline.predict(x, int(h)).unsqueeze(1)
                          for h in horizons], dim=1)

    def n_model_evals(self, h):
        return int(self.baseline.n_model_evals(h))


def build_predictor(model, model_name: str, info: dict) -> Predictor:
    if model_name in ("dt_fno", "sg_dt_fno"):
        return DirectPredictor(model, info["dt"], info["t_scale"], model_name)
    if model_name == "ar_fno":
        return ARPredictor(model, model_name)
    return BaselinePredictor(model, model_name)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_horizon_eval(predictor: Predictor, loader, info: dict,
                     device="cuda", light: bool = False,
                     amp_dtype: Optional[torch.dtype] = None,
                     distributed: bool = False,
                     max_batches: Optional[int] = None,
                     combustion_every: int = 1,
                     alpha_flame: float = 0.2,
                     e_threshold: float = 0.3) -> Dict[str, object]:
    """Stream anchors, fill the §25-30 accumulators, return one metrics dict."""
    horizons = list(info["horizons"])
    names = list(info["channel_names"])
    dev = torch.device(device)

    field = HorizonMetrics(horizons, names, info.get("channel_groups"),
                           device=dev)
    spec = None if light else SpectrumAccumulator(
        horizons, len(names), n_bins=min(info["H"], info["W"]) // 2 + 1,
        device=dev)
    comb = None
    if not light:
        hrr = reaction_zone_channel(names)
        temp = temperature_channel(names)
        grad_ch = [c for c in [temp, hrr, find_channel(names, r"\bOH\b", r"of_OH")]
                   if c >= 0]
        comb = CombustionMetrics(
            horizons, names, info["normalizer"], alpha=alpha_flame,
            grad_channels=sorted(set(grad_ch)),
            hrr_channel=hrr, temp_channel=temp, mask_channel=hrr)

    model = getattr(predictor, "model", None)
    if isinstance(model, torch.nn.Module):
        model.eval()

    autocast = (torch.autocast("cuda", dtype=amp_dtype)
                if amp_dtype is not None and dev.type == "cuda"
                else torch.autocast("cpu", enabled=False))

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = batch["x"].to(dev, non_blocking=True)
        y = batch["y"].to(dev, non_blocking=True)              # (B, n_h, H, W, C)
        grid = batch.get("grid")
        grid = grid.to(dev, non_blocking=True) if grid is not None else None
        ctx = getattr(predictor, "set_context", None)
        if ctx is not None:
            ctx(batch)

        with autocast:
            pred = predictor.predict(x, grid, horizons)
        pred = pred.float()

        for hi in range(len(horizons)):
            field.update(hi, pred[:, hi], y[:, hi])
            if spec is not None:
                spec.update(hi, pred[:, hi], y[:, hi])
            if comb is not None and bi % combustion_every == 0:
                comb.update(hi, pred[:, hi], y[:, hi])

    out: Dict[str, object] = {"model": predictor.name}
    out.update(field.compute(distributed=distributed,
                             normalizer=info.get("normalizer")))
    out["n_model_evals"] = [predictor.n_model_evals(h) for h in horizons]

    # §24's integrated score, plus a bounded companion. The plain mean stops
    # being a summary the moment an autoregressive rollout diverges: a single
    # E = 1e20 point drags the mean to 1e19 and the number no longer ranks
    # anything. `eval_score_bounded` caps each horizon at 1 -- "no better than
    # predicting the training mean" -- so a diverged horizon contributes its
    # worst legitimate value instead of an arbitrarily large one.
    E = [float(v) for v in out["E_field"]]
    n = max(len(horizons), 1)
    out["eval_score"] = float(sum(E) / n)
    # A non-finite E is a FAILED forecast, not a large error, and `min(nan, 1)`
    # is nan -- so without this one horizon turns the whole bounded score into
    # nan and an otherwise reportable run becomes unreportable. Score it at the
    # bound (1.0, "no better than predicting the training mean"), which is the
    # least generous defensible value, and keep the count so the failure is
    # reported rather than absorbed.
    def _bounded(v):
        return 1.0 if not (v == v and abs(v) != float("inf")) else min(v, 1.0)
    out["eval_score_bounded"] = float(sum(_bounded(v) for v in E) / n)
    out["nonfinite_horizons"] = [
        int(h) for h, v in zip(horizons, E)
        if not (v == v and abs(v) != float("inf"))]
    out["diverged_horizons"] = [
        int(h) for h, v in zip(horizons, E)
        if (v == v and abs(v) != float("inf") and v > 3.0)
        or not (v == v and abs(v) != float("inf"))]
    out["T_pred"] = predictability_horizon(horizons, out["E_field"], e_threshold)
    out["T_pred_threshold"] = e_threshold
    if spec is not None:
        out["spectral"] = spec.compute(distributed=distributed)
    if comb is not None:
        out["combustion"] = comb.compute(distributed=distributed)
    return out
