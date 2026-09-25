"""
Tasks — what to compute.  The Trainer owns only loop mechanics.

Two tasks, mirroring §3's three models:

    DirectTask   DT-FNO   (use_semigroup=False)
                 SG-DT-FNO(use_semigroup=True)  -> adds L_SG (§16) and L_id (§17)
    ARTask       AR-FNO-1 (rollout=1)
                 AR-FNO-R (rollout=R, random r per sample)

Splitting on a config flag rather than a class is what makes §3's claim
auditable: Model B and Model C run the same code path with lambda_SG = 0 and
lambda_SG > 0 respectively, so an improvement cannot come from an accidental
architectural or pipeline difference.

Loss composition (§18) is kept minimal on purpose:

    DT-FNO      L = L_pred
    SG-DT-FNO   L = L_pred + lambda_SG L_SG + lambda_id L_id

and nothing else.  No physics residual, no spectral term, no gradient term
(§35).  If the result is good, we need to know which component produced it.

The semigroup split is drawn as a + b = h from the SAMPLED target horizon (see
`data/sampling.split_horizon`), so the direct route is supervised by the actual
target frame while the composed route is constrained onto the same interval.
Sampling an unrelated (tau_a, tau_b) would constrain the operator at intervals
where the prediction loss provides no anchor, which is how latent collapse gets
started.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from losses import (PredictionLoss, semigroup_loss, identity_loss,
                    SemigroupWeight, LatentCollapseMonitor, relative_l2)
from data.sampling import split_horizon


class Task:
    name = "task"
    monitor = "eval_score"          # lower is better

    def __init__(self, cfg: dict, data_info: dict):
        self.cfg = cfg
        self.data_info = data_info
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def training_step(self, model, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError

    def validation_step(self, model, batch) -> Dict[str, float]:
        raise NotImplementedError

    def extra_logs(self) -> Dict[str, float]:
        return {}


# ---------------------------------------------------------------------------
# Direct-time
# ---------------------------------------------------------------------------

class DirectTask(Task):
    name = "direct"

    def __init__(self, cfg: dict, data_info: dict):
        super().__init__(cfg, data_info)
        self.criterion = PredictionLoss(
            kind=str(cfg.get("pred_loss", "nmse")),
            channel_weights=channel_weights(cfg, data_info))
        self.use_sg = bool(cfg.get("use_semigroup", False))
        self.use_id = bool(cfg.get("use_identity", self.use_sg))
        self.lam_id = float(cfg.get("lambda_id", 0.05))
        self.sg_split_mode = str(cfg.get("sg_split_mode", "uniform"))
        self.sg_detach = bool(cfg.get("sg_detach_direct", False))
        self.sg_weight = SemigroupWeight(
            lambda_max=float(cfg.get("lambda_sg", 0.1)),
            warmup_epochs=int(cfg.get("sg_warmup_epochs", 5)),
            ramp_epochs=int(cfg.get("sg_ramp_epochs", 10)),
            lambda_init=float(cfg.get("lambda_sg_init", 0.0)))
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)) + 1234)
        self._last_lambda = 0.0

    # -- helpers ---------------------------------------------------------
    def _tau(self, h: torch.Tensor) -> torch.Tensor:
        return h.float() * self.data_info["dt"] / self.data_info["t_scale"]

    def _split(self, h: torch.Tensor):
        """Per-sample (a, b) with a + b = h; a = 0 marks an unsplittable h = 1."""
        hs = h.detach().cpu().numpy()
        a = np.zeros_like(hs)
        for i, hv in enumerate(hs):
            s = split_horizon(int(hv), self.rng, self.sg_split_mode)
            a[i] = 0 if s is None else s[0]
        a_t = torch.as_tensor(a, device=h.device, dtype=h.dtype)
        return a_t, h - a_t, a_t > 0    # mask: False where h = 1 (unsplittable)

    # -- steps -----------------------------------------------------------
    def training_step(self, model, batch):
        x, y, h = batch["x"], batch["y"], batch["h"]
        grid = batch.get("grid")
        tau = self._tau(h)

        lam_sg = self.sg_weight(self.epoch) if self.use_sg else 0.0
        self._last_lambda = lam_sg
        want_sg = self.use_sg and lam_sg > 0

        tau_a = tau_b = ok = None
        if want_sg:
            h_a, h_b, ok = self._split(h)
            tau_a, tau_b = self._tau(h_a), self._tau(h_b)

        # ONE call, on whatever wrapper the trainer handed us (see the note in
        # DirectTimeFNO.forward about DDP gradient reduction).
        out = model(x, tau, grid, mode="train", tau_a=tau_a, tau_b=tau_b,
                    need_identity=bool(self.use_id and self.lam_id > 0))

        loss, logs = self.criterion(out["pred"], y)
        logs["h_mean"] = float(h.float().mean())

        if want_sg:
            per = semigroup_loss(out["zT"], out["z_compose"],
                                 detach_direct=self.sg_detach, reduction="none")
            m = ok.float()
            l_sg = (per * m).sum() / m.sum().clamp_min(1.0)
            loss = loss + lam_sg * l_sg
            logs["l_sg"] = float(l_sg.detach())
            logs["sg_frac"] = float(m.mean())
            logs["lambda_sg"] = lam_sg

        if "z_id" in out:
            l_id = identity_loss(out["z0"], out["z_id"])
            loss = loss + self.lam_id * l_id
            logs["l_id"] = float(l_id.detach())

        logs.update(LatentCollapseMonitor.stats(out["zT"], "zT"))
        logs["loss"] = float(loss.detach())
        return loss, logs

    @torch.no_grad()
    def validation_step(self, model, batch):
        x, y, h = batch["x"], batch["y"], batch["h"]
        pred = model(x, self._tau(h), batch.get("grid"))
        _, logs = self.criterion(pred, y)
        # keep h so the trainer can bucket the horizon-resolved validation
        logs["_h"] = h.detach().float().cpu()
        logs["_rel_per_sample"] = _per_sample_rel(pred, y).cpu()
        return logs

    def extra_logs(self):
        return {"lambda_sg": self._last_lambda}


# ---------------------------------------------------------------------------
# Autoregressive
# ---------------------------------------------------------------------------

class ARTask(Task):
    name = "ar"

    def __init__(self, cfg: dict, data_info: dict):
        super().__init__(cfg, data_info)
        self.criterion = PredictionLoss(
            kind=str(cfg.get("pred_loss", "nmse")),
            channel_weights=channel_weights(cfg, data_info))
        self.R = int(cfg.get("ar_rollout", 1))
        self.random_rollout = bool(cfg.get("ar_random_rollout", self.R > 1))
        # w_j over the short rollout (§23).  Uniform by default; 'decay' puts
        # more weight on the first step, which keeps AR-FNO-R from trading away
        # its one-step accuracy — the thing it is supposed to be best at.
        self.weight_mode = str(cfg.get("ar_step_weights", "uniform"))

    def _weights(self, r: int, device) -> torch.Tensor:
        if self.weight_mode == "decay":
            w = torch.tensor([0.5 ** j for j in range(r)], device=device)
        else:
            w = torch.ones(r, device=device)
        return w / w.sum()

    def training_step(self, model, batch):
        x, y = batch["x"], batch["y"]                 # y: (B, R, H, W, C)
        grid = batch.get("grid")

        # IMPORTANT: a previous implementation took max(r_eff) across the
        # batch. With batch_size=64 and r_eff~U{1..R}, that maximum is almost
        # always R, so the advertised random-rollout baseline silently became
        # fixed-R training. We draw one deterministic batch-level r by using
        # the first sample's seeded r_eff. This preserves r~U{1..R} without
        # ragged per-sample computation. For the rollout-depth scaling study we
        # set ar_random_rollout=false, so r is exactly R.
        if self.random_rollout and "r_eff" in batch:
            r = int(batch["r_eff"].reshape(-1)[0])
        else:
            r = self.R
        r = max(1, min(r, y.shape[1]))

        pred = model(x, grid, n_steps=r)              # (B, r, H, W, C)
        w = self._weights(r, x.device)

        total = 0.0
        logs: Dict[str, float] = {}
        for j in range(r):
            lj, lg = self.criterion(pred[:, j], y[:, j])
            total = total + w[j] * lj
            if j == 0:
                logs.update({f"step1_{k}": v for k, v in lg.items()})
        logs["loss"] = float(total.detach())
        logs["r"] = float(r)
        return total, logs

    @torch.no_grad()
    def validation_step(self, model, batch):
        x, y = batch["x"], batch["y"]
        pred = model.rollout(x, y.shape[1], batch.get("grid"))
        _, logs = self.criterion(pred[:, 0], y[:, 0])
        logs["_h"] = torch.ones(x.shape[0])
        logs["_rel_per_sample"] = _per_sample_rel(pred[:, 0], y[:, 0]).cpu()
        return logs


def _per_sample_rel(pred, target, eps: float = 1e-8) -> torch.Tensor:
    p = pred.float().reshape(pred.shape[0], -1)
    t = target.float().reshape(target.shape[0], -1)
    return torch.linalg.vector_norm(p - t, dim=1) / (
        torch.linalg.vector_norm(t, dim=1) + eps)


def channel_weights(cfg: dict, data_info: dict):
    """Per-channel weights for the prediction loss, by channel NAME.

    Motivated by a measurement, not a hunch. On RealPDEBench the
    Absolute_Pressure channel is:

      * unpredictable  -- relative L2 >= 0.94 for every model at every horizon,
        including persistence (1.93) and POD-DMD (1.91); at 4 kHz the acoustic
        field is decorrelated between consecutive saved frames;
      * actively harmful to the spectrum -- DT-FNO's per-channel E_spec on
        pressure is 39 / 52 / 63 / 87 at h = 1 / 8 / 32 / 128, while every other
        channel sits at 0.02-0.55. Averaged over 13 channels that single column
        produces the reported E_spec of 3.1-6.9 essentially on its own;
      * still consuming gradient, since normalised MSE weights it like any other
        channel.

    Zeroing it removes a term the model cannot learn and should not be scored
    on. It stays in the INPUT -- pressure gradients drive the flow -- and it can
    still be reported; this only changes what the loss optimises.

        loss_channel_weights:
          Absolute_Pressure: 0.0

    Unlisted channels default to 1.0.
    """
    spec = cfg.get("loss_channel_weights")
    if not spec:
        return None
    names = list(data_info["channel_names"])
    unknown = [k for k in spec if k not in names]
    if unknown:
        raise ValueError(f"loss_channel_weights names not in the channel set: "
                         f"{unknown}. Available: {names}")
    w = [float(spec.get(n, 1.0)) for n in names]
    if sum(w) <= 0:
        raise ValueError("loss_channel_weights zeroes every channel")
    dropped = [n for n, v in zip(names, w) if v == 0.0]
    if dropped:
        print(f"  loss excludes {dropped} (still used as model input and still "
              f"reported in E_channel)")
    return w


def build_task(model_name: str, cfg: dict, data_info: dict) -> Task:
    """Dispatch on the model's KIND, not on a hard-coded list of names.

    This used to be `if model_name == "ar_fno" ... elif "dt_fno" ...`, which is
    a second copy of information `MODEL_REGISTRY` already holds. Adding a
    second backbone made the copies disagree: `ar_unet` and `dt_unet` were
    registered as models but unknown here, and the failure arrived as
    `ValueError: no task for model 'dt_unet'` after the model had already been
    built. `evaluation/runner.build_predictor` held a third copy and failed
    differently and later --- it fell through to the BASELINE predictor, which
    calls `predict(x, int(h))`, so a neural direct-time model was handed an
    integer where it expected tau and raised

        AttributeError: 'int' object has no attribute 'reshape'

    three epochs into training, from inside the horizon validation. All three
    sites now read `model_kind`, so a model that builds can also be trained and
    evaluated, or fails immediately at registration.

    Whether the semigroup losses are on is a CONFIG question, not a name
    question: `MODEL_VARIANTS` already sets `use_semigroup` for `dt_fno`
    (False) and `sg_dt_fno` (True), and any new direct variant sets its own.
    """
    from models import model_kind

    try:
        kind = model_kind(model_name)
    except KeyError:
        from models import MODEL_REGISTRY
        raise ValueError(
            f"no task for model '{model_name}': it is not in MODEL_REGISTRY. "
            f"Known models: {sorted(MODEL_REGISTRY)}. Register it there and "
            f"every dispatch site picks it up.") from None

    if kind == "ar":
        return ARTask(cfg, data_info)
    if kind == "direct":
        use_sg = bool(cfg.get("use_semigroup", False))
        return DirectTask({**cfg, "use_semigroup": use_sg,
                           "use_identity": cfg.get("use_identity", use_sg)},
                          data_info)
    raise ValueError(f"unknown model kind {kind!r} for '{model_name}'; "
                     f"expected 'ar' or 'direct'")
