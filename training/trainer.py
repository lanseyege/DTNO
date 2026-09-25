"""
The Trainer.  Loop mechanics only — device placement, DDP, AMP, optimiser, LR
schedule, gradient accumulation and clipping, checkpointing, logging.  What to
compute lives in `training/tasks.py`.

Two things here are specific to this project rather than boilerplate:

1. Model selection (§24).  The monitored quantity is NOT one-step validation
   error.  It is

       Eval = (1/|H_val|) sum_{h in H_val} E(h)

   computed by running the shared horizon evaluator on validation-trajectory
   anchors.  Selecting on one-step error would systematically pick the
   checkpoint best at h = 1, which is the AR model's home turf and the exact
   regime the direct model is not competing in.  It also means AR-FNO and
   DT-FNO are selected by an identical criterion.

2. `hval_every`.  The horizon-integrated validation is the expensive part
   (len(H_val) forwards per anchor, and a full AR rollout to max(H_val)), so it
   runs every N epochs while the cheap same-distribution validation runs every
   epoch.  Best-checkpoint selection only ever uses the horizon-integrated
   number.

Optimiser defaults follow §24: AdamW, lr 1e-3, weight decay 1e-4, cosine decay,
BF16 autocast, gradient clipping at 1.0.  BF16 is safe here because
SpectralConv2d casts to fp32 internally.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from evaluation.runner import build_predictor, run_horizon_eval

_AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None, "off": None}


class Trainer:
    def __init__(self, model: nn.Module, task, train_loader, val_loader,
                 cfg: dict, data_info: dict, model_name: str,
                 hval_loader=None, hval_info: Optional[dict] = None):
        self.cfg = cfg
        self.task = task
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.hval_loader = hval_loader
        self.hval_info = hval_info
        self.data_info = data_info

        self.distributed = bool(cfg.get("distributed", False))
        self.device = torch.device(cfg.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.is_master = (not self.distributed) or dist.get_rank() == 0
        self.world = dist.get_world_size() if self.distributed else 1

        model = model.to(self.device)
        self.raw_model = model
        if self.distributed:
            ddp_kw = dict(find_unused_parameters=bool(
                cfg.get("ddp_find_unused", False)))
            if self.device.type == "cuda":
                ddp_kw["device_ids"] = [int(os.environ.get("LOCAL_RANK", 0))]
            model = nn.parallel.DistributedDataParallel(model, **ddp_kw)
        self.model = model

        amp_key = str(cfg.get("amp_dtype", "bf16")).lower()
        self.amp_dtype = _AMP.get(amp_key, torch.bfloat16) \
            if bool(cfg.get("amp", True)) and self.device.type == "cuda" else None
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.amp_dtype == torch.float16))

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(cfg.get("lr", 1e-3)),
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
            betas=tuple(cfg.get("betas", (0.9, 0.999))))
        self.accum = int(cfg.get("accumulate_gradients", 1))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
        self.max_epochs = int(cfg.get("max_epochs", 100))
        self.steps_per_epoch = len(train_loader)
        self._optim_steps_per_epoch = max(1, math.ceil(self.steps_per_epoch / self.accum))
        self.warmup_steps = int(cfg.get(
            "warmup_steps", 0.05 * self.max_epochs * self._optim_steps_per_epoch))
        self.scheduler = self._build_scheduler()

        self.epoch = 0
        self.global_step = 0
        self.optim_step = 0
        self.best_metric = float("inf")
        self.log_interval = int(cfg.get("log_interval", 50))
        self.hval_every = int(cfg.get("hval_every", 5))
        self.save_every = int(cfg.get("save_every", 25))
        self.save_dir = cfg.get("save_dir", "./checkpoints/default")
        self.history = []
        if self.is_master:
            os.makedirs(self.save_dir, exist_ok=True)

        self.writer = None
        if self.is_master and bool(cfg.get("tensorboard", True)):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(
                    log_dir=cfg.get("tb_dir", os.path.join("runs", cfg.get(
                        "exp_name", "default"))))
            except Exception as e:                        # tensorboard optional
                print(f"  [warn] TensorBoard disabled: {e}")

        if self.is_master:
            print(f"  Trainer | model={model_name} task={task.name} "
                  f"amp={amp_key} accum={self.accum} world={self.world} "
                  f"steps/epoch={self.steps_per_epoch}")

    # -- schedule ---------------------------------------------------------
    def _build_scheduler(self, last_epoch: int = -1):
        total = max(self.max_epochs * self._optim_steps_per_epoch, 1)
        warm = min(self.warmup_steps, total - 1)

        def fn(step):
            if step < warm:
                return (step + 1) / max(warm, 1)
            prog = (step - warm) / max(total - warm, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, fn,
                                                 last_epoch=last_epoch)

    def _autocast(self):
        if self.amp_dtype is not None:
            return torch.autocast("cuda", dtype=self.amp_dtype)
        return torch.autocast("cpu", enabled=False)

    def _to_device(self, batch):
        return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    def _sync_device(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _reset_train_peak_memory(self):
        """Reset CUDA peak counters immediately before the training epoch.

        Validation and horizon evaluation are intentionally excluded.  The
        resulting numbers therefore measure the peak resident memory of the
        actual forward/backward/optimizer path used by each training arm.
        """
        if self.device.type != "cuda":
            return
        self._sync_device()
        torch.cuda.reset_peak_memory_stats(self.device)

    def _train_memory_stats(self) -> Dict[str, float]:
        if self.device.type != "cuda":
            return {}
        self._sync_device()
        vals = torch.tensor([
            torch.cuda.max_memory_allocated(self.device) / 1e9,
            torch.cuda.max_memory_reserved(self.device) / 1e9,
            torch.cuda.memory_allocated(self.device) / 1e9,
        ], device=self.device, dtype=torch.float64)
        # DDP ranks can differ slightly because their sampled batches differ.
        # Report the maximum rank: that is the memory a job actually needs.
        if self.distributed:
            dist.all_reduce(vals, op=dist.ReduceOp.MAX)
        return {
            "train/peak_allocated_GB": float(vals[0]),
            "train/peak_reserved_GB": float(vals[1]),
            "train/end_allocated_GB": float(vals[2]),
        }

    # -- train ------------------------------------------------------------
    def train_epoch(self) -> float:
        self.model.train()
        self.task.set_epoch(self.epoch)
        for ds in (self.train_loader.dataset,):
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(self.epoch)

        total, n = 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)
            with self._autocast():
                loss, logs = self.task.training_step(self.model, batch)
                loss = loss / self.accum

            if not torch.isfinite(loss):
                # The old message named one cause -- a channel with sigma ~ 0 --
                # and it has now sent two debugging sessions in the wrong
                # direction, because in both the statistics were fine and a
                # different model on the same data trained without complaint.
                # Report what was actually observed and list the causes in the
                # order they have actually occurred.
                raise RuntimeError(
                    f"non-finite loss at epoch {self.epoch} step {i}: "
                    f"{float(loss)}.\n"
                    f"  amp={self.amp_dtype}, model={self.model_name}.\n"
                    f"  Causes seen in this codebase, most recent first:\n"
                    f"    1. a module computing in bf16 that needs fp32. "
                    f"`autocast(enabled=False)` does\n"
                    f"       NOT upcast -- ops run in the dtype of their input, "
                    f"so a bf16 tensor stays\n"
                    f"       bf16 and loses autocast's own promotion of "
                    f"normalisation layers. Upcast\n"
                    f"       explicitly, as SpectralConv2d does.\n"
                    f"    2. an autoregressive rollout diverging within the "
                    f"training step: check whether\n"
                    f"       the failure follows the sampled rollout length "
                    f"`r` in the step log.\n"
                    f"    3. normalisation statistics with a near-zero sigma "
                    f"on some channel.\n"
                    f"  If another model trains on this same config and split, "
                    f"3 is already ruled out.")

            self.scaler.scale(loss).backward()

            gn = None
            if (i + 1) % self.accum == 0:
                self.scaler.unscale_(self.optimizer)
                gn = float(nn.utils.clip_grad_norm_(self.model.parameters(),
                                                    self.max_grad_norm))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.optim_step += 1

            total += logs.get("loss", float(loss.detach()) * self.accum)
            n += 1
            self.global_step += 1

            if self.writer is not None:
                for k, v in logs.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"train/{k}", v, self.global_step)
                self.writer.add_scalar("train/lr",
                                       self.scheduler.get_last_lr()[0],
                                       self.global_step)
                if gn is not None:
                    self.writer.add_scalar("train/grad_norm", gn, self.global_step)

            if self.is_master and (i + 1) % self.log_interval == 0:
                extra = " ".join(f"{k}={v:.3e}" for k, v in logs.items()
                                 if k not in ("loss",) and isinstance(v, float))
                print(f"    e{self.epoch} [{i+1}/{self.steps_per_epoch}] "
                      f"loss={total/n:.5f} lr={self.scheduler.get_last_lr()[0]:.2e} "
                      + (f"gn={gn:.2f} " if gn else "") + extra)

        return total / max(n, 1)

    # -- validation -------------------------------------------------------
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Cheap, same-distribution validation, bucketed by horizon."""
        self.model.eval()
        hs, rels = [], []
        agg: Dict[str, float] = {}
        n = 0
        for batch in self.val_loader:
            batch = self._to_device(batch)
            with self._autocast():
                logs = self.task.validation_step(self.raw_model, batch)
            hs.append(logs.pop("_h"))
            rels.append(logs.pop("_rel_per_sample"))
            for k, v in logs.items():
                if isinstance(v, float):
                    agg[k] = agg.get(k, 0.0) + v
            n += 1

        out = {k: v / max(n, 1) for k, v in agg.items()}
        h = torch.cat(hs).numpy()
        r = torch.cat(rels).numpy()
        if self.distributed:
            t = torch.tensor([out.get("rel_l2", 0.0), 1.0], device=self.device)
            dist.all_reduce(t)
            out["rel_l2"] = float(t[0] / t[1])
        # log-bucketed error: what the multi-horizon picture looks like on the
        # training distribution, cheap enough to watch every epoch
        edges = [1, 2, 4, 8, 16, 32, 64, 128, 1 << 30]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (h >= lo) & (h < hi)
            if m.any():
                out[f"rel_l2_h[{lo},{hi})"] = float(r[m].mean())
        return out

    @torch.no_grad()
    def horizon_validate(self) -> Optional[Dict[str, object]]:
        """§24: integrated error over H_val on validation-trajectory anchors."""
        if self.hval_loader is None:
            return None
        self.model.eval()
        pred = build_predictor(self.raw_model, self.model_name, self.hval_info)
        return run_horizon_eval(
            pred, self.hval_loader, self.hval_info, device=self.device,
            light=True, amp_dtype=self.amp_dtype, distributed=self.distributed,
            max_batches=self.cfg.get("hval_max_batches"))

    # -- checkpoints ------------------------------------------------------
    def save_checkpoint(self, name: str = "checkpoint.pth"):
        if not self.is_master:
            return
        ckpt = {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch, "global_step": self.global_step,
            "optim_step": self.optim_step, "best_metric": self.best_metric,
            "model_name": self.model_name, "config": self.cfg,
            "data_info": {k: v for k, v in self.data_info.items()
                          if isinstance(v, (int, float, str, list, dict))},
        }
        path = os.path.join(self.save_dir, name)
        torch.save(ckpt, path)
        print(f"    saved {path}")

    def load_checkpoint(self, path: str, model_only: bool = False):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(ck["model"])
        if model_only:
            return
        self.optimizer.load_state_dict(ck["optimizer"])
        self.scaler.load_state_dict(ck["scaler"])
        self.epoch = ck["epoch"] + 1
        self.global_step = ck["global_step"]
        self.optim_step = ck.get("optim_step", 0)
        self.best_metric = ck.get("best_metric", float("inf"))
        self.scheduler = self._build_scheduler(last_epoch=self.optim_step - 1)
        if self.is_master:
            print(f"  resumed from {path} at epoch {self.epoch}")

    # -- main loop --------------------------------------------------------
    def train(self):
        if self.is_master:
            print(f"\n{'='*68}\nTraining {self.model_name} for {self.max_epochs} "
                  f"epochs on {self.world} device(s)\n{'='*68}")

        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            if self.distributed and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            self._reset_train_peak_memory()
            self._sync_device()
            t0 = time.time()
            train_loss = self.train_epoch()
            self._sync_device()
            t1 = time.time()
            train_mem = self._train_memory_stats()
            val = self.validate()
            t2 = time.time()

            hval = None
            run_hval = ((epoch + 1) % self.hval_every == 0
                        or epoch == self.max_epochs - 1)
            if run_hval:
                hval = self.horizon_validate()
            t3 = time.time()

            batch_size = int(getattr(self.train_loader, "batch_size", 1) or 1)
            global_examples = self.steps_per_epoch * batch_size * self.world
            rec = {"epoch": epoch, "train_loss": train_loss,
                   "time_train_s": t1 - t0, "time_val_s": t2 - t1,
                   "train/examples_per_s": global_examples / max(t1 - t0, 1e-12),
                   "train/temporal_depth": float(getattr(self.task, "R", 1)),
                   **train_mem,
                   **{f"val/{k}": v for k, v in val.items()}}
            if hval:
                rec["hval/eval_score"] = hval["eval_score"]
                rec["hval/eval_score_bounded"] = hval.get(
                    "eval_score_bounded", hval["eval_score"])
                rec["hval/E_field"] = hval["E_field"]
                rec["hval/horizons"] = hval["horizons"]
                rec["time_hval_s"] = t3 - t2

            if self.is_master:
                self.history.append(rec)
                if self.writer:
                    for k, v in rec.items():
                        if isinstance(v, float):
                            self.writer.add_scalar(k.replace("val/", "val/"), v, epoch)
                head = (f"\nEpoch {epoch}/{self.max_epochs} | loss {train_loss:.5f} "
                        f"| val rel_l2 {val.get('rel_l2', float('nan')):.5f}")
                if hval:
                    curve = " ".join(f"h{h}={e:.3f}" for h, e in
                                     zip(hval["horizons"], hval["E_field"]))
                    sel = hval.get("eval_score_bounded", hval["eval_score"])
                    head += f" | Eval {sel:.5f}"
                    if hval.get("diverged_horizons"):
                        head += f" (diverged at h={hval['diverged_horizons']})"
                    head += f"\n    {curve}"
                head += (f"\n    train {t1-t0:.0f}s val {t2-t1:.0f}s"
                         + (f" hval {t3-t2:.0f}s" if hval else ""))
                if train_mem:
                    head += (f" | peak alloc {train_mem['train/peak_allocated_GB']:.2f} GB"
                             f" reserved {train_mem['train/peak_reserved_GB']:.2f} GB")
                print(head)

                if hval:
                    # BOUNDED score, not the plain mean. The plain mean over
                    # H_val is dominated by whichever horizon the rollout blew
                    # up on: AR runs in the sampling sweep reported "Best Eval =
                    # 740" and "= 1595", meaning checkpoint selection was being
                    # driven entirely by the size of a divergence rather than by
                    # forecast quality. That silently weakens the AR baseline,
                    # which is the one thing §23 says must not happen.
                    self._maybe_save_best(
                        hval.get("eval_score_bounded", hval["eval_score"]))
                if (epoch + 1) % self.save_every == 0:
                    self.save_checkpoint(f"checkpoint_epoch{epoch}.pth")
                with open(os.path.join(self.save_dir, "history.json"), "w") as fp:
                    json.dump(self.history, fp, indent=2)

            if self.distributed:
                dist.barrier()

        if self.is_master:
            self.save_checkpoint("final_model.pth")
            print(f"\nDone. Best Eval (§24) = {self.best_metric:.6f}")
            if self.writer:
                self.writer.close()

    def _maybe_save_best(self, score: float):
        if score < self.best_metric:
            self.best_metric = score
            self.save_checkpoint("best_model.pth")
            print(f"    new best Eval={score:.6f}")
