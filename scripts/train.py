#!/usr/bin/env python
"""
Training entry point.

    # single GPU
    python scripts/train.py --config configs/sg_dt_fno.yaml

    # 4 x A800
    torchrun --standalone --nproc_per_node 4 \
        scripts/train.py --config configs/sg_dt_fno.yaml

    # a subset of a shared node
    CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node 2 \
        scripts/train.py --config configs/sg_dt_fno.yaml

    # seeds (§43 wants 3 for every headline number)
    for s in 0 1 2; do
      torchrun --standalone --nproc_per_node 4 scripts/train.py \
        --config configs/sg_dt_fno.yaml --seed $s \
        --set experiment.save_dir=./checkpoints/sg_dt_fno_s$s
    done

    # an ablation arm with no new config file
    python scripts/train.py --config configs/sg_dt_fno.yaml \
        --set training.lambda_sg=0.2 --set experiment.exp_name=sg_lam0.2

Model selection is the §24 criterion — mean E(h) over H_val on
validation-trajectory anchors — not one-step validation error.  See
`training/trainer.py` for why that distinction decides the whole comparison.
"""

from __future__ import annotations

import json
import os
import pprint

import torch

from common import (base_parser, resolve, set_seed, setup_distributed,  # noqa
                    cleanup_distributed, is_master, json_default)

from data import build_data, build_eval_loader                          # noqa: E402
from models import build_model, model_kind, count_params                # noqa: E402
from training import build_task, Trainer                                # noqa: E402


def main():
    ap = base_parser("Train AR-FNO / DT-FNO / SG-DT-FNO")
    ap.add_argument("--resume", nargs="?", const="auto", default=None)
    ap.add_argument("--checkpoint", default=None,
                    help="load weights only (fine-tune / cross-dataset start)")
    args = ap.parse_args()
    cfg = resolve(args)

    distributed = setup_distributed(cfg)
    master = is_master()
    set_seed(int(cfg["seed"]) + (int(os.environ.get("RANK", 0)) if distributed else 0))

    dataset_name = cfg.get("dataset_name", "realpde_combustion")
    model_name = cfg.get("model_name", "dt_fno")
    kind = model_kind(model_name)

    # ---- data ----------------------------------------------------------
    bundle = build_data(dataset_name, cfg, model_kind=kind,
                        distributed=distributed, seed=int(cfg["seed"]))
    train_loader = bundle["train_loader"]
    val_loader = bundle["val_loader"]
    info = bundle["data_info"]

    # §24 horizon-integrated validation on validation-trajectory anchors
    hval_loader = hval_info = None
    val_h = cfg.get("val_horizons")
    if val_h:
        hcfg = dict(cfg)
        hcfg["eval_stride"] = cfg.get("hval_stride", 200)
        hval_loader, hval_info = build_eval_loader(
            dataset_name, hcfg, subset="val", horizons=val_h,
            distributed=distributed)

    # ---- model ---------------------------------------------------------
    model = build_model(model_name, cfg, info)
    task = build_task(model_name, cfg, info)

    if master:
        print("=" * 72)
        print(f"MODEL {model_name}  |  DATASET {dataset_name}  |  "
              f"TASK {task.name}  |  SEED {cfg['seed']}")
        print("=" * 72)
        print(f"  {info['store_summary']}")
        print(f"  split      : {info['split'].summary()}")
        print(f"  channels   : {len(info['channel_names'])} -> "
              f"{info['channel_names']}")
        print(f"  groups     : { {g: len(v) for g, v in info['channel_groups'].items()} }")
        print(f"  history K  : {info['K']}   dt = {info['dt']:g} s   "
              f"t_scale = {info['t_scale']:g} s")
        print(f"  horizons   : {info['horizon_sampler']}")
        if hval_info:
            print(f"  H_val (§24): {hval_info['horizons']} on "
                  f"{hval_info['n_anchors']} anchors")
        print(f"  {model.summary()}")
        p = count_params(model)
        print(f"  params     : {p}")
        print("  NOTE §3: compare this total across ar_fno / dt_fno / "
              "sg_dt_fno before trusting any accuracy gap.")
        if cfg.get("verbose_config"):
            pprint.pprint(cfg)

    trainer = Trainer(model=model, task=task, train_loader=train_loader,
                      val_loader=val_loader, cfg=cfg, data_info=info,
                      model_name=model_name, hval_loader=hval_loader,
                      hval_info=hval_info)

    if args.resume is not None:
        path = args.resume
        if path == "auto":
            cand = os.path.join(cfg.get("save_dir", ""), "final_model.pth")
            ckpts = sorted(
                (f for f in os.listdir(cfg.get("save_dir", "."))
                 if f.startswith("checkpoint_epoch")),
                key=lambda f: int(f.split("epoch")[1].split(".")[0])
            ) if os.path.isdir(cfg.get("save_dir", "")) else []
            path = (os.path.join(cfg["save_dir"], ckpts[-1]) if ckpts
                    else (cand if os.path.isfile(cand) else ""))
        if path:
            trainer.load_checkpoint(path)
        elif master:
            print("  [resume] nothing to resume from; starting fresh")
    elif args.checkpoint:
        trainer.load_checkpoint(args.checkpoint, model_only=True)

    if master:
        os.makedirs(cfg.get("save_dir", "."), exist_ok=True)
        with open(os.path.join(cfg["save_dir"], "config_resolved.json"), "w") as fp:
            json.dump(cfg, fp, indent=2, default=json_default)

    trainer.train()
    cleanup_distributed()


if __name__ == "__main__":
    main()
