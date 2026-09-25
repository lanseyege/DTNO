#!/usr/bin/env python
"""
Phase 1-4 evaluation (§25-31) — the numbers that go in the paper.

Produces, for one model, on held-out TEST trajectories, at every horizon in
`eval_horizons`:

    E_field(h)              §25   relative L2, per channel and averaged
    E_group(h)              §26   flow / thermo / chemistry / reaction
    E_spec(h)               §27   spectral error + the full radial spectra
    E_grad(h)               §28   gradient error on T, OH, heat release
    IoU, flame area, ...    §29   reaction-zone structure
    integrated HRR, ...     §30   global physical quantities
    C_SG                    §31   semigroup consistency (direct models only)
    N_model_evals(h)        §32   exact, hardware-independent cost
    T_pred                  §45   largest h with E_field < threshold

Usage:
    python scripts/evaluate_horizon.py --config configs/sg_dt_fno.yaml \
        --checkpoint checkpoints/sg_dt_fno/best_model.pth

    # baselines need no checkpoint
    python scripts/evaluate_horizon.py --config configs/dt_fno.yaml \
        --model persistence
    python scripts/evaluate_horizon.py --config configs/dt_fno.yaml \
        --model pod_dmd --dmd artifacts/pod_dmd.npz

    # 4 GPUs (anchors are sharded, metrics all-reduced)
    torchrun --standalone --nproc_per_node 4 scripts/evaluate_horizon.py \
        --config configs/sg_dt_fno.yaml \
        --checkpoint checkpoints/sg_dt_fno/best_model.pth

Writes results/<exp_name>/horizon_metrics.json, which is the only input
`scripts/make_figures.py` needs.

The evaluation uses ONE anchor set for every horizon and every model (§25), so
the curves are comparable point by point rather than by distribution.
"""

from __future__ import annotations

import json
import os

import torch

from common import (base_parser, resolve, set_seed, setup_distributed,   # noqa
                    cleanup_distributed, is_master, json_default)

from data import build_eval_loader                                       # noqa: E402
from models import build_model                                           # noqa: E402
from models.baselines import (Persistence, PODDMD,                       # noqa: E402
                              TrajectoryClimatology, NearestTrainClimatology)
from evaluation.runner import build_predictor, run_horizon_eval          # noqa: E402
from evaluation.semigroup_metrics import (evaluate_semigroup,            # noqa: E402
                                          identity_consistency,
                                          DEFAULT_TAU_PAIRS)


def load_model(cfg, info, model_name, checkpoint, device):
    model = build_model(model_name, cfg, {
        "C": info["C"], "K": info["K"], "H": info["H"], "W": info["W"]})
    if checkpoint:
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ck.get("model", ck)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  [warn] state_dict mismatch: {len(missing)} missing, "
                  f"{len(unexpected)} unexpected. First few: "
                  f"{missing[:3]} / {unexpected[:3]}")
        saved = ck.get("model_name")
        if saved and saved != model_name:
            raise ValueError(f"checkpoint was trained as '{saved}' but the "
                             f"config says '{model_name}'")
    return model.to(device).eval()


def main():
    ap = base_parser("Horizon evaluation (§25-31)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--model", default=None,
                    help="override model_name; also accepts persistence | "
                         "pod_dmd | climatology | nearest_climatology")
    ap.add_argument("--dmd", default="artifacts/pod_dmd.npz")
    ap.add_argument("--subset", default="test", choices=["test", "val", "train"])
    ap.add_argument("--horizons", type=int, nargs="+", default=None)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--light", action="store_true",
                    help="field metrics only; skips §27-30")
    ap.add_argument("--combustion_every", type=int, default=1,
                    help="run §28-30 on every Nth batch (they are the slow ones)")
    ap.add_argument("--flame_alpha", type=float, default=0.2)
    ap.add_argument("--e_threshold", type=float, default=0.3,
                    help="E_field threshold defining T_pred (§45)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = resolve(args)
    distributed = setup_distributed(cfg)
    master = is_master()
    set_seed(int(cfg["seed"]))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available()
                                  else "cpu"))

    dataset_name = cfg.get("dataset_name", "realpde_combustion")
    model_name = args.model or cfg.get("model_name", "dt_fno")

    loader, info = build_eval_loader(dataset_name, cfg, subset=args.subset,
                                     horizons=args.horizons,
                                     distributed=distributed)
    info["K"] = int(cfg.get("history_len", 4))

    if master:
        print("=" * 72)
        print(f"EVALUATE {model_name}  |  subset={args.subset}  |  "
              f"{info['n_anchors']} anchors on trajectories {info['traj_indices']}")
        print(f"horizons: {info['horizons']}")
        print("=" * 72)

    # ---- build the thing being evaluated --------------------------------
    if model_name == "persistence":
        obj = Persistence()
    elif model_name == "nearest_climatology":
        from data.store import build_store
        from data.realpde import resolve_channels
        from data.splits import Split
        st = build_store(cfg)
        ci, _ = resolve_channels(st, cfg.get("channels"))
        sp = Split.load(cfg.get("split_path", "artifacts/split_realpde.json"))
        obj = NearestTrainClimatology(
            k=int(cfg.get("nn_clim_k", 1)),
            temperature=float(cfg.get("nn_clim_temperature", 0.0))
        ).fit(st, sp.train, ci, info["normalizer"],
              t_stride=int(cfg.get("climatology_stride", 10)), verbose=master)
    elif model_name == "climatology":
        from data.store import build_store
        from data.realpde import resolve_channels
        st = build_store(cfg)
        ci, _ = resolve_channels(st, cfg.get("channels"))
        obj = TrajectoryClimatology().fit(
            st, info["traj_indices"], ci, info["normalizer"],
            t_stride=int(cfg.get("climatology_stride", 10)), verbose=master)
    elif model_name == "pod_dmd":
        if not os.path.exists(args.dmd):
            raise FileNotFoundError(
                f"{args.dmd} not found. Fit it first:\n"
                f"    python scripts/fit_dmd.py --config {args.config}")
        obj = PODDMD.load(args.dmd)
    else:
        if not args.checkpoint:
            raise ValueError(f"--checkpoint is required for '{model_name}'")
        obj = load_model(cfg, info, model_name, args.checkpoint, device)
    if master:
        print(f"  {obj.summary()}\n")

    predictor = build_predictor(obj, model_name, info)

    amp = ({"bf16": torch.bfloat16, "fp16": torch.float16}
           .get(str(cfg.get("amp_dtype", "bf16")).lower())
           if bool(cfg.get("amp", True)) and device.type == "cuda" else None)

    results = run_horizon_eval(
        predictor, loader, info, device=device, light=args.light,
        amp_dtype=amp, distributed=distributed, max_batches=args.max_batches,
        combustion_every=args.combustion_every, alpha_flame=args.flame_alpha,
        e_threshold=args.e_threshold)

    # ---- §31 semigroup consistency (direct models only) -----------------
    if model_name in ("dt_fno", "sg_dt_fno"):
        pairs = cfg.get("sg_eval_pairs", DEFAULT_TAU_PAIRS)
        pairs = [tuple(p) for p in pairs]
        results["semigroup"] = evaluate_semigroup(
            obj, loader, info["dt"], info["t_scale"], tau_pairs=pairs,
            device=device, max_batches=args.max_batches or 16,
            distributed=distributed)
        results["semigroup"].update(
            identity_consistency(obj, loader, device=device, max_batches=8))

    results["meta"] = {
        "model_name": model_name,
        "exp_name": cfg.get("exp_name", model_name),
        "checkpoint": args.checkpoint,
        "subset": args.subset,
        "n_anchors": info["n_anchors"],
        "trajectories": info["traj_indices"],
        "channel_names": info["channel_names"],
        "channel_groups": info["channel_groups"],
        "dt": info["dt"], "t_scale": info["t_scale"],
        "h_max_train": cfg.get("h_max_train"),
        "train_horizons": cfg.get("train_horizons"),
        "seed": cfg.get("seed"),
        # Provenance for anything that changes what was TRAINED. Curves from
        # runs that differ in these fields are not comparable, and mixing them
        # on one axis has already cost this project two full sweeps.
        "train_config": {
            "loss_channel_weights": cfg.get("loss_channel_weights"),
            "h_max_train": cfg.get("h_max_train"),
            "history_len": cfg.get("history_len"),
            "time_embed_mode": cfg.get("time_embed_mode"),
            "predict_delta": cfg.get("predict_delta"),
            "horizon_sampling": cfg.get("horizon_sampling"),
            "ar_rollout": cfg.get("ar_rollout"),
            "ar_random_rollout": cfg.get("ar_random_rollout"),
            "ar_step_weights": cfg.get("ar_step_weights"),
            "batch_size": cfg.get("batch_size"),
            "samples_per_epoch": cfg.get("samples_per_epoch"),
        },
    }

    if master:
        out = args.out or os.path.join(
            cfg.get("results_dir", "./results"),
            cfg.get("exp_name", model_name), "horizon_metrics.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fp:
            json.dump(results, fp, indent=2, default=json_default)

        print(f"\n{'h':>6}{'N_eval':>8}{'E_field':>10}" +
              "".join(f"{g[:9]:>10}" for g in results.get("E_group", {})))
        for i, h in enumerate(results["horizons"]):
            row = (f"{h:>6}{results['n_model_evals'][i]:>8}"
                   f"{results['E_field'][i]:>10.4f}")
            for g, vals in results.get("E_group", {}).items():
                row += f"{vals[i]:>10.4f}"
            print(row)
        div = results.get("diverged_horizons", [])
        if div:
            print(f"\n  Eval (§24, bounded)   = "
                  f"{results['eval_score_bounded']:.5f}   <- use this one")
            print(f"  Eval (§24, plain mean) = {results['eval_score']:.5g}"
                  f"   meaningless here")
            print(f"  [!] DIVERGED at h = {div}. The rollout blew up rather "
                  f"than degraded;\n      report those horizons as 'diverged', "
                  f"not as a large error value.")
        else:
            print(f"\n  Eval (§24 integrated) = {results['eval_score']:.5f}")
        tp = results["T_pred"]
        print(f"  T_pred (E_field < {args.e_threshold}) = "
              f"{'never reached' if tp is None else f'{tp:.1f} frames'}")
        if "semigroup" in results:
            sg = results["semigroup"]
            print(f"  C_SG = {sg['C_SG_mean']:.4f} (field {sg['C_SG_field_mean']:.4f})"
                  f" | C_identity = {sg['C_identity']:.4f}"
                  f" | latent RMS = {sg['latent_rms_mean']:.4f}")
            print("  §31 reminder: C_SG falling is only meaningful together "
                  "with E_field falling.")
        print(f"\n  -> {out}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
