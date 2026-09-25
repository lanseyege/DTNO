#!/usr/bin/env python
"""
Figure 4 (§46) — qualitative fields: ground truth vs AR / DT / SG-DT.

Renders, at a chosen horizon and anchor, the four variables §46 asks for
(temperature, OH, heat release, velocity/vorticity) for every model, in
PHYSICAL units, plus the error map.  Also overlays the reaction-zone contour
used by the §29 IoU, which is usually the row that makes the argument: two
models with nearly identical MSE frequently put the flame in visibly different
places, and that is the whole point of §29 existing.

Usage:
    python scripts/plot_fields.py --config configs/sg_dt_fno.yaml \
        --checkpoints ar_fno=checkpoints/ar_fno_r/best_model.pth \
                      dt_fno=checkpoints/dt_fno/best_model.pth \
                      sg_dt_fno=checkpoints/sg_dt_fno/best_model.pth \
        --h 128 --anchor 0
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

from common import base_parser, resolve, set_seed   # noqa: E402

from data import build_eval_loader                   # noqa: E402
from models import build_model                       # noqa: E402
from evaluation.runner import build_predictor        # noqa: E402
from data.channels import (temperature_channel, reaction_zone_channel,  # noqa
                           find_channel, velocity_channels)
from evaluation.combustion_metrics import reaction_mask   # noqa: E402


def check_checkpoint(path, st, cfg, info, names):
    """Refuse a checkpoint that was not trained on the dataset being plotted.

    `load_state_dict(..., strict=False)` raises on a SHAPE mismatch and is
    silent about everything else. That is enough to catch pointing a Lifted H2
    config at a RealPDEBench checkpoint --- 13 channels against 5, 16x16 modes
    against 16x20, so it crashes --- but it is not enough in general. Three of
    our five datasets share 16x16 modes, and any two runs that agree on the
    channel count and the mode counts load silently: a second seed, a re-run,
    an ablation of the same dataset, or simply the wrong tag prefix. The figure
    would then be produced without complaint and captioned with the wrong
    trajectory and the wrong model.

    The checkpoint records `config` and `data_info`, so the check is exact
    rather than heuristic. It runs before build_model, so the message names the
    problem instead of describing tensor shapes.
    """
    ck_cfg = st.get("config") or {}
    ck_info = st.get("data_info") or {}
    want_ds = cfg.get("dataset_name", "realpde_combustion")
    got_ds = ck_cfg.get("dataset_name", "realpde_combustion")
    problems = []
    if got_ds != want_ds:
        problems.append(f"dataset: checkpoint was trained on {got_ds!r}, "
                        f"this config is {want_ds!r}")
    if ck_info.get("C") not in (None, info["C"]):
        problems.append(f"channels: checkpoint has C={ck_info['C']}, "
                        f"this dataset has C={info['C']}")
    ck_names = ck_info.get("channel_names")
    if ck_names and list(ck_names) != list(names):
        problems.append(f"channel names differ:\n"
                        f"        checkpoint {list(ck_names)}\n"
                        f"        config     {list(names)}")
    for k in ("modes1", "modes2", "width", "n_layers", "history_len"):
        a, b = ck_cfg.get(k), cfg.get(k)
        if a is not None and b is not None and a != b:
            problems.append(f"{k}: checkpoint {a}, this config {b}")
    if problems:
        raise SystemExit(
            f"\n  {path}\n  was not trained on the dataset this config "
            f"describes:\n\n    - " + "\n    - ".join(problems)
            + f"\n\n  Checkpoint tags are '<dataset-prefix>_<model>_s<seed>'. "
              f"The prefix is empty for\n  the original RealPDEBench runs and "
              f"set for every other dataset (lh2, cyl,\n  gs, rb), so "
              f"'checkpoints/ar_fno_r_s0' is the COMBUSTION run whatever config "
              f"you\n  pass alongside it. Available:\n"
            + "\n".join(f"    {d}" for d in sorted(
                  os.listdir("checkpoints"))[:40]
                  if os.path.isdir(os.path.join("checkpoints", d)))
            if os.path.isdir("checkpoints") else "")


def main():
    ap = base_parser("Figure 4 — qualitative fields")
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    metavar="NAME=PATH")
    ap.add_argument("--h", type=int, default=128)
    ap.add_argument("--anchor", type=int, default=0)
    ap.add_argument("--subset", default="test")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--max_panels", type=int, default=4,
                    help="non-reacting datasets only: how many of the "
                         "dataset's own channels to draw")
    ap.add_argument("--tag", default=None,
                    help="filename tag; defaults to meta.dataset_name. Without "
                         "it, two datasets write the same file.")
    ap.add_argument("--titles", action="store_true",
                    help="draw the 'Figure 4 -- ...' heading inside the image. "
                         "OFF by default: it duplicates the LaTeX caption in a "
                         "different font and goes stale when figures are "
                         "renumbered. Everything it said is written to a .txt "
                         "beside the figure so the caption can be written from "
                         "it.")
    ap.add_argument("--label", default=None, metavar="TEXT",
                    help="dataset tag drawn inside the first panel, e.g. "
                         "--label 'A2 Cylinder'. The three field figures sit in "
                         "the same appendix and have different row channels, so "
                         "without it the reader identifies them by guessing.")
    ap.add_argument("--names", nargs="+", default=None, metavar="KEY=LABEL",
                    help="column headers, e.g. --names ar_fno='AR-FNO-R'. "
                         "Defaults map the checkpoint keys onto the paper's "
                         "model names.")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    cfg = resolve(args)
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader, info = build_eval_loader(cfg.get("dataset_name",
                                             "realpde_combustion"),
                                     cfg, subset=args.subset,
                                     horizons=[args.h])
    info["K"] = int(cfg.get("history_len", 4))
    ds = loader.dataset
    sample = ds[min(args.anchor, len(ds) - 1)]
    x = sample["x"].unsqueeze(0).to(device)
    y = sample["y"].unsqueeze(0).to(device)          # (1, 1, H, W, C)
    grid = sample["grid"].unsqueeze(0).to(device)
    traj, t0 = int(sample["traj"]), int(sample["t0"])

    names = info["channel_names"]
    norm = info["normalizer"]

    # Column headers. The checkpoint key is whatever was typed on the command
    # line ("ar_fno"), which is a build_model() identifier and not the name the
    # paper uses for that arm; a reader comparing the figure with Table 2 has
    # to translate. --names overrides any of these.
    COLUMN = {"ar_fno": "AR-FNO-R", "ar_fno_r": "AR-FNO-R",
              "ar_fno_1": "AR-FNO-1", "dt_fno": "DT-FNO",
              "sg_dt_fno": "SG-DT-FNO", "pod_dmd": "POD-DMD"}
    for spec in (args.names or []):
        k, _, v = spec.partition("=")
        if not v:
            raise SystemExit(f"--names expects KEY=LABEL, got {spec!r}")
        COLUMN[k] = v

    preds = {}
    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            raise SystemExit(f"\n  no such checkpoint: {path}")
        st = torch.load(path, map_location="cpu", weights_only=False)
        check_checkpoint(path, st, cfg, info, names)
        model = build_model(name, cfg, {"C": info["C"], "K": info["K"]})
        sd = st.get("model", st)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # strict=False is kept because a checkpoint may carry buffers a later
        # refactor dropped, but silence about it is how a half-loaded model
        # gets plotted. Report, and refuse if anything with weights is missing.
        real = [k for k in missing if k.endswith((".weight", ".bias",
                                                  "weights1", "weights2"))]
        if real:
            raise SystemExit(
                f"\n  {path}\n  is missing {len(real)} parameter tensor(s) "
                f"the model needs, e.g. {real[:3]}.\n  This is not a "
                f"compatible checkpoint for '{name}'.")
        if missing or unexpected:
            print(f"    {name}: {len(missing)} missing / {len(unexpected)} "
                  f"unexpected key(s), none of them parameters")
        model = model.to(device).eval()
        pred = build_predictor(model, name, info).predict(x, grid, [args.h])
        preds[name] = norm.inverse_torch(pred[:, 0].float())[0].cpu().numpy()

    truth = norm.inverse_torch(y[:, 0].float())[0].cpu().numpy()

    ch_T = temperature_channel(names)
    # Same patterns reaction_zone_channel uses internally for its OH fallback
    # (data/channels.py). The original \bOH\b, of_OH pair does not match
    # BLASTNet's "YOH" -- Y is a word character immediately before O, so there
    # is no word boundary there -- which silently mislabelled the OH field as
    # "Heat release" on Lifted H2 (ch_OH resolved to -1, was filtered out by
    # the c >= 0 guard below, and ch_Q's fallback-to-OH channel was drawn under
    # the wrong title instead). Caught by testing against the real channel
    # names before running it on a checkpoint.
    ch_OH = find_channel(names, r"\bOH\b", r"_OH$", r"of_OH", r"^yoh$")
    ch_Q = reaction_zone_channel(names)
    # On datasets with no heat-release channel (Lifted H2), ch_Q falls back to
    # OH, so ch_OH and ch_Q are the SAME index -- without a dedupe this renders
    # two identical panels labelled "OH" and "Heat release". mask_label records
    # which physical quantity actually defines the contour, so the caption can
    # say the right thing regardless of which dataset this is run on.
    mask_label = "Heat release" if ch_Q != ch_OH else "OH (reaction-zone mask)"
    ch_V = velocity_channels(names)
    panels = [(ch_T, "Temperature [K]", "magma")]
    if ch_Q != ch_OH:
        panels.append((ch_OH, "OH", "viridis"))
    panels.append((ch_Q, mask_label, "inferno" if ch_Q != ch_OH else "viridis"))
    if ch_V:
        panels.append((ch_V[0], names[ch_V[0]], "RdBu_r"))
    panels = [(c, t, cm) for c, t, cm in panels if c is not None and c >= 0]

    # ------------------------------------------------------------------
    # NON-REACTING DATASETS.
    #
    # Everything above selects channels by combustion semantics, and on a
    # non-reacting flow every one of those lookups returns -1. The `c >= 0`
    # filter then leaves Gray-Scott with ZERO panels -- `plt.subplots(0, cols)`
    # -- and Cylinder and Rayleigh-Benard with exactly one, because
    # `velocity_channels` finds u and v but only `ch_V[0]` is ever used. A
    # one-panel Figure 4 showing the streamwise velocity is not a bug that
    # announces itself; it just looks like a thin figure.
    #
    # So when the combustion selection finds fewer than two channels, fall
    # back to plotting the dataset's own channels in order. This is the right
    # default for a flow with no flame: there is no distinguished subset, and
    # the figure's job is the same either way -- show that the rollout has
    # become noise while the direct model still has structure.
    has_reaction = ch_Q is not None and ch_Q >= 0
    if len(panels) < 2:
        cmaps = {"u": "RdBu_r", "v": "RdBu_r", "pressure": "coolwarm",
                 "buoyancy": "magma", "velocity_x": "RdBu_r",
                 "velocity_y": "RdBu_r"}
        order = list(range(len(names)))
        panels = [(c, names[c], cmaps.get(names[c].lower(), "viridis"))
                  for c in order[:args.max_panels]]
        print(f"  no reacting channels in {names}; plotting the first "
              f"{len(panels)} channel(s) instead and omitting the "
              f"reaction-zone contour.")
    print(f"  reaction-zone mask channel: "
          f"{names[ch_Q] if has_reaction else 'NONE'}"
          f"  ({'heat release' if has_reaction and ch_Q != ch_OH else ('OH fallback -- no heat-release channel in this dataset' if has_reaction else 'non-reacting dataset -- no contour drawn')})")

    cols = 1 + len(preds)
    fig, axes = plt.subplots(len(panels), cols,
                             figsize=(2.8 * cols, 2.7 * len(panels)))
    axes = np.atleast_2d(axes)

    for r, (c, title, cmap) in enumerate(panels):
        #vmin = float(min(truth[..., c].min(),
        #                 *[p[..., c].min() for p in preds.values()]))
        vmin = float(truth[..., c].min())
        #vmax = float(max(truth[..., c].max(),
        #                 *[p[..., c].max() for p in preds.values()]))
        vmax = float(truth[..., c].max())
        fields = ([("Ground truth", truth)]
                  + [(COLUMN.get(k, k), v) for k, v in preds.items()])
        for j, (label, f) in enumerate(fields):
            ax = axes[r, j]
            im = ax.imshow(f[..., c], cmap=cmap, vmin=vmin, vmax=vmax,
                           origin="lower")
            if has_reaction:
                m = reaction_mask(torch.from_numpy(f[..., ch_Q]).unsqueeze(0),
                                  args.alpha)[0].numpy()
                ax.contour(m.astype(float), levels=[0.5], colors="cyan",
                           linewidths=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(label, fontsize=10)
            if j == 0:
                ax.set_ylabel(title, fontsize=9)
        fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.02, pad=0.01)

    # dt spans 5 us (Lifted H2) to 10 s (Gray-Scott), so a hard-coded "ms"
    # prints h=128 on Gray-Scott as 1280000.00 ms.
    secs = args.h * float(info["dt"])
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9)):
        if secs >= scale or scale == 1e-9:
            t_str = f"{secs / scale:.4g} {unit}"
            break
    title = (f"Figure 4 — trajectory {traj}, anchor $t_0$={t0}, "
             f"$h$={args.h} ({t_str}).")
    if has_reaction:
        title += (f" Cyan contour: reaction zone at $\\alpha$={args.alpha} of "
                  f"peak {mask_label.split(' (')[0].lower()} (§29).")
    if args.titles:
        fig.suptitle(title, fontsize=10)

    # The dataset tag goes inside the first panel rather than above the figure,
    # so it survives the image being scaled, moved or reordered in the
    # document. The three field figures have DIFFERENT row channels --
    # Temperature/OH/UX on Lifted H2, Temperature/OH/heat release/velocity on
    # RealPDEBench, u/v/pressure on Cylinder -- which is correct, each dataset
    # has its own state, but it means the rows cannot identify the dataset the
    # way a shared axis would.
    if args.label:
        lab = args.label.replace("---", "\u2014").replace("--", "\u2013")
        axes[0, 0].text(0.03, 0.97, lab, transform=axes[0, 0].transAxes,
                        fontsize=9, fontweight="bold", va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                  edgecolor="0.7", linewidth=0.6, alpha=0.9),
                        zorder=6)

    # The filename used to be fig4_fields_h{h}, which is the same for every
    # dataset: running this on Cylinder after Gray-Scott silently overwrote the
    # first figure. Tag it.
    tag = args.tag or cfg.get("dataset_name", "dataset")
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"fig4_fields_{tag}_h{args.h}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=160, bbox_inches="tight")

    # With the title off, nothing inside the image records WHICH trajectory,
    # anchor or horizon it shows, and that is exactly the metadata a caption
    # must state for the figure to be checkable. Write it beside the file.
    rows = ", ".join(t for _, t, _ in panels)
    note = [f"Caption material for {os.path.basename(stem)}.pdf",
            "",
            f"  dataset      {cfg.get('dataset_name', '?')}",
            f"  label        {args.label or '(none)'}",
            f"  trajectory   {traj}   (test split, index into the frozen split)",
            f"  anchor t0    {t0}",
            f"  horizon      h = {args.h}  =  {t_str}",
            f"  rows         {rows}",
            f"  columns      Ground truth, "
            + ", ".join(COLUMN.get(k, k) for k in preds),
            f"  colour scale per row, fixed to the ground truth's range",
            ""]
    if has_reaction:
        note += [f"  contour      reaction zone at alpha = {args.alpha} of peak "
                 f"{mask_label.split(' (')[0].lower()} (see §29)",
                 "               State this in the caption; it is no longer in "
                 "the image.", ""]
    else:
        note += ["  contour      none -- non-reacting dataset. Do not copy the "
                 "reaction-zone", "               sentence from the combustion "
                 "captions into this one.", ""]
    with open(f"{stem}.txt", "w") as fp:
        fp.write("\n".join(note))
    print(f"  -> {stem}.png/.pdf/.txt")
    print(f"     rows: {rows}")
    if not args.titles:
        print("     Title omitted. The caption must carry trajectory, anchor, "
              "horizon and\n     the contour convention -- all of them are in "
              "the .txt beside the figure.")


if __name__ == "__main__":
    main()
