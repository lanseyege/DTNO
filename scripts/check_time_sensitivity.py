#!/usr/bin/env python
"""
Does the direct-time operator actually USE the query time? (§38 Test 2)

A flat E(h) curve has two possible readings and Figure 1 cannot tell them apart:

  (a) the operator is horizon-robust -- it keeps predicting well as h grows;
  (b) the operator IGNORES tau -- past some horizon it emits one
      near-climatological field regardless of what was asked, and the error
      curve is flat because the prediction stopped changing.

Reading (b) still beats persistence in L2 and still produces a beautifully flat
inference-cost figure, so it is entirely possible to publish it by accident.

TWO TESTS, and the second is the decisive one.

--- Test 1: sensitivity ratio (diagnostic only) ------------------------------

    S_model(h1, h2) = || G(X, h2) - G(X, h1) || / || G(X, h1) ||
    S_truth(h1, h2) = || U(t+h2)  - U(t+h1)  || / || U(t+h1)  ||
    ratio           = S_model / S_truth

READ THIS WITH CARE. A small ratio at long horizons is NOT by itself evidence
of reading (b). The minimum-MSE predictor is the conditional mean
E[U(t+h) | X_t], and as the flow decorrelates that conditional mean converges to
the climatology, which does not depend on h. So an OPTIMAL predictor also has
S_model -> 0 and ratio -> 0 at long horizons. A low ratio is consistent with
both "ignoring tau" and "correctly hedging towards the climatological mean",
and Test 1 cannot separate them.

What Test 1 is good for: catching S_model == 0 exactly (a frozen output), and
showing where the model's tau-response starts to depart from the physics.

--- Test 2: tau-swap matrix (decisive) ---------------------------------------

Ask the model for the WRONG horizon and see whether it matters:

    M[h][h'] = E_field( G(X, tau(h')),  U(t+h) )

If the operator uses tau, each row's minimum sits on the diagonal h' = h --
being asked the right question produces the best answer for that question. If
tau is decorative past some horizon, the row goes flat and the diagonal stops
winning.

This separates the two readings cleanly, because hedging towards the
climatology is still a tau-DEPENDENT choice: an optimal predictor hedges LESS at
h = 8 than at h = 128, so querying tau(128) when the target is h = 8 must be
worse. A model that ignores tau has nothing to lose from the swap.

Reported per row: the diagonal value, the best value, argmin, and the
"tau advantage" = (worst - diagonal) / diagonal, i.e. how much being asked
correctly is worth.

Usage:
    python scripts/check_time_sensitivity.py --config configs/dt_fno.yaml \
        --checkpoint checkpoints/dt_fno_s0/best_model.pth
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from common import base_parser, resolve, set_seed, json_default   # noqa: E402

from data import build_eval_loader                                 # noqa: E402
from models import build_model                                     # noqa: E402


@torch.no_grad()
def main():
    ap = base_parser("Query-time sensitivity diagnostic (§38 Test 2)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--horizons", type=int, nargs="+", default=None,
                    help="default: the config's eval_horizons. The old hardcoded "
                         "[1..256] was a RealPDEBench list and crashed on any "
                         "shorter dataset.")
    ap.add_argument("--max_batches", type=int, default=8)
    ap.add_argument("--subset", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = resolve(args)
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model or cfg.get("model_name", "dt_fno")
    if model_name not in ("dt_fno", "sg_dt_fno"):
        raise SystemExit("this diagnostic only applies to direct-time models")

    # Horizons come from the config unless overridden, and are then clipped to
    # what the trajectories can actually support: T - K - 1. A dataset with 151
    # frames cannot be asked for h = 256, and the failure should be a dropped
    # column with a note, not a crash three hours into a sweep.
    horizons = list(args.horizons or cfg.get("eval_horizons", [1, 2, 4, 8, 16]))
    from data.store import build_store
    T = build_store(cfg).T
    h_cap = T - int(cfg.get("history_len", 4)) - 1
    kept = [h for h in horizons if h <= h_cap]
    if len(kept) < len(horizons):
        print(f"  [i] dropped horizons {[h for h in horizons if h > h_cap]}: "
              f"T={T} with K={cfg.get('history_len', 4)} supports h <= {h_cap}")
    if not kept:
        raise SystemExit(f"no usable horizons: T={T} supports h <= {h_cap}")
    horizons = kept

    loader, info = build_eval_loader(cfg.get("dataset_name", "realpde_combustion"),
                                     cfg, subset=args.subset,
                                     horizons=horizons)
    print(f"  {info['n_anchors']} anchors on trajectories {info['traj_indices']}")
    if info["n_anchors"] < 30:
        print(f"  [!] only {info['n_anchors']} anchors. Every number below is a "
              f"small-sample\n      estimate; lower data.eval_stride before "
              f"quoting them.")
    info["K"] = int(cfg.get("history_len", 4))
    model = build_model(model_name, cfg, {"C": info["C"], "K": info["K"]})
    st = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(st.get("model", st), strict=False)
    model = model.to(device).eval()

    hs = list(info["horizons"])
    pairs = [(hs[i], hs[i + 1]) for i in range(len(hs) - 1)]
    acc = {p: {"model": 0.0, "truth": 0.0, "n": 0} for p in pairs}

    print("=" * 74)
    print(f"QUERY-TIME SENSITIVITY — {model_name} on {args.subset}")
    print("=" * 74)

    for bi, batch in enumerate(loader):
        if bi >= args.max_batches:
            break
        x = batch["x"].to(device)
        y = batch["y"].to(device)                      # (B, n_h, H, W, C)
        grid = batch["grid"].to(device)
        B = x.shape[0]

        preds = {}
        for h in hs:
            tau = torch.full((B,), h * info["dt"] / info["t_scale"], device=device)
            preds[h] = model(x, tau, grid).float()

        def rel(a, b):
            af = a.reshape(a.shape[0], -1)
            bf = b.reshape(b.shape[0], -1)
            return (torch.linalg.vector_norm(af - bf, dim=1)
                    / (torch.linalg.vector_norm(bf, dim=1) + 1e-8))

        for (h1, h2) in pairs:
            i1, i2 = hs.index(h1), hs.index(h2)
            acc[(h1, h2)]["model"] += float(rel(preds[h2], preds[h1]).sum())
            acc[(h1, h2)]["truth"] += float(rel(y[:, i2].float(),
                                                y[:, i1].float()).sum())
            acc[(h1, h2)]["n"] += B

    print(f"\n{'h1 -> h2':<14}{'S_model':>10}{'S_truth':>10}{'ratio':>9}  reading")
    print("-" * 66)
    rows = []
    for (h1, h2) in pairs:
        a = acc[(h1, h2)]
        sm, stt = a["model"] / max(a["n"], 1), a["truth"] / max(a["n"], 1)
        ratio = sm / max(stt, 1e-12)
        note = ("ignores tau" if ratio < 0.25 else
                "weak" if ratio < 0.6 else
                "over-reacts" if ratio > 2.0 else "tracks physics")
        print(f"{f'{h1} -> {h2}':<14}{sm:>10.4f}{stt:>10.4f}{ratio:>9.3f}  {note}")
        rows.append({"h1": h1, "h2": h2, "S_model": sm, "S_truth": stt,
                     "ratio": ratio})

    # ---- Test 2: tau-swap matrix ------------------------------------
    print(f"\n{'='*74}\nTAU-SWAP MATRIX — E_field( G(X, tau=col), U(t+row) )")
    print("Row minimum on the diagonal => the operator uses the query time.")
    print(f"{'='*74}")
    M = np.zeros((len(hs), len(hs)))
    cnt = 0
    for bi, batch in enumerate(loader):
        if bi >= args.max_batches:
            break
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        grid = batch["grid"].to(device)
        B = x.shape[0]
        preds = {}
        for h in hs:
            tau = torch.full((B,), h * info["dt"] / info["t_scale"], device=device)
            preds[h] = model(x, tau, grid).float()
        for i, h_t in enumerate(hs):                 # target
            tgt = y[:, i].float()
            for j, h_q in enumerate(hs):             # queried
                p_, t_ = preds[h_q], tgt
                num = torch.linalg.vector_norm(
                    (p_ - t_).reshape(B, -1, p_.shape[-1]), dim=1)
                den = torch.linalg.vector_norm(
                    t_.reshape(B, -1, t_.shape[-1]), dim=1) + 1e-8
                M[i, j] += float((num / den).mean(dim=1).sum())
        cnt += B
    M /= max(cnt, 1)

    print(f"\n{'target\\query':<14}" + "".join(f"{h:>8}" for h in hs) +
          f"{'argmin':>9}{'tau adv':>9}")
    print("-" * (14 + 8 * len(hs) + 18))
    swap_rows = []
    for i, h_t in enumerate(hs):
        j_best = int(np.argmin(M[i]))
        adv = (M[i].max() - M[i, i]) / max(M[i, i], 1e-12)
        cells = "".join(
            (f"{M[i, j]:>7.3f}" + ("*" if j == j_best else " "))
            for j in range(len(hs)))
        print(f"{h_t:<14}" + cells + f"{hs[j_best]:>9}{adv:>9.3f}")
        swap_rows.append({"target": h_t, "argmin_query": hs[j_best],
                          "diagonal": float(M[i, i]),
                          "best": float(M[i].min()),
                          "worst": float(M[i].max()),
                          "tau_advantage": float(adv)})

    on_diag = sum(1 for r in swap_rows if r["argmin_query"] == r["target"])
    print(f"\n  diagonal is the row minimum for {on_diag}/{len(hs)} horizons")
    # The decisive number is the DIAGONAL PENALTY: how much worse the correct
    # query is than the best available one. `tau_advantage` (worst vs diagonal)
    # only shows that SOME queries are bad, which is a much weaker statement and
    # stays large even when the diagonal is not the winner.
    print(f"\n{'target':>8}{'diagonal':>11}{'best':>9}{'argmin':>9}"
          f"{'diag penalty':>14}")
    for r in swap_rows:
        pen = (r["diagonal"] - r["best"]) / max(r["best"], 1e-12)
        flag = "  <-- correct query is not the best" if pen > 0.05 else ""
        print(f"{r['target']:>8}{r['diagonal']:>11.4f}{r['best']:>9.4f}"
              f"{r['argmin_query']:>9}{100*pen:>13.1f}%{flag}")
        r["diagonal_penalty"] = float(pen)

    late = [r for r in swap_rows if r["target"] >= 16]
    if late:
        adv_late = float(np.mean([r["tau_advantage"] for r in late]))
        pen_late = float(np.mean([r["diagonal_penalty"] for r in late]))
        print(f"\n  mean diagonal penalty at h >= 16: {100*pen_late:.1f}%")
        print(f"  mean tau advantage  at h >= 16: {adv_late:.3f}")
        if pen_late > 0.05:
            print("\n  [!] Past h = 16 the operator's own long-tau answers are")
            print("      beaten by one of its short-tau answers. tau is being")
            print("      used, but beyond this range it is used HARMFULLY --")
            print("      which is a trainable defect, not a predictability limit.")
        if adv_late < 0.02:
            print("\n  [!] Being asked the correct horizon is worth < 2% at long")
            print("      range. The operator is effectively ignoring tau there,")
            print("      so the flat E(h) curve is reading (b): one field for")
            print("      every question. Report it as such.")
        else:
            print("\n  The correct query still buys a measurable improvement, so")
            print("  the operator is using tau -- the flat E(h) reflects the")
            print("  predictability limit, not a frozen output.")

    lo = [r for r in rows if r["h2"] <= 8]
    hi = [r for r in rows if r["h1"] >= 16]
    if lo and hi:
        rl = float(np.mean([r["ratio"] for r in lo]))
        rh = float(np.mean([r["ratio"] for r in hi]))
        print(f"\n  mean ratio, short horizons (h <= 8):   {rl:.3f}")
        print(f"  mean ratio, long horizons  (h >= 16):  {rh:.3f}")
        if rh < 0.25 <= rl:
            print("\n  [!] Sensitivity to tau collapses at long horizons while the")
            print("      short end still tracks. A flat E(h) curve here means the")
            print("      operator stopped answering the question, not that it")
            print("      stayed accurate. Report the flat curve WITH this ratio.")
        elif rh >= 0.6:
            print("\n  The operator keeps responding to tau across the range, so a")
            print("  flat E(h) is genuine horizon robustness rather than a")
            print("  collapse onto one climatological field.")

    out = args.out or os.path.join(
        cfg.get("results_dir", "./results"),
        cfg.get("exp_name", model_name), "time_sensitivity.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fp:
        json.dump({"model": model_name, "checkpoint": args.checkpoint,
                   "pairs": rows, "swap_matrix": M.tolist(),
                   "swap_horizons": hs, "swap_rows": swap_rows},
                  fp, indent=2, default=json_default)
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
