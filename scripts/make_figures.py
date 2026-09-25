#!/usr/bin/env python
"""
Figures (§46, §33, §26-31) from the JSONs written by evaluate_horizon.py.

The four core figures §46 asks for:

    Figure 1   E_field vs prediction horizon        AR / DT / SG-DT / DMD
    Figure 2   inference time vs horizon           AR linear, direct flat
    Figure 3   temporal interpolation / extrapolation (Experiment B)
    Figure 4   qualitative fields                  -> scripts/plot_fields.py

plus:

    Figure 5   accuracy-cost Pareto (§33)          the actual thesis
    Figure 6   per-variable error (§26)
    Figure 7   radial spectra at selected horizons (§27)
    Figure 8   semigroup consistency vs error (§31)

Multi-seed handling (§43): pass several JSONs for the same model and they are
aggregated to mean +- std, drawn as a band.  A single seed is drawn as a line
with a note in the caption file, because a single-seed curve should not be shown
in a way that implies it has error bars.

Usage:
    python scripts/make_figures.py \
        --results results/ar_fno_r/horizon_metrics.json \
                  results/dt_fno/horizon_metrics.json \
                  results/sg_dt_fno/horizon_metrics.json \
                  results/persistence/horizon_metrics.json \
                  results/pod_dmd/horizon_metrics.json \
        --timing results/timing/inference_cost.json \
        --out results/figures

    # multi-seed: repeat the flag, labels are taken from meta.exp_name
    python scripts/make_figures.py --results results/sg_dt_fno_s*/horizon_metrics.json ...
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

from common import REPO_ROOT      # noqa: E402,F401
from evaluation.field_metrics import crossover_horizon, predictability_horizon  # noqa: E402

# configs/ar_fno.yaml IS the one-step arm, so the bare key gets the AR-FNO-1
# style. Giving both AR arms the same red made them indistinguishable in
# Figure 1, which is the one plot where the two baselines must be told apart.
STYLE = {
    "ar_fno_1":   dict(color="#ff9896", marker="o", label="AR-FNO-1"),
    "ar_fno_r":   dict(color="#d62728", marker="o", label="AR-FNO-R"),
    "ar_fno":     dict(color="#ff9896", marker="o", label="AR-FNO-1"),
    "dt_fno":     dict(color="#1f77b4", marker="s", label="DT-FNO"),

    "dt_fno_K1": dict(color="#1f77b4", marker="s", label="DT-FNO K=1"),
    "dt_fno_K2": dict(color="#ff7f0e", marker="o", label="DT-FNO K=2"),
    "dt_fno_K4": dict(color="#2ca02c", marker="^", label="DT-FNO K=4"),
    "dt_fno_K8": dict(color="#d62728", marker="D", label="DT-FNO K=8"),

    "sg_dt_fno":  dict(color="#2ca02c", marker="^", label="SG-DT-FNO"),
    "pod_dmd":    dict(color="#9467bd", marker="d", label="POD-DMD"),
    "persistence": dict(color="#7f7f7f", marker="x", label="Persistence"),
    "climatology": dict(color="#000000", marker="_", label="Traj. climatology (oracle)"),
    "nearest_climatology": dict(color="#8c564b", marker="P",
                                label="Nearest-train climatology"),
}


_FALLBACK = [dict(color="#8c564b", marker="."), dict(color="#e377c2", marker="v"),
             dict(color="#17becf", marker="P"), dict(color="#bcbd22", marker="*")]
_ASSIGNED: Dict[str, dict] = {}

# --------------------------------------------------------------------------
# Presentation options, set once in main().
#
# Titles are OFF by default. A figure that carries its own "Figure 1 — ..."
# heading and then gets a LaTeX \caption underneath says everything twice, in
# two different fonts, and the two disagree the moment the paper is
# reorganised: these files were still calling themselves Figure 1, 2, 3 after
# the panels had been merged and renumbered, and a stale figure number inside
# an image is not something a proof-read catches. The caption is the single
# place that should name and number a figure. Pass --titles to restore the old
# behaviour for quick looks at results outside a paper.
# --------------------------------------------------------------------------
OPTS = {
    "titles": False,     # draw the "Figure N — ..." headings
    "label": None,       # panel tag drawn inside the axes, e.g. "A1 Gray-Scott"
    "legend": "auto",    # auto | on | off
    "rename": {},        # display-name overrides, key -> label
}


def _fig_title(ax_or_fig, text, **kw):
    """Draw a figure-level title only when --titles is on."""
    if not OPTS["titles"]:
        return
    if hasattr(ax_or_fig, "suptitle"):
        ax_or_fig.suptitle(text, **kw)
    else:
        ax_or_fig.set_title(text, **kw)


def _panel_label(ax):
    r"""Identify the dataset inside the axes, for multi-panel LaTeX figures.

    Figure 1 of the paper is three panels from three datasets placed side by
    side with \includegraphics. Nothing inside the images says which is which,
    so the reader has to map them by position from the caption. A tag drawn in
    the corner of the axes travels with the image and survives being moved,
    rescaled or reordered.
    """
    if not OPTS["label"]:
        return
    ax.text(0.025, 0.975, _tex_dashes(OPTS["label"]), transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor="0.75", linewidth=0.6, alpha=0.92),
            zorder=6)


def _tex_dashes(text: str) -> str:
    """LaTeX dash conventions do not survive into matplotlib.

    `--label 'A1 Gray--Scott'` is the natural thing to type when the surrounding
    document is LaTeX, and matplotlib renders it literally as two hyphens.
    Translate the ligatures rather than making the caller remember which
    renderer they are talking to.
    """
    return text.replace("---", "\u2014").replace("--", "\u2013")


def _legend(ax, **kw):
    if OPTS["legend"] == "off":
        return
    ax.legend(**kw)


# Prefixes that name the dataset rather than the model. A run tag is
# "<dataset>_<model>_s<seed>", so without stripping these the legend reads
# "gs_ar_fno_r (n=3)" on a panel that is already labelled Gray-Scott.
_DATASET_PREFIX = ("gray_scott", "rayleigh_benard", "cylinder", "realpde",
                   "lifted_h2", "comb", "lh2", "gs", "cyl", "rb",
                   "expA", "expB")


def _display_key(key: str) -> str:
    for pre in sorted(_DATASET_PREFIX, key=len, reverse=True):
        if key.startswith(pre + "_"):
            return key[len(pre) + 1:]
    return key


def style_for(key: str, model_name: str = "") -> dict:
    """Colour/marker by experiment key, falling back to the model name.

    Experiment names are free-form (`sg_dt_fno_lam0.2`, `expB_dt_fno`), so match
    the longest known prefix first and only then the model_name recorded in the
    result metadata.  Anything unrecognised gets a stable assigned style rather
    than colliding with a known model's colour.
    """
    if key in _ASSIGNED:
        return dict(_ASSIGNED[key])
    # Strip the dataset prefix before matching, so `gs_ar_fno_r` resolves to
    # the AR-FNO-R style AND to the paper's name for it. A tag that is a known
    # model plus a SUFFIX (`ar_fno_r_fixsel`) keeps its full name on purpose:
    # it is a different run and the legend should say so rather than quietly
    # presenting it as the baseline.
    core = _display_key(key)
    for k in sorted(STYLE, key=len, reverse=True):
        if core.startswith(k) or k in core:
            st = dict(STYLE[k])
            st["label"] = st["label"] if core == k else core
            _ASSIGNED[key] = st
            return dict(st)
    if model_name and model_name in STYLE:
        st = dict(STYLE[model_name], label=core)
        _ASSIGNED[key] = st
        return dict(st)
    st = dict(_FALLBACK[len(_ASSIGNED) % len(_FALLBACK)], label=core)
    _ASSIGNED[key] = st
    return dict(st)


# ---------------------------------------------------------------------------
# loading / aggregation
# ---------------------------------------------------------------------------

def load_results(paths: List[str]) -> Dict[str, dict]:
    """Group JSONs by model key, aggregating seeds into mean/std curves."""
    groups: Dict[str, List[dict]] = defaultdict(list)
    for p in paths:
        with open(p) as fp:
            r = json.load(fp)
        meta = r.get("meta", {})
        key = meta.get("model_name", r.get("model", os.path.basename(p)))
        # a seed suffix (_s0, _s1) collapses onto the same key
        exp = meta.get("exp_name", key)
        base = exp.rsplit("_s", 1)[0] if "_s" in exp[-4:] else exp
        groups[base].append(r)

    out: Dict[str, dict] = {}
    for key, runs in groups.items():
        hs = runs[0]["horizons"]
        for r in runs:
            if r["horizons"] != hs:
                raise ValueError(
                    f"{key}: runs evaluated on different horizon grids "
                    f"({r['horizons']} vs {hs}). Re-run evaluate_horizon.py "
                    f"with the same --horizons for every seed.")
        E = np.array([r["E_field"] for r in runs], dtype=float)
        # A single non-finite seed would turn the whole group's mean into nan
        # and drop it from every figure and table -- including, in practice,
        # the seed that best demonstrates the instability being reported.
        # Mapping non-finite to +inf keeps the horizon in the "diverged"
        # bucket, where it is drawn as a triangle and counted, instead of
        # silently disappearing. `eval_score_bounded` (below) is the number to
        # rank on; this array is for the plot.
        E = np.where(np.isfinite(E), E, np.inf)
        with np.errstate(invalid="ignore"):
            rec = {
                "horizons": np.array(hs, dtype=float),
                "E_mean": E.mean(axis=0), "E_std": E.std(axis=0),
                "n_seeds": len(runs),
                "n_model_evals": np.array(runs[0]["n_model_evals"],
                                          dtype=float),
                "runs": runs,
                "meta": runs[0].get("meta", {}),
                "model_name": runs[0].get("meta", {}).get("model_name", key),
            }
        groups_e = runs[0].get("E_group", {})
        if groups_e:
            rec["E_group"] = {
                g: np.array([r["E_group"][g] for r in runs]).mean(axis=0)
                for g in groups_e}
        if "spectral" in runs[0]:
            rec["spectral"] = runs[0]["spectral"]
        if "combustion" in runs[0]:
            rec["combustion"] = runs[0]["combustion"]
        if "semigroup" in runs[0]:
            rec["semigroup"] = runs[0]["semigroup"]
        # The plain score is unusable once anything diverges (one 1e20 point
        # makes the mean 1e19) and undefined once anything is non-finite.
        # Average the per-run BOUNDED scores, which is also what
        # evaluate_horizon.py prints per run -- so the table and the logs agree.
        rec["eval_score"] = float(np.mean(
            [_bounded_score(r) for r in runs]))
        out[key] = rec
    return out


def _band(ax, x, mean, std, n_seeds, **kw):
    label = kw.pop("label", None)
    label = OPTS["rename"].get(label, label)
    if n_seeds > 1:
        label = f"{label} (n={n_seeds})"
        # On a log axis mean - std is frequently negative for a group whose
        # seeds straddle divergence (41.0 +- 52.1 on Gray-Scott AR-FNO-R), and
        # a negative lower edge is silently dropped, leaving a band that looks
        # symmetric and is not. Clamp it so the asymmetry is visible.
        lo = np.maximum(np.asarray(mean) - np.asarray(std),
                        np.asarray(mean) * 1e-3)
        ax.fill_between(x, lo, np.asarray(mean) + np.asarray(std), alpha=0.18,
                        color=kw.get("color"), linewidth=0)
    ax.plot(x, mean, label=label, **kw)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

DIVERGENCE = 3.0     # E_field above this is a blown-up rollout, not an error


def _bounded_score(run):
    """§24's bounded score for one run, recomputed if the JSON predates the fix.

    `min(nan, 1.0)` is nan, so a single non-finite horizon used to turn an
    otherwise reportable run's summary into nan. A non-finite E is a FAILED
    forecast, not a large error, and the bounded score's own convention already
    has a slot for that: 1.0, "no better than predicting the training mean".
    """
    E = np.asarray(run["E_field"], dtype=float)
    b = np.where(np.isfinite(E), np.minimum(E, 1.0), 1.0)
    return float(b.mean())


def figure1(res, out_dir, h_max_train: Optional[int]):
    """E_field vs horizon (§46 Fig 1) — the feasibility answer to H1/H2.

    Log y-axis with the view capped just above the last converged point.

    An autoregressive rollout that diverges does not degrade gracefully: at
    h = 512 the measured E_field was 5.9e20. On a linear axis that single point
    compresses every other curve onto the zero line and the figure conveys
    nothing. Capping the view and marking off-scale points with an upward
    triangle keeps the divergence visible AND keeps the 0.2-0.8 band -- where
    the actual comparison lives -- readable.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    finite = [v for r in res.values() for v in r["E_mean"] if v < DIVERGENCE]
    ymax = 1.6 * max(finite) if finite else DIVERGENCE
    ymin = 0.6 * min(v for r in res.values() for v in r["E_mean"] if v > 0)

    n_div = {}
    for key, r in res.items():
        st = style_for(key, r.get("model_name", ""))
        x, y = np.asarray(r["horizons"]), np.asarray(r["E_mean"])
        ok = y <= ymax
        _band(ax, x[ok], y[ok], np.asarray(r["E_std"])[ok], r["n_seeds"],
              marker=st["marker"], color=st["color"], label=st["label"],
              markersize=4, linewidth=1.6)
        if (~ok).any():
            ax.plot(x[~ok], np.full((~ok).sum(), ymax * 0.94), linestyle="none",
                    marker="^", color=st["color"], markersize=9, clip_on=False)
            n_div[st["label"]] = (int((~ok).sum()), float(y[~ok].max()))

    if h_max_train:
        ax.axvline(h_max_train, color="0.5", linestyle=":", linewidth=1)
        ax.text(h_max_train * 1.05, ymax * 0.55, "training\nhorizon limit",
                fontsize=7, color="0.35")
    ax.axhline(1.0, color="0.6", linestyle="--", linewidth=0.9)
    # Placed in AXES coordinates on purpose. Reading ax.get_xlim() before
    # set_xscale returns the default linear limits, whose left edge is
    # negative; feeding that x back onto a log axis puts the text at an
    # undefined position and bbox_inches="tight" then expands the saved figure
    # to tens of thousands of pixels.
    # Right-aligned: the panel tag drawn by _panel_label() occupies the
    # top-left corner, and on a panel whose curves reach E=1 the two overlap
    # exactly where both are least readable.
    ax.text(0.99, 1.0, "$E=1$: no better than predicting the training mean ",
            transform=ax.get_yaxis_transform(), fontsize=6.5, color="0.4",
            va="top", ha="right")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("prediction horizon $h$ (frames)")
    ax.set_ylabel(r"$E_{\mathrm{field}}(h)$   relative $L_2$")
    title = "Figure 1 — Field error vs prediction horizon"
    if n_div:
        title += "\n$\\blacktriangle$ = diverged, off scale (see caption)"
    _fig_title(ax, title, fontsize=10)
    _panel_label(ax)
    ax.grid(alpha=0.3, which="both")
    _legend(ax, fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, out_dir, "fig1_error_vs_horizon")

    if n_div:
        _note(out_dir, "fig1_diverged.txt",
              ["Off-scale points in Figure 1 (E_field > "
               f"{DIVERGENCE}), drawn as triangles at the top edge:", ""]
              + [f"  {k}: {n} horizon(s), worst {m:.4g}"
                 for k, (n, m) in n_div.items()]
              + ["",
                 "A diverged rollout is a qualitatively different outcome from a",
                 "large error. Report those horizons as 'diverged' rather than",
                 "quoting the number -- 5.9e20 is not 5.9e20 times worse than a",
                 "competitor, it is a blown-up integration with no meaning."])

    # the crossover point §2 calls the important experimental result
    lines = []
    ar_key = next((k for k in res if "ar" in k), None)
    if ar_key:
        for key in res:
            if key == ar_key or "persist" in key or "dmd" in key:
                continue
            hs = res[key]["horizons"].tolist()
            c = crossover_horizon(hs, res[ar_key]["E_mean"].tolist(),
                                  res[key]["E_mean"].tolist())
            lines.append(f"{key} beats {ar_key} from h = "
                         + (f"{c:.1f}" if c else "never (inside this range)"))
    for key, r in res.items():
        tp = predictability_horizon(r["horizons"].tolist(),
                                    r["E_mean"].tolist(), 0.3)
        lines.append(f"T_pred({key}, E<0.3) = "
                     + (f"{tp:.1f} frames" if tp else "not reached"))
    _note(out_dir, "fig1_crossover.txt", lines)


def figure2(timing, out_dir):
    """Inference cost vs horizon (§32, §46 Fig 2)."""
    if not timing:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    # Timing depends on the ARCHITECTURE, not on the training variant: AR-FNO-1
    # and AR-FNO-R are the same network (§23) and the benchmark records only the
    # model name. Labelling the curve "AR-FNO-1" here would imply a variant was
    # chosen when none was.
    timing_label = {"ar_fno": "AR-FNO (either variant)"}
    for name, blob in timing["models"].items():
        s = style_for(name)
        s["label"] = timing_label.get(name, s["label"])
        if name == "ar_fno":
            s["color"] = "#d62728"
        h = [r["h"] for r in blob["rows"]]
        ms = [r["median_ms"] for r in blob["rows"]]
        iqr = [r["iqr_ms"] for r in blob["rows"]]
        ne = [r["n_model_evals"] for r in blob["rows"]]
        axes[0].errorbar(h, ms, yerr=iqr, marker=s["marker"], color=s["color"],
                         label=s["label"], markersize=4, linewidth=1.6,
                         capsize=2)
        axes[1].plot(h, ne, marker=s["marker"], color=s["color"],
                     label=s["label"], markersize=4, linewidth=1.6)
    for ax, yl, ti in ((axes[0], "wall-clock inference time (ms)",
                        "measured (median $\\pm$ IQR, batch 1)"),
                       (axes[1], "$N_{\\mathrm{model\\ evaluations}}$",
                        "exact, hardware-independent")):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("prediction horizon $h$ (frames)")
        ax.set_ylabel(yl)
        ax.set_title(ti, fontsize=10)
        ax.grid(alpha=0.3, which="both")
        _legend(ax, fontsize=8)
    _panel_label(axes[0])
    _fig_title(fig, "Figure 2 — Inference cost vs prediction horizon", y=1.0)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_cost_vs_horizon")


def figure3(res, out_dir):
    """Experiment B: interpolation / extrapolation at unseen query times (§21)."""
    trained = None
    for r in res.values():
        trained = r["meta"].get("train_horizons") or trained
    if not trained:
        return
    trained = set(int(t) for t in trained)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    h_max = max(trained)
    for key, r in res.items():
        s = style_for(key, r.get("model_name", ""))
        hs = r["horizons"]
        seen = np.array([int(h) in trained for h in hs])
        ax.plot(hs, r["E_mean"], color=s["color"], linewidth=1.4, alpha=0.55,
                label=s["label"], zorder=1)
        ax.scatter(hs[seen], r["E_mean"][seen], color=s["color"], marker="o",
                   s=34, zorder=3, edgecolor="white", linewidth=0.6)
        interp = (~seen) & (hs <= h_max)
        extra = (~seen) & (hs > h_max)
        ax.scatter(hs[interp], r["E_mean"][interp], color=s["color"],
                   marker="D", s=42, zorder=3, facecolor="none", linewidth=1.4)
        ax.scatter(hs[extra], r["E_mean"][extra], color=s["color"],
                   marker="*", s=90, zorder=3, facecolor="none", linewidth=1.2)
    ax.axvline(h_max, color="0.5", linestyle=":", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("prediction horizon $h$ (frames)")
    ax.set_ylabel(r"$E_{\mathrm{field}}(h)$")
    # The marker legend is not a figure number and must survive --titles off:
    # without it the circles, diamonds and stars are unreadable.
    _fig_title(ax, "Figure 3 — Unseen query-time generalization\n"
                   "circles: trained $h$   diamonds: interpolation   "
                   "stars: extrapolation", fontsize=10)
    if not OPTS["titles"]:
        ax.set_title("circles: trained $h$   diamonds: interpolation   "
                     "stars: extrapolation", fontsize=8.5, color="0.3")
    _panel_label(ax)
    ax.grid(alpha=0.3, which="both")
    _legend(ax, fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "fig3_unseen_query_time")
    _note(out_dir, "fig3_caveat.txt", [
        "A smooth curve through the diamonds is the evidence that the model",
        "learned query-time conditioned dynamics rather than memorising frame",
        "indices (§21).",
        "",
        "The stars need a caveat: with time_embed_mode=fourier the embedding is",
        "periodic in tau, so tau > 1 aliases onto values seen in training.",
        "Re-run with time_embed_mode=fourier_log before concluding anything",
        "about temporal extrapolation.",
    ])


def figure5_pareto(res, timing, out_dir, horizons=(8, 32, 128, 256)):
    """Accuracy-cost Pareto (§33) — the figure the thesis actually rests on."""
    if not timing:
        return
    cost = {}
    for name, blob in timing["models"].items():
        cost[name] = {r["h"]: r for r in blob["rows"]}

    hs = [h for h in horizons]
    fig, axes = plt.subplots(1, len(hs), figsize=(3.3 * len(hs), 3.6),
                             sharey=False)
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, hs):
        for key, r in res.items():
            mname = r["model_name"]
            if mname not in cost or h not in cost[mname]:
                continue
            grid = r["horizons"].tolist()
            if h not in grid:
                continue
            i = grid.index(h)
            s = style_for(key, r.get("model_name", ""))
            ax.errorbar(cost[mname][h]["median_ms"], r["E_mean"][i],
                        yerr=(r["E_std"][i] if r["n_seeds"] > 1 else None),
                        marker=s["marker"], color=s["color"], markersize=9,
                        capsize=2, label=s["label"])
        ax.set_xscale("log")
        ax.set_xlabel("inference time (ms)")
        ax.set_title(f"$h = {h}$", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel(r"$E_{\mathrm{field}}$")
    axes[-1].legend(fontsize=7)
    _fig_title(fig, "Figure 5 — Accuracy-cost Pareto (§33): the claim is a "
                 "different trade-off, not uniformly lower error", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "fig5_pareto")


def figure6_per_variable(res, out_dir):
    """Per-variable error (§26) — the mean must never be the only number."""
    keys = [k for k in res if "E_group" in res[k]]
    if not keys:
        return
    groups = list(res[keys[0]]["E_group"].keys())
    fig, axes = plt.subplots(1, len(groups), figsize=(3.2 * len(groups), 3.6),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, groups):
        for key in keys:
            r = res[key]
            s = style_for(key, r.get("model_name", ""))
            ax.plot(r["horizons"], r["E_group"][g], marker=s["marker"],
                    color=s["color"], label=s["label"], markersize=3.5,
                    linewidth=1.4)
        ax.set_xscale("log", base=2)
        ax.set_title(g, fontsize=10)
        ax.set_xlabel("horizon $h$")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel(r"$E(h)$")
    axes[-1].legend(fontsize=7)
    _fig_title(fig, "Figure 6 — Per-variable error (§26)", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, "fig6_per_variable")


def figure7_spectra(res, out_dir, show_h=(8, 128)):
    """show_h defaults to (8, 128); pass --spectra_horizons to change it.
    A dataset whose grid stops at 64 silently gets a single panel."""
    """Radial spectra (§27) — is the direct model buying MSE with smoothing?"""
    keys = [k for k in res if "spectral" in res[k]]
    if not keys:
        return
    ref = res[keys[0]]["spectral"]
    hs = ref["horizons"]
    picks = [h for h in show_h if h in hs] or [hs[len(hs) // 2]]

    fig, axes = plt.subplots(1, len(picks), figsize=(5.0 * len(picks), 4.0))
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, picks):
        i = hs.index(h)
        tgt = np.array(ref["spectrum_target"][i]).mean(axis=0)
        k = np.arange(len(tgt))
        ax.loglog(k[1:], tgt[1:], color="k", linewidth=2.0, label="DNS")
        for key in keys:
            sp = res[key]["spectral"]
            pr = np.array(sp["spectrum_pred"][i]).mean(axis=0)
            st = style_for(key, res[key].get("model_name", ""))
            ax.loglog(k[1:], pr[1:], color=st["color"], linewidth=1.3,
                      label=f"{st['label']}  $E_{{spec}}$={sp['E_spec'][i]:.2f}")
        # E_spec per model used to be crammed into the title; with four models
        # the strings overlapped into an unreadable smear. It goes in the legend
        # instead, where matplotlib lays it out.
        ax.set_title(f"$h = {h}$", fontsize=11)
        ax.set_xlabel("wavenumber $k$")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("$E(k)$")
    for ax in axes:
        ax.legend(fontsize=7, loc="lower left")
    _fig_title(fig, "Figure 7 — Energy spectra (§27): a deficit at high $k$ is "
                 "over-smoothing.\nThe curve is the channel-MEAN spectrum; "
                 "$E_{spec}$ is the mean of per-channel ratios — they can "
                 "disagree by 10x.", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "fig7_spectra")


def figure8_semigroup(res, out_dir):
    """C_SG together with E_field (§31) — never C_SG alone."""
    keys = [k for k in res if "semigroup" in res[k]]
    if not keys:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))

    for key in keys:
        sg = res[key]["semigroup"]
        s = style_for(key, res[key].get("model_name", ""))
        tot = [p["h_total"] for p in sg["pairs"]]
        csg = [p["C_SG"] for p in sg["pairs"]]
        order = np.argsort(tot)
        axes[0].plot(np.array(tot)[order], np.array(csg)[order], marker=s["marker"],
                     color=s["color"], label=s["label"], markersize=4)
        axes[1].bar(s["label"], sg["latent_rms_mean"], color=s["color"], alpha=0.8)
        axes[2].scatter(sg["C_SG_mean"], res[key]["eval_score"], s=90,
                        color=s["color"], marker=s["marker"], label=s["label"])

    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("$h_a + h_b$")
    axes[0].set_ylabel("$C_{SG}$")
    axes[0].set_title("consistency vs composed interval", fontsize=10)
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)

    axes[1].set_ylabel("latent RMS")
    axes[1].set_title("collapse tripwire (§31)", fontsize=10)
    axes[1].grid(alpha=0.3, axis="y")

    axes[2].set_xlabel("$C_{SG}$")
    axes[2].set_ylabel("integrated $E_{field}$")
    axes[2].set_title("the joint claim: both must fall", fontsize=10)
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    _fig_title(fig, "Figure 8 — Semigroup consistency (§31)", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, "fig8_semigroup")


def summary_table(res, out_dir):
    lines = ["model                       n_seeds   Eval*   E(h=1)   "
             "E(h=8)  E(h=32)  E(h=128)  E(h=256)  diverged",
             "Eval* = mean of min(E, 1) over all evaluated horizons. The plain "
             "mean is not a summary",
             "statistic once a rollout diverges: one 1e20 point makes it 1e19 "
             "regardless of everything else.",
             ""]
    for key, r in sorted(res.items()):
        hs = r["horizons"].tolist()
        E = np.asarray(r["E_mean"])
        def at(h):
            return f"{E[hs.index(h)]:.4f}" if h in hs else "   -  "
        # Per seed, then averaged -- NOT min(mean(E), 1). The two differ:
        # seeds giving E = 0.84, 9.0, 113.2 have a mean of 41.0, so
        # min(mean, 1) = 1.000, while the per-seed bounded values are
        # 0.84, 1.0, 1.0 and average to 0.947. The per-seed version is the one
        # evaluate_horizon.py prints, so this keeps the table and the logs
        # consistent.
        bounded = float(np.mean([_bounded_score(run) for run in r["runs"]]))
        ndiv = int(np.mean([
            int((np.asarray(run["E_field"], dtype=float) > DIVERGENCE).sum()
                + (~np.isfinite(np.asarray(run["E_field"], dtype=float))).sum())
            for run in r["runs"]]))
        lines.append(f"{key:<28}{r['n_seeds']:>7}{bounded:>8.4f}  "
                     f"{at(1):>7} {at(8):>8} {at(32):>8} {at(128):>9} "
                     f"{at(256):>9}{ndiv:>10}")
    txt = "\n".join(lines)
    print("\n" + txt + "\n")
    _note(out_dir, "summary_table.txt", lines)


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=160,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.join(out_dir, name)}.png/.pdf")


def _note(out_dir, name, lines):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), "w") as fp:
        fp.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Figures (§46, §33)")
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--include", nargs="+", default=None, metavar="GLOB",
                    help="keep only groups matching these patterns, e.g. "
                         "--include 'ar_fno_r*' 'dt_fno' 'sg_dt_fno' "
                         "persistence climatology nearest_climatology")
    ap.add_argument("--exclude", nargs="+", default=None, metavar="GLOB",
                    help="drop groups matching these, e.g. --exclude 'expB_*' "
                         "'*nodelta*'")
    ap.add_argument("--max_groups", type=int, default=8)
    ap.add_argument("--spectra_horizons", type=int, nargs="+", default=[8, 128],
                    help="which horizons get a Figure 7 panel. Horizons absent "
                         "from the evaluated grid are skipped, so a dataset "
                         "capped at h=64 would otherwise show only one panel.")
    ap.add_argument("--timing", default=None)
    ap.add_argument("--out", default="results/figures")

    # ---- presentation -----------------------------------------------------
    ap.add_argument("--titles", action="store_true",
                    help="draw the 'Figure N -- ...' heading inside each image. "
                         "OFF by default: a heading inside the image duplicates "
                         "the LaTeX caption in a different font and goes stale "
                         "the moment figures are merged or renumbered. Use it "
                         "for quick looks at results outside a paper.")
    ap.add_argument("--label", default=None, metavar="TEXT",
                    help="tag drawn inside the axes, e.g. --label 'A1 Gray-Scott'. "
                         "Use it when the figure becomes one panel of a "
                         "multi-panel LaTeX figure, so the image says which "
                         "dataset it is without relying on its position.")
    ap.add_argument("--legend", choices=["auto", "on", "off"], default="auto",
                    help="'off' for the second and third panels of a "
                         "multi-panel figure, where repeating an identical "
                         "legend three times wastes the space the panels need.")
    ap.add_argument("--rename", nargs="+", default=None, metavar="OLD=NEW",
                    help="override a legend label, e.g. "
                         "--rename ar_fno_r_fixsel='AR-FNO-R (contaminated)'")
    ap.add_argument("--pareto_horizons", type=int, nargs="+",
                    default=[8, 32, 128, 256])
    args = ap.parse_args()

    OPTS["titles"] = bool(args.titles)
    OPTS["label"] = args.label
    OPTS["legend"] = args.legend
    if args.rename:
        for spec in args.rename:
            k, _, v = spec.partition("=")
            if not v:
                raise SystemExit(f"--rename expects OLD=NEW, got {spec!r}")
            OPTS["rename"][k] = v

    res = load_results(args.results)

    # `results/*/horizon_metrics.json` sweeps up everything ever run --
    # Experiment B arms trained on a different horizon protocol, one-off
    # ablations, duplicate re-runs. Putting them on one axis produces a figure
    # with seventeen curves that answers nothing. Filter explicitly.
    if args.include:
        res = {k: v for k, v in res.items()
               if any(fnmatch.fnmatch(k, p) for p in args.include)}
    if args.exclude:
        res = {k: v for k, v in res.items()
               if not any(fnmatch.fnmatch(k, p) for p in args.exclude)}
    if not res:
        raise SystemExit("no result groups left after --include/--exclude")
    if len(res) > args.max_groups:
        print(f"\n[!] {len(res)} groups on one figure: {sorted(res)}")
        print("    Figure 1 stops being readable past about 6. Experiment B "
              "arms in\n    particular do NOT belong on the same axes as the "
              "main runs -- they were\n    trained on a sparse horizon set, so "
              "their curve means something different.")
        print("    Suggested: --exclude 'expB_*' '*nodelta*'\n")
    timing = None
    if args.timing and os.path.exists(args.timing):
        with open(args.timing) as fp:
            timing = json.load(fp)

    h_max_train = None
    for r in res.values():
        h_max_train = r["meta"].get("h_max_train") or h_max_train

    print(f"Loaded {len(res)} model group(s): {list(res)}")
    if not OPTS["titles"]:
        print("    Titles are off, so the LaTeX caption is now the only place "
              "that names each\n    figure. Divergence markers are still drawn "
              "and still listed in\n    fig1_diverged.txt, but the '= diverged, "
              "off scale' note is not in the\n    image -- put it in the "
              "caption.")
    if OPTS["label"]:
        print(f"    Panel label: {OPTS['label']!r}")

    # Flag runs that differ in a field that changes what was trained. A curve
    # trained with a different loss or a different h_max_train is a different
    # experiment; putting it on the same axis as the baseline is how two
    # sweeps in this project ended up uninterpretable.
    tc = {k: (v["meta"].get("train_config") or {}) for k, v in res.items()}
    fields = sorted({f for d in tc.values() for f in d})
    for f in fields:
        vals = {k: repr(d.get(f)) for k, d in tc.items() if f in d}
        if len(set(vals.values())) > 1:
            print(f"  [!] '{f}' differs across groups:")
            for k, v in sorted(vals.items()):
                print(f"        {k:<34} {v}")
            print(f"      These are different experiments. Separate them, or "
                  f"say so in the caption.")
    figure1(res, args.out, h_max_train)
    figure2(timing, args.out)
    figure3(res, args.out)
    figure5_pareto(res, timing, args.out, args.pareto_horizons)
    figure6_per_variable(res, args.out)
    figure7_spectra(res, args.out, tuple(args.spectra_horizons))
    figure8_semigroup(res, args.out)
    summary_table(res, args.out)
    print(f"\nFigure 4 (qualitative fields) needs checkpoints:\n"
          f"    python scripts/plot_fields.py --config <config> "
          f"--checkpoints ar_fno=... dt_fno=... sg_dt_fno=...")


if __name__ == "__main__":
    main()
