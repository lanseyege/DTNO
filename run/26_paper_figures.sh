#!/usr/bin/env bash
# Regenerate EVERY figure the paper takes from scripts/make_figures.py, with
# titles off and a dataset label on each error panel, then copy them into the
# LaTeX project under the names main.tex expects.
#
#   bash run/26_paper_figures.sh /path/to/DTNO_ICLR2026/figures
#
# WHY ALL TWELVE, NOT JUST THE THREE IN FIGURE 1
# ----------------------------------------------
# Twelve of the paper's figures come from this script, three in the body and
# nine in the appendix, and they must be regenerated together. A document in
# which some figures carry an embedded "Figure 1 -- ..." heading and others do
# not looks like a mistake, and the embedded numbers are already wrong: the
# panels were merged and the sections reordered after those images were made,
# so the file still calling itself Figure 1 is Figure 3 in the current build.
# Mixed state is worse than either state.
#
# A3 Rayleigh-Benard and B1 Lifted H2 ARE still used --- they are the two
# panels of Figure 7 in the appendix (\ref{fig:error-extra}), moved out of the
# body only to reach the nine-page limit. Dropping them would remove the two
# datasets from the figures entirely while the tables still report them.
set -euo pipefail

PAPER_FIGS=${1:-}
RES=${RES:-./results}
OUT=${OUT:-./results/figures_paper_alls}
TIMING_B2=${TIMING_B2:-$RES/timing/inference_cost_s0.json}

M="ar_fno_r dt_fno sg_dt_fno persistence climatology nearest_climatology pod_dmd"
mk() {  # mk <outdir> <label> <legend> <prefix-or-empty> [extra args...]
  local dir="$1" label="$2" legend="$3" pre="$4"; shift 4
  local inc=""
  for m in $M; do inc="$inc ${pre}${m}"; done
  echo ">>> $dir  [$label]"
  python scripts/make_figures.py \
    --results "$RES"/*/horizon_metrics.json \
    --include $inc \
    --label "$label" --legend "$legend" \
    --out "$OUT/$dir" "$@"
}

# --- the five error panels ------------------------------------------------
# Legend on the leftmost body panel only: three identical legends across three
# 0.325-width panels cost more space than they carry. This is safe ONLY while
# the three panels contain the same model set -- check the "Loaded N model
# group(s)" line each run prints, and put the legend back on if they differ.
mk gs   "A1 Gray--Scott"        on  "gs_"
mk cyl  "A2 Cylinder"           off "cyl_"
mk rb   "A3 Rayleigh--Benard"  on  "rb_"
mk lh2  "B1 Lifted H2"          off "lh2_"
mk comb "B2 RealPDEBench"       off ""   --timing "$TIMING_B2"

# --- lead-time panels -----------------------------------------------------
# NOTE the include names carry no seed suffix: make_figures strips "_s<N>" when
# it forms the group key, so --include gs_leadtime_fourier_s0 matches nothing
# and the figure comes out holding only the climatology baselines.
for p in gs cyl; do
  lbl="A1 Gray--Scott"; [ "$p" = cyl ] && lbl="A2 Cylinder"
  ls "$RES"/${p}_leadtime_*/horizon_metrics.json >/dev/null 2>&1 || continue
  echo ">>> ${p}_leadtime  [$lbl]"
  python scripts/make_figures.py \
    --results "$RES"/*/horizon_metrics.json \
    --include ${p}_leadtime_fourier ${p}_leadtime_fourier_log \
              ${p}_leadtime_climatology ${p}_leadtime_nearest_climatology \
    --label "$lbl" --legend on --out "$OUT/${p}_leadtime"
done

# --- copy into the paper under the names main.tex uses --------------------
if [ -n "$PAPER_FIGS" ]; then
  echo
  echo ">>> installing into $PAPER_FIGS"
  cp_or_warn() { [ -f "$1" ] && cp "$1" "$2" && echo "    $(basename "$2")" \
                  || echo "    [!] missing: $1"; }
  cp_or_warn "$OUT/gs/fig1_error_vs_horizon.pdf"      "$PAPER_FIGS/fig1_gs_error.pdf"
  cp_or_warn "$OUT/cyl/fig1_error_vs_horizon.pdf"     "$PAPER_FIGS/fig1_cyl_error.pdf"
  cp_or_warn "$OUT/rb/fig1_error_vs_horizon.pdf"      "$PAPER_FIGS/fig1_rb_error.pdf"
  # PDF, not the PNG the paper currently references: fig1_lh2_error.png and
  # fig7_lh2_spectra.png are the only rasters among the plots, and they are
  # visibly softer at print size than their PDF siblings.
  cp_or_warn "$OUT/lh2/fig1_error_vs_horizon.pdf"     "$PAPER_FIGS/fig1_lh2_error.pdf"
  cp_or_warn "$OUT/comb/fig1_error_vs_horizon.pdf"    "$PAPER_FIGS/fig1_realpde_error.pdf"
  cp_or_warn "$OUT/comb/fig2_cost_vs_horizon.pdf"     "$PAPER_FIGS/fig2_cost.pdf"
  cp_or_warn "$OUT/comb/fig5_pareto.pdf"              "$PAPER_FIGS/fig5_pareto.pdf"
  cp_or_warn "$OUT/comb/fig6_per_variable.pdf"        "$PAPER_FIGS/fig6_realpde_pervariable.pdf"
  cp_or_warn "$OUT/lh2/fig7_spectra.pdf"              "$PAPER_FIGS/fig7_lh2_spectra.pdf"
  cp_or_warn "$OUT/comb/fig8_semigroup.pdf"           "$PAPER_FIGS/fig8_semigroup.pdf"
  cp_or_warn "$OUT/gs_leadtime/fig3_unseen_query_time.pdf"  "$PAPER_FIGS/fig3_gs_leadtime.pdf"
  cp_or_warn "$OUT/cyl_leadtime/fig3_unseen_query_time.pdf" "$PAPER_FIGS/fig3_cyl_leadtime.pdf"
  echo
  echo "    Then in main.tex change two extensions to .pdf:"
  echo "        fig1_lh2_error.png    -> fig1_lh2_error.pdf"
  echo "        fig7_lh2_spectra.png  -> fig7_lh2_spectra.pdf"
fi

cat <<'MSG'

================================================================
CHECK BEFORE COMMITTING THE FIGURES

  * The "Loaded N model group(s)" line for the three Figure 1 panels. The
    legend is drawn only on the Gray-Scott panel, which is correct only while
    all three hold the same model set. If one is missing POD-DMD, put its
    legend back with --legend on.

  * No panel should show `ar_fno_r_fixsel`. --include matches exactly, so a
    bare `ar_fno_r` cannot pull it in; if it appears, the include list was
    written with a glob somewhere.

  * The captions now carry everything the removed titles used to say. In
    particular the triangle convention: the marker is still drawn and still
    listed in fig1_diverged.txt, but "= diverged, off scale" is no longer in
    the image.

  * fig4_* come from scripts/plot_fields.py, not from here, and are not
    regenerated by this script.
================================================================
MSG
