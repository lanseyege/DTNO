#!/usr/bin/env bash
# Phase A4 — figures. One set per dataset, plus the cross-dataset summary.
#
# ONE FIGURE PER DATASET, NOT ONE FIGURE WITH FIVE DATASETS ON IT.
# E(h) is normalised per dataset but the horizon axis is not comparable across
# them: h = 8 is 20 ms on RealPDEBench combustion, 40 us on Lifted H2, 20 ms on
# Cylinder and 80 s on Gray-Scott. Overlaying the curves makes a visual claim
# that the numbers do not support. The cross-dataset comparison belongs in a
# TABLE of derived quantities -- crossover h, T_pred, Eval*, per-step AR error
# -- where the units are stated per row.
#
#   bash run/24_expA_figures.sh
set -euo pipefail

OUT=${OUT:-results/figures_expA}
mkdir -p "$OUT"

declare -A PREFIX=( [gray_scott]=gs [cylinder]=cyl [rayleigh_benard]=rb )
DATASETS=${DATASETS:-"rayleigh_benard"}

for DS in $DATASETS; do
  P="${PREFIX[$DS]:-$DS}"
  TIMING="results/timing/inference_cost_${P}.json"
  echo ">>> figures for $DS -> $OUT/$P"

  # EXACT NAMES, NOT A GLOB. §6B of the handover: Figure 1's left panel shipped
  # with a contaminated run in it because '--include ar_fno_r*' also matched
  # ar_fno_r_fixsel. Every include below is an exact tag.
  python scripts/make_figures.py \
    --results results/*/horizon_metrics.json \
    --include ${P}_ar_fno_r ${P}_dt_fno ${P}_sg_dt_fno \
              ${P}_persistence ${P}_climatology ${P}_nearest_climatology \
              ${P}_pod_dmd \
    ${TIMING:+--timing "$TIMING"} \
    --out "$OUT/$P"
done


