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
DATASETS=${DATASETS:-"gray_scott cylinder rayleigh_benard"}

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

# Lead-time interpolation, where it was run.
for P in gs cyl; do
  ls results/${P}_leadtime_*/horizon_metrics.json >/dev/null 2>&1 || continue
  echo ">>> lead-time figures for $P"
  python scripts/make_figures.py \
    --results results/*/horizon_metrics.json \
    --include ${P}_leadtime_fourier_s0 ${P}_leadtime_fourier_log_s0 \
              ${P}_leadtime_climatology ${P}_leadtime_nearest_climatology \
    --out "$OUT/${P}_leadtime"
done

cat <<'MSG'

================================================================
CHECK BEFORE USING ANY OF THESE

  * make_figures.py warns when groups differ in a field that changes
    training (meta.train_config is recorded in every results JSON). Read
    those warnings. They exist because run/11 silently compared runs that
    carried loss_channel_weights against runs that did not.

  * E_spec on Rayleigh-Benard is not a physical spectrum along z if you used
    The Well's `rayleigh_benard` rather than `rayleigh_benard_uniform` -- the
    vertical axis is sampled at Chebyshev nodes. Either caption it or use the
    uniform version.

  * Run scripts/spec_channel_report.py before quoting any E_spec number.
    §9 of the handover: DT-FNO's reported E_spec of 3.1-6.9 on RealPDEBench
    came almost entirely from one pathological channel.

  * The cross-dataset table is assembled by hand from
    results/*/horizon_metrics.json. Per row, state: dataset, dt, T, T_pred in
    SECONDS (never frames), crossover h, per-step AR error, Eval*, and both
    climatology baselines. Reporting T_pred in frames across datasets whose
    dt differs by five orders of magnitude is the single easiest way to make
    a false cross-dataset claim.
================================================================
MSG
