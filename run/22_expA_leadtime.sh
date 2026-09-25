#!/usr/bin/env bash
# Phase A2 — RQ3 (unseen lead-time interpolation) and RQ4 (extrapolation).
#
# Trains on a sparse horizon set and scores horizons that were never trained.
# Both time embeddings, because §13.1's aliasing caveat makes the extrapolation
# column uninterpretable otherwise, and because Experiment B on RealPDEBench
# found the ranking between them REVERSES under dense horizon sampling
# (roughness 0.228 Fourier vs 0.067 log-Fourier; worst unseen horizon 0.939 vs
# 0.626). Two more datasets are what turn that from an anomaly into a finding.
#
#   bash run/22_expA_leadtime.sh
#   DATASETS="gray_scott" MODES="fourier_log" bash run/22_expA_leadtime.sh
#
# Rayleigh-Benard is excluded by default: with T = 200 and h_max = 128 there
# are 17 anchors per trajectory, and an interpolation curve needs to be read
# point by point rather than in aggregate. Add it back with DATASETS if you
# want it, and report the anchor count in the caption.
set -euo pipefail

NPROC=${NPROC:-4}
SEED=${SEED:-0}
DATASETS=${DATASETS:-"gray_scott cylinder"}
MODES=${MODES:-"fourier fourier_log"}

declare -A PREFIX=( [gray_scott]=gs [cylinder]=cyl [rayleigh_benard]=rb )

for DS in $DATASETS; do
  CFG="configs/expA_leadtime_${DS}.yaml"
  [ -f "$CFG" ] || { echo "missing $CFG" >&2; exit 1; }
  P="${PREFIX[$DS]:-$DS}"

  for MODE in $MODES; do
    TAG="${P}_leadtime_${MODE}_s${SEED}"
    if [ -f "./checkpoints/${TAG}/best_model.pth" ]; then
      echo ">>> ${TAG}: checkpoint exists, skipping training"
    else
      echo ">>> training ${TAG}"
      torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
        --config "$CFG" --seed "$SEED" \
        --set model.time_embed_mode=$MODE \
        --set experiment.save_dir=./checkpoints/${TAG} \
        --set experiment.tb_dir=./runs/${TAG} \
        --set experiment.exp_name=${TAG}
    fi

    python scripts/evaluate_horizon.py --config "$CFG" \
      --set model.time_embed_mode=$MODE \
      --checkpoint ./checkpoints/${TAG}/best_model.pth \
      --set experiment.exp_name=${TAG}

    # Query-time sensitivity: does the output actually move when only tau
    # moves? A model that ignores tau can still score well on the trained
    # anchors, and this is the diagnostic that separates the two.
    python scripts/check_time_sensitivity.py --config "$CFG" \
      --set model.time_embed_mode=$MODE \
      --checkpoint ./checkpoints/${TAG}/best_model.pth \
      --set experiment.exp_name=${TAG}
  done

  # The reference line without which a flat interpolation curve looks like
  # success. Evaluated on the SAME grid, or the columns do not line up.
  python scripts/evaluate_horizon.py --config "$CFG" \
    --model climatology --set experiment.exp_name=${P}_leadtime_climatology
  python scripts/evaluate_horizon.py --config "$CFG" \
    --model nearest_climatology \
    --set experiment.exp_name=${P}_leadtime_nearest_climatology
done

cat <<'MSG'

================================================================
READING IT

  RQ3 (interpolation). Compare E(h) at the UNSEEN horizons against a smooth
  interpolant of the trained ones. Report the roughness statistic, not just
  the curve: a lead-time-conditioned operator has small roughness, a bank of
  memorised heads has large roughness and visible steps at the anchors.

  Read it only in the range where E(h) still MOVES. Where the climatology
  line is flat and the model sits on it, interpolation is trivially perfect
  and means nothing -- this is the exact reason the RealPDEBench version of
  this experiment had to be redesigned (see the header of
  configs/expB_dt_fno.yaml).

  RQ4 (extrapolation). h > 128 is past h_max_train. A5 measured that models
  do not extrapolate past their training range, so the expected result is
  degradation. Reporting it as a measured limitation is stronger than
  omitting it, and it is what makes the time-embedding comparison meaningful:
  the question is which embedding degrades more gracefully, not whether
  either succeeds.

  On Cylinder, check the shedding period from
  results/probe/timescales_cylinder.json before believing any suspiciously
  good column: at h close to a multiple of the period the field repeats, and
  persistence enjoys the same coincidence. If persistence is also good there,
  the model has learned the clock, not the dynamics.
================================================================
MSG
