#!/usr/bin/env bash
# Convenience wrapper for the two systems benchmarks added for the ICLR revision.
# Required for multi-horizon: DT_CHECKPOINT. Optional: AR_CHECKPOINT.
# Example:
#   CONFIG=configs/gray_scott.yaml DT_CHECKPOINT=checkpoints/gs_dt_fno_s0/best_model.pth \
#   AR_CHECKPOINT=checkpoints/gs_ar_fno_r_s0/best_model.pth bash run/33_revision_benchmarks.sh
set -euo pipefail

CONFIG=${CONFIG:-configs/gray_scott.yaml}
GPU=${GPU:-0}
BATCH_SIZE=${BATCH_SIZE:-2}
ROLLOUTS=${ROLLOUTS:-"1 4 8 16 32"}
DIRECT_HORIZONS=${DIRECT_HORIZONS:-"1 128"}
QUERY_COUNTS=${QUERY_COUNTS:-"1 4 8 16 32 64 128"}
CHUNKS=${CHUNKS:-"4 8 16 32 0"}
N_WARMUP=${N_WARMUP:-3}
N_REPEAT=${N_REPEAT:-10}
DT_CHECKPOINT=${DT_CHECKPOINT:-}
AR_CHECKPOINT=${AR_CHECKPOINT:-}

read -ra R_ARR <<< "$ROLLOUTS"
read -ra H_ARR <<< "$DIRECT_HORIZONS"
read -ra Q_ARR <<< "$QUERY_COUNTS"
read -ra C_ARR <<< "$CHUNKS"

CUDA_VISIBLE_DEVICES="$GPU" python scripts/benchmark_training_cost.py \
  --config "$CONFIG" \
  --rollouts "${R_ARR[@]}" \
  --direct_horizons "${H_ARR[@]}" \
  --batch_size "$BATCH_SIZE" \
  --n_warmup "$N_WARMUP" --n_repeat "$N_REPEAT"

if [ -n "$DT_CHECKPOINT" ]; then
  AR_ARGS=()
  if [ -n "$AR_CHECKPOINT" ]; then
    AR_ARGS=(--ar "$AR_CHECKPOINT")
  fi
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/benchmark_multi_horizon.py \
    --config "$CONFIG" \
    --dt "$DT_CHECKPOINT" \
    "${AR_ARGS[@]}" \
    --query_counts "${Q_ARR[@]}" \
    --chunks "${C_ARR[@]}" \
    --n_warmup "$N_WARMUP" --n_repeat "$N_REPEAT"
else
  echo "DT_CHECKPOINT not set: skipped multi-horizon benchmark."
fi
