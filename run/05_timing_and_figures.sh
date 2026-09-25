#!/usr/bin/env bash
# §32 timing + §46 figures.
#
# PIN AN IDLE GPU. On a shared node the wall-clock half of Figure 2 is
# meaningless otherwise; the script records utilisation and warns, but it
# cannot un-contaminate the measurement.
set -euo pipefail
SEED=${SEED:-0}
GPU=${GPU:-0}

CUDA_VISIBLE_DEVICES=$GPU python scripts/benchmark_timing.py \
  --config configs/sg_dt_fno.yaml \
  --ar ./checkpoints/ar_fno_r_s${SEED}/best_model.pth \
  --dt ./checkpoints/dt_fno_s${SEED}/best_model.pth \
  --sg ./checkpoints/sg_dt_fno_s${SEED}/best_model.pth \
  --max_timed_horizon 512

python scripts/make_figures.py \
  --results results/*/horizon_metrics.json \
  --timing results/timing/inference_cost.json \
  --out results/figures

python scripts/plot_fields.py --config configs/sg_dt_fno.yaml \
  --checkpoints ar_fno=./checkpoints/ar_fno_r_s${SEED}/best_model.pth \
                dt_fno=./checkpoints/dt_fno_s${SEED}/best_model.pth \
                sg_dt_fno=./checkpoints/sg_dt_fno_s${SEED}/best_model.pth \
  --h 128
