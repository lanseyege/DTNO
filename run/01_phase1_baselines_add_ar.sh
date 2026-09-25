#!/usr/bin/env bash
# Phase 1 (§37) — establish the benchmark floor: POD-DMD, persistence, AR-FNO.
#
# The AR runs are the expensive part. AR-FNO-R trains ~R times slower per sample
# than AR-FNO-1; budget accordingly.
set -euo pipefail
NPROC=${NPROC:-4}
SEED=${SEED:-0}

python scripts/fit_dmd.py --config configs/dt_fno.yaml --rank 128

for CFG in configs/ar_fno.yaml configs/ar_fno_r.yaml; do
  NAME=$(basename "$CFG" .yaml)
  torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
    --config "$CFG" --seed "$SEED" \
    --set experiment.save_dir=./checkpoints/${NAME}_s${SEED} \
    --set experiment.exp_name=${NAME}_s${SEED}

  # Evaluate immediately. Training a baseline and never scoring it leaves
  # Figure 1 without the curve the whole project is measured against.
  python scripts/evaluate_horizon.py --config "$CFG" \
    --checkpoint ./checkpoints/${NAME}_s${SEED}/best_model.pth \
    --set experiment.exp_name=${NAME}_s${SEED}
done

for M in persistence pod_dmd; do
  python scripts/evaluate_horizon.py --config configs/dt_fno.yaml \
    --model $M --set experiment.exp_name=$M
done
