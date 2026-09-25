#!/usr/bin/env bash
# Phase 3 (§39) — semigroup training on the IDENTICAL architecture.
#
# What matters here is not one-step error. It is C_SG, the interpolation and
# extrapolation columns, and E(h) at h = 32/64/128 — read them together (§31).
set -euo pipefail
NPROC=${NPROC:-4}
SEED=${SEED:-0}

torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
  --config configs/sg_dt_fno.yaml --seed "$SEED" \
  --set experiment.save_dir=./checkpoints/sg_dt_fno_s${SEED} \
  --set experiment.exp_name=sg_dt_fno_s${SEED}

python scripts/evaluate_horizon.py --config configs/sg_dt_fno.yaml \
  --checkpoint ./checkpoints/sg_dt_fno_s${SEED}/best_model.pth \
  --set experiment.exp_name=sg_dt_fno_s${SEED}
