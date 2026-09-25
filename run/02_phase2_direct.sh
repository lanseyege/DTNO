#!/usr/bin/env bash
# Phase 2 (§38) — DT-FNO, no semigroup. THE decision point of the whole project.
#
# After this finishes, compare E_DT(h) with E_AR(h) from Phase 1. If DT has
# already collapsed by h = 4-8, §44's Negative Result A applies and the next
# move is conditional Koopman / structured latent dynamics, not a bigger FNO.
set -euo pipefail
NPROC=${NPROC:-4}
SEED=${SEED:-0}

torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
  --config configs/dt_fno.yaml --seed "$SEED" \
  --set experiment.save_dir=./checkpoints/dt_fno_s${SEED} \
  --set experiment.exp_name=dt_fno_s${SEED}

python scripts/evaluate_horizon.py --config configs/dt_fno.yaml \
  --checkpoint ./checkpoints/dt_fno_s${SEED}/best_model.pth \
  --set experiment.exp_name=dt_fno_s${SEED}
