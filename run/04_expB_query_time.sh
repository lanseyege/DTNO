#!/usr/bin/env bash
# Experiment B (§21) — unseen query-time generalization.
#
# Grid redesigned after Phase 2: training anchors are sparse ({1,4,16,64,128})
# so the unseen interior points land where E(h) still moves. See the header of
# configs/expB_dt_fno.yaml for why the original grid could not answer the
# question on this dataset.
#
# Both the default Fourier embedding and the log variant, because §13.1's
# aliasing caveat makes the extrapolation column uninterpretable otherwise.
set -euo pipefail
NPROC=${NPROC:-4}
SEED=${SEED:-0}

for CFG in configs/expB_dt_fno.yaml configs/expB_sg_dt_fno.yaml; do
  NAME=$(basename "$CFG" .yaml)
  for MODE in fourier fourier_log; do
    TAG=${NAME}_${MODE}_s${SEED}
    torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
      --config "$CFG" --seed "$SEED" \
      --set model.time_embed_mode=$MODE \
      --set experiment.save_dir=./checkpoints/${TAG} \
      --set experiment.exp_name=${TAG}
    python scripts/evaluate_horizon.py --config "$CFG" \
      --set model.time_embed_mode=$MODE \
      --checkpoint ./checkpoints/${TAG}/best_model.pth \
      --set experiment.exp_name=${TAG}
    python scripts/check_time_sensitivity.py --config "$CFG" \
      --set model.time_embed_mode=$MODE \
      --checkpoint ./checkpoints/${TAG}/best_model.pth \
      --set experiment.exp_name=${TAG}
  done
done

# The reference line every Experiment B figure needs: past the predictability
# horizon all curves converge towards this, and without it a flat interpolation
# curve looks like success.
python scripts/evaluate_horizon.py --config configs/expB_dt_fno.yaml \
  --model climatology --set experiment.exp_name=expB_climatology
