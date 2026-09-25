#!/usr/bin/env bash
# P0 -- the control the U-Net comparison turns out to need.
#
#   NPROC=3 bash run/35_capacity_control.sh
#
# WHY
# ---
# The paper calls the U-Net backbone "matched". It is not. The logs report
#
#     AR-FNO   Gray-Scott     8,414,786 parameters
#     AR-UNet  Gray-Scott    16,008,626 parameters      (1.90x)
#
# The U-Net was sized to match a parameter count that was itself an analytic
# reconstruction, and that reconstruction was about a factor of two too large.
# So the finding that the U-Net rollout never diverges, on which Section 5.1's
# "the advantage tracks the instability of the rollout" now rests, is confounded
# with roughly twice the capacity. A reviewer will say so, and they will be
# right unless this is controlled.
#
# The truncation arms do NOT settle it: they were matched to the true 8.4M
# baseline (8.40M, 8.52M), so they vary bandwidth at fixed capacity and say
# nothing about capacity itself.
#
# This runs the same FNO with the width raised so that its parameter count
# lands on the U-Net's, changing nothing else. width 88 gives
# 8.41M x (88/64)^2 ~ 15.9M, against the AR-UNet's 16.01M.
#
#     still diverges  -> capacity is not the explanation; the U-Net result is
#                        about architecture and Section 5.1 stands as written
#     stops diverging -> the U-Net finding is a capacity finding, and both the
#                        main text and Appendix "Architecture and bandwidth
#                        controls" must say so
#
# The autoregressive arm alone answers it. Add dt_fno only if you also want the
# crossover at this width.
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}

NPROC=${NPROC:-3}          # must equal what the baseline used: steps/epoch, and
                           # therefore the LR schedule, depend on world size
SEED=${SEED:-0}
CFG=${CFG:-configs/gray_scott.yaml}
WIDTH=${WIDTH:-88}
MODELS=${MODELS:-"ar_fno_r"}

for M in $MODELS; do
  TAG="gs_${M}_w${WIDTH}_s${SEED}"
  if [ -f "./checkpoints/${TAG}/best_model.pth" ]; then
    echo ">>> ${TAG}: exists, skipping training"
  else
    echo ">>> training ${TAG}  (width ${WIDTH}; everything else as the baseline)"
    torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
      --config "$CFG" --seed "$SEED" \
      --set meta.model_variant=$M \
      --set model.width=$WIDTH \
      --set experiment.save_dir=./checkpoints/${TAG} \
      --set experiment.tb_dir=./runs/${TAG} \
      --set experiment.exp_name=${TAG}
  fi
  python scripts/evaluate_horizon.py --config "$CFG" \
    --set meta.model_variant=$M --set model.width=$WIDTH \
    --checkpoint ./checkpoints/${TAG}/best_model.pth \
    --set experiment.exp_name=${TAG}
done

cat <<'MSG'

================================================================
READING IT

  Check the printed parameter count first: it should be near 16.0M, i.e. the
  AR-UNet's. If it is not, adjust WIDTH and rerun -- the whole point is to hold
  capacity at the U-Net's level.

  Then compare the DIVERGED HORIZON COUNT against the baseline, not accuracy:

      AR-FNO   8.4M, Gray-Scott   diverges at 0 / 5 / 6 horizons across seeds
      AR-UNet 16.0M, Gray-Scott   diverges at 0 horizons in all three seeds

  A single seed here sits inside the baseline's 0-6 spread, so one clean run is
  suggestive, not conclusive; two would be better if there is time.
================================================================
MSG
