#!/usr/bin/env bash
# Extra seeds for the rollout-depth sweep (Section "Autoregressive training depth").
#
#   NPROC=3 DEPTHS="16" SEEDS="1 2" bash run/34_rollout_depth_seeds.sh
#
# The sweep is single-seed, and its headline comparison -- R=16 reaching
# Eval* 0.525 against the direct model's 0.347 -- is the one claim in the paper
# that rests on one run per point.
#
# TWO CONSTRAINTS, both of which make this more expensive than it looks.
#
# 1. NPROC must equal the value used for seed 0. samples_per_epoch is global, so
#    world size sets steps/epoch, which sets the LR schedule. A seed run at a
#    different world size is not a seed of the same experiment.
#
# 2. Do NOT shorten the epoch budget, tempting though it is. The R=16 arms
#    reached their best score at epoch 19 and 24 of 100, so 50 epochs looks
#    sufficient -- but the learning rate follows a cosine decay over the TOTAL
#    epoch count, so a 50-epoch run follows a different LR trajectory and is not
#    comparable with seed 0. Budget the full 100 epochs: about 8.3 h per seed at
#    R=16 on Gray-Scott, 2.2 h at R=4, 0.7 h at R=1.
#
# If time is short, one extra seed at the single depth the headline rests on
# (R=16, Gray-Scott) is worth more than a spread of cheap ones.
set -euo pipefail
NPROC=${NPROC:-3}
DATASETS=${DATASETS:-"gray_scott"}
DEPTHS=${DEPTHS:-"16"}
SEEDS=${SEEDS:-"1"}
BATCH_SIZE=${BATCH_SIZE:-32}

for S in $SEEDS; do
  for D in $DATASETS; do
    for R in $DEPTHS; do
      echo ">>> ${D} R=${R} seed ${S}"
      NPROC="$NPROC" DATASETS="$D" ROLLOUTS="$R" SEEDS="$S" \
        BATCH_SIZE="$BATCH_SIZE" bash run/32_ar_rollout_depth.sh
    done
  done
done
echo
echo "Then refresh the CSV and report mean +- sd per depth:"
echo "  python scripts/summarize_rollout_depth.py"
echo "Report the per-seed divergence counts rather than their mean: a mean over"
echo "seeds that straddle divergence is not a centre."
