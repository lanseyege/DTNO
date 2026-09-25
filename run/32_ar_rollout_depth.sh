#!/usr/bin/env bash
# ICLR 2027 revision P0: AR rollout-depth accuracy/compute sweep.
#
# Trains FIXED rollout depths R, not random Rmax training. This makes the
# experiment's x-axis exactly the temporal backpropagation depth and lets the
# trainer telemetry (epoch time + peak GPU memory) be interpreted directly.
#
# Examples:
#   NPROC=4 DATASETS="gray_scott realpde" ROLLOUTS="1 4 8 16" bash run/32_ar_rollout_depth.sh
#   NPROC=4 DATASETS="gray_scott" ROLLOUTS="32" SEEDS="0" bash run/32_ar_rollout_depth.sh
#
# If R=32 OOMs at the default micro-batch 4, rerun with BATCH_SIZE=2. Do NOT
# silently use different batch sizes inside one plotted rollout-depth sweep.
set -euo pipefail

NPROC=${NPROC:-4}
SEEDS=${SEEDS:-"0"}
ROLLOUTS=${ROLLOUTS:-"1 4 8 16"}
DATASETS=${DATASETS:-"gray_scott realpde"}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_EPOCHS=${MAX_EPOCHS:-}

for DS in $DATASETS; do
  case "$DS" in
    gray_scott|gs)
      CFG=configs/experiments/rollout_depth_gray_scott.yaml
      TAG_PREFIX=rev_gs
      ;;
    realpde|realpde_combustion|rp)
      CFG=configs/experiments/rollout_depth_realpde.yaml
      TAG_PREFIX=rev_rp
      ;;
    rayleigh_benard|rb)
      CFG=configs/experiments/rollout_depth_rayleigh_benard.yaml
      TAG_PREFIX=rev_rb
      ;;
    *) echo "unknown DATASETS entry: $DS" >&2; exit 2 ;;
  esac

  for S in $SEEDS; do
    for R in $ROLLOUTS; do
      TAG="${TAG_PREFIX}_arR${R}_s${S}"
      CKPT="./checkpoints/${TAG}/best_model.pth"
      echo
      echo "================================================================"
      echo "  $TAG  fixed rollout R=$R  batch=$BATCH_SIZE"
      echo "================================================================"

      TRAIN_ARGS=(
        --config "$CFG" --seed "$S"
        --set meta.model_variant=ar_fno_r
        --set data.ar_rollout="$R"
        --set data.ar_random_rollout=false
        --set data.batch_size="$BATCH_SIZE"
        --set experiment.exp_name="$TAG"
        --set experiment.save_dir="./checkpoints/${TAG}"
        --set experiment.tb_dir="./runs/${TAG}"
      )
      if [ -n "$MAX_EPOCHS" ]; then
        TRAIN_ARGS+=(--set training.max_epochs="$MAX_EPOCHS")
      fi

      if [ ! -f "$CKPT" ]; then
        torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py "${TRAIN_ARGS[@]}"
      else
        echo "  checkpoint exists: $CKPT (skip training)"
      fi

      python scripts/evaluate_horizon.py \
        --config "$CFG" --seed "$S" \
        --set meta.model_variant=ar_fno_r \
        --set data.ar_rollout="$R" \
        --set data.ar_random_rollout=false \
        --set data.batch_size="$BATCH_SIZE" \
        --set experiment.exp_name="$TAG" \
        --checkpoint "$CKPT"
    done
  done
done

python scripts/summarize_rollout_depth.py \
  --checkpoint_root checkpoints --results_root results \
  --prefixes rev_gs rev_rp rev_rb \
  --out results/revision/rollout_depth_summary.csv

cat <<'MSG'

================================================================
P0 sweep complete.

Use results/revision/rollout_depth_summary.csv for the paper table/plot.
For the clean training-memory claim, ALSO run scripts/benchmark_training_cost.py
on an idle single GPU with a fixed micro-batch; epoch-level peak memory includes
real training overhead and is useful corroboration, while the microbenchmark
isolates the graph itself.
================================================================
MSG
