#!/usr/bin/env bash
# P0 — the control the U-Net comparison now needs.
#
#   bash run/30_precision_control.sh
#
# WHAT HAPPENED, AND WHY THIS IS NOT OPTIONAL
# -------------------------------------------
# The matched-backbone runs came back with the crossover moving a long way:
#
#     dataset          FNO      U-Net
#     Gray-Scott       9.1      194.1
#     RealPDEBench     8.1       15.3
#
# and the Gray-Scott AR U-Net did not diverge at ANY horizon, where the AR FNO
# diverges on the same data at 4 of 13 horizons and returns non-finite fields on
# one seed in three. Read naively that says the crossover is an FNO property and
# the paper's Section 5.4 is wrong.
#
# It cannot be read naively, because two things changed at once. The U-Net would
# not train in bf16 --- it produced a non-finite loss at epoch 7 --- and was
# fixed by wrapping its backbone in `torch.autocast(enabled=False)`. So the
# U-Net runs in fp32 while the FNO runs in bf16, and PRECISION is confounded
# with ARCHITECTURE. The most interesting number here, the disappearance of
# autoregressive divergence, is exactly the number most likely to be explained
# by the precision change rather than by the architecture.
#
# This script separates them: AR-FNO-R and DT-FNO on Gray-Scott in fp32,
# everything else identical to the runs in Table 2.
#
#     if AR-FNO-R still diverges in fp32 and the crossover stays near 9
#         -> precision is not the explanation; the U-Net result stands and the
#            architecture-invariance claim must be weakened to what it is
#     if AR-FNO-R stops diverging in fp32 and the crossover moves toward 194
#         -> the headline instability result is partly a bf16 artefact. That is
#            a bigger finding than the architecture one, it changes Section 5.3,
#            and it has to be reported whether or not it is convenient.
#
# Two runs, and they gate a headline claim.
set -euo pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}

NPROC=${NPROC:-3}
SEEDS=${SEEDS:-"0"}
CFG=${CFG:-configs/gray_scott.yaml}
PRE=${PRE:-gs_}
MODELS=${MODELS:-"ar_fno_r dt_fno"}

for S in $SEEDS; do
  for M in $MODELS; do
    TAG="${PRE}${M}_fp32_s${S}"
    if [ -f "./checkpoints/${TAG}/best_model.pth" ]; then
      echo ">>> ${TAG}: exists, skipping training"
    else
      echo ">>> training ${TAG}  (amp off)"
      # training.amp=false is the ONLY difference from the Table 2 runs. Do not
      # also change the seed, the split or the horizon grid: the comparison is
      # against ${PRE}${M}_s${S}, and a second difference would make it
      # unreadable in exactly the way the U-Net comparison is unreadable now.
      torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
        --config "$CFG" --seed "$S" \
        --set meta.model_variant=$M \
        --set training.amp=false \
        --set experiment.save_dir=./checkpoints/${TAG} \
        --set experiment.tb_dir=./runs/${TAG} \
        --set experiment.exp_name=${TAG}
    fi
    python scripts/evaluate_horizon.py --config "$CFG" \
      --set meta.model_variant=$M --set training.amp=false \
      --checkpoint ./checkpoints/${TAG}/best_model.pth \
      --set experiment.exp_name=${TAG}
  done
done

echo
echo ">>> crossover, fp32 versus bf16"
python scripts/make_figures.py --results results/*/horizon_metrics.json \
  --include ${PRE}ar_fno_r_fp32 ${PRE}dt_fno_fp32 \
  --label "fp32 control" --out results/figures_fp32
cat results/figures_fp32/fig1_crossover.txt 2>/dev/null || true
echo
echo "  for comparison, from the runs already in the paper:"
cat results/figures_paper/gs/fig1_crossover.txt 2>/dev/null \
  || echo "    (regenerate the Gray-Scott figure to see it)"

cat <<'MSG'

================================================================
WHAT TO RECORD EITHER WAY

  The divergence count matters as much as the crossover. Gray-Scott AR-FNO-R
  in bf16 diverges at 4 of 13 horizons, seed-mean E(h=128) = 41.0, with one
  seed non-finite beyond h = 256. If the fp32 run diverges at 0, the paper's
  reproducibility claim -- which is currently one of its four contributions --
  is about a precision setting and not about autoregression.

  Report the precision of every run in the reproducibility appendix. It is
  currently unstated, and it is now known to be load-bearing:
  the U-Net arms are fp32 by necessity, the FNO arms bf16 by default, and no
  wall-clock comparison between the two backbones is meaningful until they
  match.
================================================================
MSG
