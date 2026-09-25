#!/usr/bin/env bash
# P0 — the cadence intervention. Turns the crossover result from a correlation
# across five datasets into an intervention on one.
#
#   bash run/29_cadence.sh
#   STRIDES="1 2 4 8" bash run/29_cadence.sh
#
# WHAT THIS TESTS, AND WHICH WAY THE PREDICTION RUNS
# ---------------------------------------------------
# Table 3 shows the crossover sitting at h = 8-20 FRAMES across five flows whose
# physical timesteps span 5 us to 10 s. Across datasets that is observational:
# five points, many confounders. Striding one dataset changes the cadence while
# holding the flow, the architecture, the protocol and the split fixed.
#
# The paper's hypothesis is that the crossover is set by how many times a
# learned map is composed, so it predicts the crossover stays at
# **8-20 FRAMES regardless of stride** --- and therefore moves proportionally in
# SECONDS: stride 4 should put the crossover at roughly 4x the physical time.
# The alternative, that the crossover is set by the flow's physical timescale,
# predicts the opposite: constant in seconds, so h* falls as 1/stride.
#
#   stride s:   composition hypothesis -> h* constant,  t* = h* s dt grows
#               physical-time hypothesis -> t* constant, h* ~ 1/s
#
# These are cleanly distinguishable at s = 4 and s = 8.
#
# ONE HONEST CAVEAT, WHICH BELONGS IN THE PAPER
# ----------------------------------------------
# Striding is not a single-variable intervention. It also raises the per-step
# autoregressive error (a 4x coarser step is a harder step) and shortens the
# predictability horizon measured in frames. So a crossover that stays at 8-20
# frames is strong evidence for the composition reading, but a crossover that
# moves is not automatically evidence for the physical-time reading --- it could
# be the per-step error moving. Report the per-step AR error at each stride
# alongside the crossover; if the crossover moves while E_AR(h=1) is flat, the
# composition reading is in trouble, and if both move together the experiment is
# inconclusive and should be reported as such.
#
# RealPDEBench is the right dataset for this: 2001 frames at 250 us leaves room
# for stride 8, and its predictability horizon of 4-6 frames is the regime where
# the crossover is tightest.
set -euo pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}

NPROC=${NPROC:-4}
SEED=${SEED:-0}
CFG=${CFG:-configs/base.yaml}
STRIDES=${STRIDES:-"1 2 4"}
MODELS=${MODELS:-"ar_fno_r dt_fno"}

# stride 1 is already trained: it is the paper's main RealPDEBench run. Do not
# retrain it, and do not compare a fresh stride-1 run against the published
# numbers -- that would introduce a seed difference into the one axis this
# experiment is supposed to hold fixed.
for S in $STRIDES; do
  for M in $MODELS; do
    if [ "$S" = 1 ]; then
      TAG="${M}_s${SEED}"
      echo ">>> stride 1 = the existing ${TAG}; reusing it"
      continue
    fi
    TAG="stride${S}_${M}_s${SEED}"
    if [ -f "./checkpoints/${TAG}/best_model.pth" ]; then
      echo ">>> ${TAG}: exists, skipping training"
    else
      echo ">>> training ${TAG}  (time_stride=${S})"
      # `data.time_stride` wraps the store in StridedStore, so every frame index
      # the samplers use is already in the strided clock. h_max stays 128 FRAMES
      # of the strided record, which is the point: the horizon axis is held
      # fixed in composition count and allowed to move in seconds.
      torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
        --config "$CFG" --seed "$SEED" \
        --set meta.model_variant=$M \
        --set data.time_stride=$S \
        --set data.split_path=artifacts/split_realpde.json \
        --set data.norm_stats_path=artifacts/norm_stats_realpde_stride${S}.json \
        --set experiment.save_dir=./checkpoints/${TAG} \
        --set experiment.tb_dir=./runs/${TAG} \
        --set experiment.exp_name=${TAG}
    fi
    python scripts/evaluate_horizon.py --config "$CFG" \
      --set meta.model_variant=$M --set data.time_stride=$S \
      --set data.norm_stats_path=artifacts/norm_stats_realpde_stride${S}.json \
      --checkpoint ./checkpoints/${TAG}/best_model.pth \
      --set experiment.exp_name=${TAG} \
      --horizons 1 2 4 8 16 32 64 128 160 192 256 384 496
  done
  # the reference lines move with the cadence too and must be re-scored
  if [ "$S" != 1 ]; then
    for B in persistence climatology nearest_climatology; do
      python scripts/evaluate_horizon.py --config "$CFG" --model "$B" \
        --set data.time_stride=$S \
        --set data.norm_stats_path=artifacts/norm_stats_realpde_stride${S}.json \
        --set experiment.exp_name=stride${S}_${B} \
        --horizons 1 2 4 8 16 32 64 128 160 192 256 384 496
    done
  fi
done

echo
echo "NOTE: each stride needs its OWN normalisation statistics, because"
echo "striding changes which frames the training split contains. Run"
echo "  python scripts/audit_data.py --config $CFG --force \\"
echo "      --set data.time_stride=S --set data.norm_stats_path=artifacts/norm_stats_strideS.json"
echo "before the first training at each stride. The split is unchanged --- it is"
echo "a trajectory split, and striding does not renumber trajectories."

cat <<'MSG'

================================================================
THE TABLE THIS PRODUCES

  stride   dt (s)    AR E(h=1)   crossover h*   crossover t* = h* x dt
  ------   -------   ---------   ------------   ----------------------
     1     250 us      0.231           8.1              2.0 ms
     2     500 us        ?              ?                 ?
     4       1 ms        ?              ?                 ?

  Composition hypothesis : h* stays 8-20, t* grows roughly with stride.
  Physical-time hypothesis: t* stays ~2 ms, h* falls roughly as 1/stride.

  Report AR E(h=1) in the same table. If it rises with stride AND h* moves,
  the experiment cannot separate the two hypotheses and should be reported as
  inconclusive rather than read in whichever direction is convenient.
================================================================
MSG
