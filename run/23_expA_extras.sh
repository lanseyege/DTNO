#!/usr/bin/env bash
# Phase A3 — optional Experiment A arms. None of these blocks the paper; each
# closes a question a reviewer is likely to ask.
#
# Pick one with BLOCK=; there is no default that runs everything, because
# together they are more GPU time than the core experiment.
#
#   BLOCK=hmax        bash run/23_expA_extras.sh   # A5 on the new datasets
#   BLOCK=long        bash run/23_expA_extras.sh   # Cylinder's long horizons
#   BLOCK=paramhold   bash run/23_expA_extras.sh   # unseen Gray-Scott patterns
#   BLOCK=native_rb   bash run/23_expA_extras.sh   # Rayleigh-Benard at 512x128
set -euo pipefail

NPROC=${NPROC:-4}
SEED=${SEED:-0}
BLOCK=${BLOCK:-""}

if [ -z "$BLOCK" ]; then
  sed -n '1,14p' "$0"; exit 1
fi

# --------------------------------------------------------------------------
# A5 on a new dataset: does h_max_train behave the same way off combustion?
#
# On RealPDEBench: h_max 16/32/64/128 -> E(h=8) 0.642/0.621/0.618/0.616 and
# Eval* 0.771/0.761/0.692/0.630. Larger is better, short-horizon accuracy is
# unaffected, and models do not extrapolate past their training range.
#
# The interesting question is whether "larger is better" survives on a dataset
# whose predictable window is LONGER than its training horizon. On combustion
# it could not be separated from "more supervision in the climatology regime".
# --------------------------------------------------------------------------
if [ "$BLOCK" = "hmax" ]; then
  DS=${DS:-gray_scott}
  HMAXES=${HMAXES:-"16 32 64 128"}
  CFG="configs/${DS}.yaml"
  for HM in $HMAXES; do
    TAG="${DS}_hmax${HM}_s${SEED}"
    torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
      --config "$CFG" --seed "$SEED" \
      --set data.h_max_train=$HM \
      --set experiment.save_dir=./checkpoints/${TAG} \
      --set experiment.tb_dir=./runs/${TAG} \
      --set experiment.exp_name=${TAG}
    # scored on the FULL grid whatever h_max_train was, so restricting
    # training shows its cost as well as its benefit
    python scripts/evaluate_horizon.py --config "$CFG" \
      --set data.h_max_train=$HM \
      --checkpoint ./checkpoints/${TAG}/best_model.pth \
      --set experiment.exp_name=${TAG}
  done
fi

# --------------------------------------------------------------------------
# Cylinder's long horizons. 3990 frames is the only place in the study where
# h = 1024 is affordable, and the constant-inference-depth claim is at its most
# dramatic there: AR needs 1024 sequential passes, DT needs one.
#
# This is a SEPARATE arm, not a change to configs/cylinder.yaml, because
# h_max_train is a confound in the cross-dataset comparison and A5 showed the
# two arms answer different questions.
# --------------------------------------------------------------------------
if [ "$BLOCK" = "long" ]; then
  CFG=configs/cylinder.yaml
  GRID="[1,2,4,8,16,32,64,128,256,384,512,768,1024]"
  for M in ar_fno_r dt_fno; do
    TAG="cyl_long_${M}_s${SEED}"
    torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
      --config "$CFG" --seed "$SEED" \
      --set meta.model_variant=$M \
      --set data.h_max_train=512 \
      --set "data.horizon_bins=[[1,2],[3,4],[5,8],[9,16],[17,32],[33,64],[65,128],[129,256],[257,512]]" \
      --set experiment.save_dir=./checkpoints/${TAG} \
      --set experiment.tb_dir=./runs/${TAG} \
      --set experiment.exp_name=${TAG}
    python scripts/evaluate_horizon.py --config "$CFG" \
      --set meta.model_variant=$M \
      --set data.h_max_train=512 \
      --set "data.eval_horizons=${GRID}" \
      --set data.eval_stride=200 \
      --checkpoint ./checkpoints/${TAG}/best_model.pth \
      --set experiment.exp_name=${TAG}
  done
  for B in persistence climatology nearest_climatology; do
    python scripts/evaluate_horizon.py --config "$CFG" --model "$B" \
      --set "data.eval_horizons=${GRID}" --set data.eval_stride=200 \
      --set experiment.exp_name=cyl_long_${B}
  done
  echo
  echo "AR-FNO-R at h = 1024 will take a long time and may diverge. That is"
  echo "the point of the figure. Model selection now uses the BOUNDED score,"
  echo "so a diverged horizon no longer silently decides which checkpoint"
  echo "was 'best' -- but check results/*/horizon_metrics.json for"
  echo "diverged_horizons before quoting the curve."
fi

# --------------------------------------------------------------------------
# Unseen pattern regime on Gray-Scott. The Well's own documentation names this
# as the interesting evaluation: "It would be impressive if a simulator --
# trained only on some of the patterns produced by a subset of the (f, k)
# parameter space -- could perform well on an unseen set of parameter values."
#
# It is a DIFFERENT and strictly harder claim than the headline (unseen initial
# condition, seen dynamics), so it gets its own split file, its own statistics
# and its own tag. Never present it as the same experiment.
# --------------------------------------------------------------------------
if [ "$BLOCK" = "paramhold" ]; then
  HOLD=${HOLD:-"spirals worms"}
  python scripts/prepare_split.py --config configs/gray_scott.yaml \
    --mode param_holdout --holdout $HOLD \
    --out artifacts/split_gray_scott_paramhold.json
  python scripts/audit_data.py --config configs/gray_scott.yaml --force \
    --set data.split_path=artifacts/split_gray_scott_paramhold.json \
    --set data.norm_stats_path=artifacts/norm_stats_gray_scott_paramhold.json
  for M in ar_fno_r dt_fno; do
    TAG="gs_paramhold_${M}_s${SEED}"
    torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
      --config configs/gray_scott.yaml --seed "$SEED" \
      --set meta.model_variant=$M \
      --set data.split_path=artifacts/split_gray_scott_paramhold.json \
      --set data.norm_stats_path=artifacts/norm_stats_gray_scott_paramhold.json \
      --set experiment.save_dir=./checkpoints/${TAG} \
      --set experiment.tb_dir=./runs/${TAG} \
      --set experiment.exp_name=${TAG}
    python scripts/evaluate_horizon.py --config configs/gray_scott.yaml \
      --set meta.model_variant=$M \
      --set data.split_path=artifacts/split_gray_scott_paramhold.json \
      --set data.norm_stats_path=artifacts/norm_stats_gray_scott_paramhold.json \
      --checkpoint ./checkpoints/${TAG}/best_model.pth \
      --set experiment.exp_name=${TAG}
  done
fi

# --------------------------------------------------------------------------
# Rayleigh-Benard at native 512 x 128. The default crops to 128 x 128 so that
# resolution is not a second uncontrolled difference between datasets. If the
# paper claims anything about small horizontal scales, that claim has to be
# made at native resolution.
# --------------------------------------------------------------------------
if [ "$BLOCK" = "native_rb" ]; then
  EXTRA=(--set data.spatial_stride=1 --set data.spatial_crop=null
         --set model.modes2=32
         --set data.split_path=artifacts/split_rayleigh_benard.json
         --set data.norm_stats_path=artifacts/norm_stats_rb_native.json)
  # The statistics are per-grid: cropping changes which points are averaged.
  python scripts/audit_data.py --config configs/rayleigh_benard.yaml --force \
    "${EXTRA[@]}"
  for M in ar_fno_r dt_fno; do
    TAG="rb_native_${M}_s${SEED}"
    torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
      --config configs/rayleigh_benard.yaml --seed "$SEED" \
      --set meta.model_variant=$M "${EXTRA[@]}" \
      --set experiment.save_dir=./checkpoints/${TAG} \
      --set experiment.tb_dir=./runs/${TAG} \
      --set experiment.exp_name=${TAG}
    python scripts/evaluate_horizon.py --config configs/rayleigh_benard.yaml \
      --set meta.model_variant=$M "${EXTRA[@]}" \
      --checkpoint ./checkpoints/${TAG}/best_model.pth \
      --set experiment.exp_name=${TAG}
  done
fi
