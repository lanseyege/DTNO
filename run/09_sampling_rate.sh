#!/usr/bin/env bash
# The controlled experiment the two datasets cannot give on their own.
#
# Measured so far:
#   RealPDEBench  dt = 250 us   AR-vs-DT crossover at h ~ 3.3   (~825 us)
#   Lifted H2     dt =   5 us   AR-vs-DT crossover at h ~ 22    (~110 us)
#
# Three things differ between those runs -- sampling rate, physics, and geometry
# -- so the comparison cannot say which one moved the crossover. Striding one
# dataset in time holds physics and trajectories fixed and varies ONLY dt.
#
# The question, stated so it can be falsified:
#   crossover constant in PHYSICAL time across strides -> set by the flow
#   crossover constant in FRAMES across strides        -> set by the model and
#                                                         the K-frame history
#
# The second outcome would be the more consequential one: it would mean the
# horizon limit we measured is an artefact of the architecture, and the whole
# §47 "why does direct prediction work" analysis would have to start there.
#
#   DATASET=realpde bash run/09_sampling_rate.sh
#   DATASET=lifted_h2 STRIDES="1 2 4" bash run/09_sampling_rate.sh
#
# Striding shortens the trajectories, so h_max and the eval grid are scaled down
# with the stride to keep the PHYSICAL horizon range fixed. That is the point:
# every stride must cover the same span in seconds, or the curves are not
# comparable.
set -euo pipefail

NPROC=${NPROC:-4}
SEED=${SEED:-0}
DATASET=${DATASET:-realpde}
STRIDES=${STRIDES:-"1 2 4 8"}
MODELS=${MODELS:-"ar_fno_r dt_fno"}

case "$DATASET" in
  realpde)   CFG=configs/dt_fno.yaml;    H_MAX=128; EVAL="1 2 4 8 16 32 64 128" ;;
  lifted_h2) CFG=configs/lifted_h2.yaml; H_MAX=64;  EVAL="1 2 4 8 16 32 64" ;;
  *) echo "DATASET must be realpde | lifted_h2" >&2; exit 1 ;;
esac

scale () {   # divide each horizon by the stride, drop anything below 1
  local s=$1; shift
  local out=""
  for h in "$@"; do
    local v=$(( h / s ))
    [ "$v" -ge 1 ] && out="$out $v"
  done
  echo "$out" | tr ' ' '\n' | sort -n -u | tr '\n' ',' | sed 's/^,//;s/,$//'
}

for S in $STRIDES; do
  HM=$(( H_MAX / S )); [ "$HM" -lt 4 ] && { echo "stride $S leaves h_max=$HM, skipping"; continue; }
  EV=$(scale "$S" $EVAL)
  # statistics and split are per-store: a strided store has different frames,
  # so it needs its own norm_stats. The SPLIT is unchanged (same trajectories).
  NS="artifacts/norm_stats_${DATASET}_stride${S}.json"
  echo ">>> stride $S : h_max=$HM  eval=[$EV]"

  # --force recomputes the STATISTICS only. Never --redraw-split here: the
  # trajectory split must be identical across strides, or the four curves are
  # measuring four different test sets and the comparison is meaningless.
  python scripts/audit_data.py --config "$CFG" \
    --set data.time_stride=$S --set data.norm_stats_path=$NS --force

  for M in $MODELS; do
    TAG="${DATASET}_str${S}_${M}_s${SEED}"
    torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
      --config "$CFG" --seed "$SEED" \
      --set meta.model_variant=$M \
      --set data.time_stride=$S --set data.norm_stats_path=$NS \
      --set data.h_max_train=$HM --set "data.eval_horizons=[$EV]" \
      --set experiment.save_dir=./checkpoints/${TAG} \
      --set experiment.exp_name=${TAG}

    python scripts/evaluate_horizon.py --config "$CFG" \
      --set meta.model_variant=$M \
      --set data.time_stride=$S --set data.norm_stats_path=$NS \
      --set "data.eval_horizons=[$EV]" \
      --checkpoint ./checkpoints/${TAG}/best_model.pth \
      --set experiment.exp_name=${TAG}
  done

  python scripts/evaluate_horizon.py --config "$CFG" \
    --model nearest_climatology \
    --set data.time_stride=$S --set data.norm_stats_path=$NS \
    --set "data.eval_horizons=[$EV]" \
    --set experiment.exp_name=${DATASET}_str${S}_nearest_climatology
done

echo
echo "Plot one figure per stride, then read the crossover from each:"
for S in $STRIDES; do
  echo "  python scripts/make_figures.py --results 'results/${DATASET}_str${S}_*/horizon_metrics.json' \\"
  echo "      --out results/figures_${DATASET}_stride${S}"
done
echo
echo "Then compare fig1_crossover.txt across strides, converting h to seconds"
echo "with dt_effective = dt_base * stride. Constant in seconds -> physics."
echo "Constant in frames -> model."
