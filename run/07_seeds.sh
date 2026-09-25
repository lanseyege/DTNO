#!/usr/bin/env bash
# §43 — three seeds for every headline model, so Figure 1 gets mean +- std
# bands instead of single lines that only look like they have error bars.
#
#   bash run/07_seeds.sh                    # AR-FNO-R only (fills the gap)
#   MODELS="ar_fno_r dt_fno sg_dt_fno" bash run/07_seeds.sh
#   SEEDS="1 2" bash run/07_seeds.sh        # seed 0 already done
#   EMBED=fourier_log bash run/07_seeds.sh  # the A3 variant
#
#   # another dataset: CONFIG pins one config for every run, and the training
#   # variant is selected via meta.model_variant (models.MODEL_VARIANTS).
#   # NOTE ar_fno_r is a VARIANT, not a model: AR-FNO-1 and AR-FNO-R are the
#   # same network (§23), differing only in ar_rollout.
#   CONFIG=configs/lifted_h2.yaml TAG_PREFIX=lh2 \
#     MODELS="ar_fno_r dt_fno sg_dt_fno" bash run/07_seeds.sh
#
# Naming: checkpoints/<model><suffix>_s<seed>, results/<model><suffix>_s<seed>.
# make_figures.py collapses the _s<seed> suffix, so all seeds of one model land
# in one group automatically.
set -euo pipefail

NPROC=${NPROC:-4}
MODELS=${MODELS:-"ar_fno_r"}
SEEDS=${SEEDS:-"0 1 2"}
EMBED=${EMBED:-""}                 # "" = whatever the config says
CONFIG=${CONFIG:-""}               # "" = per-model config from config_for()
TAG_PREFIX=${TAG_PREFIX:-""}       # prepended to run names, e.g. lh2

config_for () {                    # model tag -> config file
  # CONFIG pins one config for every model; the model itself is then selected
  # with --set meta.model_name. That is how a second dataset reuses this script
  # without a parallel set of per-model config files.
  if [ -n "$CONFIG" ]; then echo "$CONFIG"; return; fi
  case "$1" in
    ar_fno_r) echo configs/ar_fno_r.yaml ;;
    ar_fno)   echo configs/ar_fno.yaml ;;
    dt_fno)   echo configs/dt_fno.yaml ;;
    sg_dt_fno) echo configs/sg_dt_fno.yaml ;;
    *) echo "unknown model '$1'" >&2; exit 1 ;;
  esac
}

EXTRA=()
SUFFIX=""
if [ -n "$EMBED" ]; then
  EXTRA=(--set "model.time_embed_mode=$EMBED")
  SUFFIX="_${EMBED}"
fi

for M in $MODELS; do
  CFG=$(config_for "$M")
  for S in $SEEDS; do
    TAG="${TAG_PREFIX:+${TAG_PREFIX}_}${M}${SUFFIX}_s${S}"
    if [ -f "./checkpoints/${TAG}/best_model.pth" ]; then
      echo ">>> ${TAG}: checkpoint exists, skipping training"
    else
      echo ">>> training ${TAG}"
      torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
        --config "$CFG" --seed "$S" "${EXTRA[@]+"${EXTRA[@]}"}" \
        ${CONFIG:+--set meta.model_variant=$M} \
        --set experiment.save_dir=./checkpoints/${TAG} \
        --set experiment.exp_name=${TAG}
    fi

    echo ">>> evaluating ${TAG}"
    python scripts/evaluate_horizon.py --config "$CFG" \
      "${EXTRA[@]+"${EXTRA[@]}"}" ${CONFIG:+--set meta.model_variant=$M} \
      --checkpoint ./checkpoints/${TAG}/best_model.pth \
      --set experiment.exp_name=${TAG}

    # the query-time diagnostic is cheap and only meaningful for direct models
    case "$M" in
      dt_fno|sg_dt_fno)
        python scripts/check_time_sensitivity.py --config "$CFG" \
          "${EXTRA[@]+"${EXTRA[@]}"}" ${CONFIG:+--set meta.model_variant=$M} \
          --checkpoint ./checkpoints/${TAG}/best_model.pth \
          --set experiment.exp_name=${TAG} ;;
    esac
  done
done

# Reference lines. Cheap, no training, and every figure needs them:
#   climatology          ORACLE  — uses the test trajectory's own mean
#   nearest_climatology  DEPLOYABLE — training data only, no dynamics
REF_CFG=${CONFIG:-configs/dt_fno.yaml}
for B in persistence climatology nearest_climatology; do
  python scripts/evaluate_horizon.py --config "$REF_CFG" \
    --model "$B" --set experiment.exp_name="${TAG_PREFIX:+${TAG_PREFIX}_}$B"
done

echo
echo "Done. Rebuild the figures with:"
echo "  python scripts/make_figures.py --results results/*/horizon_metrics.json \\"
echo "      --timing results/timing/inference_cost.json --out results/figures"
