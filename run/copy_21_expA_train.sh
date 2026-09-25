#!/usr/bin/env bash
# Phase A1 — Experiment A core: the three models plus every reference line,
# on each of the three general PDE/CFD datasets.
#
# This is a thin wrapper around run/07_seeds.sh, which already does exactly the
# right thing when CONFIG pins one config and the model is selected through
# meta.model_variant. Do not fork it; the naming convention it uses
# (checkpoints/<tag>_s<seed>, results/<tag>_s<seed>) is what lets
# make_figures.py collapse seeds into one group automatically.
#
#   bash run/21_expA_train.sh                          # 1 seed, 3 models, 3 datasets
#   SEEDS="0 1 2" bash run/21_expA_train.sh            # add error bars
#   DATASETS="cylinder" MODELS="dt_fno" bash run/21_expA_train.sh
#   MODELS="ar_fno ar_fno_r dt_fno sg_dt_fno" bash run/21_expA_train.sh
#
# ONE THING WORTH KNOWING ABOUT THE AR BASELINE
# ---------------------------------------------
# §6E of the handover records an open gap: there was no clean "AR with
# corrected model selection" run, because the trainer used to select AR
# checkpoints on the plain mean of E(h), which is dominated by whichever
# horizon diverged. That is fixed (bounded score), but the only runs using the
# fix were contaminated by loss_channel_weights.
#
# Every run started here uses the fixed trainer and carries no channel
# weighting, so on these three datasets the gap simply does not exist. That is
# worth stating in the paper: Experiment A's AR baseline is selected correctly
# by construction, and the RealPDEBench AR-fixsel run remains the only place
# the question is open.
set -euo pipefail

NPROC=${NPROC:-4}
SEEDS=${SEEDS:-"0"}
MODELS=${MODELS:-"ar_fno_r dt_fno sg_dt_fno"}
DATASETS=${DATASETS:-"gray_scott cylinder rayleigh_benard"}

declare -A PREFIX=( [gray_scott]=gs [cylinder]=cyl [rayleigh_benard]=rb )

for DS in $DATASETS; do
  CFG="configs/${DS}.yaml"
  TAG="${PREFIX[$DS]:-$DS}"
  [ -f "$CFG" ] || { echo "missing $CFG" >&2; exit 1; }

  for F in artifacts/split_${DS}.json artifacts/norm_stats_${DS}.json; do
    [ -f "$F" ] || {
      echo "missing $F -- run run/20_expA_prepare.sh first" >&2; exit 1; }
  done

  echo
  echo "================================================================"
  echo "  Experiment A: $DS   models='$MODELS'  seeds='$SEEDS'"
  echo "================================================================"
  CONFIG="$CFG" TAG_PREFIX="$TAG" MODELS="$MODELS" SEEDS="$SEEDS" \
    NPROC="$NPROC" bash run/07_seeds.sh

  # POD-DMD floor. Cheap, no training, and Figure 1 has a place for it.
  # Rank 128 captured 96.1% of POD energy on RealPDEBench; §8 of the handover
  # asks for a higher rank so the baseline sits at its ceiling, and these
  # datasets have far fewer channels, so 256 is affordable here.
  python scripts/fit_dmd.py --config "$CFG" --rank ${DMD_RANK:-256}
  python scripts/evaluate_horizon.py --config "$CFG" --model pod_dmd \
    --set experiment.exp_name=${TAG}_pod_dmd

  # Wall-clock cost. The 386.8x figure is architecture-level, but the
  # BREAK-EVEN horizon is not: it is set by per-step AR error, which is a
  # property of the flow. Each dataset needs its own timing run, on an idle
  # GPU with a sustained warm-up (§9: an idle A800 sits at low clocks and short
  # measurements land 32% slow).
  # Timing does not depend on weights, so --random_weights lets this run
  # before or after training; what it DOES depend on is an idle card and a
  # sustained warm-up, which is why it is skippable here and worth re-running
  # alone when the machine is quiet.
  if [ "${SKIP_TIMING:-0}" != "1" ]; then
    python scripts/benchmark_timing.py --config "$CFG" --random_weights \
      --out results/timing/inference_cost_${TAG}.json || \
      echo "  [warn] timing failed for $DS; run it later on an idle GPU"
  fi
done

cat <<'MSG'

================================================================
NEXT
  bash run/24_expA_figures.sh

WHAT TO LOOK AT FIRST, IN THIS ORDER
  1. E_DT(h) against BOTH climatology baselines, per dataset. The deployable
     one (nearest_climatology, training data only) is the honest bar; the
     oracle one (climatology, uses the test trajectory's own mean) is the
     ceiling. A curve that beats neither is a curve that carries no
     information at that horizon, however smooth it looks.
  2. The crossover h where DT overtakes AR-FNO-R, per dataset, and per-step
     AR error next to it. On the two combustion datasets the crossover
     tracked per-frame flow change (0.231 vs 0.111 -> h ~ 8 vs h ~ 19), not
     the architecture. Three more points either confirm that or break it, and
     either outcome is a paper-grade claim.
  3. C_SG on Gray-Scott specifically. It is the only autonomous,
     fully-observed, closed-state system in the study. If the semigroup
     constraint ever helps, it helps there. Two nulls and a positive would be
     the most interesting single number this extension can produce; three
     nulls settles the question.
================================================================
MSG
