#!/usr/bin/env bash
# Phase A0 — data preparation for the three Experiment A datasets.
#
# No GPU. Nothing here trains anything. Everything here is a decision that
# cannot be changed afterwards without invalidating every result that follows,
# which is why it is one script with a fixed order:
#
#   probe    ->  is the horizon grid honest, and is dt right?
#   screen   ->  which trajectories are quiescent and must be excluded?
#   split    ->  frozen, stratified, written to JSON
#   audit    ->  frozen normalisation statistics, fitted on TRAIN only
#
# Reversing any two of these silently produces a leak. The screen changes
# n_traj, so it must precede the split; the split determines which trajectories
# the statistics see, so it must precede the audit.
#
#   bash run/20_expA_prepare.sh                    # all three
#   DATASETS="cylinder" bash run/20_expA_prepare.sh
#   SKIP_SCREEN=1 bash run/20_expA_prepare.sh      # second pass, exclusions set
set -euo pipefail

DATASETS=${DATASETS:-"gray_scott cylinder rayleigh_benard"}
SKIP_PROBE=${SKIP_PROBE:-0}
SKIP_SCREEN=${SKIP_SCREEN:-0}

for DS in $DATASETS; do
  CFG="configs/${DS}.yaml"
  echo
  echo "================================================================"
  echo "  $DS   ($CFG)"
  echo "================================================================"

  if [ "$SKIP_PROBE" != "1" ]; then
    echo ">>> [1/4] timescale probe"
    python scripts/probe_timescales.py --config "$CFG" \
      --subset all --n_traj 8 --n_anchors 24
  fi

  # The screen is only meaningful where quiescence is a documented risk.
  # Gray-Scott: ~12% of two parameter sets reach a fixed point at ~12% of the
  # record. Rayleigh-Benard: run it to check the spin-up head, not the tail.
  # Cylinder: a shedding wake does not go quiescent; skip unless curious.
  case "$DS" in
    gray_scott|rayleigh_benard)
      if [ "$SKIP_SCREEN" != "1" ]; then
        echo ">>> [2/4] stationarity screen"
        python scripts/screen_stationary.py --config "$CFG" \
          --out "artifacts/stationary_${DS}.json"
        echo
        echo "    If it found stationary trajectories, set"
        echo "      data.exclude_trajectories: artifacts/stationary_${DS}.json"
        echo "    in ${CFG} and re-run this script with SKIP_SCREEN=1."
        echo "    Excluding trajectories changes n_traj, so the split below"
        echo "    would otherwise be built against the wrong index set."
      fi ;;
    *) echo ">>> [2/4] stationarity screen skipped for $DS" ;;
  esac

  echo ">>> [3/4] freeze the split"
  python scripts/prepare_split.py --config "$CFG"

  # --strata is the labels prepare_split just wrote. Without it the audit
  # prints "[!] Unstratified", which is a false alarm here: the split on disk
  # IS stratified, the audit simply has no way to know. Passing them also gets
  # the real per-subset breakdown printed, which is what you actually want to
  # read before training.
  echo ">>> [4/4] audit"
  python scripts/audit_data.py --config "$CFG" \
    --strata "artifacts/split_${DS}_strata.json"
done

cat <<'MSG'

================================================================
READ BEFORE TRAINING

  1. The CHANNEL GROUPING the audit printed, for every dataset. §9 of the
     handover: "Regex channel matching breaks silently across datasets."
     A channel in the wrong group does not fail; it disables the §26
     per-variable reporting without saying so.

  2. The TRANSFORM TABLE. A channel left on the wrong transform will
     dominate the loss.

  3. The PROBE's section [6]. If it says a dataset's E_pers(h) is already
     near 1 by h = 8, the long-horizon columns of eval_horizons are
     climatology, not physics -- keep them, they are honest, but do not
     expect them to discriminate, and make sure BOTH climatology baselines
     are on every figure for that dataset.

  4. The probe's dominant period against dt. A period that is physically
     absurd means data.dt is wrong, and every reported time is wrong with it.
================================================================
MSG
