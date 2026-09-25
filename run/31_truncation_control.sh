#!/usr/bin/env bash
# Is the autoregressive instability caused by SPECTRAL TRUNCATION?
#
#   bash run/31_truncation_control.sh
#
# THE HYPOTHESIS, AND WHY IT IS WORTH ONE NIGHT
# ---------------------------------------------
# The matched-backbone runs left one thing unexplained. On Gray-Scott the U-Net
# rollout does not destabilise at all --- 0 diverged horizons, and
# E(h=128) = 0.4267 against the direct model's 0.4268, identical --- while the
# FNO rollout on the same data diverges at 0/5/6 horizons across three seeds.
# The fp32 control ruled out precision. So it is the architecture, and the most
# obvious difference is that the FNO keeps 16x16 of the 64x65 available Fourier
# modes and throws the rest away at every layer, while a U-Net truncates
# nothing.
#
# If that is the mechanism, each rollout step discards high-wavenumber content
# the next step needs and the deficit compounds. Gray-Scott, whose fields are
# the sharpest in the study, is where it should bite hardest --- and
# RealPDEBench, whose fields are smoother, shows a much smaller architecture
# shift (8.1 -> 15.3 against 9.1 -> 194.1).
#
# THE CONFOUND THIS SCRIPT EXISTS TO AVOID
# -----------------------------------------
# Raising the mode counts also raises the parameter count, because the spectral
# weights scale as width^2 * m1 * m2. Going 16x16 -> 32x32 at width 64 takes the
# backbone from 16.8 M to 67.1 M, and a model four times larger might be more
# stable for reasons that have nothing to do with truncation. Reducing the width
# at the same time restores the count exactly:
#
#   arm  width  modes    params    fraction of Nyquist kept
#   A     64    16x16    16.80 M    25% x 25%     <- the run already in the paper
#   B     64    32x32    67.14 M    50% x 49%     more modes AND 4x capacity
#   C     32    32x32    16.79 M    50% x 49%     more modes, capacity matched
#   D     16    64x65    17.04 M   100% x 100%    NO TRUNCATION, capacity matched
#
# Arm D really is lossless, which is worth stating because it is easy to assume
# otherwise. `SpectralConv2d.forward` clamps with
# `m1 = min(modes1, H//2)` and `m2 = min(modes2, W//2+1)`, so on a 128x128 grid
# D resolves to m1 = 64, m2 = 65, and the two writes `out_ft[..., :64, :65]` and
# `out_ft[..., -64:, :65]` tile all 128 rows exactly once. Nothing is dropped
# and nothing is written twice: arm D is an FNO whose spectral layer is a full,
# untruncated linear operator in Fourier space.
#
# D is therefore the decisive arm: a Fourier layer that discards nothing, at the
# same parameter count as the paper's run. If the rollout still diverges there, the
# truncation hypothesis is dead and the U-Net result needs a different
# explanation. C is the dose-response point in between. B is optional and only
# separates "more modes" from "more parameters" if C and D disagree.
#
# Default is C and D, which are the two that answer the question.
set -euo pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}

NPROC=${NPROC:-3}
SEEDS=${SEEDS:-"0"}
CFG=${CFG:-configs/gray_scott.yaml}
PRE=${PRE:-gs_}
# "C D" answers the mechanism question. Add B only if C and D disagree.
ARMS=${ARMS:-"C D"}
# AR alone tells you whether the rollout still diverges, which IS the mechanism
# question, and halves the cost. DT is needed only for the crossover number.
MODELS=${MODELS:-"ar_fno_r dt_fno"}

arm_cfg() {                       # -> "<width> <modes1> <modes2> <label>"
  case "$1" in
    B) echo "64 32 32 more-modes-4x-params" ;;
    C) echo "32 32 32 half-Nyquist-matched" ;;
    D) echo "16 64 65 no-truncation-matched" ;;
    *) echo "" ;;
  esac
}

for S in $SEEDS; do
  for A in $ARMS; do
    read -r W M1 M2 LABEL <<< "$(arm_cfg "$A")"
    [ -n "$W" ] || { echo "  unknown arm '$A' (expected B, C or D)"; continue; }
    for M in $MODELS; do
      TAG="${PRE}${M}_m${M1}x${M2}w${W}_s${S}"
      if [ -f "./checkpoints/${TAG}/best_model.pth" ]; then
        echo ">>> ${TAG}: exists, skipping training"
      else
        echo ">>> training ${TAG}   arm $A ($LABEL)"
        # ONLY width and the mode counts change. Not the seed, not the split,
        # not the horizon grid, not the padding (Gray-Scott is periodic, so it
        # stays 0). A second difference would make this as unreadable as the
        # U-Net comparison was before the precision control.
        torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
          --config "$CFG" --seed "$S" \
          --set meta.model_variant=$M \
          --set model.width=$W --set model.modes1=$M1 --set model.modes2=$M2 \
          --set experiment.save_dir=./checkpoints/${TAG} \
          --set experiment.tb_dir=./runs/${TAG} \
          --set experiment.exp_name=${TAG} \
        || { echo "  [!] ${TAG} failed -- see the traceback above."; continue; }
      fi
      python scripts/evaluate_horizon.py --config "$CFG" \
        --set meta.model_variant=$M \
        --set model.width=$W --set model.modes1=$M1 --set model.modes2=$M2 \
        --checkpoint ./checkpoints/${TAG}/best_model.pth \
        --set experiment.exp_name=${TAG}
    done

    if [ "$MODELS" = "ar_fno_r dt_fno" ]; then
      echo ">>> crossover for arm $A"
      python scripts/make_figures.py --results results/*/horizon_metrics.json \
        --include ${PRE}ar_fno_r_m${M1}x${M2}w${W} ${PRE}dt_fno_m${M1}x${M2}w${W} \
        --label "modes ${M1}x${M2}, width ${W}" \
        --out results/figures_truncation/${A}
      cat results/figures_truncation/${A}/fig1_crossover.txt 2>/dev/null || true
      cat results/figures_truncation/${A}/fig1_diverged.txt 2>/dev/null \
        || echo "    no off-scale points: the rollout did not diverge"
    fi
  done
done

cat <<'MSG'

================================================================
HOW TO READ IT

  The number that decides the hypothesis is the DIVERGED HORIZON COUNT of the
  autoregressive arm, not the accuracy. Baseline, Gray-Scott, modes 16x16:

      AR-FNO-R diverges at 0 / 5 / 6 horizons across three seeds
      AR-UNet  diverges at 0 horizons, and crossover moves 9.1 -> 194.1

  arm D (no truncation) still diverges
      -> truncation is not the mechanism. The U-Net's stability comes from
         something else, and the paper should say the architecture dependence
         is real and unexplained rather than attach a story to it.

  arm D stops diverging, arm C is intermediate
      -> dose-response in the retained bandwidth. That is a mechanism, it is
         measurable, and it predicts the RealPDEBench/Gray-Scott asymmetry from
         the fields' spectral content. Worth a paragraph and a figure.

  arm D stops diverging but arm C does not
      -> the effect is all-or-nothing rather than graded. Report it as such;
         do not interpolate between two points.

  ONE SEED IS NOT ENOUGH TO CONCLUDE. The baseline spans 0 to 6 diverged
  horizons across seeds, so a single arm-D run with 0 could be the same luck
  that gave seed 0 its clean baseline. Confirm any positive result with
  SEEDS="0 1 2" before it goes in the paper.

  Also note E_spec will change across arms by construction: a model that keeps
  more modes can represent more of the spectrum. Do not read that as evidence
  for the hypothesis; the rollout stability is the evidence.
================================================================
MSG
