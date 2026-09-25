#!/usr/bin/env bash
# P0 — architecture robustness. The single largest gap in the paper.
#
#   bash run/28_matched_backbone.sh
#
# WHY A SINGLE NON-FNO DIRECT MODEL WOULD NOT ANSWER THE QUESTION
# ---------------------------------------------------------------
# The obvious reading of "add a non-FNO baseline" is: train one TC-UNet or a
# fine-tuned Poseidon as a direct-time predictor and report its error. That
# tests whether *some other architecture* can do direct-time prediction, which
# is not in doubt --- Poseidon and TC-UNet already exist.
#
# What the paper claims is about the CROSSOVER: that the horizon at which one
# conditioned pass overtakes an h-step rollout sits at h = 8-20 and follows from
# composing a learned map with itself rather than from the operator's
# parameterisation. Testing that needs a MATCHED PAIR in the second
# architecture --- AR-UNet against DT-UNet, trained under the identical
# protocol --- because a crossover is a property of two curves, not one. A lone
# direct model produces no crossover to compare.
#
# So this is four runs per dataset, not two, and two datasets are enough:
# Gray-Scott (long predictable window, crossover 9.1) and RealPDEBench (short
# window, crossover 8.1). If the crossover lands near 8-20 on both under a
# different backbone, the claim survives; if it moves by a factor of several,
# the paper's central empirical law is an FNO property and must be restated as
# one.
#
# WHAT YOU MUST ADD FIRST
# -----------------------
# `models/MODEL_REGISTRY` currently holds only {ar_fno, dt_fno, sg_dt_fno}, all
# of which wrap `FNOBackbone`. A second backbone means one new module exporting
# the same interface the FNO one does:
#
#   forward(z, e_t) : (B, width, H, W) x (B, cond_dim) -> (B, width, H, W)
#
# with `cond_dim=0` for the AR arm and FiLM conditioning for the direct arm, so
# that `models/ar_fno.py` and `models/direct_fno.py` can instantiate it by
# swapping one class. A U-Net with FiLM at each resolution is the least
# work and the most standard comparison. Register it as `ar_unet` / `dt_unet`
# and add the matching MODEL_VARIANTS entries (ar_unet_r needs
# `ar_rollout: 4, ar_random_rollout: true`, mirroring ar_fno_r exactly ---
# if the AR arm is not rollout-trained the comparison is against a straw man).
#
# Everything below assumes those names exist. It checks first and explains
# rather than failing inside torch.
set -euo pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}

NPROC=${NPROC:-4}
SEED=${SEED:-0}
DATASETS=${DATASETS:-"gray_scott realpde_combustion"}
MODELS=${MODELS:-"ar_unet_r dt_unet"}

python - <<'PY' || exit 1
import sys
try:
    from models import MODEL_REGISTRY, MODEL_VARIANTS
except Exception as e:
    print(f"  cannot import the model registry: {e}"); sys.exit(1)
need_m = [m for m in ("ar_unet", "dt_unet") if m not in MODEL_REGISTRY]
need_v = [v for v in ("ar_unet_r", "dt_unet") if v not in MODEL_VARIANTS]
if need_m or need_v:
    print("  A second backbone is not registered yet.")
    if need_m: print(f"    missing from MODEL_REGISTRY: {need_m}")
    if need_v: print(f"    missing from MODEL_VARIANTS: {need_v}")
    print("""
  Add a U-Net backbone exposing the same interface as FNOBackbone
  (forward(z, e_t), cond_dim=0 for AR and FiLM for direct), then register:

      MODEL_REGISTRY["ar_unet"] = "ar"
      MODEL_REGISTRY["dt_unet"] = "direct"
      MODEL_VARIANTS["ar_unet_r"] = {"model_name": "ar_unet", "ar_rollout": 4,
                                     "ar_random_rollout": True,
                                     "horizon_sampling": "fixed"}
      MODEL_VARIANTS["dt_unet"]   = {"model_name": "dt_unet"}

  The AR arm MUST be rollout-trained, exactly as ar_fno_r is. An AR baseline
  trained on one-step supervision would diverge earlier for a reason that has
  nothing to do with the architecture, and the crossover it produced would not
  be comparable with the one in the paper.""")
    sys.exit(1)
print("  both backbones registered")
PY

# Trailing underscore included. Without it the tag came out `gsar_unet_r_s0`
# instead of `gs_ar_unet_r_s0`, which is not just cosmetic: make_figures strips
# `_s<N>` and then a dataset prefix, and `gsar_unet_r` matches neither, so the
# run would have been plotted as its own ungrouped series. It also made the
# model name unreadable, which is how a hand-patched dispatch ended up keyed on
# `gsar_unet_r_s0`.
declare -A PREFIX=( [gray_scott]=gs_ [realpde_combustion]="" [cylinder]=cyl_ )
declare -A CONFIG=( [gray_scott]=configs/gray_scott.yaml
                    [realpde_combustion]=configs/base.yaml
                    [cylinder]=configs/cylinder.yaml )

for DS in $DATASETS; do
  CFG=${CONFIG[$DS]}; P=${PREFIX[$DS]-}
  for M in $MODELS; do
    TAG="${P}${M}_s${SEED}"
    if [ -f "./checkpoints/${TAG}/best_model.pth" ]; then
      echo ">>> ${TAG}: exists, skipping training"
    else
      echo ">>> training ${TAG}"
      torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
        --config "$CFG" --seed "$SEED" \
        --set meta.model_variant=$M \
        --set experiment.save_dir=./checkpoints/${TAG} \
        --set experiment.tb_dir=./runs/${TAG} \
        --set experiment.exp_name=${TAG}
    fi
    python scripts/evaluate_horizon.py --config "$CFG" \
      --set meta.model_variant=$M \
      --checkpoint ./checkpoints/${TAG}/best_model.pth \
      --set experiment.exp_name=${TAG}
  done
done

cat <<'MSG'

================================================================
READING IT

  The number to extract is the CROSSOVER, from fig1_crossover.txt, not the
  absolute error. A U-Net that is uniformly worse than the FNO is fine and
  expected; the question is only where its two curves cross.

    FNO crossovers, for comparison:  Gray-Scott 9.1   RealPDEBench 8.1

  If the U-Net crossovers land in 8-20, the paper's claim holds across
  architectures and Section 5.4 can say so. If they land at, say, 3 or 60, the
  claim is an FNO property and the sentence "we read this as a property of the
  comparison rather than of any flow" must become "...of this architecture
  class", with the measured numbers given. Either outcome is publishable; only
  the untested version is not.

  Regenerate the figures with the new tags included, and add a row to the
  crossover table rather than a new table.
================================================================
MSG
