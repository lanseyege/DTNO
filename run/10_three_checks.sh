#!/usr/bin/env bash
# The three cheap checks that decide whether the deterministic route is
# actually exhausted, before committing to a generative one.
#
#   STEP=1 bash run/10_three_checks.sh   # pressure out of the loss (RealPDEBench)
#   STEP=2 bash run/10_three_checks.sh   # sampling-rate sweep, split preserved
#   STEP=3 bash run/10_three_checks.sh   # Lifted H2 physics metrics
#   bash run/10_three_checks.sh          # all three, in order
#
# Each step ends by printing what to look at and what each outcome means. None
# of them takes more than a few hours; together they decide a 1-2 year fork.
set -euo pipefail

NPROC=${NPROC:-4}
SEED=${SEED:-0}
STEP=${STEP:-all}

hr () { printf '%.0s=' {1..74}; echo; }

# ---------------------------------------------------------------------------
# STEP 1 — is the deterministic model handicapped by an unlearnable channel?
# ---------------------------------------------------------------------------
step1 () {
  hr; echo "STEP 1 — Absolute_Pressure out of the loss (RealPDEBench only)"; hr
  cat <<'MSG'
Three independent measurements say this channel is dead weight:
  * unpredictable      relative L2 >= 0.94 for EVERY model at EVERY horizon
                       (persistence 1.93, POD-DMD 1.91, DT-FNO 1.18)
  * spectrally toxic   DT-FNO per-channel E_spec = 39 / 52 / 63 / 87 at
                       h = 1 / 8 / 32 / 128, vs 0.02-0.55 for every other
                       channel. Averaged over 13 channels it produces the whole
                       reported E_spec of 3.1-6.9 by itself.
  * still costly       normalised MSE gives it 1/13 of the gradient.

It stays in the INPUT (pressure gradients drive the flow) and stays in the
per-channel report. Only the loss changes.

If the other channels improve, "deterministic MSE is exhausted" is NOT yet
established and the paradigm question reopens.
MSG
  echo

  TAG=dt_fno_nopress_s${SEED}
  torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
    --config configs/dt_fno.yaml --seed "$SEED" \
    --set "training.loss_channel_weights={Absolute_Pressure: 0.0}" \
    --set experiment.save_dir=./checkpoints/${TAG} \
    --set experiment.exp_name=${TAG}

  python scripts/evaluate_horizon.py --config configs/dt_fno.yaml \
    --set "training.loss_channel_weights={Absolute_Pressure: 0.0}" \
    --checkpoint ./checkpoints/${TAG}/best_model.pth \
    --set experiment.exp_name=${TAG}

  echo; hr
  echo "Compare against the unweighted baseline, per channel:"
  echo
  python scripts/per_channel_report.py \
    results/dt_fno_s${SEED}/horizon_metrics.json \
    results/${TAG}/horizon_metrics.json
  python scripts/spec_channel_report.py \
    results/dt_fno_s${SEED}/horizon_metrics.json \
    results/${TAG}/horizon_metrics.json
  cat <<'MSG'

READING IT
  Temperature / OH / HRR / velocity improve at h = 8-32
      -> the model was spending capacity on acoustic noise. Re-run the 3-seed
         headline with the weight in place and treat every earlier number as
         superseded. The deterministic route is not exhausted.
  No change outside pressure
      -> the channel was inert, and the h >= 8 ceiling is real. Step 2 then
         decides whether that ceiling is physics or sampling.
MSG
}

# ---------------------------------------------------------------------------
# STEP 2 — is the horizon ceiling physics or sampling rate?
# ---------------------------------------------------------------------------
step2 () {
  hr; echo "STEP 2 — sampling-rate sweep (split preserved)"; hr
  cat <<'MSG'
The first attempt at this was invalid: audit_data.py --force used to redraw the
frozen split, so four strides trained on an unstratified split that differed
from every other experiment. --force now recomputes statistics only.

Verify the split is the stratified one before starting.
MSG
  echo
  python - <<'PY'
import json, sys
sp = json.load(open("artifacts/split_realpde.json"))
print(f"  split: test={sp['test']} val={sp['val']}")
print(f"  note : {sp['note']}")
if sp["test"] != [8, 12, 14, 21, 26]:
    print("\n  [!] This is NOT the stratified split (expected test=[8,12,14,21,26]).")
    print("      Restore it before running:")
    print("        rm artifacts/split_realpde.json")
    print("        python scripts/derive_strata.py --config configs/dt_fno.yaml --levels 5")
    print("        python scripts/audit_data.py --config configs/dt_fno.yaml \\")
    print("            --redraw-split --strata artifacts/strata.json --force")
    sys.exit(1)
PY
  echo
  NPROC=$NPROC SEED=$SEED DATASET=realpde STRIDES="1 2 4 8" \
    bash run/09_sampling_rate.sh

  cat <<'MSG'

READING IT
  Convert each stride's crossover from frames to seconds: dt_eff = 250us * stride.
    constant in SECONDS  -> the ceiling is set by the flow. Nothing in the loss
                            or the architecture will move it, and a generative
                            model buys statistically plausible samples, not
                            synchronised ones.
    constant in FRAMES   -> the ceiling is set by the model and the K-frame
                            history. That is a much better problem to have, and
                            it is fixable without changing paradigm.
MSG
}

# ---------------------------------------------------------------------------
# STEP 3 — the dataset where DT already wins on L2: does it win on physics?
# ---------------------------------------------------------------------------
step3 () {
  hr; echo "STEP 3 — Lifted H2 physics metrics"; hr
  cat <<'MSG'
Lifted H2 is the only dataset where DT-FNO beats the retrieval climatology on
L2 (0.165 vs 0.382 at h=1, still ahead at h=16). Its spectra and flame metrics
have never been computed. This is where a positive physics result is most
likely, and it is pure evaluation -- no training.

NOTE: do NOT put loss_channel_weights: {Absolute_Pressure: 0.0} in
configs/lifted_h2.yaml. This dataset has channels [UX, UY, T, YH2O, YOH] and no
pressure at all; the weight lookup validates names and will abort. The pressure
finding is RealPDEBench-specific and belongs in configs/base.yaml or
configs/dt_fno.yaml.
MSG
  echo

  for B in climatology nearest_climatology persistence; do
    python scripts/evaluate_horizon.py --config configs/lifted_h2.yaml \
      --model "$B" --set experiment.exp_name=lh2_${B}_full
  done

  # re-score the trained models with the full suite (§27-30), same anchors
  for M in ar_fno_r dt_fno sg_dt_fno; do
    CK=./checkpoints/lh2_${M}_s${SEED}/best_model.pth
    [ -f "$CK" ] || { echo "  missing $CK, skipping"; continue; }
    python scripts/evaluate_horizon.py --config configs/lifted_h2.yaml \
      --set meta.model_variant=$M --checkpoint "$CK" \
      --set experiment.exp_name=lh2_${M}_s${SEED}
  done

  python scripts/make_figures.py \
    --results results/lh2_*/horizon_metrics.json \
    --out results/figures_lh2_physics

  echo
  python scripts/spec_channel_report.py \
    results/lh2_dt_fno_s${SEED}/horizon_metrics.json \
    results/lh2_climatology_full/horizon_metrics.json
  cat <<'MSG'

READING IT
  DT spectra clearly better than climatology through h ~ 16-24
      -> the operator does real dynamics inside the predictability window on a
         chaotic-free flow. Paper claim: "physically faithful fields at constant
         inference cost, within the predictability horizon."
  DT spectra no better than climatology
      -> the L2 win was a mean-field win. Then the deterministic route really
         is capped and the generative proposal is the next project.

  The flame mask here falls back to OH (this dataset has no heat-release
  channel). Say so in the caption -- an OH iso-contour and a heat-release
  iso-contour are not the same object.
MSG
}

case "$STEP" in
  1) step1 ;;
  2) step2 ;;
  3) step3 ;;
  all) step1; step2; step3 ;;
  *) echo "STEP must be 1 | 2 | 3 | all" >&2; exit 1 ;;
esac
