#!/usr/bin/env bash
# Two model changes the measurements point at, plus a corrected AR baseline.
#
# This is the "fix the model" script, not another diagnostic. Every arm here
# targets something that was measured, not something that seemed plausible.
#
#   bash run/11_model_fixes.sh          # all three blocks
#   BLOCK=ar bash run/11_model_fixes.sh
#   BLOCK=a5 bash run/11_model_fixes.sh
#   BLOCK=a2 bash run/11_model_fixes.sh
set -euo pipefail

NPROC=${NPROC:-4}
SEED=${SEED:-0}
BLOCK=${BLOCK:-all}

hr () { printf '%.0s=' {1..74}; echo; }

# ---------------------------------------------------------------------------
# AR — re-train with corrected checkpoint selection
# ---------------------------------------------------------------------------
ar () {
  hr; echo "AR baseline, corrected model selection"; hr
  cat <<'MSG'
The trainer was selecting AR checkpoints on the PLAIN mean of E(h) over H_val.
For a model whose rollout diverges, that mean is the size of the divergence:
the sampling sweep logged "Best Eval = 740" and "= 1595". Checkpoint choice was
therefore driven by how badly the longest horizon blew up, not by forecast
quality -- which weakens exactly the baseline §23 insists must be strong.

Selection now uses the bounded score (each horizon capped at 1). Every AR
number in the paper has to come from a re-trained checkpoint; the old ones are
not a fair baseline and a reviewer would be right to say so.
MSG
  echo
  for S in 0 1 2; do
    TAG=ar_fno_r_fixsel_s${S}
    torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
      --config configs/ar_fno_r.yaml --seed "$S" \
      --set experiment.save_dir=./checkpoints/${TAG} \
      --set experiment.exp_name=${TAG} \
      --set experiment.tb_dir=./runs/${TAG}
    python scripts/evaluate_horizon.py --config configs/ar_fno_r.yaml \
      --checkpoint ./checkpoints/${TAG}/best_model.pth \
      --set experiment.exp_name=${TAG} \
      --set experiment.tb_dir=./runs/${TAG}
  done
}

# ---------------------------------------------------------------------------
# A5 — training horizon ceiling (§34)
# ---------------------------------------------------------------------------
a5 () {
  hr; echo "A5 — training horizon ceiling"; hr
  cat <<'MSG'
The tau-swap matrix showed the model's own long-tau answers are beaten by its
own short-tau answers: querying tau(16) instead of the correct tau(256) was 24%
BETTER at h = 256, and 10.7% better at h = 128. Capacity is being spent on
horizons past the predictability window, and spent badly.

Log-binned sampling makes this worse than it looks: the bin [65,128] gets 1/7 of
the draws spread over 64 distinct horizons, so h = 128 individually sees ~32x
fewer samples than h = 1 -- undertrained AND useless.

h_max_train in {16, 32, 64, 128}. If 32 matches or beats 128 on h <= 32, the
model should simply not be trained past the horizon it can serve, and every
headline number improves for free.
MSG
  echo
  for HM in 32 64 128; do
    TAG=dt_fno_hmax${HM}_s${SEED}
    torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
      --config configs/dt_fno.yaml --seed "$SEED" \
      --set data.h_max_train=$HM \
      --set experiment.save_dir=./checkpoints/${TAG} \
      --set experiment.exp_name=${TAG} \
      --set experiment.tb_dir=./runs/${TAG}
    # evaluated on the FULL grid regardless of h_max_train, so the cost of
    # restricting training shows up as well as the benefit
    python scripts/evaluate_horizon.py --config configs/dt_fno.yaml \
      --set data.h_max_train=$HM \
      --checkpoint ./checkpoints/${TAG}/best_model.pth \
      --set experiment.exp_name=${TAG} \
      --set experiment.tb_dir=./runs/${TAG}
  done
}

# ---------------------------------------------------------------------------
# A2 — history length (§34, §10)
# ---------------------------------------------------------------------------
a2 () {
  hr; echo "A2 — history length K"; hr
  cat <<'MSG'
The direct model loses to a retrieval climatology from h ~ 8 on RealPDEBench.
Climatology wins by identifying the operating point; the model has to infer it
from K frames. K = 4 was fixed by §10 to avoid confounding, never tested.

More history gives a better state and operating-point estimate, which is
precisely the deficit the climatology comparison exposed. This is the one
architectural knob the measurement points at directly.

K in {1, 2, 4, 8}. Note K changes the input width, so the parameter count moves
slightly -- the run prints it, and §3 wants that recorded.
MSG
  echo
  for K in 2 4 8; do
    TAG=dt_fno_K${K}_s${SEED}
    torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
      --config configs/dt_fno.yaml --seed "$SEED" \
      --set data.history_len=$K \
      --set experiment.save_dir=./checkpoints/${TAG} \
      --set experiment.exp_name=${TAG} \
      --set experiment.tb_dir=./runs/${TAG}
    python scripts/evaluate_horizon.py --config configs/dt_fno.yaml \
      --set data.history_len=$K \
      --checkpoint ./checkpoints/${TAG}/best_model.pth \
      --set experiment.exp_name=${TAG} \
      --set experiment.tb_dir=./runs/${TAG}
  done
}

case "$BLOCK" in
  ar) ar ;; a5) a5 ;; a2) a2 ;;
  all) ar; a5; a2 ;;
  *) echo "BLOCK must be ar | a5 | a2 | all" >&2; exit 1 ;;
esac

hr
cat <<'MSG'
Compare against the reference lines, main-run filter only:

  python scripts/make_figures.py --results results/*/horizon_metrics.json \
      --include 'dt_fno_hmax*' 'dt_fno_K*' 'ar_fno_r_fixsel*' \
                nearest_climatology climatology \
      --out results/figures_fixes

The target is explicit: DT below nearest_climatology (0.574 flat) at h >= 8.
That is the number the deterministic route has to clear.
MSG
