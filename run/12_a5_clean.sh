#!/usr/bin/env bash
# A5 (training horizon ceiling) — CLEAN. The last blocking experiment.
#
# Guards against the exact mistake that spoiled run/11: it refuses to start if
# loss_channel_weights is set anywhere in the resolved config, because that flag
# was measured to cost ~20% across all channels and makes the runs
# incomparable with the 3-seed baseline.
set -euo pipefail

NPROC=${NPROC:-4}
SEED=${SEED:-0}
HMAXES=${HMAXES:-"16 32 64 128"}

python - <<'PY'
import sys, argparse
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from common import resolve
f = resolve(argparse.Namespace(config="configs/dt_fno.yaml", overrides=[], seed=0))
w = f.get("loss_channel_weights")
if w:
    print(f"\nREFUSING TO START: loss_channel_weights = {w}\n")
    print("Excluding Absolute_Pressure from the loss was measured to make every")
    print("channel worse (E without pressure 0.213 -> 0.256 at h=1; E_spec")
    print("3.09 -> 10.91), and it is what made run/11's results incomparable")
    print("with the 3-seed baseline.")
    print("\nRemove `loss_channel_weights` from configs/base.yaml and")
    print("configs/dt_fno.yaml, then re-run.\n")
    sys.exit(1)
print("config check: no loss_channel_weights, baseline-comparable")
PY

for HM in $HMAXES; do
  TAG=a5clean_hmax${HM}_s${SEED}
  torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
    --config configs/dt_fno.yaml --seed "$SEED" \
    --set data.h_max_train=$HM \
    --set experiment.save_dir=./checkpoints/${TAG} \
    --set experiment.tb_dir=./runs/${TAG} \
    --set experiment.exp_name=${TAG}

  # evaluated on the FULL grid whatever h_max_train was, so restricting
  # training shows its cost as well as its benefit
  python scripts/evaluate_horizon.py --config configs/dt_fno.yaml \
    --set data.h_max_train=$HM \
    --checkpoint ./checkpoints/${TAG}/best_model.pth \
    --set experiment.exp_name=${TAG}
done

python scripts/make_figures.py --results results/*/horizon_metrics.json \
  --include 'a5clean_*' 'dt_fno' 'ar_fno_r' nearest_climatology climatology \
  --out results/figures_a5clean

cat <<'MSG'

READING IT
  The target is explicit: DT below nearest_climatology (0.574 flat) at h >= 8.

  Cleared        -> DT beats AR and both climatology baselines on the chaotic
                    dataset. Re-run the winning h_max with 3 seeds, refresh the
                    headline figures, and the paper has an unqualified result.
  Not cleared    -> the deterministic route is characterised. Freeze the
                    numbers, write §5 of docs/RESULTS.md as the claim, and open
                    docs/proposal_flow_matching.md as the next project.
MSG
