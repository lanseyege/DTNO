#!/usr/bin/env bash
# The two seeds that decide whether the U-Net's freedom from divergence is
# architectural or a capacity effect.
#
#   NPROC=3 bash run/36_capacity_seeds.sh          # seeds 1 and 2, ~9 h total
#
# WHERE THIS STANDS
# -----------------
# The width-88 arm (15,904,010 parameters, against the AR-UNet's 16,008,626)
# already settled the ACCURACY question: it reaches Eval* 0.617, worse than the
# 8.4M baseline's 0.593, while the U-Net at the same size reaches 0.295. So the
# U-Net's accuracy advantage is not bought with parameters.
#
# It did not settle DIVERGENCE. That single seed diverged at zero horizons, but
# the 8.4M baseline diverges at 0, 5 and 6 horizons across its three seeds, so a
# single zero is an unremarkable draw from that distribution. The R=16 arms made
# the same point the hard way: seed 0 diverged at no horizon and seed 1 at six.
#
# Two more seeds at width 88 give three, matching the baseline and the U-Net:
#
#   all three clean        -> the FNO stops diverging when widened, so the U-Net
#                             result is a capacity effect and Appendix
#                             "Architecture and bandwidth controls" must say so
#   one or more diverges   -> widening does not buy stability, the U-Net's 6/6
#                             clean runs stand as architectural, and Section 5.1
#                             can drop its hedge
#
# Either answer is worth having. The hedge currently in the appendix ("two
# further seeds would settle it") is the sentence this removes.
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}

NPROC=${NPROC:-3}          # must match seed 0: world size sets steps/epoch and
                           # therefore the LR schedule
SEEDS=${SEEDS:-"1 2"}
WIDTH=${WIDTH:-88}
CFG=${CFG:-configs/gray_scott.yaml}
M=${M:-ar_fno_r}           # the AR arm alone answers the question

for S in $SEEDS; do
  TAG="gs_${M}_w${WIDTH}_s${S}"
  if [ -f "./checkpoints/${TAG}/best_model.pth" ]; then
    echo ">>> ${TAG}: exists, skipping training"
  else
    echo ">>> training ${TAG}"
    torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
      --config "$CFG" --seed "$S" \
      --set meta.model_variant=$M \
      --set model.width=$WIDTH \
      --set experiment.save_dir=./checkpoints/${TAG} \
      --set experiment.tb_dir=./runs/${TAG} \
      --set experiment.exp_name=${TAG}
  fi
  python scripts/evaluate_horizon.py --config "$CFG" \
    --set meta.model_variant=$M --set model.width=$WIDTH \
    --checkpoint ./checkpoints/${TAG}/best_model.pth \
    --set experiment.exp_name=${TAG}
done

echo
echo ">>> divergence across the three width-88 seeds"
for S in 0 $SEEDS; do
  f="results/gs_${M}_w${WIDTH}_s${S}/horizon_metrics.json"
  [ -f "$f" ] && python3 -c "
import json,sys,math
r=json.load(open('$f')); E=[float(v) for v in r['E_field']]
d=[int(h) for h,v in zip(r['horizons'],E) if (not math.isfinite(v)) or v>3]
b=sum(1.0 if not math.isfinite(v) else min(v,1.0) for v in E)/len(E)
print(f'    seed $S: Eval* {b:.4f}   diverged at {d or \"no horizon\"}')"
done
cat <<'MSG'

  Compare against, on the same dataset:
      AR-FNO   8.4M    0 / 5 / 6 diverged horizons across three seeds
      AR-UNet 16.0M    0 / 0 / 0

  Report the three counts individually. A mean over seeds that straddle
  divergence is not a centre -- the paper says so about AR-FNO-R and the same
  applies here.
MSG
