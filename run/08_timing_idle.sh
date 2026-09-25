#!/usr/bin/env bash
# §32 — inference cost, measured properly.
#
#   GPU=3 bash run/08_timing_idle.sh
#   GPU=3 SEED=0 SUFFIX=_fourier_log bash run/08_timing_idle.sh
#   GPU=3 FORCE=1 bash run/08_timing_idle.sh    # measure anyway (not for the paper)
#
# The previous run was taken on a card that was already 100% busy with another
# job, and it showed: the AR point at h = 64 came out at 148 ms where linear
# scaling predicts ~200 ms, with an IQR 25x larger than every other point.
# Those numbers cannot go in a figure. This script refuses to start on a busy
# GPU rather than producing a plausible-looking contaminated result.
#
# N_model_evals is exact and unaffected by contention; it is the column that
# survives any argument about methodology. The wall-clock half is the one that
# needs an idle card.
set -euo pipefail

GPU=${GPU:-0}
SEED=${SEED:-0}
SUFFIX=${SUFFIX:-""}               # e.g. _fourier_log
FORCE=${FORCE:-0}
UTIL_LIMIT=${UTIL_LIMIT:-5}        # percent
MEM_LIMIT=${MEM_LIMIT:-2000}       # MiB

read -r UTIL MEM < <(nvidia-smi --query-gpu=utilization.gpu,memory.used \
  --format=csv,noheader,nounits -i "$GPU" | tr -d ',')
echo "GPU $GPU: utilisation ${UTIL}%, memory ${MEM} MiB in use"

if [ "$FORCE" != "1" ] && { [ "$UTIL" -gt "$UTIL_LIMIT" ] || [ "$MEM" -gt "$MEM_LIMIT" ]; }; then
  cat >&2 <<MSG

REFUSING TO MEASURE. GPU $GPU is busy (${UTIL}% util, ${MEM} MiB).

Wall-clock timing on a shared card is not a measurement of the model, it is a
measurement of the queue. Pick an idle GPU:

    nvidia-smi --query-gpu=index,utilization.gpu,memory.used \\
        --format=csv,noheader
    GPU=<idle index> bash run/08_timing_idle.sh

Set FORCE=1 to measure anyway -- useful while iterating, never for a figure.
MSG
  exit 1
fi

AR=./checkpoints/ar_fno_r_s${SEED}/best_model.pth
DT=./checkpoints/dt_fno${SUFFIX}_s${SEED}/best_model.pth
SG=./checkpoints/sg_dt_fno${SUFFIX}_s${SEED}/best_model.pth
for f in "$AR" "$DT" "$SG"; do
  [ -f "$f" ] || { echo "missing checkpoint: $f" >&2; exit 1; }
done

OUT=./results/timing/inference_cost${SUFFIX}_s${SEED}.json
mkdir -p ./results/timing

# max_timed_horizon 512 = every point genuinely measured, nothing extrapolated.
# Costs a few minutes because the h=512 AR rollout is 512 sequential forwards
# repeated 10 times; that is the price of a figure with no asterisks in it.
CUDA_VISIBLE_DEVICES=$GPU python scripts/benchmark_timing.py \
  --config configs/sg_dt_fno.yaml \
  ${SUFFIX:+--set model.time_embed_mode=${SUFFIX#_}} \
  --ar "$AR" --dt "$DT" --sg "$SG" \
  --horizons 1 2 4 8 16 32 64 128 256 512 \
  --max_timed_horizon 512 \
  --n_warmup 10 --n_repeat 30 \
  --out "$OUT"

echo
echo "-> $OUT"
python - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
g = d["gpu"]
print(f"\nrecorded GPU state: {g.get('gpu_util_percent_at_start','?')}% util, "
      f"{g.get('gpu_mem_used_MB_at_start','?')} MiB at start")
if "warning" in g:
    print(f"[!] {g['warning']}")
else:
    print("card was idle: wall-clock numbers are usable in the figure")
rows = {m: {r["h"]: r for r in b["rows"]} for m, b in d["models"].items()}
if "ar_fno" in rows:
    ar = rows["ar_fno"]
    base = ar[1]["median_ms"]
    print(f"\nAR linearity check (median_ms / h, should be ~constant = "
          f"{base:.2f} ms):")
    for h in sorted(ar):
        per = ar[h]["median_ms"] / h
        flag = "  <-- off-linear, suspect contention" if abs(per/base - 1) > 0.25 else ""
        print(f"  h={h:>4}  {ar[h]['median_ms']:>9.2f} ms  "
              f"{per:>7.3f} ms/step{flag}")
PY
