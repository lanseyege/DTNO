#!/usr/bin/env bash
# Phase A5 — finalize Experiment A. No training. Roughly 30 minutes.
#
# Four things, in this order, because each depends on the previous one:
#
#   [1] re-score every run whose JSON predates the non-finite fix
#   [2] per-dataset wall-clock timing
#   [3] regenerate every figure with all seeds
#   [4] print what still has no error bar
#
# WHY [1] EXISTS
# §24's bounded score caps each horizon at 1.0 so that a diverged rollout
# (E = 1e20) still produces a rankable number. It does not survive a NON-FINITE
# one: `min(nan, 1.0)` is nan, so a single blown-up horizon turned the whole
# summary into nan. Gray-Scott AR-FNO-R at seed 1 hit exactly that -- and it is
# the seed that best demonstrates the instability being reported, so losing it
# would have understated the very thing the seeds were run to measure.
#
# evaluation/runner.py now scores a non-finite horizon at the bound and counts
# it in `nonfinite_horizons`; scripts/make_figures.py recomputes the bounded
# score per seed rather than from the seed mean. Neither touches the model, so
# this is a re-score of existing checkpoints, not a re-run.
#
#   bash run/25_expA_finalize.sh
#   DATASETS="gray_scott" bash run/25_expA_finalize.sh
#   SKIP_TIMING=1 bash run/25_expA_finalize.sh
set -euo pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}

DATASETS=${DATASETS:-"gray_scott cylinder rayleigh_benard"}
SEEDS=${SEEDS:-"0 1 2"}
OUT=${OUT:-results/figures_expA}
declare -A PREFIX=( [gray_scott]=gs [cylinder]=cyl [rayleigh_benard]=rb )

# --------------------------------------------------------------------------
echo "=================================================================="
echo "  [0] Confirm the fix is in"
echo "=================================================================="
python - <<'PY'
import sys
bad = []
for path, needle, what in (
        ("evaluation/runner.py", "nonfinite_horizons",
         "bounded score survives NaN"),
        ("scripts/make_figures.py", "_bounded_score",
         "per-seed bounded score")):
    try:
        if needle not in open(path).read():
            bad.append(f"  {path}: {what} NOT applied")
    except FileNotFoundError:
        bad.append(f"  {path}: file not found")
if bad:
    print("\n".join(bad))
    print("\nCopy the shipped evaluation/runner.py and scripts/make_figures.py "
          "before running this.\nRe-scoring with the old code reproduces the "
          "nan and wastes the pass.")
    sys.exit(1)
print("  both applied")
PY

# --------------------------------------------------------------------------
echo
echo "=================================================================="
echo "  [1] Re-score AR runs (the only ones that can go non-finite)"
echo "=================================================================="
for DS in $DATASETS; do
  CFG="configs/${DS}.yaml"; P="${PREFIX[$DS]:-$DS}"
  [ -f "$CFG" ] || { echo "  missing $CFG, skipping $DS"; continue; }
  for S in $SEEDS; do
    for M in ar_fno_r dt_fno sg_dt_fno; do
      TAG="${P}_${M}_s${S}"
      CK="./checkpoints/${TAG}/best_model.pth"
      [ -f "$CK" ] || continue
      # Only AR rollouts produce non-finite fields; DT is a single pass and
      # cannot. Re-scoring DT anyway costs a few minutes and guarantees every
      # JSON in the figure comes from one code version -- which is the actual
      # requirement, since make_figures compares them against each other.
      echo ">>> ${TAG}"
      python scripts/evaluate_horizon.py --config "$CFG" \
        --set meta.model_variant=$M \
        --checkpoint "$CK" \
        --set experiment.exp_name=${TAG}
    done
  done
done

# --------------------------------------------------------------------------
echo
echo "=================================================================="
echo "  [2] Per-dataset timing"
echo "=================================================================="
# The 386x figure is architecture-level, but the BREAK-EVEN horizon is not:
# it is set by per-step AR error, which is a property of the flow. run/21 wrote
# one shared inference_cost.json, so fig5_pareto only exists for whichever
# dataset happened to run last. Needs an idle GPU and a sustained warm-up --
# an idle A800 sits at low clocks and short measurements land 32% slow.
if [ "${SKIP_TIMING:-0}" != "1" ]; then
  mkdir -p results/timing
  for DS in $DATASETS; do
    CFG="configs/${DS}.yaml"; P="${PREFIX[$DS]:-$DS}"
    [ -f "$CFG" ] || continue
    echo ">>> timing $DS"
    python scripts/benchmark_timing.py --config "$CFG" --random_weights \
      --out results/timing/inference_cost_${P}.json \
      || echo "  [warn] timing failed for $DS; re-run alone on an idle GPU"
  done
else
  echo "  skipped (SKIP_TIMING=1)"
fi

# --------------------------------------------------------------------------
echo
echo "=================================================================="
echo "  [3] Figures, all seeds"
echo "=================================================================="
mkdir -p "$OUT"
# fig1_diverged.txt and fig3_caveat.txt are written ONLY when the condition
# they describe occurs, so a corrected re-run leaves the previous run's file
# sitting in the output directory describing results that were thrown away.
# The Rayleigh-Benard directory carried "rb_climatology: 8 horizon(s), worst
# 437.2" from the invalid spin-up run long after the corrected figure showed no
# divergence at all -- exactly the kind of file that gets cited by mistake.
find "$OUT" -name "fig1_diverged.txt" -o -name "fig3_caveat.txt" \
  | xargs -r rm -f
for DS in $DATASETS; do
  P="${PREFIX[$DS]:-$DS}"
  TIMING="results/timing/inference_cost_${P}.json"
  echo ">>> $DS -> $OUT/$P"
  # EXACT names, not a glob: '--include ar_fno_r*' also matches
  # ar_fno_r_fixsel, which is how a contaminated run reached Figure 1 once
  # already. make_figures collapses <tag>_s<seed> onto <tag> by itself, so the
  # seed suffix does not appear here.
  python scripts/make_figures.py \
    --results results/*/horizon_metrics.json \
    --include ${P}_ar_fno_r ${P}_dt_fno ${P}_sg_dt_fno \
              ${P}_persistence ${P}_climatology ${P}_nearest_climatology \
              ${P}_pod_dmd \
    $([ -f "$TIMING" ] && echo "--timing $TIMING") \
    --out "$OUT/$P"
done

for P in gs cyl; do
  ls results/${P}_leadtime_*/horizon_metrics.json >/dev/null 2>&1 || continue
  echo ">>> lead-time figures for $P"
  python scripts/make_figures.py \
    --results results/*/horizon_metrics.json \
    --include ${P}_leadtime_fourier ${P}_leadtime_fourier_log \
              ${P}_leadtime_climatology ${P}_leadtime_nearest_climatology \
    --out "$OUT/${P}_leadtime"
done

# --------------------------------------------------------------------------
echo
echo "=================================================================="
echo "  [4] What still has no error bar"
echo "=================================================================="
python - <<'PY'
import glob, json, os, re
from collections import defaultdict
seeds = defaultdict(set)
for p in glob.glob("results/*/horizon_metrics.json"):
    tag = os.path.basename(os.path.dirname(p))
    m = re.search(r"_s(\d+)$", tag)
    seeds[tag[:m.start()] if m else tag].add(m.group(1) if m else "-")
    try:
        nf = json.load(open(p)).get("nonfinite_horizons") or []
    except Exception:
        nf = []
    if nf:
        print(f"  non-finite horizons in {tag}: {nf}")
print()
for k in sorted(seeds):
    n = len(seeds[k])
    if any(m in k for m in ("ar_fno", "dt_fno")) and n < 3:
        print(f"  {k:<34} {n} seed(s)  <- no band in Figure 1")
print("""
Read before using any of these:
  * make_figures warns that `horizon_sampling` differs across groups
    (ar_fno_r is 'fixed', the rest 'log_binned'). That is BY DESIGN --
    configs/ar_fno_r.yaml sets it, because the AR arm uses ar_rollout and no
    horizon sampler. It fires on every dataset. Do not "fix" it; say in the
    caption that the AR arm is trained on short rollouts by construction.
  * A group whose seeds straddle divergence has a mean that is not a centre.
    Gray-Scott AR-FNO-R at h=128 gives 0.84, 9.0 and 113.2; the mean 41.0
    describes none of them. Quote the seeds, not the mean.
  * scripts/spec_channel_report.py before quoting any E_spec.
""")
PY
