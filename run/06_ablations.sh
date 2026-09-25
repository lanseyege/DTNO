#!/usr/bin/env bash
# §34 — the five core ablations. One arm per line, no new config files.
#
# This is the expensive script: 5 ablations x several arms x 3 seeds is a lot of
# A800-hours. Run A1 and A2 first; they are the two that change what the paper
# claims. A3-A5 refine it.
set -euo pipefail
NPROC=${NPROC:-4}
SEED=${SEED:-0}

run () {  # run <tag> <config> <extra --set args...>
  local TAG=$1; local CFG=$2; shift 2
  torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py \
    --config "$CFG" --seed "$SEED" "$@" \
    --set experiment.save_dir=./checkpoints/${TAG}_s${SEED} \
    --set experiment.exp_name=${TAG}_s${SEED}
  python scripts/evaluate_horizon.py --config "$CFG" "$@" \
    --checkpoint ./checkpoints/${TAG}_s${SEED}/best_model.pth \
    --set experiment.exp_name=${TAG}_s${SEED}
}

# A1 — semigroup weight
for LAM in 0.0 0.05 0.1 0.2; do
  run a1_lam${LAM} configs/sg_dt_fno.yaml --set training.lambda_sg=$LAM
done

# A2 — history length K
for K in 1 2 4 8; do
  run a2_K${K} configs/dt_fno.yaml --set data.history_len=$K
done

# A3 — time conditioning
run a3_film    configs/dt_fno.yaml --set model.time_cond=film
run a3_concat  configs/dt_fno.yaml --set model.time_cond=concat \
                                   --set model.time_embed_mode=scalar
run a3_fourier_concat configs/dt_fno.yaml --set model.time_cond=concat

# A4 — horizon curriculum
run a4_logbin  configs/dt_fno.yaml --set data.horizon_sampling=log_binned
run a4_uniform configs/dt_fno.yaml --set data.horizon_sampling=uniform

# A5 — training horizon ceiling
for HMAX in 32 64 128; do
  run a5_hmax${HMAX} configs/dt_fno.yaml --set data.h_max_train=$HMAX
done
