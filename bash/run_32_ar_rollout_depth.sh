NPROC=3 \
DATASETS="gray_scott realpde" \
ROLLOUTS="1 4 8 16" \
SEEDS="0" \
BATCH_SIZE=32 \
bash run/32_ar_rollout_depth.sh
