CUDA_VISIBLE_DEVICES=3 python scripts/benchmark_training_cost.py \
  --config configs/gray_scott.yaml \
  --rollouts 1 4 8 16 32 \
  --direct_horizons 1 128 \
  --batch_size 2 \
  --n_warmup 3 \
  --n_repeat 10
