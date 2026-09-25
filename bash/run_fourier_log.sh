# 1. 主实验换 fourier_log，3 个 seed（这是关键路径）
for S in 0 1 2; do
  for M in dt_fno sg_dt_fno; do
    torchrun --standalone --nproc_per_node 4 scripts/train.py \
      --config configs/${M}.yaml --seed $S \
      --set model.time_embed_mode=fourier_log \
      --set experiment.save_dir=./checkpoints/${M}_log_s${S} \
      --set experiment.exp_name=${M}_log_s${S} \
      --set experiment.tb_dir=./runs/${M}_log_s${S}
    python scripts/evaluate_horizon.py --config configs/${M}.yaml \
      --set model.time_embed_mode=fourier_log \
      --checkpoint ./checkpoints/${M}_log_s${S}/best_model.pth \
      --set experiment.exp_name=${M}_log_s${S}
      --set experiment.tb_dir=./runs/${M}_log_s${S}
  done
done

# 2. AR-FNO-R 也补到 3 seeds（§43 要求 headline 都有 mean±std）
# 3. 空闲卡重测 timing
