torchrun --standalone --nproc_per_node 4 scripts/train.py \
  --config configs/dt_fno.yaml --seed 0 \
  --set model.predict_delta=false \
  --set experiment.save_dir=./checkpoints/dt_fno_nodelta_s0 \
  --set experiment.exp_name=dt_fno_nodelta_s0

python scripts/evaluate_horizon.py --config configs/dt_fno.yaml \
  --set model.predict_delta=false \
  --checkpoint ./checkpoints/dt_fno_nodelta_s0/best_model.pth \
  --set experiment.exp_name=dt_fno_nodelta_s0
