GPU=3

CUDA_VISIBLE_DEVICES=$GPU python scripts/check_time_sensitivity.py --config configs/dt_fno.yaml \
    --checkpoint checkpoints/dt_fno_s0/best_model.pth \
    --set experiment.exp_name=dt_fno_s0
