for N in ar_fno ar_fno_r; do
  python scripts/evaluate_horizon.py --config configs/${N}.yaml \
    --checkpoint ./checkpoints/${N}_s0/best_model.pth \
    --set experiment.exp_name=${N}_s0
done
