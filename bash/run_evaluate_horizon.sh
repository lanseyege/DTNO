
for M in persistence pod_dmd; do
  python scripts/evaluate_horizon.py --config configs/dt_fno.yaml \
    --model $M --set experiment.exp_name=$M
done
