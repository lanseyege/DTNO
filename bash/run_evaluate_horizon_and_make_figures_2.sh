for M in climatology nearest_climatology; do
  python scripts/evaluate_horizon.py --config configs/lifted_h2.yaml \
    --model $M --set experiment.exp_name=lh2_${M}_full
done
python scripts/make_figures.py \
  --results results/lh2_*/horizon_metrics.json --out results/figures_lh2_physics

