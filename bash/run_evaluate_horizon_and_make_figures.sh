# 完整指标，不加 --light
GPU=1


for M in climatology nearest_climatology; do
    CUDA_VISIBLE_DEVICES=$GPU python scripts/evaluate_horizon.py --config configs/dt_fno.yaml --model $M \
    --set experiment.exp_name=${M}_full
done

CUDA_VISIBLE_DEVICES=$GPU python scripts/make_figures.py --results results/dt_fno_s*/horizon_metrics.json \
    results/ar_fno_r_s*/horizon_metrics.json results/*_full/horizon_metrics.json \
    --out results/figures_physics

