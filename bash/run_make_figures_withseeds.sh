#python scripts/make_figures.py --results results/*/horizon_metrics.json \
#    --timing results/timing/inference_cost.json --out results/figures_seeds

python scripts/make_figures.py --results results/*/horizon_metrics.json \
    --include 'ar_fno_r*' 'dt_fno' 'sg_dt_fno' persistence climatology nearest_climatology \
    --timing results/timing/inference_cost.json --out results/figures_main_2
