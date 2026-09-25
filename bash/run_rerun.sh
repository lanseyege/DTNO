python scripts/make_figures.py --results results/*/horizon_metrics.json \
    --exclude 'expB_*' '*nodelta*' 'dt_fno_log*' 'sg_dt_fno_log*' \
    --timing results/timing/inference_cost.json --out results/figures_main

python scripts/make_figures.py --results results/expB_*/horizon_metrics.json \
    --out results/figures_expB
