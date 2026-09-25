
python ./scripts/make_figures.py --results ./results/*/horizon_metrics.json \
    --include ar_fno_r dt_fno sg_dt_fno persistence climatology nearest_climatology \
    --timing ./results/timing/inference_cost_s0.json \
    --out ./results/figures_paper

