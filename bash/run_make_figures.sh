python scripts/make_figures.py \
    --results results/ar_fno_r_s0/horizon_metrics.json \
        results/dt_fno_s0/horizon_metrics.json \
        results/sg_dt_fno_s0/horizon_metrics.json \
        results/persistence/horizon_metrics.json \
        results/pod_dmd/horizon_metrics.json \
    --timing results/timing/inference_cost2.json \
    --out results/figures2
