python ./scripts/make_figures.py --results ./results/*/horizon_metrics.json \
    --include gs_ar_fno_r gs_dt_fno gs_sg_dt_fno gs_persistence \
              gs_climatology gs_nearest_climatology gs_pod_dmd \
    --label "A1 Gray--Scott" --out ./results/figures_paper_all/gs

python ./scripts/make_figures.py --results ./results/*/horizon_metrics.json \
    --include cyl_ar_fno_r cyl_dt_fno cyl_sg_dt_fno cyl_persistence \
              cyl_climatology cyl_nearest_climatology cyl_pod_dmd \
    --label "A2 Cylinder" --legend off --out ./results/figures_paper_all/cyl

python ./scripts/make_figures.py --results ./results/*/horizon_metrics.json \
    --include ar_fno_r dt_fno sg_dt_fno persistence \
              climatology nearest_climatology pod_dmd \
    --timing ./results/timing/inference_cost_s0.json \
    --label "B2 RealPDEBench" --legend off --out ./results/figures_paper_all/comb

