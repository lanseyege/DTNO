
#python scripts/physics_metrics_table.py --results "results/*/horizon_metrics.json" \
#    --include ar_fno_r dt_fno sg_dt_fno climatology nearest_climatology persistence \
#    --horizons 1 8 32 128
python scripts/physics_metrics_table.py --results "results/*/horizon_metrics.json" \
    --include ar_fno_r dt_fno sg_dt_fno climatology --horizons 1 32 --latex \
    > paper/tab_physics.tex
