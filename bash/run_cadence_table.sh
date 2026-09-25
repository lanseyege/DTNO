#python scripts/cadence_table.py --results "results/*/horizon_metrics.json" --timestep 2.5e-4
python scripts/cadence_table.py --results "results/*/horizon_metrics.json" \
    --timestep 2.5e-4 --exclude-channels Absolute_Pressure
