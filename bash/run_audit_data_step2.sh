GPU=0

CUDA_VISIBLE_DEVICES=$GPU python scripts/audit_data.py --config configs/dt_fno.yaml \
    --redraw-split --strata artifacts/strata.json --force

