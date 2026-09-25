
python scripts/screen_stationary.py --config configs/gray_scott.yaml \
    --out artifacts/stationary_gray_scott.json

# 用它过滤 artifacts/split_gray_scott.json 的 test 列表 -> split_gs_dynamic.json
python scripts/evaluate_horizon.py --config configs/gray_scott.yaml \
    --set data.split_path=artifacts/split_gs_dynamic.json \
    --checkpoint ./checkpoints/gs_dt_fno_s0/best_model.pth \
    --set experiment.exp_name=gs_dt_fno_dynamic_s0


