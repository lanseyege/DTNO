#python scripts/filter_split.py --config configs/gray_scott.yaml \
#    --split artifacts/split_gray_scott.json \
#    --exclude artifacts/stationary_gs_h128.json \
#    --out artifacts/split_gray_scott_dynamic.json
python scripts/filter_split.py --config configs/gray_scott.yaml \
    --split artifacts/split_gray_scott.json \
    --exclude artifacts/stationary_gs_h128.json --keep-only-excluded \
    --out artifacts/split_gray_scott_stationary.json 

