torchrun --standalone --nproc_per_node 4 scripts/train.py \
    --config configs/cylinder.yaml --seed 0 --set meta.model_variant=dt_fno \
    --set data.num_workers=6 --set data.num_workers_val=2 --set data.num_workers_eval=4 \
    --set experiment.save_dir=./checkpoints/cyl_dt_fno_s0 \
    --resume ./checkpoints/cyl_dt_fno_s0/best_model.pth
