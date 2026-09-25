TAG=ar_fno_r_fixsel_s3

python scripts/evaluate_horizon.py --config configs/ar_fno_r.yaml \
    --checkpoint ./checkpoints/${TAG}/best_model.pth \
    --set experiment.exp_name=${TAG} \
    --set experiment.tb_dir=./runs/${TAG}

