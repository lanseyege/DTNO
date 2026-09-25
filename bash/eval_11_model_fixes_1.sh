HM=16
SEED=0
TAG=dt_fno_hmax${HM}_s${SEED}
python scripts/evaluate_horizon.py --config configs/dt_fno.yaml \
    --set data.h_max_train=$HM \
    --checkpoint ./checkpoints/${TAG}/best_model.pth \
    --set experiment.exp_name=${TAG} \
    --set experiment.tb_dir=./runs/${TAG}

K=1
TAG=dt_fno_K${K}_s${SEED}

python scripts/evaluate_horizon.py --config configs/dt_fno.yaml \
    --set data.history_len=$K \
    --checkpoint ./checkpoints/${TAG}/best_model.pth \
    --set experiment.exp_name=${TAG} \
    --set experiment.tb_dir=./runs/${TAG}

