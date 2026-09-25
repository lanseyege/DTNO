# 0. 先把 Δt 找出来
python scripts/convert_lifted_h2.py \
    --loader ./data/lifted_h2_dataset.py \
    --data_root /mnt/sdb/yuanye/datasets/blastnet/lifted_hydrogen_jet_flame \
    --out /mnt/sdb/yuanye/datasets/lifted_h2_npy --dt 5e-06 \
    --test_re 7500 --val_re 9000 \

python scripts/convert_lifted_h2.py \
    --loader ./data/lifted_h2_dataset.py \
    --data_root /mnt/sdb/yuanye/datasets/blastnet/lifted_hydrogen_jet_flame \
    --out /mnt/sdb/yuanye/datasets/lifted_h2_npy \
    --dt 5.0e-6 --test_re 7500 --val_re 9000

# 1. split 手动固定，别靠随机：test 必须是 Re=7500（转换后的 traj_003）
#    审计会警告未分层 —— 这里 8 条、1 val 1 test，分层没有意义，
#    但要显式选择而不是碰运气
#python scripts/audit_data.py --config configs/lifted_h2.yaml --force

# 2. 三个模型 + 参考线，协议一字不改
#NPROC=4 SEED=0 MODELS="ar_fno_r dt_fno sg_dt_fno" bash run/07_seeds.sh
