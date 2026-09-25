python scripts/plot_fields.py \
    --config configs/lifted_h2.yaml \
    --checkpoints ar_fno=./checkpoints/lh2_ar_fno_r_s0/best_model.pth \
                  dt_fno=./checkpoints/lh2_dt_fno_s0/best_model.pth \
                  sg_dt_fno=./checkpoints/lh2_sg_dt_fno_s0/best_model.pth \
    --h 128 --anchor 0 --alpha 0.2 \
    --out results/figures_paper
