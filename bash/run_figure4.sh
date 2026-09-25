python scripts/plot_fields.py --config configs/dt_fno.yaml \
    --checkpoints ar_fno=./checkpoints/ar_fno_r_s0/best_model.pth \
                  dt_fno=./checkpoints/dt_fno_s0/best_model.pth \
                  sg_dt_fno=./checkpoints/sg_dt_fno_s0/best_model.pth \
    --h 128 --anchor 0 --out results/figures_paper

#cp results/figures_paper/fig4_fields_h128.pdf paper/figures/
