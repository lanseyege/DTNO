
#python scripts/plot_fields.py --config configs/cylinder.yaml \
#    --checkpoints ar_fno=checkpoints/cyl_ar_fno_r_s0/best_model.pth \
#                dt_fno=checkpoints/cyl_dt_fno_s0/best_model.pth \
#                sg_dt_fno=checkpoints/cyl_sg_dt_fno_s0/best_model.pth \
#    --h 128 --anchor 0 \
#    --out ./results/figures4/

python scripts/plot_fields.py --config configs/gray_scott.yaml \
    --checkpoints ar_fno=checkpoints/gs_ar_fno_r_s0/best_model.pth \
                dt_fno=checkpoints/gs_dt_fno_s0/best_model.pth \
                sg_dt_fno=checkpoints/gs_sg_dt_fno_s0/best_model.pth \
    --h 128 --anchor 0 \
    --out ./results/figures4/

