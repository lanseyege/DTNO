for P in "" gs_; do
  python scripts/make_figures.py --results results/*/horizon_metrics.json \
    --include ${P}ar_unet_r ${P}dt_unet --out results/figures_unet/${P:-comb}
done
cat results/figures_unet/*/fig1_crossover.txt
