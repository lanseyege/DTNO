# DTNO — direct-time neural operators, and a controlled comparison against autoregressive rollout

Code for the five-dataset study of finite-time (direct) prediction against
rollout-trained autoregression under a matched spatial backbone.

Three models share an encoder, an FNO backbone and a decoder, and differ only in
how time enters:

| variant | what it is |
|---|---|
| `ar_fno_r` | autoregressive, trained on four-step rollouts of its own predictions |
| `dt_fno` | direct: one forward pass conditioned on the requested lead time |
| `sg_dt_fno` | `dt_fno` plus the latent semigroup and identity losses |

`ar_unet_r` / `dt_unet` are the same two schemes on a U-Net backbone, used as the
architecture control.

---

## 1. Install

There is no packaging file; the code runs from the repository root with the
repository root on `PYTHONPATH` (the `run/*.sh` scripts assume you are standing
in it).

```
python >= 3.10
torch >= 2.1          # CUDA build; bf16 autocast and torchrun are used
numpy, pyyaml, matplotlib
h5py                  # The Well  (Gray-Scott, Rayleigh-Benard)
pyarrow               # RealPDEBench Cylinder
zarr                  # RealPDEBench combustion
```

Nothing else is imported. A CPU-only install is enough for the smoke test below
and for every analysis script, but not for training.

## 2. Two-minute smoke test, no datasets required

This exercises the whole pipeline — audit, split, normalisation, both dataset
classes, all three models, the semigroup losses, the horizon evaluator and the
figures — on synthetic fields shaped like the real ones.

```bash
python scripts/make_synthetic_data.py --out /tmp/smoke_data
python scripts/audit_data.py --config configs/dt_fno.yaml \
    --set data.data_path=/tmp/smoke_data --force
python scripts/train.py --config configs/dt_fno.yaml \
    --set data.data_path=/tmp/smoke_data \
    --set training.epochs=2 --set training.samples_per_epoch=64
```

Run it before queuing anything on a GPU. It is also the fastest way to check
that a change to the data layer has not broken the channel grouping or the
transform heuristics.

## 3. Layout

```
configs/         one YAML per dataset + one per model variant; `_base_:` chains them
  experiments/   configs for the revision experiments (rollout depth)
data/            one module per dataset behind a common store interface
models/          fno.py, unet.py (backbones), ar_fno.py, direct_fno.py (schemes)
losses/          prediction, semigroup, identity, collapse monitor
training/        trainer.py (loop mechanics only), tasks.py (what to compute)
evaluation/      runner.py (horizon evaluation), field/spectral/combustion metrics
scripts/         every entry point; see §5
run/             numbered shell scripts, one per phase or experiment
artifacts/       frozen splits and normalisation statistics  ** empty in this archive **
results/         one directory per run, each with horizon_metrics.json
```

`artifacts/` is empty here. Splits and normalisation statistics are decisions,
not outputs: regenerating them **changes every number in the paper**, because a
redrawn split puts different trajectories in test. If you have the originals,
restore them before reproducing anything; if you do not, see §4 and expect
results near but not equal to the published ones.

## 4. Datasets

Every `configs/<dataset>.yaml` carries an absolute `data.data_path` pointing at
the machine this was developed on. Edit it, or override on the command line with
`--set data.data_path=...`.

| config | source |
|---|---|
| `gray_scott.yaml`, `rayleigh_benard.yaml` | The Well |
| `cylinder.yaml` | RealPDEBench, numerical cylinder (Arrow) |
| `base.yaml` | RealPDEBench combustion (Zarr) |
| `lifted_h2.yaml` | BLASTNet lifted flame, via `scripts/convert_lifted_h2.py` |

Two notes that are easy to get wrong. Cylinder is used at its native
$128\times256$, not the benchmark's subsampled $64\times128$, and with a
trajectory-held-out split rather than the released window-level index — under
that index all 92 trajectories appear in training and 43 also in test, so
numbers from it are not comparable with ours or with the leaderboard.
Rayleigh-Benard is the Chebyshev-node release, subsampled $512\times128 \to
128\times128$ along the periodic axis; `E_field` is unaffected but a vertical
spectrum computed on that grid is not a physical spectrum.

## 5. The order things must happen in

```
audit  ->  split  ->  train  ->  evaluate  ->  figures
```

`run/20_expA_prepare.sh` enforces this for the Experiment A datasets and
documents why reversing any two steps silently leaks: the stationarity screen
changes the trajectory count, so it must precede the split; the split decides
which trajectories the statistics see, so it must precede the audit.

Every entry point takes the same three arguments:

```bash
python scripts/<tool>.py --config configs/<dataset>.yaml \
    --set some.nested.key=value --set another=value --seed 0
```

Training is launched through `torchrun`:

```bash
torchrun --standalone --nproc_per_node 3 scripts/train.py \
    --config configs/gray_scott.yaml --seed 0 \
    --set meta.model_variant=dt_fno \
    --set experiment.exp_name=gs_dt_fno_s0
```

Main tools:

| script | what it does |
|---|---|
| `audit_data.py` | per-channel statistics, transform choices, writes `artifacts/norm_stats_*.json` |
| `prepare_split.py` | freezes a trajectory split to JSON with a manifest |
| `train.py` | training; selects on the horizon-integrated score, not one-step error |
| `evaluate_horizon.py` | the horizon grid, climatology baselines, spectra, combustion metrics |
| `make_figures.py` | all plots, `summary_table.txt`, `fig1_crossover.txt` |
| `param_table.py` | true parameter counts, built from the configs |
| `rescore_bounded.py` | recomputes the bounded score from an existing JSON, no GPU |
| `physics_metrics_table.py` | tabulates reaction-zone IoU, gradient and heat-release error |
| `summarize_rollout_depth.py` | the rollout-depth CSV |
| `benchmark_training_cost.py` | forward+backward step time and activation memory vs depth |
| `check_state.py` | behavioural probe: which patches are applied |

## 6. Things that will bite you

These are all mistakes that were actually made here, each of which produced a
plausible-looking wrong number rather than an error.

**`--nproc_per_node` must match across seeds.** `samples_per_epoch` is global,
so world size sets steps per epoch, which sets the learning-rate schedule. A
seed run at a different world size is not a seed of the same experiment.

**Do not shorten the epoch budget to save time.** The learning rate follows a
cosine decay over the *total* epoch count, so a 50-epoch run is not a truncated
100-epoch run even if the best checkpoint arrived at epoch 20.

**`--include` in `make_figures.py` is exact, not glob.** `ar_fno_r` will not
match `ar_fno_r_fixsel`; `'ar_fno_r*'` will, and that is how a contaminated run
once reached a figure. The seed suffix `_s<N>` is stripped automatically, so
pass `gs_dt_fno`, never `gs_dt_fno_s0`.

**Anchor-check any archived results before reusing them.** Two sets of runs on
disk looked reusable and were not: the strided sweep and the `dt_fno_K*`
history ablation both carried an old loss configuration, and the K arms miss the
current `dt_fno` by 31%. Before using an old run, check that its closest arm
reproduces a published number.

**The AR rollout depth is a batch-level maximum.** Each sample draws
$r \in \{1..4\}$ but the batch unrolls to the largest draw, so in practice
nearly every batch unrolls four steps. `ar_fno_r` is a four-step baseline, not a
uniform-over-1-to-4 one. The fixed-depth arms in `configs/experiments/` set
`ar_random_rollout: false` and are unaffected.

**Non-finite horizons make the plain mean useless and the bounded score `nan`
unless guarded.** `min(nan, 1)` is `nan`. A non-finite field is a failed
forecast and is scored at the bound, with the count reported separately; use
`rescore_bounded.py` on any JSON that predates that convention.

**Some note files are written only when their condition fires.** `fig1_diverged.txt`
and `fig3_caveat.txt` survive a corrected re-run and then describe results that
were thrown away. `run/24` and `run/25` delete them before regenerating.

**Precision is not uniform.** The FNO arms train in bf16; the U-Net arms need
fp32 inside the backbone and would otherwise produce a non-finite loss within a
dozen epochs, at a different epoch per seed. No wall-clock comparison between
the two backbones is meaningful until that is equalised.

## 7. Parameter counts

Read them from `scripts/param_table.py` or from the training log, never from a
formula. The spectral layer stores one complex weight tensor per block, so a
block costs $2w^2 m_1 m_2$ real parameters; assuming two tensors doubles every
count, which is how the paper briefly claimed 16.80 M for a model that has
8.41 M. Gray-Scott, Rayleigh-Benard and RealPDEBench are 8.4 M, Lifted H2
10.5 M, Cylinder 16.8 M; the direct variants add exactly 150,912 for the four
FiLM heads and the time-embedding MLP.
