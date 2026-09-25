# Reproduction checklist

Every claim in the paper, the command that produces it, and the number to expect.
Work top to bottom: the later sections depend on the earlier ones.

Conventions used below: `NPROC` is the value passed to `--nproc_per_node`, and
it must be the same for every run you intend to compare. Wall-clock figures are
the ones actually recorded on 3×A800 at batch 32; blanks mean it was not logged.

---

## 0. Before anything (no GPU)

- [ ] Point the configs at your copies of the data (`data.data_path` in each
      `configs/<dataset>.yaml`).
- [ ] Restore `artifacts/` if you have the original splits and statistics.
      **If you regenerate them, every number below shifts**, because a redrawn
      split puts different trajectories in test. Reproduction is then
      qualitative, not numerical.
- [ ] Smoke test (README §2). Two minutes, no datasets.
- [ ] `python scripts/check_state.py --repo .` — behavioural probe for whether
      the data-layer and evaluation patches are in place.

```bash
bash run/20_expA_prepare.sh            # probe -> screen -> split -> audit
```

This order is load-bearing and the script says why. It writes
`artifacts/split_*.json` (with a manifest recording the file and realization
behind each index) and `artifacts/norm_stats_*.json`, fitted on training
trajectories only.

Checks: the probe's `[2b] FIELD AMPLITUDE vs TIME` block must show no spin-up
transient inside the evaluation window. On Rayleigh-Bénard it does — set
`traj_cut` from it and reduce `h_max` to fit the remaining record. Skipping this
is how an earlier run reported a climatology error of 437, which a climatology
cannot have.

---

## 1. Main comparison — Table 2, Figure 1, Table 3

```bash
NPROC=3 bash run/21_expA_train.sh      # Experiment A: three datasets x three models
NPROC=3 bash run/25_expA_finalize.sh   # re-score, per-dataset timing, figures
bash run/26_paper_figures.sh  /path/to/paper/figures
bash run/27_paper_fields.py   /path/to/paper/figures   # yes, .py: it is a
                                                       # bash script saved
                                                       # under the wrong
                                                       # extension; rename it
```

Expected `Eval*` (mean of `min(E,1)` over the horizon grid; three seeds for
AR/DT everywhere, and for SG on B1/B2):

| | AR-FNO-R | DT-FNO | SG-DT-FNO | climatology |
|---|---|---|---|---|
| Gray–Scott | 0.5926 | **0.3468** | 0.3461 | 0.5027 |
| Cylinder | 0.5978 | **0.4340** | 0.4447 | 0.6140 |
| Rayleigh–Bénard | 0.5494 | **0.4707** | 0.4704 | 0.5286 |
| Lifted H2 | 0.3835 | 0.3403 | **0.3397** | 0.3781 |
| RealPDEBench | 0.7861 | 0.5953 | 0.6068 | **0.4549** |

Crossovers, from `fig1_crossover.txt` in each figure directory: **9.1, 9.6,
11.5, 19.1, 8.1**. Cost: direct latency flat at 2.62 ms from h=1 to h=512
against 1.992 ms per rollout step — 512× fewer model evaluations, 386.8× faster
wall-clock at h=512.

Checks:

- [ ] RealPDEBench is the one dataset where the climatology oracle beats the
      direct model. That is a result, not a bug; the paper states it.
- [ ] `make_figures.py` warns that `horizon_sampling` differs across groups.
      By design — the AR arm is `fixed` at h=1. Do not "fix" it.
- [ ] AR-FNO-R diverges on Gray–Scott at 0/5/6 horizons across seeds. A single
      seed showing 0 is a normal draw, not a clean run.

---

## 2. Rollout depth — Table 5, §5.4

The experiment that answers "is the long-horizon gap just supervision?"

```bash
NPROC=3 DATASETS="gray_scott realpde" ROLLOUTS="1 4 8 16" SEEDS="0" \
  BATCH_SIZE=32 bash run/32_ar_rollout_depth.sh
NPROC=3 DEPTHS="16" SEEDS="1" bash run/34_rollout_depth_seeds.sh
python scripts/summarize_rollout_depth.py
```

Gray–Scott, per depth: `Eval*` 0.667 / 0.606 / 0.573 / {0.525, 0.598};
s per epoch 23 / 80 / 151 / 299; peak 2.6 / 9.0 / 17.5 / 34.6 GB; total train
2352 / 8022 / 15195 / 30058 s. RealPDEBench: 0.849 / 0.842 / 0.829 / 0.804 and
3.4 / 11.8 / 23.1 / 45.8 GB, 8874 / 11116 / 21736 / 43734 s.

- [ ] **Anchor check first.** R=4 must reproduce `ar_fno_r` to within a couple of
      percent (0.606 against 0.593 on Gray–Scott). If it does not, stop and find
      out why before reading anything else.
- [ ] The two R=16 seeds diverge at **0 and 6** horizons. Deeper rollouts move
      the average and leave the variance; do not report "divergence disappears".
- [ ] One-step error gets *worse* with depth (0.0055 → 0.0188). By R=16 the
      direct model wins at every tabulated horizon including h=1.

Microbenchmark (isolates the training step from data loading; one idle GPU,
minutes):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_training_cost.py \
  --config configs/gray_scott.yaml --rollouts 1 4 8 16 32 \
  --direct_horizons 1 128 --batch_size 2 --n_warmup 3 --n_repeat 10
```

AR 35.0 ms / 0.090 GB at R=1 → 678.5 ms / 4.216 GB at R=32. Direct: **48.0 ms
and 0.112 GB at both h=1 and h=128**. The per-step memory slope (0.133 GB at
batch 2) scales to the training sweep's (2.14 GB at batch 32) by exactly 16.

---

## 3. Controls that bound the main claim

| control | command | expected |
|---|---|---|
| backbone | `NPROC=3 bash run/28_matched_backbone.sh` (+ `SEED=1`, `SEED=2`) | U-Net: Gray–Scott AR 0.2952±0.0191 vs DT 0.2980±0.0077, crossover gone; RealPDEBench AR 0.8165 vs DT 0.5801, crossover 15.3; **0 diverged horizons in all six runs** |
| precision | `NPROC=3 bash run/30_precision_control.sh` | fp32 AR still diverges (1 horizon), crossover 9.1 → 11.4 |
| bandwidth | `NPROC=3 bash run/31_truncation_control.sh` | half-Nyquist 8.40 M, 4 diverged, crossover 10.5; **no truncation at all** 8.52 M, 5 diverged, crossover 8.2 |
| capacity | `NPROC=3 bash run/35_capacity_control.sh` then `run/36_capacity_seeds.sh` | width 88 = 15.90 M: `Eval*` 0.5915±0.0171 (baseline 0.5926), diverged 0/3/2 |

Read all four by **divergence count and crossover**, not by absolute accuracy.
Together they say: the direct model's advantage tracks how unstable the rollout
it is compared with happens to be; that instability survives more capacity, more
bandwidth and fp32, and does not survive a change of backbone.

---

## 4. Ablations

```bash
DATASETS="gray_scott cylinder rayleigh_benard" MODES="fourier" \
  bash run/22_expA_leadtime.sh          # sparse-horizon generalization
NPROC=3 bash run/23_expA_extras.sh      # semigroup, horizon range, POD-DMD
```

Lead-time ratios (error at unseen horizons over the interpolant of trained
neighbours): Rayleigh–Bénard 1.06, Gray–Scott 1.24, **Cylinder 3.62** with all
ten unseen horizons above 2×. log-Fourier: identity residual 190.0 and 101.6
against 2.15 and 3.35.

Semigroup: `C_SG` falls 4.34→0.12, 4.21→0.09, 3.31→0.08 with latent RMS
*rising*; `Eval*` moves −0.20%, +2.47%, −0.06%, −0.18%, +1.93%. Five nulls.

- [ ] The lead-time `--include` names carry **no** seed suffix. With one, the
      figure comes out holding only the climatology baselines.

---

## 5. Analysis-only, no GPU

```bash
python scripts/param_table.py --configs configs/*.yaml --latex
python scripts/rescore_bounded.py "results/*/horizon_metrics.json"     # add --write
python scripts/physics_metrics_table.py --results "results/*/horizon_metrics.json" \
    --include ar_fno_r dt_fno sg_dt_fno climatology --horizons 1 32 --latex
```

Parameter counts: Gray–Scott / Rayleigh–Bénard / RealPDEBench 8.4 M, Lifted H2
10.5 M, Cylinder 16.8 M; direct variants add exactly 150,912.

Physics metrics at h=128 on RealPDEBench: flame IoU 0.092 (AR) vs 0.304 (DT) vs
0.393 (climatology); mean |ΔT| 314 K vs 12.2 K vs 8.1 K.

---

## 6. What does **not** reproduce, and should not be attempted

- **Temporal cadence.** `run/29_cadence.sh` runs, but striding shortens the
  usable record while `h_max` is fixed in frames, so anchors collapse: 150 at
  stride 1, 55 at stride 2, **5** at stride 4. Five anchors cannot support a
  crossover. Reported in the appendix as attempted and not obtained.
- **The archived `realpde_str*` sweep.** Its AR/DT ordering is inverted at h=1,
  so no crossover can be read from it even after excluding the pressure channel.
- **The `dt_fno_K*` history ablation.** `dt_fno_K4` gives 0.7794 against the
  current `dt_fno`'s 0.5953 — a different protocol. Not reported.
- **Architecture-insensitivity of the crossover.** Claimed in an early draft,
  falsified by the U-Net control.
- **Spectral truncation as the cause of rollout divergence.** Proposed, then
  ruled out by the untruncated arm, which diverges slightly *more*.
- **Advection as the cause of the lead-time interpolation failure.** Proposed,
  then falsified by Rayleigh–Bénard, which advects and interpolates best.

The last three are in the paper as falsified hypotheses. They are listed here so
that a reproducer who finds the same negative results knows they are expected.

---

## 7. Rough cost of a full reproduction

| stage | GPU-hours (3×A800) |
|---|---|
| preparation | 0, but hours of I/O |
| Experiment A, three datasets × three models × three seeds | recorded per run in `history.json` |
| rollout depth, two datasets × four depths | ~24 |
| rollout depth, one extra seed at R=16 | ~8 |
| backbone control, two datasets × two models × three seeds | not recorded |
| capacity control, three seeds | ~13 |
| precision and bandwidth controls | not recorded |
| everything in §5 | 0 |

If you only want to check the central claim, §1 and §2 are enough, and §2 alone
answers the question a reader is most likely to ask.
