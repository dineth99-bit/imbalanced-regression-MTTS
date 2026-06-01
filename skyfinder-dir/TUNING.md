# Tuning SkyFinder-DIR (LDS / FDS / reweighting)

DIR methods help when **some temperatures have few training images** and errors on those bins matter. On a 5-camera pilot they may lose to **Vanilla** if labels are clean, data is small, or hyperparameters are still set for AgeDB (age 0–120, dense bins).

## Method naming (Yang et al.)

All runs use **ResNet-50 + L1**. Paper-style names:

| Name | `--reweight` | LDS | FDS |
|------|--------------|-----|-----|
| Vanilla | `none` | off | off |
| SQINV | `sqrt_inv` | off | off |
| SQINV + LDS | `sqrt_inv` | on | off |
| Vanilla + FDS | `none` | off | on |
| SQINV + LDS + FDS | `sqrt_inv` | on | on |

LDS requires a reweighting base (we use **SQINV**, as in the AgeDB **SQINV** block). FDS can be trained on **Vanilla** or on top of **SQINV + LDS**.

## 1. Check imbalance first

```bash
cd data
python analyze_imbalance.py
```

Look at per-°C counts in **train**. If most bins have 10+ samples, LDS/FDS gain less. If many bins have &lt;5 samples, tune low-shot metrics and FDS buckets.

## 2. Knobs that matter (in order)

### Shot metrics (evaluation only, but drives what you optimize for)

| Flag | Default (AgeDB) | SkyFinder suggestion | Why |
|------|-----------------|----------------------|-----|
| `--many_shot_thr` | 50 | **15–25** (tuned runs: **20**) | Small train set → few bins reach 50+ |
| `--low_shot_thr` | 10 | **3–8** (tuned runs: **5**) | Defines Med. / Few strata |

These do **not** change training; they change how you read `Many / Median / Low` in logs.

### LDS (label distribution smoothing) — use with SQINV

Requires `--reweight sqrt_inv` (or `inverse`). This is **SQINV + LDS** in the paper.

| Flag | Paper default | Try |
|------|---------------|-----|
| `--lds_ks` | 5 | 5, 7, 9 (odd only) |
| `--lds_sigma` | 2 | **1**, 2, **3** — larger σ = smoother target density |
| `--lds_kernel` | gaussian | gaussian, triang, laplace |

**Symptom:** SQINV + LDS hurts overall → try **smaller σ (1)** or weaker reweight (`sqrt_inv` not `inverse`).

### FDS (feature distribution smoothing) — Vanilla + FDS or SQINV + LDS + FDS

| Flag | Default | SkyFinder suggestion |
|------|---------|----------------------|
| `--bucket_num` | 81 | Match label span: `--label_min -15 --label_max 40` → 56 buckets |
| `--bucket_start` | 0 | Keep 0 (FDS uses bucket index = round(temp) - label_min) |
| `--fds_ks` / `--fds_sigma` | 5 / 2 | Same grid as LDS |
| `--start_update` | 0 | 0, **5** — delay stats until features stabilize |
| `--start_smooth` | 1 | 1, **5**, **10** (10-cam benchmark uses **10**) |
| `--fds_mmt` | 0.9 | 0.9, **0.95** |

**Symptom:** Vanilla + FDS hurts → raise `start_smooth`, reduce `fds_sigma`, or narrow `label_min`/`label_max` to actual train range.

### Training (all methods)

| Flag | Default | Try |
|------|---------|-----|
| `--epoch` | 90 (agedb) / 30 (sweep) | 30 tune, **60–90** final |
| `--lr` | 1e-3 | 5e-4, 1e-3 |
| `--schedule` | 60 80 | 20 25 for 30-epoch runs |
| `--batch_size` | 64 | 32 if OOM |
| `--loss` | l1 | l1, huber |

### Reweighting (DIR bases)

| `--reweight` | Paper name | When |
|--------------|------------|------|
| `none` | **Vanilla** (or Vanilla + FDS) | No sample reweighting |
| `sqrt_inv` | **SQINV** (required before LDS) | Use with LDS / LDS+FDS |
| `inverse` | INV | Stronger; can overfit rare bins on small data |

## 3. Quick manual examples

```bash
cd skyfinder-dir

# SQINV + LDS (softer σ)
python train.py --meta_dir ./data --data_dir ./data/images --mask_dir ./data \
  --reweight sqrt_inv --lds --lds_ks 5 --lds_sigma 1 \
  --many_shot_thr 20 --low_shot_thr 5 --store_name sqinv_lds_s1 --epoch 30

# Vanilla + FDS (warmup)
python train.py --meta_dir ./data --data_dir ./data/images --mask_dir ./data \
  --fds --fds_ks 5 --fds_sigma 2 --start_smooth 5 --start_update 5 \
  --label_min -15 --label_max 40 --bucket_num 56 \
  --many_shot_thr 20 --low_shot_thr 5 --store_name vanilla_fds_warm --epoch 30

# SQINV + LDS + FDS (adjusted buckets)
python train.py --meta_dir ./data --data_dir ./data/images --mask_dir ./data \
  --reweight sqrt_inv --lds --lds_ks 5 --lds_sigma 2 \
  --fds --fds_ks 5 --fds_sigma 2 --start_smooth 10 --start_update 5 \
  --label_min -15 --label_max 40 --bucket_num 56 \
  --many_shot_thr 20 --low_shot_thr 5 --store_name sqinv_lds_fds --epoch 30
```

## 4. Automated small grid

```bash
# Fast search (~15 epochs per run; set TUNE_EPOCHS=30 for serious search)
TUNE_EPOCHS=20 python tune_hyperparams.py
cat tune_results.csv
```

Picks best configs by **test low-shot L1** and **test overall L1** (see script output). Pilot winner for FDS-only: **Vanilla + FDS** with `start_smooth=10`.

## 5. What to report

For SkyFinder write-ups, report at least:

- **Overall test L1 (All)** on the held-out test camera (**623** for 10-cam benchmark)
- **Many / Med. / Few** MAE with documented `--many_shot_thr` and `--low_shot_thr` (note if Few is empty)
- Method names as **Vanilla**, **SQINV**, **SQINV + LDS**, **Vanilla + FDS**, **SQINV + LDS + FDS**
- Train temperature histogram (from `analyze_imbalance.py`)

Compare against **Vanilla** and **SQINV** before claiming LDS/FDS help. Do not compare absolute MAE to AgeDB-DIR numbers (different task and split).
