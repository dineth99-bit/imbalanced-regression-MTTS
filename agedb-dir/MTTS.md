# MTTS on AgeDB-DIR

**Meta-Learned Target Topology Smoothing** replaces fixed Gaussian LDS with a learned, row-stochastic kernel \(K_\theta(b,b')\), initialized from LDS and meta-trained with a bin-balanced objective on a held-out meta split (10% of train).

## Files

| File | Role |
|------|------|
| `mtts.py` | Learnable kernel, effective density, sample weights, regularizers |
| `train_mtts.py` | Training + first-order meta kernel updates |
| `run_experiments_agedb.sh` | Baselines (vanilla, SQINV, LDS, FDS, LDS+FDS) + MTTS |

## Data setup (full AgeDB)

```bash
cd agedb-dir/data
# Option A: official zip (password-free mirror)
gdown 1vbMP-6_tGzOyaOiTa6s16rnhbGjU1BUi -O AgeDB_full.zip
unzip -q -o AgeDB_full.zip -d .
# Option B: Hugging Face partial (~8k images)
python download_agedb_hf.py --data_dir .
python filter_agedb_on_disk.py --data_dir .   # if using partial only
```

Bundled `agedb.csv` (16,488 images, balanced val/test) is used as-is.

## Train MTTS

```bash
CUDA_VISIBLE_DEVICES=0 python train_mtts.py --data_dir ./data --epoch 90 --batch_size 256
```

## Full benchmark

```bash
CUDA_VISIBLE_DEVICES=0 EPOCHS=90 bash run_experiments_agedb.sh
python summarize_results.py
```

## Key hyperparameters

| Flag | Default | Meaning |
|------|---------|---------|
| `--meta_ratio` | 0.1 | Fraction of train for meta set |
| `--meta_lr` | 1e-2 | Kernel optimizer LR |
| `--inner_lr` | 1e-4 | Inner MAML step on train batch |
| `--lds_sigma` / `--lds_ks` | 2 / 9 | Gaussian init for \(K_0\) |
| `--mtts_beta` | 1.0 | Inverse-density exponent |
| `--lam_local/smooth/sym` | 0.01 | Kernel regularizers |

## Inference

At test time only `f_\phi(x)` is used; the kernel is training-only.
