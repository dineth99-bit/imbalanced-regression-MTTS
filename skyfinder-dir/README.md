# SkyFinder-DIR

Deep imbalanced regression for **ambient temperature prediction** from outdoor webcam images ([SkyFinder](https://zenodo.org/records/5884485)), using [LDS and FDS](https://github.com/YyzHarry/imbalanced-regression) from *Delving into Deep Imbalanced Regression* (ICML 2021).

Labels come from the official SkyFinder metadata (`skyfinder_metadata.csv`, column `TempM` in °C), joined to Zenodo images by `CamId` + filename. Cameras: **10-camera default** — 1093, 3888, 4795, 5021, 3297 plus 684, 623, 3837, 5020, 65 (3297 replaces 858). Use `CAMERAS_5` in `download_skyfinder.py` for the original 5-camera pilot slice.

All methods use **ResNet-50 + L1** on the same images; LDS and FDS are training add-ons, not separate models (see [method naming](#methods-dir-paper-naming) below).

## Setup

```bash
pip install -r requirements.txt
cd data
python download_skyfinder.py              # Zenodo zips + masks
python build_skyfinder_meta.py            # skyfinder_raw.csv (hour 10–12, TempM)
python preprocess_skyfinder.py            # skyfinder.csv (DIR splits)
```

## Methods (DIR paper naming)

Following Yang et al., each row is a **base** plus optional LDS/FDS on the same ResNet-50:

| Paper-style name | `--reweight` | LDS | FDS | Notes |
|------------------|--------------|-----|-----|--------|
| **Vanilla** | `none` | off | off | No imbalance handling |
| **SQINV** | `sqrt_inv` | off | off | Inverse-frequency weights only |
| **SQINV + LDS** | `sqrt_inv` | on | off | Gaussian LDS (default σ=2, ks=5) |
| **Vanilla + FDS** | `none` | off | on | FDS only; 10-cam tuned: `start_smooth=10`, `start_update=5` |
| **SQINV + LDS + FDS** | `sqrt_inv` | on | on | Full DIR combination (SQINV block) |

Checkpoint `store_name` tags (e.g. `10cam_lds`, `fds_tuned`) map to these rows; see [RESULTS.md](RESULTS.md).

The pipeline is modular: camera lists, split seeds, shot thresholds, and FDS bucket ranges are CLI arguments, so the benchmark can scale beyond 10 cameras without changing the training loop.

## Training

```bash
cd skyfinder-dir
pip install -r requirements.txt

# Vanilla
python train.py --meta_dir ./data --data_dir ./data/images --mask_dir ./data --reweight none

# SQINV
python train.py --meta_dir ./data --data_dir ./data/images --mask_dir ./data --reweight sqrt_inv

# SQINV + LDS
python train.py --meta_dir ./data --data_dir ./data/images --mask_dir ./data \
  --reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2

# Vanilla + FDS (tuned 10-cam example)
python train.py --meta_dir ./data --data_dir ./data/images --mask_dir ./data \
  --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2 \
  --start_update 5 --start_smooth 10 \
  --label_min -15 --label_max 40 --bucket_num 56 \
  --many_shot_thr 20 --low_shot_thr 5 --schedule 20 25

# SQINV + LDS + FDS (paper combination; tune FDS warmup for SkyFinder)
python train.py --meta_dir ./data --data_dir ./data/images --mask_dir ./data \
  --reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 \
  --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2 \
  --start_update 5 --start_smooth 10 \
  --label_min -15 --label_max 40 --bucket_num 56
```

**10-camera benchmark (recommended):**

```bash
EPOCHS=30 bash run_experiments_10cam_tuned.sh
python summarize_results.py
```

**5-camera pilot** (AgeDB-style defaults): `EPOCHS=30 bash run_experiments.sh`

## Splits

- **Default:** camera-disjoint train/val/test, then balanced val/test caps per integer °C (DIR protocol).
- **Optional:** `python preprocess_skyfinder.py --random_split` for random image splits.

## Tuning LDS / FDS

On small SkyFinder slices, AgeDB-DIR defaults often need adjustment. See **[TUNING.md](TUNING.md)**.

```bash
python data/analyze_imbalance.py          # bin counts + suggested shot thresholds
TUNE_EPOCHS=20 python tune_hyperparams.py   # small grid on 5-camera pilot
```

## Options

| Flag | Description |
|------|-------------|
| `--crop_mode sky_bbox` | Crop to sky mask bounding box |
| `--crop_mode sky_masked` | Zero ground pixels via mask |
| `--many_shot_thr` / `--low_shot_thr` | Shot metrics thresholds (AgeDB 50/10; SkyFinder tuned runs use 20/5) |
| `--label_min` / `--label_max` / `--bucket_num` | FDS bucket range (match train temp span) |
| `--lds_sigma` / `--fds_sigma` | LDS/FDS smoothing strength |
| `--start_smooth` / `--start_update` | When FDS kicks in (try 5–10 on small data) |

## Data layout

```
data/
  images/<camera_id>/<YYYYMMDD_HHMMSS>.jpg
  masks/skyfinder_masks/<camera_id>.png
  skyfinder_metadata.csv # official weather + MCR table
  skyfinder.csv          # temperature,path,split,camera_id
```

## Results

See **[RESULTS.md](RESULTS.md)** for the full benchmark. **Main result (10 cameras, test camera 623):** Vanilla **5.70°C** overall L1; **SQINV + LDS** best median-shot (**5.35°C**). Pilot history (5 cameras) is in the same file.

```bash
python summarize_results.py
```

## Citation

Please cite the DIR paper and the SkyFinder dataset if you use this benchmark.
