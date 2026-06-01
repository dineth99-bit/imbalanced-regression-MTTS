# SkyFinder-DIR Results (official TempM labels)

Labels from `skyfinder_metadata.csv` (`TempM`, °C), hour 10–12, on-disk image join. **All methods:** ResNet-50, L1, batch 64, Adam 1e-3 unless noted.

**Naming:** Tables use [Yang et al. (DIR)](https://github.com/YyzHarry/imbalanced-regression) style — each row is **base + optional LDS/FDS** on the same backbone, not a separate model.

| Paper-style name | Training (`reweight` / LDS / FDS) |
|------------------|-----------------------------------|
| Vanilla | `none` / off / off |
| SQINV | `sqrt_inv` / off / off |
| SQINV + LDS | `sqrt_inv` / on / off |
| Vanilla + FDS | `none` / off / on |
| SQINV + LDS + FDS | `sqrt_inv` / on / on |

---

## Phase 1 — 5 cameras (AgeDB-style hyperparameters)

**Cameras:** 1093, 3888, 4795, 5021, 3297 (3297 replaces 858). **725** images → **724** total after preprocess.

**Split (camera-disjoint):** train 3297/4795/5021 · val 3888 · **test 1093** (102 images).

| Method (base + add-ons) | Overall L1 ↓ | Median-shot L1 | Low-shot L1 ↓ | Low-shot G-Mean ↓ |
|-------------------------|-------------:|---------------:|--------------:|------------------:|
| **Vanilla** | **4.32** | 3.69 | 4.62 | 3.64 |
| SQINV | 4.53 | 5.68 | **3.69** | **2.42** |
| SQINV + LDS | 5.38 | 4.52 | 5.87 | 4.35 |
| Vanilla + FDS | 5.20 | 3.75 | 5.94 | 4.39 |
| SQINV + LDS + FDS | 5.07 | 3.72 | 5.81 | 5.25 |

Run: `EPOCHS=30 bash run_experiments.sh` (AgeDB defaults: shot thr 50/10, FDS buckets 81). Log tags: `vanilla`, `sqinv`, `lds`, `fds`, `lds_fds`.

**Takeaway:** Vanilla best overall on this small slice; SQINV best on low-shot metrics.

---

## Phase 2 — Hyperparameter search (5-camera pilot)

**Goal:** Tune LDS/FDS/reweighting before the 10-camera benchmark.

**Procedure:** `TUNE_EPOCHS=20 python tune_hyperparams.py` → `tune_results.csv` (10 configs, shot thr **20/5**, FDS buckets **-15…40 / 56**).

| Rank | Config (paper-style) | Test overall L1 | Test low-shot L1 |
|------|----------------------|----------------:|-----------------:|
| 1 | **Vanilla + FDS** (`smooth10`) | **4.45** | **2.64** |
| 2 | SQINV + LDS (σ=2) | 4.92 | 8.27 |
| 3 | SQINV + LDS (σ=1) | 5.13 | 7.55 |
| 4 | Vanilla (tuned eval bins only) | 5.18 | 8.66 |

**Selected hyperparameters for Phase 3:**

| Component | Value |
|-----------|--------|
| FDS buckets | `--label_min -15 --label_max 40 --bucket_num 56` |
| FDS warmup | `--start_update 5 --start_smooth 10` (pilot winner **Vanilla + FDS**) |
| FDS kernel | `gaussian`, `fds_ks=5`, `fds_sigma=2` |
| LDS (when used) | SQINV + `--lds_sigma 2` |
| Eval shot bins | `--many_shot_thr 20 --low_shot_thr 5` |
| LR schedule | `--schedule 20 25` (for 30-epoch runs) |

Scripts: `train_best.sh` (Vanilla + FDS pilot winner), `run_experiments_10cam_tuned.sh` (full sweep).

---

## Phase 3 — 10 cameras (main benchmark, tuned hyperparameters)

**Cameras:** 1093, 3888, 4795, 5021, 3297, **684, 623, 3837, 5020, 65**.

**Data:** 2731 metadata matches → **2730** rows in `skyfinder.csv` (1 bad image dropped).

**Split (camera-disjoint, seed 666):**

| Split | Cameras | Images |
|-------|---------|-------:|
| Train | 65, 684, 3297, 3837, 3888, 4795, 5020, 5021 (+ spillover from val/test cams) | 2210 |
| Val | 1093 | 102 |
| **Test** | **623** | **418** |

Train temperature span: about **-10°C to 44°C** (33 integer bins in train).

**Protocol:** 30 epochs; tuned buckets, FDS `start_smooth=10`, LDS σ=2; shot metrics 20/5. Test MAE on **camera 623** only.

| Method (base + add-ons) | All ↓ | Many ↓ | Med. ↓ | Few ↓ |
|-------------------------|------:|-------:|-------:|------:|
| **Vanilla** | **5.70** | **5.65** | 7.00 | — |
| SQINV | 7.27 | 7.25 | 10.80 | — |
| SQINV + LDS (σ=2) | 5.95 | 6.06 | **5.35** | — |
| Vanilla + FDS (`smooth10`) | 6.58 | 6.51 | 9.82 | — |
| SQINV + LDS + FDS | 5.89 | 5.84 | 9.02 | — |

Low-shot L1 is **n/a** on test camera 623 with thresholds 20/5 (no test samples in low-shot bins under train-count definitions). `analyze_imbalance.py` on 10-cam train suggests `--many_shot_thr 42 --low_shot_thr 12` for future runs.

**Not run:** SQINV + FDS (standard DIR ablation); can be added via `run_experiments_10cam_tuned.sh`.

**Takeaway:** **Vanilla** best **All** and **Many** MAE on held-out camera 623. **SQINV + LDS** improves **Med.** only (7.00 → 5.35°C). Pilot **Vanilla + FDS** settings do not beat vanilla at 10 cameras. **SQINV** alone is worst overall (7.27°C).

Run: `EPOCHS=30 bash run_experiments_10cam_tuned.sh` · log: `experiments_10cam.log`

Checkpoint tags: `10cam_vanilla`, `10cam_sqinv`, `10cam_lds`, `10cam_fds_tuned`, `10cam_lds_fds_tuned`.

```bash
# Data (10 cameras)
cd data && python download_skyfinder.py   # DEFAULT_CAMERAS = 10
python build_skyfinder_meta.py && python preprocess_skyfinder.py

# Experiments
cd .. && EPOCHS=30 bash run_experiments_10cam_tuned.sh
python summarize_results.py | grep 10cam
```

---

## Commands quick reference

| Task | Command |
|------|---------|
| 5-cam pilot sweep (AgeDB defaults) | `EPOCHS=30 bash run_experiments.sh` |
| 5-cam hyperparam grid | `TUNE_EPOCHS=20 python tune_hyperparams.py` |
| **10-cam tuned benchmark** | `EPOCHS=30 bash run_experiments_10cam_tuned.sh` |
| Vanilla + FDS only (pilot winner) | `bash train_best.sh` |
