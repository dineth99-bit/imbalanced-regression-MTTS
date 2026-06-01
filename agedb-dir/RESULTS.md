# AgeDB-DIR Full Benchmark Results

Generated: 2026-05-31 05:27

**Setup:** ResNet50, 90 epochs, L1 (FOCAL-R rows use `focal_l1`), LDS/FDS Gaussian ks=5 σ=2.
**Data:** `data/agedb.csv` (on-disk images). **Train:** 25 runs completed.

**Note:** SMOTER / SMOGN rows skipped (not implemented in this repo).

## Results table (Test set)

| Method | MAE All | MAE Many | MAE Med. | MAE Few | GM All | GM Many | GM Med. | GM Few |
|--------|--------:|---------:|---------:|--------:|-------:|--------:|--------:|-------:|
| VANILLA | 7.72 | 6.81 | 8.70 | 13.81 | 4.98 | 4.37 | 5.93 | 10.76 |
| SMOTER | *SKIPPED* | — | — | — | — | — | — | — |
| SMOGN | *SKIPPED* | — | — | — | — | — | — | — |
| SMOGN+LDS | *SKIPPED* | — | — | — | — | — | — | — |
| SMOGN+FDS | *SKIPPED* | — | — | — | — | — | — | — |
| SMOGN+LDS+FDS | *SKIPPED* | — | — | — | — | — | — | — |
| FOCAL-R | 7.60 | 6.64 | 8.93 | 13.09 | 4.70 | 4.05 | 5.99 | 9.96 |
| FOCAL-R+LDS | 7.87 | 7.17 | 8.83 | 11.97 | 5.00 | 4.60 | 5.59 | 8.31 |
| FOCAL-R+FDS | 7.68 | 6.88 | 8.66 | 12.65 | 4.88 | 4.30 | 6.01 | 9.33 |
| FOCAL-R+LDS+FDS | 7.59 | 7.01 | 8.40 | 10.89 | 4.79 | 4.45 | 5.22 | 7.54 |
| RRT stage-1 | 7.67 | 7.02 | 8.61 | 11.21 | 4.84 | 4.44 | 5.67 | 7.14 |
| RRT | 7.67 | 7.02 | 8.61 | 11.21 | 4.84 | 4.44 | 5.67 | 7.14 |
| RRT+LDS | 7.67 | 7.06 | 8.60 | 10.93 | 4.80 | 4.38 | 5.74 | 7.02 |
| RRT+FDS | 7.68 | 7.06 | 8.62 | 10.97 | 4.78 | 4.35 | 5.75 | 7.05 |
| RRT+LDS+FDS | 7.67 | 7.07 | 8.57 | 10.93 | 4.77 | 4.35 | 5.63 | 7.09 |
| SQINV | 7.46 | 6.83 | 8.39 | 10.84 | 4.60 | 4.18 | 5.48 | 7.03 |
| SQINV+LDS | 8.03 | 7.18 | 9.33 | 12.45 | 5.15 | 4.67 | 6.19 | 8.03 |
| SQINV+FDS | 7.81 | 7.08 | 8.92 | 11.70 | 4.96 | 4.45 | 6.04 | 8.04 |
| SQINV+LDS+FDS | 7.56 | 7.01 | 8.40 | 10.48 | 4.76 | 4.40 | 5.41 | 6.92 |
| SMOTER+MTTS | *SKIPPED* | — | — | — | — | — | — | — |
| SMOGN+MTTS | *SKIPPED* | — | — | — | — | — | — | — |
| SMOGN+MTTS+FDS | *SKIPPED* | — | — | — | — | — | — | — |
| FOCAL-R+MTTS | 7.71 | 6.86 | 9.08 | 12.04 | 4.92 | 4.33 | 6.16 | 8.88 |
| FOCAL-R+MTTS+FDS | 7.56 | 6.79 | 8.78 | 11.48 | 4.82 | 4.27 | 6.09 | 8.11 |
| RRT+MTTS | 7.72 | 7.03 | 8.85 | 11.23 | 4.90 | 4.48 | 5.82 | 7.07 |
| RRT+MTTS+FDS | 16.99 | 12.03 | 27.79 | 34.24 | 12.12 | 8.60 | 27.23 | 33.68 |
| SQINV+MTTS | 7.32 | 6.77 | 8.10 | 10.43 | 4.66 | 4.26 | 5.45 | 7.18 |
| SQINV+MTTS+FDS | 7.47 | 6.82 | 8.42 | 11.04 | 4.66 | 4.20 | 5.52 | 7.67 |
| SQINV+MTTS (no meta) | 7.40 | 6.82 | 8.14 | 10.86 | 4.64 | 4.25 | 5.40 | 7.07 |
| SQINV+MTTS (unconstrained) | 7.93 | 7.04 | 9.14 | 13.05 | 5.05 | 4.44 | 6.21 | 9.72 |
| RankSim+SQINV | 7.20 | 6.65 | 8.09 | 9.95 | 4.43 | 4.07 | 5.14 | 6.67 |
| VIR | 6.82 | 6.37 | 7.48 | 9.26 | 4.38 | 4.10 | 4.87 | 6.15 |
| PRIME+PRW | 7.47 | 6.81 | 8.53 | 10.87 | 4.77 | 4.33 | 5.70 | 7.33 |

## MTTS (BEST) vs VANILLA (Test MAE / GM deltas)

Best MTTS config: **SQINV+MTTS**

| Shot | Δ MAE (Vanilla − Best MTTS) | Δ GM (Vanilla − Best MTTS) |
|------|---------------------------:|---------------------------:|
| All | +0.40 | +0.32 |
| Many | +0.00 | +0.00 |
| Med. | +0.00 | +0.00 |
| Few | +3.38 | +3.58 |

## Reproduce

```bash
cd agedb-dir
CUDA_VISIBLE_DEVICES=0 bash run_full_benchmark.sh
python3 collect_results.py
```
