#!/usr/bin/env bash
# Best FDS config from tune_hyperparams.py (5-cam pilot, 20-epoch grid).
# Paper name: Vanilla + FDS (smooth10) — reweight none, start_smooth=10.
set -euo pipefail
cd "$(dirname "$0")"

EPOCHS="${EPOCHS:-30}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python3 train.py \
  --meta_dir ./data \
  --data_dir ./data/images \
  --mask_dir ./data \
  --epoch "$EPOCHS" \
  --batch_size 64 \
  --workers 8 \
  --schedule 20 25 \
  --store_name best_fds_smooth10 \
  --reweight none \
  --fds \
  --fds_ks 5 \
  --fds_sigma 2 \
  --start_update 5 \
  --start_smooth 10 \
  --label_min -15 \
  --label_max 40 \
  --bucket_num 56 \
  --many_shot_thr 20 \
  --low_shot_thr 5
