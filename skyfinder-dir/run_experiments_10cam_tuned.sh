#!/usr/bin/env bash
# 10-camera benchmark (DIR paper naming: base + LDS/FDS on ResNet-50).
# Tuned from 5-cam pilot: Vanilla+FDS winner — start_update 5, start_smooth 10, buckets -15..40/56, shot 20/5.
# Runs: Vanilla | SQINV | SQINV+LDS | Vanilla+FDS | SQINV+LDS+FDS  (SQINV+FDS not included)
set -euo pipefail
cd "$(dirname "$0")"
META_DIR=./data
DATA_DIR=./data/images
EPOCHS=${EPOCHS:-30}
BS=${BS:-64}
WORKERS=${WORKERS:-8}
TUNED="--many_shot_thr 20 --low_shot_thr 5 --label_min -15 --label_max 40 --bucket_num 56 --schedule 20 25"
COMMON="--meta_dir $META_DIR --data_dir $DATA_DIR --mask_dir $META_DIR --epoch $EPOCHS --batch_size $BS --workers $WORKERS $TUNED"

run() {
  echo "========== $* =========="
  python3 train.py $COMMON "$@"
}

run --reweight none --store_name 10cam_vanilla                    # Vanilla
run --reweight sqrt_inv --store_name 10cam_sqinv                 # SQINV
run --reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 --store_name 10cam_lds  # SQINV+LDS
run --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2 \
  --start_update 5 --start_smooth 10 --store_name 10cam_fds_tuned   # Vanilla+FDS
run --reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 \
  --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2 \
  --start_update 5 --start_smooth 10 --store_name 10cam_lds_fds_tuned  # SQINV+LDS+FDS

echo "10-camera tuned sweep finished."
