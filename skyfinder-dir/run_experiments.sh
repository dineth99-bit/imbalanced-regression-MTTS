#!/usr/bin/env bash
# 5-camera pilot sweep (AgeDB-style defaults). DIR naming: Vanilla | SQINV | SQINV+LDS | Vanilla+FDS | SQINV+LDS+FDS
set -euo pipefail
cd "$(dirname "$0")"
META_DIR=./data
DATA_DIR=./data/images
EPOCHS=${EPOCHS:-30}
BS=${BS:-64}
WORKERS=${WORKERS:-8}
COMMON="--meta_dir $META_DIR --data_dir $DATA_DIR --mask_dir $META_DIR --epoch $EPOCHS --batch_size $BS --workers $WORKERS"

run() {
  echo "========== $* =========="
  python train.py $COMMON "$@"
}

run --reweight none --store_name vanilla                              # Vanilla
run --reweight sqrt_inv --store_name sqinv                          # SQINV
run --reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 --store_name lds   # SQINV+LDS
run --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2 --store_name fds                       # Vanilla+FDS
run --reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 \
    --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2 --store_name lds_fds                   # SQINV+LDS+FDS

echo "All runs finished."
