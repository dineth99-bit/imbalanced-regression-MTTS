#!/usr/bin/env bash
# Full AgeDB-DIR benchmark: baselines + MTTS (requires images under data/AgeDB/).
set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-./data}"
EPOCHS="${EPOCHS:-90}"
BS="${BS:-256}"
WORKERS="${WORKERS:-8}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

if ! python3 data/check_agedb_images.py --data_dir "$DATA_DIR" --min_found 100; then
  echo "Place AgeDB images in $DATA_DIR/AgeDB/ (see README.md) and re-run."
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
COMMON="--data_dir $DATA_DIR --epoch $EPOCHS --batch_size $BS --workers $WORKERS"

run_base() {
  echo "========== $* =========="
  python3 train.py $COMMON "$@"
}

echo "=== Baselines (paper-style) ==="
run_base --reweight none --store_name vanilla
run_base --reweight sqrt_inv --store_name sqinv
run_base --reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 --store_name lds
run_base --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2 --store_name fds
run_base --reweight sqrt_inv --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2 \
  --fds --fds_kernel gaussian --fds_ks 5 --fds_sigma 2 --store_name lds_fds

echo "=== MTTS ==="
python3 train_mtts.py $COMMON --reweight sqrt_inv --lds_sigma 2 --lds_ks 5 --store_name mtts_full

echo "Done. Summarize:"
python3 summarize_results.py --checkpoint_root checkpoint
