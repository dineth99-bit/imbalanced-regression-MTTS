#!/usr/bin/env bash
# FOCAL-R+MTTS and FOCAL-R+MTTS+FDS only (kernel-only meta).
set -uo pipefail
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-./data}"
EPOCHS="${EPOCHS:-90}"
BS="${BS:-128}"
WORKERS="${WORKERS:-4}"
META_BS="${META_BS:-32}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

FDS_KS=5
FDS_SIG=2
COMMON="--data_dir $DATA_DIR --epoch $EPOCHS --batch_size $BS --workers $WORKERS --schedule 60 80 --meta_batch_size $META_BS"
MANIFEST="$PWD/experiments_manifest.csv"
LOG="$PWD/rerun_focal_mtts.log"

log_done() { sed -i "s/^$1,$2,.*/$1,$2,DONE/" "$MANIFEST"; }
log_failed() { sed -i "s/^$1,$2,.*/$1,$2,FAILED/" "$MANIFEST"; }

run_mtts() {
  local method="$1" store="$2"
  shift 2
  echo "========== $method ($store) ==========" | tee -a "$LOG"
  if python3 train_mtts.py $COMMON --store_name "$store" "$@" 2>&1 | tee -a "$LOG"; then
    log_done "$method" "$store"
    echo "DONE: $method" | tee -a "$LOG"
  else
    log_failed "$method" "$store"
    echo "FAILED: $method" | tee -a "$LOG"
    return 1
  fi
}

echo "=== FOCAL MTTS rerun $(date -Iseconds) ===" | tee "$LOG"

run_mtts "FOCAL-R+MTTS" "md_focal_r_mtts" --loss focal_l1 --reweight sqrt_inv
run_mtts "FOCAL-R+MTTS+FDS" "md_focal_r_mtts_fds" --loss focal_l1 --reweight sqrt_inv --fds --fds_ks $FDS_KS --fds_sigma $FDS_SIG

python3 collect_results.py --manifest "$MANIFEST" --output RESULTS.md 2>&1 | tee -a "$LOG"
echo "=== Finished $(date -Iseconds) ===" | tee -a "$LOG"
