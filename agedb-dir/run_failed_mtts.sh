#!/usr/bin/env bash
# Re-run MTTS experiments that still FAILED (OOM / FDS forward / original DataParallel bugs).
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

RRT_CKPT="$PWD/checkpoint/agedb_resnet50_md_rrt_s1_sqrt_inv_adam_l1_0.001_${BS}/ckpt.best.pth.tar"
if [[ ! -f "$RRT_CKPT" ]]; then
  RRT_CKPT=$(find checkpoint -path "*md_rrt_s1*ckpt.best.pth.tar" | head -1)
fi
echo "RRT checkpoint: $RRT_CKPT"

MANIFEST="$PWD/experiments_manifest.csv"

log_done() {
  local method="$1" store="$2"
  if grep -q "^${method}," "$MANIFEST" 2>/dev/null; then
    sed -i "s/^${method},${store},.*/${method},${store},DONE/" "$MANIFEST"
  else
    echo "${method},${store},DONE" >> "$MANIFEST"
  fi
}

log_failed() {
  local method="$1" store="$2"
  if grep -q "^${method}," "$MANIFEST" 2>/dev/null; then
    sed -i "s/^${method},${store},.*/${method},${store},FAILED/" "$MANIFEST"
  else
    echo "${method},${store},FAILED" >> "$MANIFEST"
  fi
}

run_mtts() {
  local method="$1" store="$2"
  shift 2
  echo ""
  echo "========== $method ($store) =========="
  if python3 train_mtts.py $COMMON --store_name "$store" "$@"; then
    log_done "$method" "$store"
  else
    echo "FAILED: $method"
    log_failed "$method" "$store"
    return 1
  fi
}

echo "=== Re-run remaining failed MTTS (meta_batch_size=$META_BS) ==="

run_mtts "FOCAL-R+MTTS" "md_focal_r_mtts" --loss focal_l1 --reweight sqrt_inv
run_mtts "FOCAL-R+MTTS+FDS" "md_focal_r_mtts_fds" --loss focal_l1 --reweight sqrt_inv \
  --fds --fds_ks $FDS_KS --fds_sigma $FDS_SIG

run_mtts "RRT+MTTS+FDS" "md_rrt_mtts_fds" --reweight sqrt_inv --retrain_fc --pretrained "$RRT_CKPT" \
  --fds --fds_ks $FDS_KS --fds_sigma $FDS_SIG

run_mtts "SQINV+MTTS" "md_sqinv_mtts" --reweight sqrt_inv
run_mtts "SQINV+MTTS+FDS" "md_sqinv_mtts_fds" --reweight sqrt_inv \
  --fds --fds_ks $FDS_KS --fds_sigma $FDS_SIG

run_mtts "SQINV+MTTS (unconstrained)" "md_sqinv_mtts_uncon" --reweight sqrt_inv --unconstrained_kernel

echo "=== Collect results ==="
python3 collect_results.py --manifest "$MANIFEST" --output RESULTS.md

echo "Failed MTTS re-runs finished."
