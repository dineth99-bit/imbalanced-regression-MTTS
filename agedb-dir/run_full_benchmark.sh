#!/usr/bin/env bash
# Full AgeDB-DIR matrix: LDS block + MTTS block (+ ablations).
# SMOTER/SMOGN skipped (not in repo). Results -> collect_results.py -> RESULTS.md
set -uo pipefail
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-./data}"
EPOCHS="${EPOCHS:-90}"
BS="${BS:-128}"
WORKERS="${WORKERS:-4}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

LDS_KS=5
LDS_SIG=2
FDS_KS=5
FDS_SIG=2
COMMON="--data_dir $DATA_DIR --epoch $EPOCHS --batch_size $BS --workers $WORKERS --schedule 60 80"
LDS_FLAGS="--lds --lds_kernel gaussian --lds_ks $LDS_KS --lds_sigma $LDS_SIG"
FDS_FLAGS="--fds --fds_kernel gaussian --fds_ks $FDS_KS --fds_sigma $FDS_SIG"

MANIFEST="$PWD/experiments_manifest.csv"
echo "method,store_substr,status" > "$MANIFEST"

log_skip() {
  echo "$1,$2,SKIPPED" >> "$MANIFEST"
}

log_done() {
  echo "$1,$2,DONE" >> "$MANIFEST"
}

run_train() {
  local method="$1"
  local store="$2"
  shift 2
  echo ""
  echo "========== $method ($store) =========="
  python3 train.py $COMMON --store_name "$store" "$@" || { echo "FAILED: $method"; echo "$method,$store,FAILED" >> "$MANIFEST"; return 0; }
  log_done "$method" "$store"
}

run_mtts() {
  local method="$1"
  local store="$2"
  shift 2
  echo ""
  echo "========== $method ($store) =========="
  python3 train_mtts.py $COMMON --store_name "$store" "$@" || { echo "FAILED: $method"; echo "$method,$store,FAILED" >> "$MANIFEST"; return 0; }
  log_done "$method" "$store"
}

python3 data/check_agedb_images.py --data_dir "$DATA_DIR" --min_found 400

echo "=== LDS block (paper) ==="

run_train "VANILLA" "md_vanilla" --reweight none
log_skip "SMOTER" "md_smoter"
log_skip "SMOGN" "md_smogn"
log_skip "SMOGN+LDS" "md_smogn_lds"
log_skip "SMOGN+FDS" "md_smogn_fds"
log_skip "SMOGN+LDS+FDS" "md_smogn_lds_fds"

run_train "FOCAL-R" "md_focal_r" --loss focal_l1
run_train "FOCAL-R+LDS" "md_focal_r_lds" --loss focal_l1 --reweight sqrt_inv $LDS_FLAGS
run_train "FOCAL-R+FDS" "md_focal_r_fds" --loss focal_l1 $FDS_FLAGS
run_train "FOCAL-R+LDS+FDS" "md_focal_r_lds_fds" --loss focal_l1 --reweight sqrt_inv $LDS_FLAGS $FDS_FLAGS

run_train "RRT stage-1" "md_rrt_s1" --reweight sqrt_inv
RRT_CKPT="$PWD/checkpoint/agedb_resnet50_sqrt_inv_md_rrt_s1_adam_l1_0.001_${BS}/ckpt.best.pth.tar"
if [[ ! -f "$RRT_CKPT" ]]; then
  RRT_CKPT=$(find checkpoint -path "*md_rrt_s1*ckpt.best.pth.tar" | head -1)
fi
echo "RRT checkpoint: $RRT_CKPT"

run_train "RRT" "md_rrt" --reweight sqrt_inv --retrain_fc --pretrained "$RRT_CKPT"
run_train "RRT+LDS" "md_rrt_lds" --reweight sqrt_inv --retrain_fc --pretrained "$RRT_CKPT" $LDS_FLAGS
run_train "RRT+FDS" "md_rrt_fds" --reweight sqrt_inv --retrain_fc --pretrained "$RRT_CKPT" $FDS_FLAGS
run_train "RRT+LDS+FDS" "md_rrt_lds_fds" --reweight sqrt_inv --retrain_fc --pretrained "$RRT_CKPT" $LDS_FLAGS $FDS_FLAGS

run_train "SQINV" "md_sqinv" --reweight sqrt_inv
run_train "SQINV+LDS" "md_sqinv_lds" --reweight sqrt_inv $LDS_FLAGS
run_train "SQINV+FDS" "md_sqinv_fds" --reweight sqrt_inv $FDS_FLAGS
run_train "SQINV+LDS+FDS" "md_sqinv_lds_fds" --reweight sqrt_inv $LDS_FLAGS $FDS_FLAGS

echo "=== MTTS block ==="
log_skip "SMOTER+MTTS" "md_smoter_mtts"
log_skip "SMOGN+MTTS" "md_smogn_mtts"
log_skip "SMOGN+MTTS+FDS" "md_smogn_mtts_fds"

run_mtts "FOCAL-R+MTTS" "md_focal_r_mtts" --loss focal_l1 --reweight sqrt_inv
run_mtts "FOCAL-R+MTTS+FDS" "md_focal_r_mtts_fds" --loss focal_l1 --reweight sqrt_inv --fds --fds_ks $FDS_KS --fds_sigma $FDS_SIG

run_mtts "RRT+MTTS" "md_rrt_mtts" --reweight sqrt_inv --retrain_fc --pretrained "$RRT_CKPT"
run_mtts "RRT+MTTS+FDS" "md_rrt_mtts_fds" --reweight sqrt_inv --retrain_fc --pretrained "$RRT_CKPT" --fds --fds_ks $FDS_KS --fds_sigma $FDS_SIG

run_mtts "SQINV+MTTS" "md_sqinv_mtts" --reweight sqrt_inv
run_mtts "SQINV+MTTS+FDS" "md_sqinv_mtts_fds" --reweight sqrt_inv --fds --fds_ks $FDS_KS --fds_sigma $FDS_SIG

run_mtts "SQINV+MTTS (no meta)" "md_sqinv_mtts_nomet" --reweight sqrt_inv --meta_freq 0
run_mtts "SQINV+MTTS (unconstrained)" "md_sqinv_mtts_uncon" --reweight sqrt_inv --unconstrained_kernel

echo "=== Collect results ==="
python3 collect_results.py --manifest "$MANIFEST" --output RESULTS.md

echo "Full benchmark finished."
