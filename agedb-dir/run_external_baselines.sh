#!/usr/bin/env bash
# Run external SOTA baselines on AgeDB-DIR (RankSim, VIR, PRIME).
# Uses shared data/checkpoint under agedb-dir. Results -> collect_results.py
set -uo pipefail
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-$PWD/data}"
CKPT_ROOT="${CKPT_ROOT:-$PWD/checkpoint}"
EPOCHS="${EPOCHS:-90}"
BS="${BS:-64}"
LR="${LR:-2.5e-4}"
WORKERS="${WORKERS:-4}"
MANIFEST="${MANIFEST:-$PWD/experiments_manifest.csv}"
LOG="${LOG:-$PWD/external_baselines.log}"

RANKSIM_DIR="${RANKSIM_DIR:-$PWD/../external-baselines/ranksim/agedb-dir}"
VIR_DIR="${VIR_DIR:-$PWD/../external-baselines/vir}"

GPU_RANKSIM="${GPU_RANKSIM:-0}"
GPU_VIR="${GPU_VIR:-1}"
GPU_PRIME="${GPU_PRIME:-0}"

exec > >(tee -a "$LOG") 2>&1

log_row() {
  local method="$1" store="$2" status="$3"
  if ! grep -q "^${method}," "$MANIFEST" 2>/dev/null; then
    echo "${method},${store},${status}" >> "$MANIFEST"
  else
    sed -i "s|^${method},.*|${method},${store},${status}|" "$MANIFEST"
  fi
}

run_ranksim() {
  echo ""
  echo "========== RankSim + SQINV (md_ranksim_sqinv) GPU=$GPU_RANKSIM =========="
  (
    cd "$RANKSIM_DIR"
    CUDA_VISIBLE_DEVICES="$GPU_RANKSIM" python3 train.py \
      --data_dir "$DATA_DIR" \
      --store_root "$CKPT_ROOT" \
      --store_name md_ranksim_sqinv \
      --reweight sqrt_inv \
      --batch_size "$BS" --lr "$LR" --epoch "$EPOCHS" \
      --schedule 60 80 --workers "$WORKERS" \
      --regularization_weight 100.0 --interpolation_lambda 2.0
  ) && log_row "RankSim+SQINV" "md_ranksim_sqinv" "DONE" \
    || log_row "RankSim+SQINV" "md_ranksim_sqinv" "FAILED"
}

run_vir() {
  echo ""
  echo "========== VIR (md_vir) GPU=$GPU_VIR =========="
  (
    cd "$VIR_DIR"
    CUDA_VISIBLE_DEVICES="$GPU_VIR" python3 rerun.py \
      --data_dir "$DATA_DIR/" \
      --store_root "$CKPT_ROOT" \
      --store_name md_vir \
      --gpu 0 \
      --batch_size "$BS" --lr 0.001 --epoch "$EPOCHS" \
      --schedule 60 80 --workers "$WORKERS" \
      --lds --lds_ks 5 --lds_sigma 2 \
      --fds --fds_ks 5 --fds_sigma 2 \
      --reweight sqrt_inv \
      --use_prm --use_edl --use_cdm --use_recons \
      --lambda_recons 0.7 --lambda_reg 0.01 --seeds 1223
  ) && log_row "VIR" "md_vir" "DONE" \
    || log_row "VIR" "md_vir" "FAILED"
}

run_prime() {
  echo ""
  echo "========== PRIME+PRW (md_prime_prw) GPU=$GPU_PRIME =========="
  CUDA_VISIBLE_DEVICES="$GPU_PRIME" python3 train_prime.py \
    --data_dir "$DATA_DIR" \
    --store_root "$CKPT_ROOT" \
    --store_name md_prime_prw \
    --reweight sqrt_inv \
    --batch_size "$BS" --lr "$LR" --epoch "$EPOCHS" \
    --workers "$WORKERS" --schedule 60 80 \
    --num_proxies 20 --lambda_p 10 --lambda_a 50 \
    --tau_f 10 --tau_t 2 --alpha 0.0005 --prw \
  && log_row "PRIME+PRW" "md_prime_prw" "DONE" \
    || log_row "PRIME+PRW" "md_prime_prw" "FAILED"
}

# Append manifest rows if missing
for row in "RankSim+SQINV,md_ranksim_sqinv,PENDING" "VIR,md_vir,PENDING" "PRIME+PRW,md_prime_prw,PENDING"; do
  m="${row%%,*}"
  grep -q "^${m}," "$MANIFEST" 2>/dev/null || echo "$row" >> "$MANIFEST"
done

# RankSim and VIR in parallel on two GPUs; PRIME after RankSim finishes.
run_ranksim &
pid_rs=$!
run_vir &
pid_vir=$!
wait "$pid_rs" || true
wait "$pid_vir" || true
run_prime

python3 collect_results.py --manifest "$MANIFEST"
echo "External baselines finished. See $LOG and RESULTS.md"
