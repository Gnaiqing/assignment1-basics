#!/bin/bash
# run_grid.sh — grid search on a single GPU (Runpod)
# Usage: bash run_grid_pod.sh
set -euo pipefail

############################
# User settings
############################
uv sync
source .venv/bin/activate
PYTHON=python
SCRIPT=./cs336_basics/train_lm_on_cluster.py
CONFIG=./configs/tinystories_local.yaml

# Grid
LRS=(3e-4 1e-4 3e-5 1e-5)
BATCH_SIZES=(1024 512 256 128 64)

# Keep total tokens processed roughly constant across batch sizes:
TOTAL_TOKENS=1280000   # adjust if you want a longer/shorter run

# Output
RUNS_ROOT=./runs        # where to put logs/checkpoints per run
LOGS_ROOT=./logs        # where to put .log files (symlinked from run dirs)
mkdir -p "$RUNS_ROOT" "$LOGS_ROOT"

# Concurrency (1 = sequential; increase to run a few in parallel)
MAX_JOBS=${MAX_JOBS:-1}

# GPU selection (Runpod single GPU typically is 0)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# (Optional) W&B setup — uncomment if you want online logging
# export WANDB_MODE=online
# export WANDB_ENTITY=uoft-db
# export WANDB_PROJECT=cs336-hw1

############################
# Helpers
############################
sanitize() {
  # make a string safe for a directory name
  local s="$1"
  s="${s//./}"   # remove dots
  s="${s//-/m}"  # replace minus with 'm' (e.g., 3e-4 -> 3em4)
  echo "$s"
}

running_jobs=0
pids=()

run_one() {
  local lr="$1"
  local bs="$2"
  # ceil division for safety
  local ts=$(( (TOTAL_TOKENS + bs - 1) / bs ))

  local lr_clean
  lr_clean="$(sanitize "$lr")"
  local run_name="bs${bs}_lr${lr_clean}"

  local out_dir="${RUNS_ROOT}/${run_name}"
  local log="${LOGS_ROOT}/${run_name}.log"
  mkdir -p "$out_dir"

  echo "[run] $run_name  (bs=${bs}, lr=${lr}, total_steps=${ts})"
  echo "  -> out_dir: $out_dir"
  echo "  -> log:     $log"

  # Launch
  # Add any other flags your script supports (e.g., --checkpoint "$out_dir")
  # Keeping your requested syntax: --batch_size {bs} --total_steps {ts} --lr {lr}
  (
    set -x
    $PYTHON "$SCRIPT" \
      --config "$CONFIG" \
      --batch_size "$bs" \
      --total_steps "$ts" \
      --lr "$lr" \
      --checkpoint "$out_dir" \
      2>&1 | tee "$log"
  )
}

launch_or_queue() {
  if (( MAX_JOBS > 1 )); then
    run_one "$1" "$2" &
    pids+=($!)
    running_jobs=$((running_jobs + 1))
    if (( running_jobs >= MAX_JOBS )); then
      wait -n
      running_jobs=$((running_jobs - 1))
    fi
  else
    run_one "$1" "$2"
  fi
}

cleanup() {
  echo "[cleanup] Terminating child jobs..."
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

############################
# Sweep
############################
for lr in "${LRS[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    launch_or_queue "$lr" "$bs"
  done
done

# wait for any remaining background jobs
wait
echo "[done] All runs completed."
