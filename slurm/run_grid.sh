#!/bin/bash
for lr in 3e-4 1e-4 3e-5 1e-5; do
  for bs in 1024 512 256 128 64; do
    total_steps=$((1280000 / ${bs}))
    RUN_NAME=bs${bs}_lr${lr//./} \
    EXTRA_ARGS="lr=${lr} batch_size=${bs} total_steps=${total_steps} checkpoint=/scratch/$USER/lm_runs/bs${bs}_lr${lr//./}" \
    sbatch train_tinystories.sbatch
  done
done
