#!/bin/bash
#SBATCH --job-name=tinyLM
#SBATCH --account=def-koudas          # your Alliance account (ask your PI if unsure)
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out

############################
# 0) Modules / CUDA / Python
############################
# Many Alliance clusters provide CUDA & Python via modules. If your project is pure-PyTorch wheels,
# you can sometimes skip CUDA modules (PyTorch wheels ship their own CUDA). If you rely on nvcc, load CUDA.
set -euo pipefail

# module load StdEnv/2023 cuda/12.9 cudnn/9.13 python/3.11

########################################
# 1) Prepare virtual environment
########################################
# Create/activate a project venv under $HOME (persists across jobs)
uv sync
UV_VENV="$HOME/cs336/assignment1-basics/.venv"
source "$UV_VENV/bin/activate"

########################################
# 2) Staging paths
########################################
PROJECT_ROOT=/project/def-koudas
DATA_DIR=$PROJECT_ROOT/datasets/tinystories           # read-only source
RUNTIME_DATA_DIR=$SLURM_TMPDIR/data                   # fast local SSD during job
CKPT_DIR=/scratch/$USER/lm_runs/run1                  # frequent writes go here
ARCHIVE_DIR=$PROJECT_ROOT/archives/lm_2025_v1         # long-term copies

mkdir -p "$RUNTIME_DATA_DIR" "$CKPT_DIR" "$ARCHIVE_DIR"

echo "[stage] Copy dataset to \$SLURM_TMPDIR…"
rsync -a "$DATA_DIR/" "$RUNTIME_DATA_DIR/"

########################################
# 3) Configuration
########################################
# cat > "$SLURM_TMPDIR/runtime.yaml" <<EOF
#runtime_data_dir: "$RUNTIME_DATA_DIR"
#checkpoint: "$CKPT_DIR"
#archive_dir: "$ARCHIVE_DIR"
#save_wandb: true
#eval_interval: 200
#num_val_batches: 100
#device: "auto"
#EOF

MAIN_CFG="$HOME/cs336/assignment1-basics/configs/tinystories_base.yaml"

########################################
# 4) Run
########################################

python -u $HOME/cs336/assignment1-basics/cs336_basics/train_lm_on_cluster.py \
  --config "$MAIN_CFG"

echo "[archive] Copying final/best checkpoints to \$PROJECT…"
rsync -a --include="*/" --include="*final.pt" --exclude="*" "$CKPT_DIR/" "$ARCHIVE_DIR/"
