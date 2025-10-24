#!/bin/bash
#SBATCH --job-name=tinyLM
#SBATCH --account=def-koudas          # your Alliance account (ask your PI if unsure)
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out

############################
# 0) Modules / CUDA / Python
############################
# Many Alliance clusters provide CUDA & Python via modules. If your project is pure-PyTorch wheels,
# you can sometimes skip CUDA modules (PyTorch wheels ship their own CUDA). If you rely on nvcc, load CUDA.
module purge
# If your site uses StdEnv + python modules, uncomment and adjust:
# module load StdEnv/2023
# module load cuda/12.1
# module load python/3.10

########################################
# 1) Prepare uv & virtual environment
########################################
# Recommended: install uv once in $HOME, keep PATH.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "[env] Installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Create/activate a project venv under $HOME (persists across jobs)
UV_VENV="$HOME/cs336/assignment1-basics/.venv"
uv venv "$UV_VENV"
source "$UV_VENV/bin/activate"

# Choose ONLINE or OFFLINE dependency sync (see section 2 below).
# --- ONLINE (simple; needs outbound internet):
uv pip sync $HOME/cs336/assignment1-basics/pyproject.toml
# --- OFFLINE (robust; prebuild wheels to $PROJECT/wheels):
# uv pip sync --no-index --find-links /project/<prof_group>/wheels \
#     $HOME/projects/transformer_lm/requirements.txt

# If you want PyTorch to use the SLURM-selected GPU only:
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$SLURM_JOB_GPUS}

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
# 3) Runtime YAML that overrides base config
########################################
cat > "$SLURM_TMPDIR/runtime.yaml" <<EOF
runtime_data_dir: "$RUNTIME_DATA_DIR"
checkpoint: "$CKPT_DIR"
archive_dir: "$ARCHIVE_DIR"
save_wandb: true
eval_interval: 200
num_val_batches: 100
device: "auto"
EOF

MAIN_CFG="$HOME/cs336/assignment1-basics/configs/tinystories_base.yaml"

########################################
# 4) Run
########################################
set -euo pipefail
python -u $HOME/cs336/assignment1-basics/sc336_basics/train_lm_on_cluster.py \
  --config "$MAIN_CFG" \
  --config "$SLURM_TMPDIR/runtime.yaml" \
  --resume

echo "[archive] Copying final/best checkpoints to \$PROJECT…"
rsync -a --include="*/" --include="*final.pt" --include="*best.pt" --exclude="*" "$CKPT_DIR/" "$ARCHIVE_DIR/"
