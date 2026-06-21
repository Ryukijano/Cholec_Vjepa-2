#!/bin/bash

# Interactive 3-GPU supervised DETR+ReID run without SLURM.
# Keeps W&B enabled by default; make sure `wandb login` has already been run.

set -euo pipefail

cd "$(dirname "$0")" || exit 1

echo "=== V-JEPA 2.1 Supervised DETR+ReID (Interactive 3-GPU) ==="
echo "Config: cholec20-supervised-detr-reid-base384"
echo "GPUs: cuda:0 cuda:1 cuda:2"
echo ""

echo "Loading miniforge..."
module load miniforge

echo "Activating dino_wm environment..."
conda activate dino_wm

echo ""
echo "Starting training..."
echo ""

CFG="$PWD/configs/train_2_1/vitb16/cholec20-supervised-detr-reid-base384.yaml"

env -u SLURM_JOB_ID \
  -u SLURM_NTASKS \
  -u SLURM_PROCID \
  -u SLURM_LOCALID \
  -u MASTER_ADDR \
  -u MASTER_PORT \
  -u RANK \
  -u WORLD_SIZE \
  PYTHONPATH=. \
  python -m core_app.main \
    --fname "$CFG" \
    --devices cuda:0 cuda:1 cuda:2
