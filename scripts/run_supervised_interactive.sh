#!/bin/bash

# Interactive supervised scaffold training script
# Activate conda and run training directly

set -e

cd "$(dirname "$0")" || exit 1

echo "=== V-JEPA 2.1 Supervised DETR+ReID (Interactive) ==="
echo "Config: cholec20-supervised-detr-reid-base384"
echo "GPUs: cuda:0"
echo ""

# Load miniforge and activate environment
echo "Loading miniforge..."
module load miniforge

echo "Activating dino_wm environment..."
conda activate dino_wm

echo ""
echo "Starting training..."
echo ""

# Run training with correct arguments
CFG="$PWD/configs/train_2_1/vitb16/cholec20-supervised-detr-reid-base384.yaml"

PYTHONPATH=. python -m core_app.main \
  --fname "$CFG" \
  --devices cuda:0 \
  --debugmode True
