#!/bin/bash
# V-JEPA 2.1 World Model Training Script (Multi-GPU)

set -e

cd "$(dirname "$0")" || exit 1

echo "=== V-JEPA 2.1 World Model Training (3x GPU) ==="
echo "Config: cholec20-world-model-detr-reid"
echo "GPUs: cuda:0,1,2"
echo ""

# Load environment
module load miniforge 2>/dev/null || true
conda activate dino_wm 2>/dev/null || true

# Configuration
CFG="$PWD/configs/train_2_1/vitb16/cholec20-world-model-detr-reid.yaml"

# Run training with 3 GPUs
PYTHONPATH=. python -m core_app.main \
  --fname "$CFG" \
  --devices cuda:0 cuda:1 cuda:2 \
  --debugmode False

echo ""
echo "Training complete!"
