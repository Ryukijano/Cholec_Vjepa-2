#!/bin/bash
# V-JEPA 2.1 World Model Training Script
# Multi-scale future prediction + DETR/ReID tracking

set -e

cd "$(dirname "$0")" || exit 1

echo "=== V-JEPA 2.1 World Model Training ==="
echo "Config: cholec20-world-model-detr-reid"
echo "Features: Multi-scale prediction (1/4/16 frames) + DETR + ReID"
echo ""

# Load environment
echo "Loading environment..."
module load miniforge 2>/dev/null || echo "miniforge not available, skipping"
conda activate dino_wm 2>/dev/null || echo "dino_wm env not found, using current env"

echo ""
echo "Starting training..."
echo ""

# Configuration
CFG="$PWD/configs/train_2_1/vitb16/cholec20-world-model-detr-reid.yaml"

# Run training
PYTHONPATH=. python -m core_app.main \
  --fname "$CFG" \
  --devices cuda:0 \
  --debugmode False

echo ""
echo "Training complete!"
