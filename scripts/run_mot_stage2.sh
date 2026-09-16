#!/bin/bash
# Stage 2 — GOT-JEPA teacher-student predictor pretraining.
# Teacher = frozen copy of Stage-1 predictor.
# Student + ProjNet + Expander are the only trainable modules.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "=== Surgical MOT — Stage 2 (GOT-JEPA pretrain) ==="

module load miniforge 2>/dev/null || true
conda activate dino_wm 2>/dev/null || true

CFG="configs/train_mot/dinov2/cholec20-mot-stage2-jepa-pretrain.yaml"
STAGE1_CKPT="outputs/mot/cholec20-stage1-supervised/best.pth.tar"

if [ ! -f "$STAGE1_CKPT" ]; then
  echo "ERROR: Stage 1 checkpoint not found at $STAGE1_CKPT"
  echo "Run scripts/run_mot_stage1.sh first."
  exit 1
fi

PYTHONPATH=. python -m core_app.mot.main \
  --fname "$CFG" \
  --devices cuda:0 \
  --resume "$STAGE1_CKPT" \
  --debugmode False

echo ""
echo "Stage 2 training complete!"
