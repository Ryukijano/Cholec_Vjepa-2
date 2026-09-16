#!/bin/bash
# Stage 3 — Joint MOT fine-tune with GOT-JEPA-pretrained student.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "=== Surgical MOT — Stage 3 (Joint fine-tune) ==="

module load miniforge 2>/dev/null || true
conda activate dino_wm 2>/dev/null || true

CFG="configs/train_mot/dinov2/cholec20-mot-stage3-joint-finetune.yaml"
STAGE2_CKPT="outputs/mot/cholec20-stage2-jepa-pretrain/best.pth.tar"

if [ ! -f "$STAGE2_CKPT" ]; then
  echo "ERROR: Stage 2 checkpoint not found at $STAGE2_CKPT"
  echo "Run scripts/run_mot_stage2.sh first."
  exit 1
fi

PYTHONPATH=. python -m core_app.mot.main \
  --fname "$CFG" \
  --devices cuda:0 \
  --resume "$STAGE2_CKPT" \
  --debugmode False

echo ""
echo "Stage 3 training complete!"
