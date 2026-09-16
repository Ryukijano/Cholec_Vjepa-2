#!/bin/bash
# Stage 4 — Full GOT stack: Stage 3 + VGGT geometry + OccuSolver.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "=== Surgical MOT — Stage 4 (VGGT + OccuSolver) ==="

module load miniforge 2>/dev/null || true
conda activate dino_wm 2>/dev/null || true

CFG="configs/train_mot/dinov2/cholec20-mot-stage4-full.yaml"
STAGE3_CKPT="outputs/mot/cholec20-stage3-joint-finetune/best.pth.tar"

if [ ! -f "$STAGE3_CKPT" ]; then
  echo "ERROR: Stage 3 checkpoint not found at $STAGE3_CKPT"
  echo "Run scripts/run_mot_stage3.sh first."
  exit 1
fi

PYTHONPATH=. python -m core_app.mot.main \
  --fname "$CFG" \
  --devices cuda:0 \
  --resume "$STAGE3_CKPT" \
  --debugmode False

echo ""
echo "Stage 4 training complete!"
