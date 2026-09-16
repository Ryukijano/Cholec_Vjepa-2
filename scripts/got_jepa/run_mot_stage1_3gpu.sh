#!/bin/bash
# Stage 1 — Supervised MOT scaffolding (3x L40s GPUs).
# NOTE: the current trainer is single-device. For 3-GPU DDP,
# launch via `torchrun` once the DDP wiring is added. Provided here
# as a placeholder and for SLURM templating.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "=== Surgical MOT — Stage 1 (Supervised, 3x GPU) ==="
echo "Repo root: $REPO_ROOT"

module load miniforge 2>/dev/null || true
conda activate dino_wm 2>/dev/null || true

CFG="configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml"

PYTHONPATH=. python -m core_app.mot.main \
  --fname "$CFG" \
  --devices cuda:0 cuda:1 cuda:2 \
  --debugmode False

echo ""
echo "Stage 1 training complete!"
