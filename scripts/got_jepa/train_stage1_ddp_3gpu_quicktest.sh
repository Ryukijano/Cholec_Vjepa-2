#!/bin/bash
# Quick 10-epoch test to validate query-diversity fix before full 100-epoch run.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

module load miniforge 2>/dev/null || true
conda activate surgi_world_track_cuda 2>/dev/null || true

GPUS=3
CONFIG="configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml"

# Backup old outputs so we don't overwrite the previous run
mkdir -p outputs/mot/cholec20-stage1-supervised-backup
if [ -f outputs/mot/cholec20-stage1-supervised/best.pth.tar ]; then
  cp outputs/mot/cholec20-stage1-supervised/best.pth.tar \
     outputs/mot/cholec20-stage1-supervised-backup/best-old.pth.tar
fi

echo "=== Stage 1 quick test (10 epochs) ==="

PYTHONPATH=. torchrun \
  --standalone \
  --nproc_per_node="${GPUS}" \
  -m core_app.mot.main \
  --fname "${CONFIG}" \
  --devices cuda

echo ""
echo "Quick test done. Check W&B for query diversity."
