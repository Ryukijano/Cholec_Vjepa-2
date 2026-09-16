#!/bin/bash
# Stage 1 — Supervised MOT scaffolding on CholecTrack20 (single GPU)
# Per-track filter predictor + Track manager + Hungarian association.
# Target: HOTA > 35 baseline on CholecTrack20 val.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "=== Surgical MOT — Stage 1 (Supervised) ==="
echo "Repo root: $REPO_ROOT"
echo "Stage   : stage1_supervised"
echo ""
export MOT_DATA_ROOT="/scratch/kcwp264/data/surgi_world_track/cholectrack20"

if [ ! -d "$MOT_DATA_ROOT" ]; then
  LOCAL_MOT_DATA_ROOT="$REPO_ROOT/data/cholectrack20"
  if [ -d "$LOCAL_MOT_DATA_ROOT" ]; then
    export MOT_DATA_ROOT="$LOCAL_MOT_DATA_ROOT"
    echo "[WARN] Default cluster dataset path not mounted. Falling back to local dataset:"
    echo "      $MOT_DATA_ROOT"
  else
    echo "[WARN] Could not find CholecTrack20 at default or local path."
    echo "      Will try configured path from YAML."
  fi
fi

module load miniforge 2>/dev/null || true
conda activate dino_wm 2>/dev/null || true

CFG="configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml"

PYTHONPATH=. python -m core_app.mot.main \
  --fname "$CFG" \
  --devices cuda:0 \
  --debugmode False

echo ""
echo "Stage 1 training complete!"
