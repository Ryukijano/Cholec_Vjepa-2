#!/usr/bin/env bash
# DGX Spark (GB10) — env + data symlink for Cholec_Vjepa-2
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CT20_DEFAULT="/home/aimsgroupuol/AIMSgeneral/Gyanateet_tracking/data/cholectrack20"
CHOLECTRACK20_ROOT="${CHOLECTRACK20_ROOT:-$CT20_DEFAULT}"

cd "$ROOT"
bash scripts/setup_lfs.sh

if [[ -d "$CHOLECTRACK20_ROOT" ]]; then
  ln -sfn "$CHOLECTRACK20_ROOT" "$ROOT/cholec_dataset"
  echo "Linked cholec_dataset -> $CHOLECTRACK20_ROOT"
else
  echo "WARN: CT20 not found at $CHOLECTRACK20_ROOT — download CholecTrack20 first" >&2
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "Ready. PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
